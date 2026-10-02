"""Mini Jinja template interpreter for the chat templates of GGUF files (standard library only).

Runs the `tokenizer.chat_template` shipped with each model, using the same configuration as llama.cpp
(trim_blocks=True, lstrip_blocks=True). Covers the subset of Jinja used by chat templates
(if/for/set, common filters and tests, namespace, string and dict methods). If a template uses something
that is not supported, NoSoportado is raised and the server falls back to its own built-in template.
"""
import json
import re
import time


class NoSoportado(Exception):
    pass


class ErrorPlantilla(Exception):
    """raise_exception() from the template, or a runtime error."""


class _Indefinido:
    def __init__(self, nombre=""):
        self.nombre = nombre

    def __bool__(self):
        return False

    def __str__(self):
        return ""

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __eq__(self, otro):
        return isinstance(otro, _Indefinido)

    def __hash__(self):
        return 0


INDEFINIDO = _Indefinido()


class _Namespace(dict):
    pass


# ---------------------------------------------------------------------------------------------- lexing
_RE_TAG = re.compile(r"\{\{-?|\{%-?|\{#-?")


def _tokenizar(src):
    """Returns a list of ('texto', s) | ('var', s) | ('bloque', s), applying -, trim_blocks and lstrip_blocks."""
    out = []
    pos = 0
    n = len(src)
    recortar_sig = False       # '-%}' or pending trim_blocks applied to the following text
    quitar_nl = False
    while pos < n:
        m = _RE_TAG.search(src, pos)
        if not m:
            texto = src[pos:]
            if recortar_sig:
                texto = texto.lstrip()
            elif quitar_nl and texto.startswith("\n"):
                texto = texto[1:]
            out.append(["texto", texto])
            break
        texto = src[pos:m.start()]
        abre = m.group()
        nl_quitado = False
        if recortar_sig:
            texto = texto.lstrip()
        elif quitar_nl and texto.startswith("\n"):
            texto = texto[1:]
            nl_quitado = True
        quitar_nl = recortar_sig = False
        es_bloque = abre.startswith("{%") or abre.startswith("{#")
        if abre.endswith("-"):
            texto = texto.rstrip()
        elif es_bloque:
            # lstrip_blocks: strip spaces/tabs between the start of the line and the tag
            i = len(texto)
            while i > 0 and texto[i - 1] in " \t":
                i -= 1
            if i < len(texto):
                inicio_linea = (i > 0 and texto[i - 1] == "\n") or (i == 0 and (nl_quitado or not out))
                if inicio_linea:
                    texto = texto[:i]
        if texto:
            out.append(["texto", texto])
        # find the closing delimiter, respecting strings
        cierre = "}}" if abre.startswith("{{") else ("%}" if abre.startswith("{%") else "#}")
        j = m.end()
        comilla = None
        while j < n:
            c = src[j]
            if comilla:
                if c == "\\":
                    j += 2
                    continue
                if c == comilla:
                    comilla = None
            elif c in "\"'" and cierre != "#}":
                comilla = c
            elif src.startswith(cierre, j) or (c == "-" and src.startswith(cierre, j + 1)):
                break
            j += 1
        if j >= n:
            raise NoSoportado("unclosed tag")
        contenido = src[m.end():j]
        if src[j] == "-":
            recortar_sig = True
            j += 1
        pos = j + 2
        if abre.startswith("{{"):
            out.append(["var", contenido.strip()])
        elif abre.startswith("{%"):
            out.append(["bloque", contenido.strip()])
            if not recortar_sig:
                quitar_nl = True
        else:
            if not recortar_sig:
                quitar_nl = True
    return out


# ---------------------------------------------------------------------------------------------- expressions
_RE_EXPR = re.compile(r"""
    \s*(?:
      (?P<num>\d+\.\d+|\d+)
     |(?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
     |(?P<name>[A-Za-z_][A-Za-z_0-9]*)
     |(?P<op>\*\*|//|==|!=|<=|>=|[-+*/%<>=~|.,:()\[\]{}])
    )""", re.X)


def _tok_expr(s):
    toks = []
    pos = 0
    s = s.rstrip()
    while pos < len(s):
        m = _RE_EXPR.match(s, pos)
        if not m or m.end() == pos:
            raise NoSoportado("unrecognized expression: " + s[pos:pos + 20])
        if m.lastgroup == "num":
            v = m.group("num")
            toks.append(("lit", float(v) if "." in v else int(v)))
        elif m.lastgroup == "str":
            raw = m.group("str")
            q = raw[0]
            cuerpo = raw[1:-1]
            cuerpo = re.sub(r"\\(.)", lambda mm: {"n": "\n", "t": "\t", "r": "\r"}.get(mm.group(1), mm.group(1)), cuerpo)
            toks.append(("lit", cuerpo))
            _ = q
        elif m.lastgroup == "name":
            toks.append(("name", m.group("name")))
        else:
            toks.append(("op", m.group("op")))
        pos = m.end()
    return toks


class _Parser:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def ver(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def sig(self):
        v = self.ver()
        self.i += 1
        return v

    def es(self, tipo, val=None):
        t = self.ver()
        return t[0] == tipo and (val is None or t[1] == val)

    def esperar(self, tipo, val=None):
        if not self.es(tipo, val):
            raise NoSoportado("expected %s %s, found %s" % (tipo, val, self.ver()))
        return self.sig()

    def fin(self):
        return self.i >= len(self.t)

    # precedence: ternary < or < and < not < comparison < ~ < +- < */ // % < unary < filter < postfix
    def expr(self):
        return self.ternario()

    def ternario(self):
        e = self.o_or()
        while self.es("name", "if"):
            self.sig()
            c = self.o_or()
            otro = ("lit", INDEFINIDO)
            if self.es("name", "else"):
                self.sig()
                otro = self.ternario()
            e = ("ternario", c, e, otro)
        return e

    def o_or(self):
        e = self.o_and()
        while self.es("name", "or"):
            self.sig()
            e = ("or", e, self.o_and())
        return e

    def o_and(self):
        e = self.o_not()
        while self.es("name", "and"):
            self.sig()
            e = ("and", e, self.o_not())
        return e

    def o_not(self):
        if self.es("name", "not"):
            self.sig()
            return ("not", self.o_not())
        return self.comparacion()

    def comparacion(self):
        e = self.concat()
        while True:
            t = self.ver()
            if t[0] == "op" and t[1] in ("==", "!=", "<", ">", "<=", ">="):
                self.sig()
                e = ("cmp", t[1], e, self.concat())
            elif t == ("name", "in"):
                self.sig()
                e = ("in", e, self.concat())
            elif t == ("name", "not") and self.i + 1 < len(self.t) and self.t[self.i + 1] == ("name", "in"):
                self.sig(); self.sig()
                e = ("not", ("in", e, self.concat()))
            elif t == ("name", "is"):
                self.sig()
                neg = False
                if self.es("name", "not"):
                    self.sig()
                    neg = True
                nombre = self.esperar("name")[1]
                arg = None
                if self.es("op", "("):
                    self.sig()
                    arg = self.expr()
                    self.esperar("op", ")")
                elif self.ver()[0] in ("lit",) or (self.ver()[0] == "name" and self.ver()[1] not in ("and", "or", "if", "else", "in", "is")):
                    arg = self.postfijo()
                e = ("test", nombre, e, arg)
                if neg:
                    e = ("not", e)
            else:
                return e

    def concat(self):
        e = self.suma()
        while self.es("op", "~"):
            self.sig()
            e = ("concat", e, self.suma())
        return e

    def suma(self):
        e = self.prod()
        while self.ver()[0] == "op" and self.ver()[1] in ("+", "-"):
            op = self.sig()[1]
            e = ("bin", op, e, self.prod())
        return e

    def prod(self):
        e = self.unario()
        while self.ver()[0] == "op" and self.ver()[1] in ("*", "/", "//", "%"):
            op = self.sig()[1]
            e = ("bin", op, e, self.unario())
        return e

    def unario(self):
        if self.es("op", "-"):
            self.sig()
            return ("neg", self.unario())
        if self.es("op", "+"):
            self.sig()
            return self.unario()
        return self.filtro()

    def filtro(self):
        e = self.postfijo()
        while self.es("op", "|"):
            self.sig()
            nombre = self.esperar("name")[1]
            while self.es("op", ".") and self.i + 1 < len(self.t) and self.t[self.i + 1][0] == "name":
                self.sig()
                nombre += "." + self.sig()[1]
            args, kw = [], {}
            if self.es("op", "("):
                args, kw = self.args()
            e = ("filtro", nombre, e, args, kw)
        return e

    def args(self):
        self.esperar("op", "(")
        args, kw = [], {}
        while not self.es("op", ")"):
            if self.ver()[0] == "name" and self.i + 1 < len(self.t) and self.t[self.i + 1] == ("op", "="):
                k = self.sig()[1]
                self.sig()
                kw[k] = self.expr()
            else:
                args.append(self.expr())
            if self.es("op", ","):
                self.sig()
        self.esperar("op", ")")
        return args, kw

    def postfijo(self):
        e = self.primario()
        while True:
            if self.es("op", "."):
                self.sig()
                e = ("attr", e, self.esperar("name")[1])
            elif self.es("op", "["):
                self.sig()
                a = None
                if not self.es("op", ":"):
                    a = self.expr()
                if self.es("op", ":"):
                    self.sig()
                    b = None
                    if not self.es("op", "]") and not self.es("op", ":"):
                        b = self.expr()
                    paso = None
                    if self.es("op", ":"):
                        self.sig()
                        if not self.es("op", "]"):
                            paso = self.expr()
                    self.esperar("op", "]")
                    e = ("slice", e, a, b, paso)
                else:
                    self.esperar("op", "]")
                    e = ("item", e, a)
            elif self.es("op", "("):
                args, kw = self.args()
                e = ("call", e, args, kw)
            else:
                return e

    def primario(self):
        t = self.sig()
        if t[0] == "lit":
            return ("lit", t[1])
        if t[0] == "name":
            v = t[1]
            if v in ("true", "True"):
                return ("lit", True)
            if v in ("false", "False"):
                return ("lit", False)
            if v in ("none", "None"):
                return ("lit", None)
            return ("var", v)
        if t == ("op", "("):
            e = self.expr()
            if self.es("op", ","):
                elems = [e]
                while self.es("op", ","):
                    self.sig()
                    if self.es("op", ")"):
                        break
                    elems.append(self.expr())
                self.esperar("op", ")")
                return ("lista", elems)
            self.esperar("op", ")")
            return e
        if t == ("op", "["):
            elems = []
            while not self.es("op", "]"):
                elems.append(self.expr())
                if self.es("op", ","):
                    self.sig()
            self.esperar("op", "]")
            return ("lista", elems)
        if t == ("op", "{"):
            pares = []
            while not self.es("op", "}"):
                k = self.expr()
                self.esperar("op", ":")
                pares.append((k, self.expr()))
                if self.es("op", ","):
                    self.sig()
            self.esperar("op", "}")
            return ("dict", pares)
        raise NoSoportado("unexpected token %s" % (t,))


def _parse_expr(s):
    p = _Parser(_tok_expr(s))
    e = p.expr()
    if not p.fin():
        raise NoSoportado("extra tokens in: " + s)
    return e


# ---------------------------------------------------------------------------------------------- statements
def _construir(tokens):
    """Tree: list of nodes. Node: ('texto', s) ('salida', expr) ('if', [(cond, cuerpo)], sino) ('for', ...) ('set', ...)"""
    pos = [0]

    def bloque(fines):
        nodos = []
        while pos[0] < len(tokens):
            tipo, cont = tokens[pos[0]]
            if tipo == "texto":
                nodos.append(("texto", cont)); pos[0] += 1
            elif tipo == "var":
                nodos.append(("salida", _parse_expr(cont))); pos[0] += 1
            else:
                palabra = cont.split(None, 1)[0] if cont else ""
                if palabra in fines:
                    return nodos, palabra
                pos[0] += 1
                resto = cont[len(palabra):].strip()
                if palabra == "if":
                    ramas = []
                    cond = _parse_expr(resto)
                    while True:
                        cuerpo, fin = bloque({"elif", "else", "endif"})
                        ramas.append((cond, cuerpo))
                        c2 = tokens[pos[0]][1]
                        pos[0] += 1
                        if fin == "elif":
                            cond = _parse_expr(c2[len("elif"):].strip())
                        elif fin == "else":
                            sino, _ = bloque({"endif"})
                            pos[0] += 1
                            nodos.append(("if", ramas, sino))
                            break
                        else:
                            nodos.append(("if", ramas, []))
                            break
                elif palabra == "for":
                    m = re.match(r"(.+?)\s+in\s+(.+)$", resto, re.S)
                    if not m:
                        raise NoSoportado("invalid for")
                    vars_ = [v.strip() for v in m.group(1).split(",")]
                    it = m.group(2)
                    cond = None
                    mm = re.match(r"(.+?)\s+if\s+(.+)$", it, re.S)
                    if mm:
                        it, cond = mm.group(1), _parse_expr(mm.group(2))
                    recursivo = False
                    if it.rstrip().endswith(" recursive"):
                        raise NoSoportado("for recursive")
                    cuerpo, fin = bloque({"else", "endfor"})
                    pos[0] += 1
                    sino = []
                    if fin == "else":
                        sino, _ = bloque({"endfor"})
                        pos[0] += 1
                    _ = recursivo
                    nodos.append(("for", vars_, _parse_expr(it), cond, cuerpo, sino))
                elif palabra == "set":
                    m = re.match(r"([A-Za-z_][\w.]*(?:\s*,\s*[A-Za-z_]\w*)*)\s*=(?!=)\s*(.+)$", resto, re.S)
                    if m:
                        nodos.append(("set", [v.strip() for v in m.group(1).split(",")], _parse_expr(m.group(2))))
                    else:
                        # {% set x %} ... {% endset %}
                        nombre = resto.strip()
                        cuerpo, _ = bloque({"endset"})
                        pos[0] += 1
                        nodos.append(("setbloque", nombre, cuerpo))
                elif palabra in ("macro", "call", "include", "import", "from", "extends", "block", "filter", "with"):
                    raise NoSoportado("{% " + palabra + " %} not supported")
                elif palabra in ("generation", "endgeneration"):
                    pass
                else:
                    raise NoSoportado("unknown statement: " + palabra)
        if fines:
            raise NoSoportado("unclosed block (%s)" % ",".join(fines))
        return nodos, None

    nodos, _ = bloque(set())
    return nodos


# ---------------------------------------------------------------------------------------------- execution
def _verdad(v):
    if isinstance(v, _Indefinido):
        return False
    return bool(v)


def _texto(v):
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (list, dict)) and not isinstance(v, _Namespace):
        return _py_repr(v)
    return str(v)


def _py_repr(v):
    if isinstance(v, str):
        return "'" + v.replace("\\", "\\\\").replace("'", "\\'") + "'"
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, list):
        return "[" + ", ".join(_py_repr(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(_py_repr(k) + ": " + _py_repr(x) for k, x in v.items()) + "}"
    return str(v)


def _tojson(v, indent=None, ensure_ascii=False, sort_keys=False, separators=None):
    return json.dumps(v, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys,
                      separators=separators)


class _Bucle:
    def __init__(self, items):
        self.items = items
        self.i = 0

    def attr(self, nombre):
        n = len(self.items)
        if nombre == "index":
            return self.i + 1
        if nombre == "index0":
            return self.i
        if nombre == "first":
            return self.i == 0
        if nombre == "last":
            return self.i == n - 1
        if nombre == "length":
            return n
        if nombre == "revindex":
            return n - self.i
        if nombre == "revindex0":
            return n - self.i - 1
        if nombre == "previtem":
            return self.items[self.i - 1] if self.i > 0 else INDEFINIDO
        if nombre == "nextitem":
            return self.items[self.i + 1] if self.i < n - 1 else INDEFINIDO
        return INDEFINIDO


class _Ejecutor:
    def __init__(self, globales):
        self.pilas = [dict(globales)]

    def buscar(self, nombre):
        for e in reversed(self.pilas):
            if nombre in e:
                return e[nombre]
        return INDEFINIDO

    def asignar(self, nombre, valor):
        self.pilas[-1][nombre] = valor

    # -- evaluation
    def ev(self, e):
        k = e[0]
        if k == "lit":
            return e[1]
        if k == "var":
            return self.buscar(e[1])
        if k == "lista":
            return [self.ev(x) for x in e[1]]
        if k == "dict":
            return {self.ev(a): self.ev(b) for a, b in e[1]}
        if k == "attr":
            return self.atributo(self.ev(e[1]), e[2])
        if k == "item":
            return self.item(self.ev(e[1]), self.ev(e[2]))
        if k == "slice":
            base = self.ev(e[1])
            a = self.ev(e[2]) if e[2] else None
            b = self.ev(e[3]) if e[3] else None
            paso = self.ev(e[4]) if e[4] else None
            if isinstance(base, _Indefinido):
                return INDEFINIDO
            return base[slice(a, b, paso)]
        if k == "call":
            return self.llamar(e)
        if k == "filtro":
            return self.filtro(e)
        if k == "test":
            return self.test(e)
        if k == "not":
            return not _verdad(self.ev(e[1]))
        if k == "and":
            a = self.ev(e[1])
            return self.ev(e[2]) if _verdad(a) else a
        if k == "or":
            a = self.ev(e[1])
            return a if _verdad(a) else self.ev(e[2])
        if k == "ternario":
            return self.ev(e[2]) if _verdad(self.ev(e[1])) else self.ev(e[3])
        if k == "neg":
            return -self.ev(e[1])
        if k == "concat":
            return _texto(self.ev(e[1])) + _texto(self.ev(e[2]))
        if k == "cmp":
            a, b = self.ev(e[2]), self.ev(e[3])
            op = e[1]
            try:
                if op == "==":
                    return a == b
                if op == "!=":
                    return a != b
                if op == "<":
                    return a < b
                if op == ">":
                    return a > b
                if op == "<=":
                    return a <= b
                return a >= b
            except TypeError:
                raise ErrorPlantilla("invalid comparison")
        if k == "in":
            a, b = self.ev(e[1]), self.ev(e[2])
            if isinstance(b, _Indefinido):
                return False
            try:
                return a in b
            except TypeError:
                return False
        if k == "bin":
            a, b = self.ev(e[2]), self.ev(e[3])
            op = e[1]
            try:
                if op == "+":
                    if isinstance(a, _Indefinido) or isinstance(b, _Indefinido):
                        raise ErrorPlantilla("addition with an undefined value")
                    return a + b
                if op == "-":
                    return a - b
                if op == "*":
                    return a * b
                if op == "/":
                    return a / b
                if op == "//":
                    return a // b
                return a % b
            except TypeError:
                raise ErrorPlantilla("invalid operation %r %s %r" % (a, op, b))
        raise NoSoportado("node " + k)

    def atributo(self, obj, nombre):
        if isinstance(obj, _Bucle):
            return obj.attr(nombre)
        if isinstance(obj, dict):
            if nombre in obj:
                return obj[nombre]
            # dict methods (items, keys, values, get) are resolved in llamar()
            if nombre in ("items", "keys", "values", "get", "update", "pop"):
                return ("metodo", obj, nombre)
            return INDEFINIDO
        if isinstance(obj, _Indefinido):
            return INDEFINIDO
        if isinstance(obj, (str, list)):
            return ("metodo", obj, nombre)
        return INDEFINIDO

    def item(self, obj, idx):
        if isinstance(obj, _Indefinido):
            return INDEFINIDO
        if isinstance(obj, dict):
            return obj.get(idx, INDEFINIDO)
        try:
            return obj[idx]
        except (IndexError, TypeError, KeyError):
            return INDEFINIDO

    def llamar(self, e):
        f = e[1]
        args = [self.ev(a) for a in e[2]]
        kw = {k: self.ev(v) for k, v in e[3].items()}
        if f[0] == "var":
            n = f[1]
            if n == "namespace":
                return _Namespace(kw)
            if n == "raise_exception":
                raise ErrorPlantilla(args[0] if args else "raise_exception")
            if n == "range":
                return list(range(*[int(a) for a in args]))
            if n == "strftime_now":
                return time.strftime(args[0]) if args else ""
            if n in ("dict",):
                return dict(kw)
            if n == "lipsum" or n == "cycler" or n == "joiner":
                raise NoSoportado(n)
            v = self.buscar(n)
            if callable(v):
                return v(*args, **kw)
            raise NoSoportado("call to " + n)
        obj = self.ev(f)
        if isinstance(obj, tuple) and obj and obj[0] == "metodo":
            return self.metodo(obj[1], obj[2], args, kw)
        if callable(obj):
            return obj(*args, **kw)
        raise ErrorPlantilla("value is not callable")

    def metodo(self, o, n, a, kw):
        if isinstance(o, str):
            if n == "split":
                return o.split(*a) if a else o.split()
            if n in ("strip", "lstrip", "rstrip"):
                return getattr(o, n)(*a)
            if n in ("startswith", "endswith", "replace", "upper", "lower", "title", "capitalize", "find", "count", "join", "format"):
                return getattr(o, n)(*a)
            if n == "splitlines":
                return o.splitlines()
        if isinstance(o, dict):
            if n == "items":
                return list(o.items())
            if n == "keys":
                return list(o.keys())
            if n == "values":
                return list(o.values())
            if n == "get":
                return o.get(a[0], a[1] if len(a) > 1 else None)
        if isinstance(o, list):
            if n == "append":
                o.append(a[0]); return None
            if n == "index":
                return o.index(*a)
            if n == "count":
                return o.count(*a)
        raise NoSoportado("method %s" % n)

    def test(self, e):
        return self.probar(e[1], self.ev(e[2]), (self.ev(e[3]) if e[3] is not None else None))

    def probar(self, nombre, v, arg):
        if nombre == "defined":
            return not isinstance(v, _Indefinido)
        if nombre == "undefined":
            return isinstance(v, _Indefinido)
        if nombre == "none":
            return v is None
        if nombre == "string":
            return isinstance(v, str)
        if nombre in ("mapping",):
            return isinstance(v, dict)
        if nombre in ("iterable", "sequence"):
            return isinstance(v, (list, tuple, str, dict))
        if nombre == "number":
            return isinstance(v, (int, float)) and not isinstance(v, bool)
        if nombre == "integer":
            return isinstance(v, int) and not isinstance(v, bool)
        if nombre == "float":
            return isinstance(v, float)
        if nombre == "boolean":
            return isinstance(v, bool)
        if nombre == "true":
            return v is True
        if nombre == "false":
            return v is False
        if nombre == "odd":
            return v % 2 == 1
        if nombre == "even":
            return v % 2 == 0
        if nombre == "divisibleby":
            return v % arg == 0
        if nombre in ("equalto", "eq", "sameas", "=="):
            return v == arg
        if nombre in ("ne", "!="):
            return v != arg
        if nombre in ("in",):
            return v in arg
        if nombre in ("lt", "<"):
            return v < arg
        if nombre in ("gt", ">"):
            return v > arg
        if nombre == "callable":
            return callable(v)
        raise NoSoportado("test " + nombre)

    def filtro(self, e):
        nombre, v = e[1], self.ev(e[2])
        a = [self.ev(x) for x in e[3]]
        kw = {k: self.ev(x) for k, x in e[4].items()}
        if nombre == "trim":
            return _texto(v).strip() if not a else _texto(v).strip(a[0])
        if nombre == "length" or nombre == "count":
            return len(v)
        if nombre == "first":
            return v[0] if len(v) else INDEFINIDO
        if nombre == "last":
            return v[-1] if len(v) else INDEFINIDO
        if nombre == "join":
            sep = a[0] if a else kw.get("d", "")
            return sep.join(_texto(x) for x in v)
        if nombre == "lower":
            return _texto(v).lower()
        if nombre == "upper":
            return _texto(v).upper()
        if nombre in ("default", "d"):
            d = a[0] if a else ""
            boolean = a[1] if len(a) > 1 else kw.get("boolean", False)
            if isinstance(v, _Indefinido) or (boolean and not v):
                return d
            return v
        if nombre == "tojson":
            return _tojson(v, indent=kw.get("indent"))
        if nombre == "string":
            return _texto(v)
        if nombre == "int":
            try:
                return int(v)
            except (TypeError, ValueError):
                return a[0] if a else 0
        if nombre == "float":
            return float(v)
        if nombre == "list":
            return list(v)
        if nombre == "reverse":
            return list(reversed(v))
        if nombre == "replace":
            return _texto(v).replace(a[0], a[1])
        if nombre in ("safe", "e", "escape", "forceescape"):
            return v
        if nombre == "capitalize":
            return _texto(v).capitalize()
        if nombre == "title":
            return _texto(v).title()
        if nombre == "items":
            return list(v.items()) if isinstance(v, dict) else []
        if nombre == "indent":
            n = a[0] if a else 4
            return ("\n" + " " * n).join(_texto(v).split("\n"))
        if nombre == "length":
            return len(v)
        if nombre == "abs":
            return abs(v)
        if nombre in ("selectattr", "rejectattr"):
            atr = a[0]
            prueba = a[1] if len(a) > 1 else None
            arg = a[2] if len(a) > 2 else None
            out = []
            for x in v:
                val = x
                for parte in str(atr).split("."):
                    val = self.atributo(val, parte) if not isinstance(val, dict) else val.get(parte, INDEFINIDO)
                ok = _verdad(val) if prueba is None else self.probar(prueba, val, arg)
                if ok == (nombre == "selectattr"):
                    out.append(x)
            return out
        if nombre == "map":
            atr = kw.get("attribute")
            if atr is None:
                raise NoSoportado("map without attribute")
            out = []
            for x in v:
                val = x
                for parte in str(atr).split("."):
                    val = val.get(parte, INDEFINIDO) if isinstance(val, dict) else INDEFINIDO
                out.append(val)
            return out
        if nombre in ("select", "reject"):
            prueba = a[0] if a else None
            arg = a[1] if len(a) > 1 else None
            return [x for x in v if (_verdad(x) if prueba is None else self.probar(prueba, x, arg)) == (nombre == "select")]
        if nombre == "sort":
            return sorted(v)
        if nombre == "unique":
            vistos = []
            for x in v:
                if x not in vistos:
                    vistos.append(x)
            return vistos
        if nombre == "sum":
            return sum(v)
        if nombre in ("min", "max"):
            return (min if nombre == "min" else max)(v)
        raise NoSoportado("filter " + nombre)

    # -- statements
    def ejecutar(self, nodos, salida):
        for n in nodos:
            k = n[0]
            if k == "texto":
                salida.append(n[1])
            elif k == "salida":
                salida.append(_texto(self.ev(n[1])))
            elif k == "if":
                for cond, cuerpo in n[1]:
                    if _verdad(self.ev(cond)):
                        self.ejecutar(cuerpo, salida)
                        break
                else:
                    self.ejecutar(n[2], salida)
            elif k == "for":
                vars_, it, cond, cuerpo, sino = n[1], n[2], n[3], n[4], n[5]
                items = self.ev(it)
                if isinstance(items, dict):
                    items = list(items.keys())
                items = list(items) if not isinstance(items, _Indefinido) else []
                self.pilas.append({})
                if cond is not None:
                    filtrados = []
                    for x in items:
                        self._vinc(vars_, x)
                        if _verdad(self.ev(cond)):
                            filtrados.append(x)
                    items = filtrados
                if not items:
                    self.pilas.pop()
                    self.ejecutar(sino, salida)
                    continue
                bucle = _Bucle(items)
                for i, x in enumerate(items):
                    bucle.i = i
                    self.pilas[-1].clear()
                    self.pilas[-1]["loop"] = bucle
                    self._vinc(vars_, x)
                    self.ejecutar(cuerpo, salida)
                self.pilas.pop()
            elif k == "set":
                valor = self.ev(n[2])
                nombres = n[1]
                if len(nombres) == 1:
                    self._set(nombres[0], valor)
                else:
                    for nm, v in zip(nombres, valor):
                        self._set(nm, v)
            elif k == "setbloque":
                parcial = []
                self.ejecutar(n[2], parcial)
                self._set(n[1], "".join(parcial))
            else:
                raise NoSoportado("node " + k)

    def _vinc(self, vars_, x):
        if len(vars_) == 1:
            self.pilas[-1][vars_[0]] = x
        else:
            for nm, v in zip(vars_, x):
                self.pilas[-1][nm] = v

    def _set(self, nombre, valor):
        if "." in nombre:
            base, atr = nombre.split(".", 1)
            ns = self.buscar(base)
            if not isinstance(ns, _Namespace):
                raise ErrorPlantilla("assignment to an attribute of something that is not a namespace")
            ns[atr] = valor
        else:
            # variables from {% set %} inside a for do not leak out of the for, same as in Jinja
            self.pilas[-1][nombre] = valor


class Plantilla:
    def __init__(self, src, bos="", eos=""):
        self.src = src
        self.bos = bos
        self.eos = eos
        self.nodos = _construir(_tokenizar(src))

    def renderizar(self, mensajes, add_generation_prompt=True, **extra):
        g = {"messages": mensajes, "add_generation_prompt": add_generation_prompt,
             "bos_token": self.bos, "eos_token": self.eos}
        g.update(extra)
        ej = _Ejecutor(g)
        salida = []
        ej.ejecutar(self.nodos, salida)
        return "".join(salida)
