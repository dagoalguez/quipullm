import os, sys, json, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from templates import Plantilla, ErrorPlantilla, NoSoportado
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tpl_examples import T

def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

def ref(src, **kw):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    def raise_exception(m): raise ErrorPlantilla(m)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda f: __import__("time").strftime(f)
    return env.from_string(src).render(**kw)

CASOS = {
  "user": [{"role": "user", "content": "Hola"}],
  "sys+user": [{"role": "system", "content": "Eres breve."}, {"role": "user", "content": "¿Cuánto es 12 por 11?"}],
  "multi": [{"role": "system", "content": "Responde en español."}, {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "  Hola, ¿en qué ayudo? "}, {"role": "user", "content": "Cuánto es 2+2"}],
  "think": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "<think>x</think>\n\nb"}, {"role": "user", "content": "c"}],
  "partes": [{"role": "system", "content": [{"type": "text", "text": "Sé breve."}]}, {"role": "user", "content": [{"type": "text", "text": "  Mira "}, {"type": "image"}]}],
  "alterna-mal": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
  "tools": ([{"role": "user", "content": "clima"}], {"tools": [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]}),
}
fallas = 0; ok = 0
for nombre, src in T.items():
    p = Plantilla(src, bos="<BOS>", eos="<EOS>")
    for cn, caso in CASOS.items():
        extra = {}
        msgs = caso
        if isinstance(caso, tuple): msgs, extra = caso
        for agp in (True, False):
            def correr(f):
                try: return ("ok", f())
                except ErrorPlantilla as e: return ("err", str(e)[:40])
                except NoSoportado as e: return ("nosop", str(e))
                except Exception as e: return ("exc", type(e).__name__ + ":" + str(e)[:60])
            a = correr(lambda: p.renderizar(copy.deepcopy(msgs), add_generation_prompt=agp, **extra))
            b = correr(lambda: ref(src, messages=copy.deepcopy(msgs), add_generation_prompt=agp, bos_token="<BOS>", eos_token="<EOS>", **extra))
            if a[0] == "err" and b[0] in ("err", "exc"): ok += 1; continue
            if a == b: ok += 1
            else:
                fallas += 1
                print("FALLA", nombre, cn, agp, "\n  mio:", repr(a)[:400], "\n  ref:", repr(b)[:400])
print(ok, "OK,", fallas, "fallas")
sys.exit(1 if fallas else 0)
