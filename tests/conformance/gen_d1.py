"""Synthetic Liquid d1 model (lfm2 + decision type lfm2-d1) and its references from llama.cpp's llama-server.

The weights are random: what is compared is the whole path (prompt, label tokens, one forward pass, softmax), not the
quality of the answers. Usage:
    python3 gen_d1.py                 # makes the model and, if LLAMA_SERVER points to a llama-server that knows d1,
                                      # asks it the requests of PEDIDOS and writes referencias_d1.json
References were made with llama.cpp 8a1a9b5 (the build that includes PR #30110, lfm2-d1), with ONE slot (-np 1): with several
slots llama-server shares the start of the prompt between the questions of a request and, on this model, the later
questions change by up to ~5e-3 in probability (and not the same each time) compared with answering each one alone;
asking one question per request, or -np 1, gives the same numbers as this engine."""
import hashlib, json, os, subprocess, sys, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_modelos as G

AQUI = os.path.dirname(os.path.abspath(__file__))
NOMBRE = "d1-test-f16"
REF = os.path.join(AQUI, "referencias_d1.json")

# Words the BPE has to know as single tokens: the labels of d1 (yes/no, letters, digits) with and without a space before.
G.CORPUS += "yes\nYes\nYES\nno\nNo\nNO\n" * 200 + (" yes Yes YES no No NO" * 200 + " A B C D E F G H I J" * 200 + " 0 1 2 3 4 5 6 7 8 9" * 60 + "\n") * 3
G.entrenar_bpe.__defaults__ = (700,)      # enough merges for those words to become single tokens


def jinja_str_or_json(name):
    return "{{ " + name + " if " + name + " is string else " + name + " | tojson }}"


def plantilla():
    """The 'systemone' template of d1: copy of D1Model._systemone_template (llama.cpp, conversion/lfm2.py, MIT license)."""
    description = jinja_str_or_json("o.description")
    choice = (
        "{{ '\\n\\nOptions:\\n' }}"
        "{% for o in options %}{{ o.label }} {% if o.description %}" + description + "{% else %}{{ o.key | replace('_', ' ') }}{% endif %}"
        "{% if not loop.last %}{{ '\\n' }}{% endif %}{% endfor %}"
        "{{ '\\n\\nReply with the option code only.' }}"
    )
    noul = (
        "{% set ns = namespace(criteria=false) %}{% for o in options %}{% if o.description is not none %}{% set ns.criteria = true %}{% endif %}{% endfor %}"
        "{% if ns.criteria %}"
        "{% for o in options %}{{ '\\nYes: ' if o.key == 'true' else '\\nNo: ' }}"
        "{% if o.description is none %}None{% else %}" + description + "{% endif %}{% endfor %}{% endif %}"
        "{{ '\\n\\nReply with yes or no only.' }}"
    )
    score = (
        "{{ '\\n\\n' }}{% for o in options %}{{ o.key }} " + description + "{{ '\\n' }}{% endfor %}"
        "{{ '\\nReply with a single digit 0-' }}{{ options | length - 1 }}{{ ' only.' }}"
    )
    return (
        "<|startoftext|><|im_start|>user\n"
        "{% for image in images %}{{ image }}{% endfor %}"
        "{% if state is not none %}{% if state is string %}{{ state }}{% else %}{{ state | tojson(indent=2) }}{% endif %}"
        "{{ '\\n\\n\\nQUESTION:\\n' }}{% endif %}"
        + jinja_str_or_json("instructions")
        + "{% if type == 'choice' %}" + choice + "{% elif type == 'noul' %}" + noul + "{% else %}" + score + "{% endif %}"
        "{{ '<|im_end|>\\n<|im_start|>assistant\\n' }}"
    )


def extra(w):
    w.add_chat_template([{"name": "systemone", "template": plantilla()}])
    w.add_string("lfm2.decision.type", "lfm2-d1")


def crear():
    # escala_salida < 1: flat probabilities, so the comparison with llama.cpp is not hidden by saturated answers (0.9999)
    return G.crear(NOMBRE, tipo_mat="F16", semilla=7, add_bos=False, extra=extra, escala_salida=0.12)


def opciones30():
    return {"opcion_%d" % i: (None if i % 3 == 0 else "descripcion %d" % i) for i in range(30)}


PEDIDOS = {
    "choice_basico": {"state": "El cliente dice: me cobraron dos veces el pedido de la semana pasada.",
                      "questions": {"ruta": {"type": "choice", "instructions": "Que equipo lo atiende?",
                                             "criteria": {"billing": None, "shipping": None, "tech_support": "problemas tecnicos"}}}},
    "choice_letras": {"state": "texto corto", "questions": {"q": {"type": "choice", "instructions": "Elige.",
                                                                 "criteria": {"A": "primera", "B": "segunda", "C": None}}}},
    "noul_simple": {"state": "El servidor responde en el puerto 1234.", "questions": {"q": {"type": "noul", "instructions": "Es una frase sobre redes?"}}},
    "noul_criterios": {"state": "Hola", "questions": {"q": {"type": "noul", "instructions": "Esta enojado?",
                                                           "criteria": {"true": "usa mayusculas o insultos"}}}},
    "score": {"state": "La base de datos guarda la informacion.", "questions": {"q": {"type": "score", "instructions": "Que tan urgente es?",
                                                                                     "criteria": ["puede esperar", "esta semana", "hoy", "ahora mismo"]}}},
    "score_no_texto": {"state": "x", "questions": {"q": {"type": "score", "instructions": "Nota", "criteria": [{"min": 0}, [1, 2.5], None]}}},
    "estado_objeto": {"state": {"cliente": "Ñandú Pérez", "total": 12.5, "items": [1, 2, {"a": "b\"c"}], "vacio": {}, "lista": [], "ok": True, "nada": None, "nota": "línea1\nlínea2\t!"},
                      "questions": {"q": {"type": "noul", "instructions": {"pregunta": "es grande?"}}}},
    "estado_nulo": {"state": None, "questions": {"q": {"type": "noul", "instructions": "Hay algo?"}}},
    "choice_30": {"state": "muchas opciones", "questions": {"q": {"type": "choice", "instructions": "Cual?", "criteria": opciones30()}}},
    "varias": {"state": "El cliente pide un reembolso hoy mismo.",
               "questions": {"ruta": {"type": "choice", "instructions": "Equipo?", "criteria": {"ventas": None, "soporte": "ayuda"}},
                             "enojado": {"type": "noul", "instructions": "Esta enojado?"},
                             "urgencia": {"type": "score", "instructions": "Urgencia?", "criteria": ["baja", "media", "alta"]}}},
}


def hash_modelo(ruta):
    return hashlib.sha256(open(ruta, "rb").read()).hexdigest()


def referencias(ruta, binario):
    puerto = 18390
    proc = subprocess.Popen([binario, "-m", ruta, "--port", str(puerto), "-c", "4096", "--no-warmup", "-ngl", "0", "-t", "2", "-np", "1"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = "http://127.0.0.1:%d" % puerto
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(base + "/health", timeout=2).read(); break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("llama-server did not start")
        refs = {}
        for nombre, cuerpo in PEDIDOS.items():
            req = urllib.request.Request(base + "/v1/systemone", data=json.dumps(cuerpo).encode(), headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    refs[nombre] = json.loads(r.read())
                    refs[nombre]["model"] = NOMBRE      # llama-server answers with the path of the file
            except urllib.error.HTTPError as e:
                refs[nombre] = {"error": json.loads(e.read() or b"{}"), "http": e.code}
            print(nombre, "->", json.dumps(refs[nombre])[:150])
        return refs
    finally:
        proc.terminate()


if __name__ == "__main__":
    ruta = crear()
    print("model:", ruta, os.path.getsize(ruta), "bytes")
    binario = os.environ.get("LLAMA_SERVER")
    if binario:
        refs = referencias(ruta, binario)
        with open(REF, "w", encoding="utf-8") as f:
            json.dump({"llama_cpp": os.environ.get("LLAMA_CPP_COMMIT", "8a1a9b5"), "sha256_modelo": hash_modelo(ruta), "pedidos": PEDIDOS, "respuestas": refs},
                      f, ensure_ascii=False, indent=1)
        print("written", REF)
