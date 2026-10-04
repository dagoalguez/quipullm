# -*- coding: utf-8 -*-
"""
api_tests.py - Server test suite (standard library only).

Starts server.py on a free port with a FAKE models folder (minimal GGUF files
with the same names as the real models) and a SIMULATED engine in Python,
and tests the whole API: names, chat, streaming, completions, stop, reasoning,
errors, queue, cancellation and the bridge to the engine.

Usage:  python tests/api_tests.py   (or python tests/run_all.py)
Needs neither the GPU nor the real models; it works in a temporary folder.
"""
import http.client
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root

# Same files as on the server PC (architecture as reported by LM Studio)
MODELOS = [
    ("bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf", "deepseek2"),
    ("Kondara/Qwen2.5-Coder-7B-Instruct-Q4_K_M-GGUF/qwen2.5-coder-7b-instruct-q4_k_m.gguf", "qwen2"),
    ("LiquidAI/LFM2-1.2B-Extract-GGUF/LFM2-1.2B-Extract-Q8_0.gguf", "lfm2"),
    ("LiquidAI/LFM2-1.2B-RAG-GGUF/LFM2-1.2B-RAG-Q8_0.gguf", "lfm2"),
    ("LiquidAI/LFM2-350M-Math-GGUF/LFM2-350M-Math-Q8_0.gguf", "lfm2"),
    ("LiquidAI/LFM2.5-1.2B-Thinking-GGUF/LFM2.5-1.2B-Thinking-Q8_0.gguf", "lfm2"),
    ("LiquidAI/LFM2.5-VL-1.6B-GGUF/LFM2.5-VL-1.6B-Q8_0.gguf", "lfm2"),
    ("LiquidAI/LFM2.5-VL-1.6B-GGUF/mmproj-LFM2.5-VL-1.6b-F16.gguf", "clip"),
    ("lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf", "qwen2"),
    ("lmstudio-community/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf", "gemma3"),
    ("lmstudio-community/gemma-3-4b-it-GGUF/mmproj-model-f16.gguf", "clip"),
    ("lmstudio-community/granite-3.2-8b-instruct-GGUF/granite-3.2-8b-instruct-Q4_K_M.gguf", "granite"),
    ("lmstudio-community/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q8_0.gguf", "lfm2"),
    ("lmstudio-community/Mistral-Nemo-Instruct-2407-GGUF/Mistral-Nemo-Instruct-2407-Q3_K_L.gguf", "llama"),
    ("lmstudio-community/Mistral-Nemo-Instruct-2407-GGUF/mmproj-pixtral-F16.gguf", "clip"),
    ("nomic/nomic-embed-text-v1.5.Q4_K_M.gguf", "nomic-bert"),
    ("otros/Mamba-Test-Q4_0.gguf", "mamba"),
    ("acme/Tpl-Qwen-GGUF/tpl-qwen-Q4_K_M.gguf", "qwen2"),
    ("acme/Tpl-Mistral-GGUF/tpl-mistral-Q3_K_L.gguf", "llama"),
]

PLANTILLA_QWEN = ("{%- if messages[0]['role'] == 'system' %}{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
                  "{%- else %}{{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}{%- endif %}"
                  "{%- for message in messages %}{%- if message.role == 'user' or (message.role == 'assistant' and not message.tool_calls) %}"
                  "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}{%- endif %}{%- endfor %}"
                  "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}")
PLANTILLA_MISTRAL = ("{{ bos_token }}{% for m in messages %}{% if m['role'] == 'user' %}{{ '[INST]' + m['content'] + '[/INST]' }}"
                     "{% elif m['role'] == 'assistant' %}{{ m['content'] + eos_token }}"
                     "{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}")
EXTRA = {
    "tpl-qwen-Q4_K_M.gguf": [("qwen2.block_count", 4, 32), ("qwen2.attention.head_count", 4, 32),
                             ("qwen2.attention.head_count_kv", 4, 8), ("qwen2.attention.key_length", 4, 128),
                             ("qwen2.embedding_length", 4, 4096), ("tokenizer.chat_template", 8, PLANTILLA_QWEN)],
    "gemma-3-4b-it-Q4_K_M.gguf": [("gemma3.block_count", 4, 34), ("gemma3.attention.head_count", 4, 8),
                                  ("gemma3.attention.head_count_kv", 4, 4), ("gemma3.attention.key_length", 4, 256),
                                  ("gemma3.embedding_length", 4, 2560), ("gemma3.attention.sliding_window", 4, 1024),
                                  ("tokenizer.ggml.tokens", 9, ["<pad>", "<eos>", "<bos>"]),
                                  ("tokenizer.ggml.bos_token_id", 4, 2), ("tokenizer.ggml.eos_token_id", 4, 1)],
    "DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf": [("deepseek2.block_count", 4, 27), ("deepseek2.attention.head_count", 4, 16),
                                                  ("deepseek2.attention.head_count_kv", 4, 16), ("deepseek2.attention.key_length", 4, 192),
                                                  ("deepseek2.attention.kv_lora_rank", 4, 512), ("deepseek2.rope.dimension_count", 4, 64),
                                                  ("deepseek2.embedding_length", 4, 2048),
                                                  ("tokenizer.ggml.tokens", 9, ["<\uff5cbegin\u2581of\u2581sentence\uff5c>", "<\uff5cend\u2581of\u2581sentence\uff5c>"]),
                                                  ("tokenizer.ggml.bos_token_id", 4, 0), ("tokenizer.ggml.eos_token_id", 4, 1)],
    "mmproj-LFM2.5-VL-1.6b-F16.gguf": [("clip.projector_type", 8, "lfm2"), ("clip.has_vision_encoder", 7, True)],
    "mmproj-model-f16.gguf": [("clip.projector_type", 8, "gemma3"), ("clip.has_vision_encoder", 7, True)],
    "mmproj-pixtral-F16.gguf": [("clip.projector_type", 8, "pixtral"), ("clip.has_vision_encoder", 7, True)],
    "tpl-mistral-Q3_K_L.gguf": [("tokenizer.ggml.tokens", 9, ["<unk>", "<s>", "</s>"]), ("tokenizer.ggml.bos_token_id", 4, 1),
                                ("tokenizer.ggml.eos_token_id", 4, 2), ("tokenizer.chat_template", 8, PLANTILLA_MISTRAL)],
}

# requested name -> expected file (LM Studio ids and names used in the multi-agent code)
RESOLUCION = {
    "liquid/lfm2.5-1.2b": "LFM2.5-1.2B-Instruct-Q8_0.gguf",
    "lfm2.5-1.2b-instruct": "LFM2.5-1.2B-Instruct-Q8_0.gguf",
    "lfm2.5-1.2b-thinking": "LFM2.5-1.2B-Thinking-Q8_0.gguf",
    "lfm2-1.2b-rag": "LFM2-1.2B-RAG-Q8_0.gguf",
    "LiquidAI/LFM2-1.2B-RAG-GGUF": "LFM2-1.2B-RAG-Q8_0.gguf",
    "lfm2-1.2b-extract": "LFM2-1.2B-Extract-Q8_0.gguf",
    "lfm2-350m-math": "LFM2-350M-Math-Q8_0.gguf",
    "lfm2.5-vl-1.6b": "LFM2.5-VL-1.6B-Q8_0.gguf",
    "LiquidAI/LFM2.5-VL-1.6B-GGUF": "LFM2.5-VL-1.6B-Q8_0.gguf",
    "google/gemma-3-4b": "gemma-3-4b-it-Q4_K_M.gguf",
    "qwen2.5-coder-7b-instruct": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
    "Kondara/Qwen2.5-Coder-7B-Instruct-Q4_K_M-GGUF": "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
    "granite-3.2-8b-instruct": "granite-3.2-8b-instruct-Q4_K_M.gguf",
    "mistral-nemo-instruct-2407": "Mistral-Nemo-Instruct-2407-Q3_K_L.gguf",
    "deepseek-coder-v2-lite-instruct": "DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf",
    "deepseek-r1-distill-qwen-7b": "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
    "text-embedding-nomic-embed-text-v1.5": "nomic-embed-text-v1.5.Q4_K_M.gguf",
}


# ------------------------------------------------------------------ minimal GGUF
def _s(txt):
    b = txt.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def escribir_gguf(ruta, arch):
    kv = [("general.architecture", 8, arch), ("general.name", 8, os.path.basename(ruta)),
          ("general.alignment", 4, 32), (arch + ".context_length", 4, 32768),
          ("tokenizer.ggml.tokens", 9, ["<|pad|>", "hola", "mundo"])]
    for k, t, v in EXTRA.get(os.path.basename(ruta), []):
        kv = [x for x in kv if x[0] != k] + [(k, t, v)]
    cuerpo = b""
    for k, t, v in kv:
        cuerpo += _s(k) + struct.pack("<I", t)
        if t == 8:
            cuerpo += _s(v)
        elif t == 4:
            cuerpo += struct.pack("<I", v)
        elif t == 7:
            cuerpo += struct.pack("<B", 1 if v else 0)
        elif t == 9:
            cuerpo += struct.pack("<IQ", 8, len(v)) + b"".join(_s(x) for x in v)
    tinfo = _s("token_embd.weight") + struct.pack("<I", 2) + struct.pack("<QQ", 32, 2) + struct.pack("<I", 0) + struct.pack("<Q", 0)
    cab = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kv)) + cuerpo + tinfo
    relleno = (32 - len(cab) % 32) % 32
    datos = struct.pack("<64f", *[float(i) for i in range(64)])
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        f.write(cab + b"\0" * relleno + datos)
    return len(cab) + relleno


# ------------------------------------------------------------------ simulated engine
class MotorFalso(threading.Thread):
    """Mimics web/js/engine.js: long-poll, events and cancellation."""

    def __init__(self, base, mid="motorA"):
        super().__init__(daemon=True)
        self.base = base
        self.mid = mid
        self.cerrado = False
        self.activo = True
        self.recibidos = []
        self.cancelados = []
        self.listo = False
        self.reiniciar_en = None   # job id on which to simulate a page reload

    def post(self, ruta, d):
        d = dict(d, motor=self.mid)
        req = urllib.request.Request(self.base + ruta, json.dumps(d).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def run(self):
        self.post("/engine/api/hello", {"gpu": {"summary": "GPU simulada"}, "version": "prueba"})
        self.listo = True
        while self.activo:
            try:
                with urllib.request.urlopen(self.base + "/engine/api/next?motor=" + self.mid, timeout=40) as r:
                    job = json.loads(r.read())
            except Exception:
                time.sleep(0.2)
                continue
            if job.get("accion") == "cerrar":
                self.cerrado = True
                return
            if job.get("accion") in (None, "nada"):
                continue
            self.recibidos.append(job)
            try:
                self.atender(job)
            except Exception as e:
                print("motor falso:", e)

    def atender(self, job):
        jid = job["id"]
        if job["accion"] == "embeber":
            entradas = job["params"]["entradas"]
            if any("FALLA" in t for t in entradas):
                self.post("/engine/api/event", {"tipo": "error", "id": jid, "mensaje": "falla simulada", "codigo": 500})
                return
            vec = [[len(t) / 100.0, 0.5, -0.25, float(i)] for i, t in enumerate(entradas)]
            self.post("/engine/api/event", {"tipo": "fin", "id": jid, "razon": "stop", "embeddings": vec,
                                            "prompt_tokens": sum(len(t.split()) + 2 for t in entradas), "completion_tokens": 0})
            return
        if job["accion"] != "generar":
            self.post("/engine/api/event", {"tipo": "fin", "id": jid, "razon": "stop"})
            return
        prompt = job["prompt"]
        if "REINICIO" in prompt:
            return   # the "page reloads": it does not respond and asks for work again
        if "FALLA" in prompt:
            self.post("/engine/api/event", {"tipo": "error", "id": jid, "mensaje": "falla simulada", "codigo": 500})
            return
        self.post("/engine/api/event", {"tipo": "progreso", "id": jid, "fase": "cargando modelo 50%"})
        self.post("/engine/api/event", {"tipo": "inicio", "id": jid, "prompt_tokens": len(prompt)})
        if "PENSAR" in prompt:
            piezas = ["<th", "ink>", "Voy a ", "razonar", "</th", "ink>", "\n\n", "Respuesta ", "final."]
        elif "ECO" in prompt:
            piezas = [prompt[i:i + 7] for i in range(0, len(prompt), 7)]
        elif "LENTO" in prompt:
            piezas = ["uno ", "dos ", "tres ", "cuatro ", "cinco ", "seis ", "siete ", "ocho "] * 5
        else:
            piezas = ["Hola", ", ", "soy ", "el ", "motor", " simulado", ". FIN", " esto no debe salir"]
        mt = job["params"].get("max_tokens", -1)
        n = 0
        razon = "stop"
        for p in piezas:
            if mt > 0 and n >= mt:
                razon = "length"
                break
            if "LENTO" in prompt or "GOTEO" in prompt:      # GOTEO: the normal reply, one token at a time (lets the server notice a stop before the engine finishes)
                time.sleep(0.15 if "LENTO" in prompt else 0.1)
            r = self.post("/engine/api/event", {"tipo": "token", "id": jid, "texto": p, "n": 1})
            n += 1
            if r.get("cancelar"):
                self.cancelados.append(jid)
                break
        self.post("/engine/api/event", {"tipo": "fin", "id": jid, "razon": razon, "prompt_tokens": len(prompt),
                                        "completion_tokens": n, "tps": 12.5, "ttft": 0.1})


# ------------------------------------------------------------------ utilities
OK, FALLAS = [0], []


def check(nombre, cond, detalle=""):
    if cond:
        OK[0] += 1
        print("  OK    " + nombre)
    else:
        FALLAS.append(nombre)
        print("  FALLA " + nombre + ("  -> " + str(detalle)[:400] if detalle else ""))


def puerto_libre():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Cliente:
    def __init__(self, base):
        self.base = base

    def pedir(self, ruta, datos=None, metodo=None, cabeceras=None, crudo=False, timeout=60):
        cuerpo = None
        if datos is not None:
            cuerpo = datos if isinstance(datos, bytes) else json.dumps(datos).encode("utf-8")
        req = urllib.request.Request(self.base + ruta, data=cuerpo, method=metodo,
                                     headers=dict({"Content-Type": "application/json"}, **(cabeceras or {})))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                b = r.read()
                return r.status, (b if crudo else _json(b)), dict(r.headers)
        except urllib.error.HTTPError as e:
            b = e.read()
            return e.code, (b if crudo else _json(b)), dict(e.headers)

    def stream(self, ruta, datos, timeout=60):
        req = urllib.request.Request(self.base + ruta, data=json.dumps(datos).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        lineas = []
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ctype = r.headers.get("Content-Type", "")
            for raw in r:
                l = raw.decode("utf-8").rstrip("\r\n")
                if l:
                    lineas.append(l)
        return ctype, lineas


def h_panel_ok(c):
    cod, b, h = c.pedir("/", crudo=True)
    return cod == 200 and "frame-ancestors 'none'" in h.get("Content-Security-Policy", "") and h.get("X-Frame-Options") == "DENY"


def servidor_ips():
    sys.path.insert(0, AQUI)
    import server
    return server.ips_locales()


def _json(b):
    try:
        return json.loads(b.decode("utf-8"))
    except ValueError:
        return b


def chunks(lineas):
    out = []
    for l in lineas:
        if l.startswith("data:") and l[5:].strip() != "[DONE]":
            out.append(json.loads(l[5:].strip()))
    return out


# ------------------------------------------------------------------ tests
def main():
    tmp = tempfile.mkdtemp(prefix="pruebas_llm_")
    carpeta = os.path.join(tmp, "modelos")
    offsets = {}
    for rel, arch in MODELOS:
        offsets[os.path.basename(rel)] = escribir_gguf(os.path.join(carpeta, *rel.split("/")), arch)
    puerto = puerto_libre()
    base = "http://127.0.0.1:%d" % puerto
    # The server runs under a tiny wrapper that dumps every thread's stack after 20 s, so that a hang shows up in the failure log.
    envoltorio = ("import faulthandler, runpy, sys; faulthandler.dump_traceback_later(20); "
                  "import os; sys.argv = sys.argv[1:]; sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0]))); "
                  "runpy.run_path(sys.argv[0], run_name='__main__')")
    # Output goes to a file, not a pipe: nobody reads a pipe while the tests run, and when it fills up (small on Windows) the server blocks on its own log.
    ruta_salida = os.path.join(tmp, "server_output.log")
    salida_srv = open(ruta_salida, "wb")
    srv = subprocess.Popen([sys.executable, "-c", envoltorio, os.path.join(AQUI, "server.py"), "--no-engine", "--port", str(puerto), "--host", "0.0.0.0",
                            "--models-dir", carpeta, "--config", os.path.join(tmp, "config_prueba.json")], stdout=salida_srv, stderr=subprocess.STDOUT)
    c = Cliente(base)
    for _ in range(50):
        try:
            c.pedir("/api/status")
            break
        except Exception:
            time.sleep(0.2)
    motor = None
    try:
        print("\n== Sin motor conectado")
        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "messages": [{"role": "user", "content": "hola"}]})
        check("503 si el motor no está conectado", cod == 503 and "engine" in r["error"]["message"].lower(), (cod, r))

        motor = MotorFalso(base)
        motor.start()
        time.sleep(0.5)

        print("\n== Modelos y nombres")
        cod, r, _ = c.pedir("/v1/models")
        ids = [m["id"] for m in r["data"]]
        check("/v1/models lista 16 modelos (sin mmproj)", cod == 200 and len(ids) == 16, ids)
        check("ningún mmproj en la lista", not any("mmproj" in i for i in ids))
        cod, r, _ = c.pedir("/api/v0/models")
        tipos = {m["id"]: m["type"] for m in r["data"]}
        check("tipos LM Studio: embeddings / vlm / llm",
              tipos.get("nomic-embed-text-v1.5") == "embeddings" and tipos.get("lfm2.5-vl-1.6b") == "vlm"
              and tipos.get("lfm2-1.2b-rag") == "llm", tipos)
        for pedido, archivo in RESOLUCION.items():
            cod, r, _ = c.pedir("/api/resolve?model=" + urllib.request.quote(pedido))
            got = r["model"]["file"].rsplit("/", 1)[-1] if r.get("model") else None
            check("resuelve %-48s -> %s" % (pedido, archivo), got == archivo, got)
        cod, lg, _ = c.pedir("/api/logs")
        check("/api/logs: cada entrada trae time, level y message (el panel lee x.time)",
              cod == 200 and bool(lg) and all({"time", "level", "message"} <= set(x) for x in lg), lg[:2] if lg else lg)
        cod, r, _ = c.pedir("/api/status")
        check("/api/status: motor conectado y GPU informada",
              r["engine"]["connected"] and "simulada" in json.dumps(r["engine"]["info"]), r["engine"])
        sop = {m["id"]: m["supported"] for m in r["models"]}
        check("lfm2, qwen2, llama, granite, gemma3 y deepseek2 figuran como soportadas; mamba no",
              sop["lfm2-1.2b-rag"] and sop["lfm2.5-1.2b-thinking"] and sop["qwen2.5-coder-7b-instruct"]
              and sop["granite-3.2-8b-instruct"] and sop["mistral-nemo-instruct-2407"]
              and sop["gemma-3-4b-it"] and sop["deepseek-coder-v2-lite-instruct"] and not sop["mamba-test"], sop)

        print("\n== Validaciones y errores")
        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "no-existe", "messages": [{"role": "user", "content": "x"}]})
        check("404 modelo inexistente", cod == 404, (cod, r))
        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "mamba-test", "messages": [{"role": "user", "content": "x"}]})
        check("400 arquitectura aún no soportada (mamba)", cod == 400 and "mamba" in r["error"]["message"], r)
        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "text-embedding-nomic-embed-text-v1.5", "messages": [{"role": "user", "content": "x"}]})
        check("400 modelo de embeddings usado para chat", cod == 400, r)
        cod, r, _ = c.pedir("/v1/embeddings", {"model": "text-embedding-nomic-embed-text-v1.5", "input": ["hola mundo", "a", "tres palabras sueltas"]})
        check("/v1/embeddings 200: formato OpenAI, un vector por texto y en orden",
              cod == 200 and r["object"] == "list" and [d["index"] for d in r["data"]] == [0, 1, 2]
              and all(d["object"] == "embedding" for d in r["data"]) and r["data"][1]["embedding"][0] == 0.01
              and r["data"][0]["embedding"][3] == 0.0 and r["data"][2]["embedding"][3] == 2.0
              and r["model"] == "text-embedding-nomic-embed-text-v1.5", (cod, r))
        check("... usage con los tokens del motor", cod == 200 and r["usage"] == {"prompt_tokens": 4 + 3 + 5, "total_tokens": 12}, r.get("usage"))
        check("... el motor recibió accion 'embeber' con las entradas",
              motor.recibidos[-1]["accion"] == "embeber" and motor.recibidos[-1]["params"]["entradas"][1] == "a"
              and motor.recibidos[-1]["modelo"]["arch"] == "nomic-bert", motor.recibidos[-1])
        cod, r, _ = c.pedir("/v1/embeddings", {"model": "nomic-embed-text-v1.5", "input": "solo uno"})
        check("/v1/embeddings acepta 'input' como texto suelto y alias del modelo", cod == 200 and len(r["data"]) == 1, (cod, r))
        cod, r, _ = c.pedir("/v1/embeddings", {"input": "sin modelo"})
        check("/v1/embeddings sin 'model' usa el modelo de embeddings disponible", cod == 200 and len(r["data"]) == 1, (cod, r))
        cod, r, _ = c.pedir("/v1/embeddings", {"model": "nomic-embed-text-v1.5", "input": "x", "encoding_format": "base64"})
        ok64 = False
        if cod == 200:
            import base64 as _b64
            ok64 = tuple(round(x, 6) for x in struct.unpack("<4f", _b64.b64decode(r["data"][0]["embedding"]))) == (0.01, 0.5, -0.25, 0.0)
        check("/v1/embeddings base64 = float32 little-endian", ok64, (cod, r))
        for cuerpo, que in (({"model": "nomic-embed-text-v1.5"}, "sin input"), ({"model": "nomic-embed-text-v1.5", "input": []}, "lista vacía"),
                            ({"model": "nomic-embed-text-v1.5", "input": [1, 2, 3]}, "tokens numéricos"),
                            ({"model": "nomic-embed-text-v1.5", "input": "x", "encoding_format": "hex"}, "formato desconocido"),
                            ({"model": "qwen2.5-coder-7b-instruct", "input": "x"}, "modelo de chat")):
            cod, r, _ = c.pedir("/v1/embeddings", cuerpo)
            check("/v1/embeddings 400: " + que, cod == 400, (cod, r))
        cod, r, _ = c.pedir("/v1/embeddings", {"model": "no-existe-xyz", "input": "x"})
        check("/v1/embeddings 404 modelo inexistente", cod == 404, (cod, r))
        cod, r, _ = c.pedir("/v1/embeddings", {"model": "nomic-embed-text-v1.5", "input": ["ok", "FALLA"]})
        check("/v1/embeddings propaga el error del motor", cod == 500 and "falla simulada" in r["error"]["message"], (cod, r))
        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "messages": [{"role": "user", "content": "x"}],
                                                     "response_format": {"type": "json_object"}})
        check("json_object rechazado igual que LM Studio",
              cod == 400 and r["error"]["message"] == "'response_format.type' must be 'json_schema' or 'text'", r)
        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "lfm2.5-vl-1.6b", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, {"type": "text", "text": "lee"}]}]})
        check("imagen a un modelo con visión soportada (LFM2-VL + mmproj lfm2): 200", cod == 200, r)
        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "gemma-3-4b-it", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, {"type": "text", "text": "lee"}]}]})
        check("imagen a Gemma 3 con mmproj gemma3: 200", cod == 200, r)
        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "mistral-nemo-instruct-2407", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, {"type": "text", "text": "lee"}]}]})
        check("400 imagen a un modelo cuyo mmproj no es de un tipo soportado (pixtral)", cod == 400 and r["error"]["type"] == "vision_not_supported"
              and "not a supported type" in r["error"]["message"], r)
        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, {"type": "text", "text": "lee"}]}]})
        check("400 imagen a un modelo sin mmproj", cod == 400 and r["error"]["type"] == "vision_not_supported"
              and "there is no mmproj file" in r["error"]["message"], r)
        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag"})
        check("400 sin 'messages'", cod == 400, r)
        cod, r, _ = c.pedir("/v1/chat/completions", b"{esto no es json")
        check("400 JSON inválido", cod == 400, r)
        cod, r, _ = c.pedir("/ruta/rara")
        check("404 ruta desconocida", cod == 404)
        cod, r, h = c.pedir("/v1/chat/completions", metodo="OPTIONS")
        check("OPTIONS (CORS) 204", cod == 204 and h.get("Access-Control-Allow-Origin") == "*", (cod, h))

        print("\n== Chat normal")
        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "messages": [{"role": "user", "content": "hola"}],
                                                     "temperature": 0, "max_tokens": 100})
        check("200 chat", cod == 200, r)
        msg = r["choices"][0]["message"]
        check("content correcto", msg["content"] == "Hola, soy el motor simulado. FIN esto no debe salir", msg)
        check("forma OpenAI: id, object, model, usage",
              r["id"].startswith("chatcmpl-") and r["object"] == "chat.completion" and r["model"] == "lfm2-1.2b-rag"
              and r["usage"]["total_tokens"] == r["usage"]["prompt_tokens"] + r["usage"]["completion_tokens"], r)
        check("finish_reason stop", r["choices"][0]["finish_reason"] == "stop")
        job = motor.recibidos[-1]
        check("parámetros al motor: temperature 0, max_tokens 100, top_k por defecto 40",
              job["params"]["temperature"] == 0 and job["params"]["max_tokens"] == 100 and job["params"]["top_k"] == 40, job["params"])
        check("el motor recibe URL de metadatos y archivo",
              job["modelo"]["meta_url"].startswith("/engine/api/meta/") and job["modelo"]["archivo_url"].startswith("/engine/file/"))

        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "liquid/lfm2.5-1.2b", "messages": [
            {"role": "system", "content": "Eres breve."}, {"role": "user", "content": "Hola"},
            {"role": "assistant", "content": "<think>pensé</think>\n\n¡Hola!"},
            {"role": "user", "content": [{"type": "text", "text": "ECO ¿12 por 11?"}]}]})
        esperado = ("<|im_start|>system\nEres breve.<|im_end|>\n<|im_start|>user\nHola<|im_end|>\n"
                    "<|im_start|>assistant\n¡Hola!<|im_end|>\n<|im_start|>user\nECO ¿12 por 11?<|im_end|>\n"
                    "<|im_start|>assistant\n")
        check("plantilla ChatML de LFM2 exacta (sin razonamiento previo, content en partes)",
              motor.recibidos[-1]["prompt"] == esperado, motor.recibidos[-1]["prompt"])
        check("respuesta usa el nombre pedido", r["model"] == "liquid/lfm2.5-1.2b")

        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "lfm2.5-1.2b-thinking", "messages": [{"role": "user", "content": "PENSAR"}]})
        m = r["choices"][0]["message"]
        check("razonamiento separado en reasoning_content (como LM Studio)",
              m.get("reasoning_content") == "Voy a razonar" and m["content"] == "Respuesta final.", m)

        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "messages": [{"role": "user", "content": "hola GOTEO"}],
                                                     "stop": [". FIN"]})
        check("stop recorta el texto", r["choices"][0]["message"]["content"] == "Hola, soy el motor simulado", r)
        for _ in range(50):                      # the engine learns about the cancel in the reply to its next event
            if motor.recibidos[-1]["id"] in motor.cancelados:
                break
            time.sleep(0.1)
        check("stop avisa al motor que cancele", motor.recibidos[-1]["id"] in motor.cancelados)

        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "messages": [{"role": "user", "content": "hola"}],
                                                     "max_tokens": 3})
        check("max_tokens -> finish_reason length", r["choices"][0]["finish_reason"] == "length" and
              r["choices"][0]["message"]["content"] == "Hola, soy ", r)

        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "messages": [{"role": "user", "content": "FALLA"}]})
        check("error del motor -> 500 con mensaje", cod == 500 and "falla simulada" in r["error"]["message"], (cod, r))

        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "messages": [{"role": "user", "content": "x"}],
                                                     "response_format": {"type": "json_schema", "json_schema": {"schema": {}}}})
        check("json_schema aceptado", cod == 200, r)

        print("\n== Streaming (SSE)")
        ctype, lineas = c.stream("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "stream": True,
                                                          "messages": [{"role": "user", "content": "hola"}]})
        ch = chunks(lineas)
        texto = "".join(x["choices"][0]["delta"].get("content", "") for x in ch)
        check("Content-Type text/event-stream", ctype.startswith("text/event-stream"), ctype)
        check("texto reconstruido desde delta.content", texto == "Hola, soy el motor simulado. FIN esto no debe salir", texto)
        check("primer delta trae role assistant", ch[0]["choices"][0]["delta"].get("role") == "assistant", ch[0])
        check("último chunk: delta vacío, finish_reason y usage",
              ch[-1]["choices"][0]["delta"] == {} and ch[-1]["choices"][0]["finish_reason"] == "stop" and "usage" in ch[-1], ch[-1])
        check("termina en data: [DONE]", lineas[-1] == "data: [DONE]", lineas[-3:])
        check("chunks con object chat.completion.chunk e id constante",
              all(x["object"] == "chat.completion.chunk" and x["id"] == ch[0]["id"] for x in ch))

        _, lineas = c.stream("/v1/chat/completions", {"model": "lfm2.5-1.2b-thinking", "stream": True,
                                                      "messages": [{"role": "user", "content": "PENSAR"}]})
        ch = chunks(lineas)
        razon = "".join(x["choices"][0]["delta"].get("reasoning_content", "") for x in ch)
        cont = "".join(x["choices"][0]["delta"].get("content", "") for x in ch)
        check("stream: razonamiento en delta.reasoning_content, respuesta en delta.content",
              razon == "Voy a razonar" and cont == "Respuesta final.", (razon, cont))

        _, lineas = c.stream("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "stream": True, "stop": "FIN",
                                                      "messages": [{"role": "user", "content": "hola"}]})
        cont = "".join(x["choices"][0]["delta"].get("content", "") for x in chunks(lineas))
        check("stream con stop (stop partido entre tokens)", cont == "Hola, soy el motor simulado. ", cont)

        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "stream": True,
                                                     "messages": [{"role": "user", "content": "FALLA"}]})
        check("stream con error antes de empezar -> JSON de error", cod == 500 and "falla simulada" in r["error"]["message"], (cod, r))

        print("\n== /v1/completions")
        cod, r, _ = c.pedir("/v1/completions", {"model": "lfm2-350m-math", "prompt": "ECO La capital de Francia es"})
        check("completion devuelve texto crudo", cod == 200 and r["choices"][0]["text"] == "ECO La capital de Francia es"
              and r["object"] == "text_completion", r)
        check("completion: el prompt llega sin plantilla", motor.recibidos[-1]["prompt"] == "ECO La capital de Francia es")
        _, lineas = c.stream("/v1/completions", {"model": "lfm2-350m-math", "prompt": "ECO abc", "stream": True})
        check("completion en stream", "".join(x["choices"][0].get("text", "") for x in chunks(lineas)) == "ECO abc")

        print("\n== Cola, cancelación y reinicio del motor")
        res = {}

        def pedir_en_hilo(clave, contenido):
            res[clave] = c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "max_tokens": 6,
                                                          "messages": [{"role": "user", "content": contenido}]}, timeout=60)
        h1 = threading.Thread(target=pedir_en_hilo, args=("a", "LENTO a"))
        h2 = threading.Thread(target=pedir_en_hilo, args=("b", "LENTO b"))
        h1.start(); time.sleep(0.2); h2.start()
        time.sleep(0.4)
        cod, st, _ = c.pedir("/api/status")
        check("status muestra trabajo actual y cola", st["current_job"] is not None and st["queued"] == 1, (st["current_job"], st["queued"]))
        h1.join(); h2.join()
        check("dos peticiones simultáneas: ambas terminan", res["a"][0] == 200 and res["b"][0] == 200)
        orden = [j["prompt"] for j in motor.recibidos[-2:]]
        check("se atienden en orden de llegada (una a la vez)", "LENTO a" in orden[0] and "LENTO b" in orden[1], orden)

        # client that cuts the stream halfway
        conn = http.client.HTTPConnection("127.0.0.1", puerto, timeout=30)
        conn.request("POST", "/v1/chat/completions", json.dumps({"model": "lfm2-1.2b-rag", "stream": True,
                     "messages": [{"role": "user", "content": "LENTO corte"}]}), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read1(200)
        conn.close()
        time.sleep(2.0)
        jid = motor.recibidos[-1]["id"]
        check("si el cliente corta el stream, el motor recibe cancelar", jid in motor.cancelados, motor.cancelados[-3:])

        res.clear()
        h = threading.Thread(target=lambda: res.update(r=c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag",
                             "messages": [{"role": "user", "content": "REINICIO"}]})))
        h.start()
        time.sleep(0.8)
        motor.activo = False
        motor_b = MotorFalso(base, "motorB")   # the page reloaded (or another window was opened)
        motor_b.start()
        h.join(timeout=20)
        for _ in range(100):   # wait until the new window has registered (avoids a race in the test)
            if motor_b.listo:
                break
            time.sleep(0.05)
        cod, r, _ = res["r"]
        check("si la página del motor se recarga a mitad, la petición falla con mensaje claro",
              cod >= 500 and "restarted" in r["error"]["message"], (cod, r))
        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "messages": [{"role": "user", "content": "hola"}]})
        check("la ventana nueva atiende las peticiones siguientes", cod == 200 and motor_b.recibidos, (cod, r))
        cod, j, _ = c.pedir("/engine/api/next?motor=motorA")
        check("la ventana vieja recibe 'cerrar' (un solo motor activo)", j.get("accion") == "cerrar", j)
        cod, j, _ = c.pedir("/engine/api/event", {"tipo": "token", "id": "x", "texto": "a", "motor": "motorA"})
        check("eventos de la ventana vieja se ignoran", j.get("cerrar") is True, j)
        motor = motor_b

        print("\n== Visión (LFM2-VL)")
        IMG1, IMG2 = "AAAABBBBCCCC", "DDDDEEEEFFFF"
        def url(b): return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}}
        def tx(t): return {"type": "text", "text": t}
        def chat_vl(mensajes, **extra):
            cod, r, _ = c.pedir("/v1/chat/completions", dict({"model": "lfm2.5-vl-1.6b", "messages": mensajes}, **extra))
            return cod, r
        cod, r = chat_vl([{"role": "user", "content": [tx("ECO mira"), url(IMG1), tx("qué ves")]}])
        job = motor.recibidos[-1]
        check("imagen entre dos textos: 200 y el marcador va pegado, sin saltos de línea",
              cod == 200 and "user\nECO mira<__media__>qué ves<|im_end|>" in job["prompt"], job["prompt"])
        check("el motor recibe la imagen en base64 tal cual, con su tipo",
              job["params"]["imagenes"] == [{"tipo": "image/png", "b64": IMG1}], job["params"].get("imagenes"))
        check("el motor recibe las URL del proyector de visión",
              job["modelo"]["vision"]["meta_url"].startswith("/engine/api/meta-mmproj/")
              and job["modelo"]["vision"]["archivo_url"].startswith("/engine/file-mmproj/") and job["modelo"]["vision"]["bytes"] > 0, job["modelo"])
        cod, r = chat_vl([{"role": "user", "content": [url(IMG1), tx("ECO lee esto")]}])
        check("imagen primero: '<__media__>' y el texto pegados", "user\n<__media__>ECO lee esto<|im_end|>" in motor.recibidos[-1]["prompt"], motor.recibidos[-1]["prompt"])
        cod, r = chat_vl([{"role": "user", "content": [tx("ECO a"), tx("b"), url(IMG1), url(IMG2), tx("c")]}])
        pr = motor.recibidos[-1]["prompt"]
        check("varios textos se unen con salto de línea; dos imágenes seguidas llevan dos marcadores",
              "user\nECO a\nb<__media__><__media__>c<|im_end|>" in pr and [i["b64"] for i in motor.recibidos[-1]["params"]["imagenes"]] == [IMG1, IMG2], pr)
        cod, r = chat_vl([{"role": "user", "content": [tx("ECO uno"), url(IMG1)]}, {"role": "assistant", "content": "ok"},
                          {"role": "user", "content": [url(IMG2), tx("y esta")]}])
        j = motor.recibidos[-1]
        check("varios turnos con imágenes: marcadores en orden y una imagen por marcador",
              j["prompt"].count("<__media__>") == 2 and [i["b64"] for i in j["params"]["imagenes"]] == [IMG1, IMG2], j["prompt"])
        cod, r = chat_vl([{"role": "user", "content": [tx("ECO hola <__media__> falso"), url(IMG1)]}])
        j = motor.recibidos[-1]
        check("un marcador escrito a mano por el usuario se elimina (marcadores = imágenes)",
              cod == 200 and j["prompt"].count("<__media__>") == 1, j["prompt"])
        cod, r = chat_vl([{"role": "user", "content": [tx("ECO sin imágenes")]}])
        check("texto sin imágenes: no se envía 'imagenes' al motor", cod == 200 and "imagenes" not in motor.recibidos[-1]["params"], motor.recibidos[-1]["params"])
        cod, r = chat_vl([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://ejemplo.com/a.png"}}]}])
        check("400 URL http(s): solo data URL en base64", cod == 400 and "data URL" in r["error"]["message"], r)
        cod, r = chat_vl([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@@"}}]}])
        check("400 base64 inválido", cod == 400 and "base64" in r["error"]["message"], r)
        cod, r = chat_vl([{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}])
        check("400 imagen sin URL", cod == 400, r)
        cod, r = chat_vl([{"role": "user", "content": [{"type": "input_image", "image_url": "data:image/jpeg;base64," + IMG1}, tx("ECO x")]}])
        check("acepta 'input_image' con la URL como texto y conserva image/jpeg",
              cod == 200 and motor.recibidos[-1]["params"]["imagenes"] == [{"tipo": "image/jpeg", "b64": IMG1}], (cod, r))
        cod, r, _ = c.pedir("/api/status")
        vok = {m["id"]: m["vision_ok"] for m in r["models"]}
        check("/api/status: vision_ok solo en LFM2-VL (lfm2) y Gemma 3 (gemma3)", vok["lfm2.5-vl-1.6b"] and vok["gemma-3-4b-it"]
              and not vok["lfm2-1.2b-rag"] and not vok["mistral-nemo-instruct-2407"], vok)
        cod, r, _ = c.pedir("/engine/api/meta-mmproj/lfm2.5-vl-1.6b")
        check("meta del mmproj para el motor (solo local)", cod == 200 and r["kv"]["clip.projector_type"] == "lfm2" and r["tensors"], (cod, str(r)[:200]))
        cod, r, _ = c.pedir("/engine/api/meta-mmproj/lfm2-1.2b-rag")
        check("meta del mmproj: 404 si el modelo no tiene", cod == 404, (cod, r))
        cod, b, h = c.pedir("/engine/file-mmproj/lfm2.5-vl-1.6b", cabeceras={"Range": "bytes=0-3"}, crudo=True)
        check("archivo del mmproj por rangos (206, 'GGUF')", cod == 206 and b == b"GGUF", (cod, b))

        print("\n== Plantillas de chat (v2: qwen2 / llama / granite)")
        def eco(modelo, mensajes, **extra):
            cod, r, _ = c.pedir("/v1/chat/completions", dict({"model": modelo, "messages": mensajes}, **extra))
            return cod, r
        QW = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
        cod, r = eco("qwen2.5-coder-7b-instruct", [{"role": "user", "content": "ECO hola"}])
        check("qwen2 sin plantilla en el GGUF: ChatML de respaldo con system por defecto",
              cod == 200 and r["choices"][0]["message"]["content"] == QW + "<|im_start|>user\nECO hola<|im_end|>\n<|im_start|>assistant\n", r)
        check("... y pide BOS por defecto", motor.recibidos[-1]["params"]["add_bos"] is True, motor.recibidos[-1]["params"])
        cod, r = eco("qwen2.5-coder-7b-instruct", [{"role": "system", "content": "S"}, {"role": "user", "content": "ECO a"},
                                                    {"role": "assistant", "content": "b"}, {"role": "user", "content": "ECO c"}])
        check("qwen2 respaldo con system propio y varios turnos",
              cod == 200 and r["choices"][0]["message"]["content"] ==
              "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nECO a<|im_end|>\n<|im_start|>assistant\nb<|im_end|>\n<|im_start|>user\nECO c<|im_end|>\n<|im_start|>assistant\n", r)
        cod, r = eco("tpl-qwen-Q4_K_M", [{"role": "user", "content": [{"type": "text", "text": "ECO hola"}]}])
        check("qwen2 con plantilla Jinja del GGUF (contenido en partes)",
              cod == 200 and r["choices"][0]["message"]["content"] == QW + "<|im_start|>user\nECO hola<|im_end|>\n<|im_start|>assistant\n", r)
        cod, r = eco("granite-3.2-8b-instruct", [{"role": "user", "content": "ECO hola"}])
        t = r["choices"][0]["message"]["content"] if cod == 200 else ""
        check("granite respaldo: roles start_of_role/end_of_role",
              t.startswith("<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.") and
              t.endswith("<|start_of_role|>user<|end_of_role|>ECO hola<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"), t)
        cod, r = eco("mistral-nemo-instruct-2407", [{"role": "system", "content": "SIS"}, {"role": "user", "content": "ECO hola"}])
        check("llama (Nemo) respaldo: [INST] con system dentro del último turno de usuario",
              cod == 200 and r["choices"][0]["message"]["content"] == "<s>[INST]SIS\n\nECO hola[/INST]", r)
        check("... el BOS ya va en el texto: add_bos=false", motor.recibidos[-1]["params"]["add_bos"] is False)
        cod, r = eco("gemma-3-4b-it", [{"role": "system", "content": "SIS"}, {"role": "user", "content": "ECO hola"},
                                        {"role": "assistant", "content": "ok"}, {"role": "user", "content": "ECO otra"}])
        check("gemma3 respaldo: start_of_turn/end_of_turn, system pegado al primer turno de usuario",
              cod == 200 and r["choices"][0]["message"]["content"] ==
              "<bos><start_of_turn>user\nSIS\n\nECO hola<end_of_turn>\n<start_of_turn>model\nok<end_of_turn>\n<start_of_turn>user\nECO otra<end_of_turn>\n<start_of_turn>model\n", r)
        check("... el BOS ya va en el texto: add_bos=false", motor.recibidos[-1]["params"]["add_bos"] is False)
        cod, r = eco("tpl-mistral-Q3_K_L", [{"role": "user", "content": "ECO hola"}, {"role": "assistant", "content": "ok"},
                                             {"role": "user", "content": "ECO otra"}])
        check("llama con plantilla del GGUF: bos_token y eos_token reales del vocabulario",
              cod == 200 and r["choices"][0]["message"]["content"] == "<s>[INST]ECO hola[/INST]ok</s>[INST]ECO otra[/INST]", r)
        check("... add_bos=false porque la plantilla ya lo incluye", motor.recibidos[-1]["params"]["add_bos"] is False)
        cod, r = eco("tpl-mistral-Q3_K_L", [{"role": "system", "content": "S"}, {"role": "user", "content": "ECO hola"}])
        check("raise_exception de la plantilla -> 400 con su mensaje",
              cod == 400 and "Only user and assistant roles" in r["error"]["message"], (cod, r))
        c.pedir("/api/config", {"kv_max_mb": 640})     # manual cap of 640 MB (automatic mode is tested further below)
        cod, r, _ = c.pedir("/api/status")
        ctxs = {m["file"].rsplit("/", 1)[-1]: m["ctx"] for m in r["models"]}
        check("tope de contexto por memoria de la caché KV (32 capas x 8 cabezas x 128 -> 2304)",
              ctxs["tpl-qwen-Q4_K_M.gguf"] == 2304, ctxs)
        check("gemma3: el tope de KV cuenta solo las capas globales + ventana fija de las locales (34 capas, ventana 1024: conserva 8192; contándolas todas como globales bajaría a 2048)",
              ctxs["gemma-3-4b-it-Q4_K_M.gguf"] == 8192, ctxs)
        check("deepseek2: la caché comprimida (27 capas x 576 valores) conserva 8192 (como K/V expandidos por cabeza bajaría a 2048)",
              ctxs["DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf"] == 8192, ctxs)
        c.pedir("/api/config", {"kv_max_mb": 0})
        cod, r = eco("deepseek-coder-v2-lite-instruct", [{"role": "system", "content": "SIS"}, {"role": "user", "content": "ECO hola"},
                                                          {"role": "assistant", "content": "ok"}, {"role": "user", "content": "ECO otra"}])
        check("deepseek2 respaldo: BOS, system suelto, 'User: ...' / 'Assistant: ...<eos>'",
              cod == 200 and r["choices"][0]["message"]["content"] ==
              "<\uff5cbegin\u2581of\u2581sentence\uff5c>SIS\n\nUser: ECO hola\n\nAssistant: ok<\uff5cend\u2581of\u2581sentence\uff5c>User: ECO otra\n\nAssistant:", r)
        check("... el BOS ya va en el texto: add_bos=false", motor.recibidos[-1]["params"]["add_bos"] is False)
        check("modelos sin datos de atención conservan default_ctx", ctxs["LFM2-1.2B-RAG-Q8_0.gguf"] == 8192, ctxs)

        print("\n== Puente con el motor y archivos")
        mid = "lfm2-1.2b-rag"
        cod, meta, _ = c.pedir("/engine/api/meta/" + mid)
        check("metadatos completos para el motor (vocabulario incluido)",
              cod == 200 and meta["kv"]["tokenizer.ggml.tokens"] == ["<|pad|>", "hola", "mundo"] and meta["tensors"][0]["nombre"] == "token_embd.weight", meta)
        ini = meta["tensors"][0]["abs"]
        cod, b, h = c.pedir("/engine/file/" + mid, cabeceras={"Range": "bytes=%d-%d" % (ini, ini + 15)}, crudo=True)
        check("archivo GGUF por rangos (206 y bytes exactos)",
              cod == 206 and struct.unpack("<4f", b) == (0.0, 1.0, 2.0, 3.0) and h.get("Content-Range", "").startswith("bytes %d-" % ini), (cod, b[:16], h.get("Content-Range")))
        cod, r, _ = c.pedir("/api/load", {"model": "mamba-test"})
        check("/api/load rechaza arquitectura no soportada", cod == 400)
        cod, r, _ = c.pedir("/api/load", {"model": "lfm2-1.2b-rag"})
        time.sleep(0.5)
        check("/api/load envía 'cargar' al motor", cod == 202 and motor.recibidos[-1]["accion"] == "cargar", motor.recibidos[-1])
        cod, r, _ = c.pedir("/api/rescan", {})
        check("/api/rescan vuelve a leer la carpeta", cod == 200 and len(r["models"]) == 16)

        print("\n== Configuración desde el panel")
        cod, st, _ = c.pedir("/api/status")
        check("status informa nombre de la PC y que el panel se ve desde la PC servidora",
              bool(st["pc_name"]) and st["from_server_pc"] is True, (st.get("pc_name"), st.get("from_server_pc")))
        check("la primera dirección usa el nombre de la PC", st["reachable_at"][0] == "http://%s:%d" % (st["pc_name"], puerto), st["reachable_at"][:2])
        cod, cf, _ = c.pedir("/api/config")
        check("GET /api/config: valores y editable", cod == 200 and cf["models_dir"] == carpeta and cf["editable"] is True and cf["default_ctx"] == 8192, cf)
        cod, r, _ = c.pedir("/api/config", {"models_dir": os.path.join(tmp, "no_existe")})
        check("rechaza una carpeta que no existe", cod == 400 and "does not exist" in r["error"]["message"], r)
        cod, r, _ = c.pedir("/api/config", {"default_ctx": "abc", "timeout_seconds": 5})
        check("rechaza valores inválidos", cod == 400, r)
        cod, r, _ = c.pedir("/api/config", {"default_ctx": 4096, "timeout_seconds": 900, "default_sampling": {"temperature": 0.3}})
        check("guarda ajustes válidos", cod == 200 and r["config"]["default_ctx"] == 4096 and r["config"]["default_sampling"]["temperature"] == 0.3, r)
        ruta_cfg = os.path.join(tmp, "config_prueba.json")
        with open(ruta_cfg, encoding="utf-8") as f:
            en_disco = json.load(f)
        check("config.json solo recibe las claves cambiadas", en_disco.get("default_ctx") == 4096 and en_disco["default_sampling"] == {"temperature": 0.3}
              and "models_dir" not in en_disco, en_disco)
        otra = os.path.join(tmp, "otra")
        os.makedirs(os.path.join(otra, "sub"))
        cod, r, _ = c.pedir("/api/config", {"models_dir": otra})
        check("cambiar la carpeta reescanea (vacía = 0 modelos)", cod == 200 and r["models"] == 0 and r["restart_required"] is False, r)
        cod, st, _ = c.pedir("/api/status")
        check("status refleja la carpeta nueva", st["models_dir"] == otra and st["models"] == [], st["models_dir"])
        cod, r, _ = c.pedir("/api/config", {"models_dir": carpeta, "port": puerto + 1})
        check("volver a la carpeta original recupera los modelos; puerto nuevo pide reiniciar", cod == 200 and r["models"] == 16 and r["restart_required"] is True, r)
        check("el puerto no cambia en caliente", r["config"]["port"] == puerto, r["config"]["port"])
        cod, r, _ = c.pedir("/api/folders?path=" + urllib.request.quote(carpeta))
        check("explorador: lista subcarpetas y cuenta .gguf", cod == 200 and r["path"] == os.path.abspath(carpeta) and len(r["folders"]) >= 1, r)
        cod, r, _ = c.pedir("/api/folders?path=" + urllib.request.quote(os.path.join(tmp, "zzz")))
        check("explorador: carpeta inexistente -> 400", cod == 400, r)
        cod, r, _ = c.pedir("/api/folders")
        check("explorador sin ruta lista la raíz/unidades", cod == 200 and isinstance(r["folders"], list) and len(r["folders"]) > 0, r)

        print("\n== Autoajuste")
        sys.path.insert(0, AQUI)
        import server as SV
        GB = 1e9
        e = SV.estimar_memoria(4.68 * GB, 114688, 0, 5376, 0, 7.8 * GB)
        check("qwen 7B Q4_K_M en 7,8 GB (la PC real): cabe", e["state"] == "fits", e)
        e = SV.estimar_memoria(6.56 * GB, 327680, 0, 2048, 0, 7.8 * GB)
        check("nemo Q3_K_L en 7,8 GB (cargó en la PC real): justo, no 'no cabe'", e["state"] == "tight", e)
        e = SV.estimar_memoria(6.43 * GB, 62208, 0, 8192, 0, 7.8 * GB)
        check("deepseek Q2_K en 7,8 GB (cargó en la PC real): justo", e["state"] == "tight", e)
        e = SV.estimar_memoria(4.68 * GB, 114688, 0, 5376, 0, 4 * GB)
        check("el mismo qwen en una PC con 4 GB: no cabe", e["state"] == "no_fit" and e["ratio"] > 1, e)
        e = SV.estimar_memoria(4.68 * GB, 114688, 0, 5376, 0.85 * GB, 7.8 * GB)
        check("la visión suma su proyector", e["necesita_bytes"] > SV.estimar_memoria(4.68 * GB, 114688, 0, 5376, 0, 7.8 * GB)["necesita_bytes"])
        check("sin presupuesto conocido: 'unknown' (no bloquea)", SV.estimar_memoria(1e9, 1, 0, 100, 0, None)["state"] == "unknown")
        check("clasifica GPU: intel gen-12lp integrada, nvidia dedicada, intel arc dedicada, rara desconocida",
              [SV.tipo_gpu({"summary": d}) for d in ("intel gen-12lp", "nvidia ada lovelace", "intel xe-hpg arc a770", "")]
              == ["integrated", "discrete", "discrete", "unknown"])
        cod, st, _ = c.pedir("/api/status")
        hw = st["hardware"]
        check("status incluye el hardware y el presupuesto estimado", cod == 200 and hw["ram_gb"] and hw["budget_gb"] and hw["source"] and hw["memory_check"] is True, hw)
        check("cada modelo soportado trae su estimación de memoria", all(m["memory"] and m["memory"]["state"] in ("fits", "tight", "no_fit") for m in st["models"] if m["supported"]),
              [m["memory"] for m in st["models"][:3]])
        check("el modelo no soportado no trae estimación", all(m["memory"] is None for m in st["models"] if not m["supported"]))
        cod, r, _ = c.pedir("/api/config", {"gpu_memory_gb": 0.2})
        check("«Memoria para modelos» fijada: se usa como presupuesto", cod == 200, r)
        cod, st, _ = c.pedir("/api/status")
        check("... el presupuesto del status es 0,2 GB y viene de la configuración",
              st["hardware"]["budget_gb"] == 0.2 and "configuration" in st["hardware"]["source"], st["hardware"])
        cod, r, _ = c.pedir("/api/load", {"model": "qwen2.5-coder-7b-instruct"})
        check("/api/load rechaza lo que no cabe (400 model_too_large) con el mensaje claro",
              cod == 400 and r["error"]["type"] == "model_too_large" and "GB" in r["error"]["message"], (cod, r))
        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "qwen2.5-coder-7b-instruct", "messages": [{"role": "user", "content": "hola"}]})
        check("el chat también lo rechaza", cod == 400 and r["error"]["type"] == "model_too_large", (cod, r))
        c.pedir("/api/config", {"memory_check": False})
        cod, r, _ = c.pedir("/api/load", {"model": "qwen2.5-coder-7b-instruct"})
        check("con el autoajuste desactivado se permite", cod == 202, (cod, r))
        c.pedir("/api/config", {"memory_check": True, "gpu_memory_gb": 0})
        cod, st, _ = c.pedir("/api/status")
        check("al volver a automático el presupuesto vuelve a calcularse", st["hardware"]["source"] != "set in the configuration", st["hardware"])
        cod, cf, _ = c.pedir("/api/config")
        check("kv_max_mb por defecto es 0 (automático)", cf["kv_max_mb"] == 0, cf)

        print("\n== Seguridad: clave de API y límite de cola")
        cod, r, _ = c.pedir("/api/config", {"api_key": "con espacios"})
        check("rechaza una clave con espacios", cod == 400, r)
        cod, r, _ = c.pedir("/api/config", {"api_key": "clave-de-prueba"})
        check("fija la clave desde el panel", cod == 200 and r["config"]["api_key_set"] is True, r)
        check("la configuración pública nunca devuelve la clave", "clave-de-prueba" not in json.dumps(r))
        cod, r, h = c.pedir("/v1/models")
        check("sin clave: /v1/models -> 401 con WWW-Authenticate", cod == 401 and r["error"]["type"] == "invalid_api_key" and h.get("WWW-Authenticate") == "Bearer", (cod, r, h))
        cod, r, _ = c.pedir("/v1/models", cabeceras={"Authorization": "Bearer otra"})
        check("clave incorrecta -> 401", cod == 401, cod)
        cod, r, _ = c.pedir("/v1/models", cabeceras={"Authorization": "Bearer clave-de-prueba"})
        check("clave correcta -> 200", cod == 200 and len(r["data"]) > 0, cod)
        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "messages": [{"role": "user", "content": "hola"}]})
        check("chat sin clave -> 401", cod == 401, cod)
        cod, r, _ = c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "max_tokens": 4, "messages": [{"role": "user", "content": "hola"}]},
                            cabeceras={"Authorization": "Bearer clave-de-prueba"})
        check("chat con clave -> 200 (el motor local sigue conectado sin clave)", cod == 200, (cod, r))
        cod, r, _ = c.pedir("/v1/embeddings", {"model": "nomic-embed-text-v1.5", "input": "x"})
        check("embeddings sin clave -> 401", cod == 401, cod)
        cod, r, _ = c.pedir("/api/load", {"model": "lfm2-1.2b-rag"})
        check("/api/load sin clave -> 401", cod == 401, cod)
        cod, r, _ = c.pedir("/api/v0/models")
        check("/api/v0/models sin clave -> 401", cod == 401, cod)
        cod, st, _ = c.pedir("/api/status")
        check("el panel y /api/status siguen abiertos e informan que se exige clave", cod == 200 and st["requires_key"] is True, cod)
        cod, b, _ = c.pedir("/", crudo=True)
        check("el panel (HTML) es accesible sin clave", cod == 200)
        cod, b, h = c.pedir("/chat", crudo=True)
        check("la página de chat (/chat) es accesible sin clave y no se puede enmarcar",
              cod == 200 and b"quipullm" in b and h.get("X-Frame-Options") == "DENY", cod)
        cod, b, _ = c.pedir("/api/load", {"model": "lfm2-1.2b-rag"})
        check("cargar un modelo sin la clave -> 401", cod == 401, cod)
        cod, r, _ = c.pedir("/api/config", {"api_key": ""})
        cod, r, _ = c.pedir("/v1/models")
        check("al quitar la clave la API vuelve a estar abierta", cod == 200, cod)

        print("\n== Modelos visibles en el chat y panel solo en la PC servidora")
        cod, r, _ = c.pedir("/api/config", {"chat_models": ["gemma-3-4b-it", "lfm2-1.2b-rag", "gemma-3-4b-it"]})
        check("chat_models: se guarda sin duplicados y en orden", cod == 200 and r["config"]["chat_models"] == ["gemma-3-4b-it", "lfm2-1.2b-rag"], r)
        cod, st, _ = c.pedir("/api/status")
        check("status publica chat_models", st["chat_models"] == ["gemma-3-4b-it", "lfm2-1.2b-rag"], st.get("chat_models"))
        cod, r, _ = c.pedir("/api/config", {"chat_models": "no-es-lista"})
        check("chat_models inválido -> 400", cod == 400, r)
        cod, r, _ = c.pedir("/api/config", {"chat_models": [1, 2]})
        check("chat_models con no-cadenas -> 400", cod == 400, r)
        cod, r, _ = c.pedir("/api/config", {"chat_models": []})
        cod, st, _ = c.pedir("/api/status")
        check("chat_models vacío = todos los modelos", cod == 200 and st["chat_models"] == [], st.get("chat_models"))
        check("la PC servidora ve el panel y los ajustes (panel_allowed)", st["panel_allowed"] is True and isinstance(st["history"], list))
        print("\n== Longitud de respuesta y temperatura por modelo")
        peti = lambda extra=None: c.pedir("/v1/chat/completions", dict({"model": "lfm2-1.2b-rag", "messages": [{"role": "user", "content": "hola"}]}, **(extra or {})))
        cod, r, _ = c.pedir("/api/config", {"default_max_tokens": 50, "max_tokens_limit": 100})
        check("fija longitud por defecto y límite", cod == 200 and r["config"]["default_max_tokens"] == 50 and r["config"]["max_tokens_limit"] == 100, r)
        peti(); check("sin max_tokens: se usa la longitud por defecto del administrador", motor.recibidos[-1]["params"]["max_tokens"] == 50, motor.recibidos[-1]["params"])
        peti({"max_tokens": 30}); check("max_tokens menor que el límite se respeta", motor.recibidos[-1]["params"]["max_tokens"] == 30)
        peti({"max_tokens": 5000}); check("max_tokens por encima del límite se recorta al límite", motor.recibidos[-1]["params"]["max_tokens"] == 100, motor.recibidos[-1]["params"])
        peti({"max_tokens": -1}); check("max_tokens -1 (sin tope) también se recorta al límite", motor.recibidos[-1]["params"]["max_tokens"] == 100)
        cod, r, _ = c.pedir("/api/config", {"model_overrides": {"lfm2-1.2b-rag": {"temperature": 0.3, "max_tokens": 20, "max_tokens_limit": 40}}})
        check("fija valores por modelo", cod == 200, r)
        cod, st, _ = c.pedir("/api/status")
        mod = next(x for x in st["models"] if x["id"] == "lfm2-1.2b-rag")
        check("status publica los valores del modelo", mod["gen"] == {"temperature": 0.3, "max_tokens": 20, "max_tokens_limit": 40} and mod["overrides"]["temperature"] == 0.3, mod.get("gen"))
        peti(); p_ = motor.recibidos[-1]["params"]
        check("el modelo usa su temperatura y su longitud propias", p_["temperature"] == 0.3 and p_["max_tokens"] == 20, p_)
        peti({"max_tokens": 5000, "temperature": 1.0}); p_ = motor.recibidos[-1]["params"]
        check("la petición manda su temperatura, pero el límite del modelo se aplica", p_["temperature"] == 1.0 and p_["max_tokens"] == 40, p_)
        otro = next(x for x in st["models"] if x["id"] != "lfm2-1.2b-rag" and x["supported"] and not x["embedding"])
        check("otro modelo no cambia", otro["gen"]["max_tokens"] == 50 and otro["gen"]["max_tokens_limit"] == 100, otro.get("gen"))
        cod, r, _ = c.pedir("/api/config", {"model_overrides": {"lfm2-1.2b-rag": {"temperature": None}}})
        cod, st, _ = c.pedir("/api/status")
        mod = next(x for x in st["models"] if x["id"] == "lfm2-1.2b-rag")
        check("vaciar un valor vuelve al ajuste general y conserva los otros", "temperature" not in mod["overrides"] and mod["overrides"].get("max_tokens") == 20, mod.get("overrides"))
        cod, r, _ = c.pedir("/api/config", {"model_overrides": {"lfm2-1.2b-rag": {"max_tokens": "abc"}}})
        check("valor por modelo inválido -> 400", cod == 400, r)
        cod, r, _ = c.pedir("/api/config", {"default_max_tokens": 0, "max_tokens_limit": 0, "model_overrides": {"lfm2-1.2b-rag": {"max_tokens": None, "max_tokens_limit": None}}})
        peti(); check("sin valores: vuelve a 'hasta llenar el contexto' (-1)", motor.recibidos[-1]["params"]["max_tokens"] == -1, motor.recibidos[-1]["params"])

        # "another PC": a request coming from 127.0.0.2 is not the server PC for the server (it only trusts 127.0.0.1)
        def desde_otra_pc(metodo, ruta, cuerpo=None):
            cn = http.client.HTTPConnection("127.0.0.1", puerto, timeout=10, source_address=("127.0.0.2", 0))
            try:
                cn.request(metodo, ruta, body=json.dumps(cuerpo) if cuerpo is not None else None,
                           headers={"Content-Type": "application/json"})
                rr = cn.getresponse(); datos = rr.read()
                return rr.status, dict(rr.getheaders()), datos
            finally:
                cn.close()
        try:
            cod_o, cab_o, _ = desde_otra_pc("GET", "/")
        except OSError as e:
            print("   (omitido: este sistema no permite conectar desde 127.0.0.2: %s)" % e)
            cod_o = None
        if cod_o is not None:
            check("otra PC: / redirige al chat", cod_o == 302 and cab_o.get("Location") == "/chat", (cod_o, cab_o.get("Location")))
            cod_o, _, cuerpo_o = desde_otra_pc("GET", "/chat")
            check("otra PC: /chat se sirve", cod_o == 200 and b"quipullm" in cuerpo_o, cod_o)
            cod_o, _, cuerpo_o = desde_otra_pc("GET", "/api/status")
            st_o = json.loads(cuerpo_o)
            check("otra PC: status sin historial, sin hardware, sin clave de compartir y con panel_allowed falso",
                  cod_o == 200 and st_o["history"] == [] and st_o["hardware"] == {} and st_o["share_key"] is None and st_o["panel_allowed"] is False, st_o.get("panel_allowed"))
            check("otra PC: status sigue trayendo los modelos para el chat", len(st_o["models"]) > 0 and "chat_models" in st_o)
            check("otra PC: /api/logs -> 403", desde_otra_pc("GET", "/api/logs")[0] == 403)
            check("otra PC: /api/config (lectura) -> 403", desde_otra_pc("GET", "/api/config")[0] == 403)
            check("otra PC: guardar ajustes -> 403", desde_otra_pc("POST", "/api/config", {"chat_models": ["x"]})[0] == 403)
            check("otra PC sin clave: cargar un modelo -> 403", desde_otra_pc("POST", "/api/load", {"model": "lfm2-1.2b-rag"})[0] == 403)
            check("otra PC: /v1/models sigue funcionando", desde_otra_pc("GET", "/v1/models")[0] == 200)

        cod, r, _ = c.pedir("/api/config", {"max_queue": 1})
        check("fija el límite de cola", cod == 200 and r["config"]["max_queue"] == 1, r)
        res_c = {}
        def pedir_c(clave, contenido):
            res_c[clave] = c.pedir("/v1/chat/completions", {"model": "lfm2-1.2b-rag", "max_tokens": 6,
                                                            "messages": [{"role": "user", "content": contenido}]}, timeout=60)
        hs = []
        for k, txt, espera in (("a", "LENTO a", 0.0), ("b", "LENTO b", 0.25), ("c", "LENTO c", 0.25)):
            time.sleep(espera)
            hs.append(threading.Thread(target=pedir_c, args=(k, txt))); hs[-1].start()
        for h in hs:
            h.join()
        check("cola con límite 1: la primera se atiende, la segunda espera y la tercera recibe 429",
              res_c["a"][0] == 200 and res_c["b"][0] == 200 and res_c["c"][0] == 429, {k: v[0] for k, v in res_c.items()})
        check("... el 429 trae Retry-After y el tipo rate_limit_exceeded", res_c["c"][2].get("Retry-After") == "10" and res_c["c"][1]["error"]["type"] == "rate_limit_exceeded", res_c["c"][1])
        c.pedir("/api/config", {"max_queue": 64})
        cod, st, _ = c.pedir("/api/status")
        check("la petición rechazada no cuenta como atendida ni deja la cola atascada", st["queued"] == 0, st["queued"])

        print("\n== Arquitecturas enchufables")
        cod, lista, _ = c.pedir("/api/architectures")
        ids_arch = {d["id"] for d in lista}
        check("/api/architectures lista las del núcleo y el ejemplo declarativo qwen3",
              cod == 200 and {"lfm2", "qwen2", "llama", "granite", "gemma3", "nomic-bert", "deepseek2", "qwen3"} <= ids_arch, sorted(ids_arch))
        check("qwen3 es declarativa (base transformer, sin módulo)", next(d for d in lista if d["id"] == "qwen3").get("base") == "transformer")
        gg = os.path.join(carpeta, "zz", "ZZ-Test-Q4_0.gguf")
        man = os.path.join(AQUI, "web", "arch", "zz_prueba.json")
        malo = os.path.join(AQUI, "web", "arch", "zz_malo.json")
        escribir_gguf(gg, "zzarch")
        try:
            c.pedir("/api/rescan", {})
            cod, st, _ = c.pedir("/api/status")
            zz = next(m for m in st["models"] if m["arch"] == "zzarch")
            check("arquitectura sin manifiesto: aparece como no soportada", zz["supported"] is False and zz["memory"] is None, zz)
            cod, r, _ = c.pedir("/api/load", {"model": zz["id"]})
            check("... y no se puede cargar (400)", cod == 400, (cod, r))
            with open(man, "w", encoding="utf-8") as f:
                json.dump({"id": "zz", "arch": ["zzarch"], "base": "transformer", "options": {"rope": "neox"}, "fallback_template": "chatml"}, f)
            with open(malo, "w", encoding="utf-8") as f:
                f.write("{ esto no es json")
            c.pedir("/api/rescan", {})
            cod, st, _ = c.pedir("/api/status")
            zz = next(m for m in st["models"] if m["arch"] == "zzarch")
            check("al soltar un manifiesto y reescanear (sin reiniciar) la arquitectura queda soportada", zz["supported"] is True, zz)
            cod, lista, _ = c.pedir("/api/architectures")
            check("un manifiesto inválido se ignora y no tumba al servidor", cod == 200 and "zz" in {d["id"] for d in lista} and len(lista) == 9, [d["id"] for d in lista])
            cod, meta, _ = c.pedir("/engine/api/meta/" + zz["id"])
            check("el meta del motor trae las opciones del manifiesto", cod == 200 and meta.get("arch_options") == {"rope": "neox"}, meta.get("arch_options"))
            cod, r = eco(zz["id"], [{"role": "user", "content": "ECO hola"}])
            check("plantilla de respaldo ChatML genérica para una arquitectura declarada",
                  cod == 200 and r["choices"][0]["message"]["content"] == "<|im_start|>user\nECO hola<|im_end|>\n<|im_start|>assistant\n", r)
            with open(man, "w", encoding="utf-8") as f:
                json.dump({"id": "zz", "arch": ["zzarch"], "base": "no_existe"}, f)
            c.pedir("/api/rescan", {})
            cod, st, _ = c.pedir("/api/status")
            check("'base' desconocida: el manifiesto se rechaza y el modelo vuelve a no soportado",
                  next(m for m in st["models"] if m["arch"] == "zzarch")["supported"] is False)
            with open(man, "w", encoding="utf-8") as f:
                json.dump({"id": "zz", "arch": ["zzarch"], "module": "no_existe.js"}, f)
            c.pedir("/api/rescan", {})
            cod, lista, _ = c.pedir("/api/architectures")
            check("'modulo' inexistente: se rechaza", "zz" not in {d["id"] for d in lista})
            with open(man, "w", encoding="utf-8") as f:
                json.dump({"id": "zz", "arch": ["zzarch"], "module": "../server.py"}, f)
            c.pedir("/api/rescan", {})
            cod, lista, _ = c.pedir("/api/architectures")
            check("'modulo' con ruta fuera de web/arch/: se rechaza", "zz" not in {d["id"] for d in lista})
        finally:
            for f in (man, malo, gg):
                if os.path.exists(f):
                    os.remove(f)
            c.pedir("/api/rescan", {})

        print("\n== Páginas")
        for ruta, tipo in (("/", "text/html"), ("/engine", "text/html"), ("/web/js/engine.js", "text/javascript"),
                           ("/web/js/gpu.js", "text/javascript")):
            cod, b, h = c.pedir(ruta, crudo=True)
            check("GET %-18s %s" % (ruta, tipo), cod == 200 and h.get("Content-Type", "").startswith(tipo), (cod, h.get("Content-Type")))
        cod, b, h = c.pedir("/web/../server.py", crudo=True)
        check("no expone archivos fuera de web/", cod == 404, cod)
        check("las páginas HTML no se pueden incrustar en otro sitio (frame-ancestors)",
              h_panel_ok(c), "")

        print("\n== Seguridad: otras páginas web (CSRF / CORS / DNS rebinding)")
        ajeno = {"Origin": "https://sitio-ajeno.example"}
        cruzado = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors"}
        cod, r, h = c.pedir("/api/config", metodo="OPTIONS", cabeceras=dict(ajeno, **{"Access-Control-Request-Method": "POST"}))
        check("OPTIONS de otro origen en /api/config: rechazado y sin CORS", cod == 403 and "Access-Control-Allow-Origin" not in h, (cod, h))
        cod, r, h = c.pedir("/api/status")
        check("/api/status (mismo origen) no envía CORS abierto", cod == 200 and "Access-Control-Allow-Origin" not in h, h)
        cod, r, _ = c.pedir("/api/folders", cabeceras=ajeno)
        check("otro origen no puede listar carpetas del disco", cod == 403, (cod, r))
        cod, cf0, _ = c.pedir("/api/config")
        cod, r, _ = c.pedir("/api/config", {"default_ctx": 1024}, cabeceras=ajeno)
        cod2, cf1, _ = c.pedir("/api/config")
        check("otro origen no puede cambiar la configuración", cod == 403 and cf1["default_ctx"] == cf0["default_ctx"], (cod, cf1.get("default_ctx")))
        cod, r, _ = c.pedir("/api/config", {"default_ctx": 1024}, cabeceras=dict({"Content-Type": "text/plain"}, **cruzado))
        check("... tampoco con un POST 'simple' sin Origin (Sec-Fetch-Site: cross-site)", cod == 403, cod)
        cod, r, _ = c.pedir("/api/config", {"default_ctx": cf0["default_ctx"]}, cabeceras={"Origin": "http://127.0.0.1:%d" % puerto})
        check("el panel (mismo origen) sí puede guardar", cod == 200, (cod, r))
        cod, r, _ = c.pedir("/api/load", {"model": "lfm2-1.2b-rag"}, cabeceras=cruzado)
        check("otro sitio no puede cargar modelos (/api/load)", cod == 403, cod)
        cod, r, _ = c.pedir("/engine/api/next?motor=intruso", cabeceras=cruzado)
        check("otro sitio no puede pedir trabajos del motor", cod == 403, cod)
        cod, r, _ = c.pedir("/engine/api/event", {"id": "x"}, cabeceras=ajeno)
        check("otro sitio no puede inyectar eventos del motor", cod == 403, cod)
        cod, r, _ = c.pedir("/web/js/engine.js", crudo=True, cabeceras=cruzado)
        check("otro sitio no puede leer los archivos del motor", cod == 403, cod)
        cod, b, _ = c.pedir("/", crudo=True, cabeceras={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"})
        check("sí se puede navegar al panel desde un enlace de otro sitio", cod == 200, cod)
        cod, b, _ = c.pedir("/engine", crudo=True, cabeceras={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"})
        check("el panel (localhost) puede abrir el motor en 127.0.0.1 (navegación entre sitios)", cod == 200, cod)
        cod, r, h = c.pedir("/v1/models", cabeceras=ajeno)
        check("la API OpenAI (/v1) mantiene CORS abierto, como LM Studio", cod == 200 and h.get("Access-Control-Allow-Origin") == "*", (cod, h))
        cod, r, _ = c.pedir("/v1/models", cabeceras={"Host": "atacante.example:%d" % puerto})
        check("Host ajeno (DNS rebinding) -> 403", cod == 403 and "allowed_hosts" in r["error"]["message"], (cod, r))
        cod, r, _ = c.pedir("/api/status", cabeceras={"Host": "localhost:%d" % puerto})
        check("Host localhost aceptado", cod == 200, cod)
        cod, r, _ = c.pedir("/api/status", cabeceras={"Host": "%s:%d" % (st["pc_name"], puerto)})
        check("Host con el nombre de la PC aceptado", cod == 200, cod)

        print("\n== Por defecto: solo este PC")
        p2 = puerto_libre()
        srv2 = subprocess.Popen([sys.executable, os.path.join(AQUI, "server.py"), "--no-engine", "--port", str(p2),
                                 "--models-dir", carpeta, "--config", os.path.join(tmp, "no_hay_config.json")],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            c2 = Cliente("http://127.0.0.1:%d" % p2)
            st2 = None
            for _ in range(50):
                try:
                    st2 = c2.pedir("/api/status")[1]
                    break
                except Exception:
                    time.sleep(0.2)
            check("sin config escucha solo en 127.0.0.1 y lo dice en el panel",
                  st2 and st2["local_only"] is True and st2["reachable_at"] == ["http://localhost:%d" % p2], st2 and (st2["local_only"], st2["reachable_at"]))
            ips = [i for i in servidor_ips()]
            if ips:
                try:
                    socket.create_connection((ips[0], p2), timeout=2).close()
                    llega = True
                except OSError:
                    llega = False
                check("... y no acepta conexiones por la IP de la red (%s)" % ips[0], not llega)
        finally:
            srv2.terminate()
            srv2.communicate(timeout=5)
    except Exception:
        import traceback
        check("uncaught exception in the suite", False, traceback.format_exc()[-1500:])
    finally:
        if motor:
            motor.activo = False
        srv.terminate()
        try:
            srv.wait(timeout=5)
        except Exception:
            srv.kill()
        salida_srv.close()
        try:
            with open(ruta_salida, "rb") as f:
                salida = f.read().decode("utf-8", "replace")
        except OSError:
            salida = ""
        shutil.rmtree(tmp, ignore_errors=True)
    resiliencia()
    print("\n%d pruebas OK, %d fallas" % (OK[0], len(FALLAS)))
    if FALLAS:
        print("\n--- log del servidor ---\n" + salida[-6000:])
        print("\nFallaron:", *FALLAS, sep="\n  - ")
    return 1 if FALLAS else 0


def resiliencia():
    """Engine-connection resilience: watchdog, handle_error, engine.js timeout, versions."""
    import importlib.util
    import io
    import logging
    import re
    print("\n== resilience ==")
    spec = importlib.util.spec_from_file_location("server_mod", os.path.join(AQUI, "server.py"))
    sv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sv)

    class Falso:
        def __init__(self):
            self.pendientes, self.activos, self.motor = [], {}, {"visto": 0.0}
    puente = Falso()
    sv.ESTADO["puente"] = puente
    aperturas = []
    paso = sv.vigilar_motor({"port": 0}, intervalo=3600, sin_senal=20, reintento=90, abrir=lambda: aperturas.append(1))
    t0 = 1000.0
    puente.motor["visto"] = t0 - 500
    check("watchdog: does not reopen with no waiting requests", paso(t0) is False and not aperturas)
    puente.pendientes.append(object())
    puente.motor["visto"] = t0 - 5
    check("watchdog: does not reopen with a recent signal", paso(t0) is False and not aperturas)
    puente.motor["visto"] = t0 - 30
    check("watchdog: reopens once with pending requests + silent engine", paso(t0) is True and len(aperturas) == 1)
    check("watchdog: does not repeat before the retry interval", paso(t0 + 60) is False and len(aperturas) == 1)
    check("watchdog: retries after the interval", paso(t0 + 95) is True and len(aperturas) == 2)
    puente.activos["x"] = object()
    check("watchdog: does not reopen while a job is active", paso(t0 + 500) is False and len(aperturas) == 2)

    # handle_error: one log line vs traceback
    flujo = io.StringIO()
    h = logging.StreamHandler(flujo)
    sv.log.addHandler(h)
    nivel = sv.log.level
    sv.log.setLevel(logging.INFO)
    srvh = sv.ServidorHTTP.__new__(sv.ServidorHTTP)
    try:
        try:
            raise ConnectionAbortedError(10053, "aborted")
        except ConnectionAbortedError:
            srvh.handle_error(None, ("10.0.0.9", 5555))
        salida = flujo.getvalue()
        check("handle_error: ConnectionAbortedError -> one line, no traceback",
              "Connection cut by the client 10.0.0.9" in salida and "Traceback" not in salida
              and len(salida.strip().splitlines()) == 1, salida)
    finally:
        sv.log.removeHandler(h)
        sv.log.setLevel(nivel)
    err = io.StringIO()
    anterior, sys.stderr = sys.stderr, err
    try:
        try:
            raise ValueError("boom")
        except ValueError:
            srvh.handle_error(None, ("10.0.0.9", 5555))
    finally:
        sys.stderr = anterior
    check("handle_error: ValueError still shows a traceback", "Traceback" in err.getvalue() and "boom" in err.getvalue())

    # Linux: prefers a Chromium-based browser on PATH over the default one (Firefox has no WebGPU by default)
    esta = {"chromium": "/usr/bin/chromium", "brave-browser": "/usr/bin/brave-browser"}
    check("browser (Linux): picks a Chromium-based browser found on PATH",
          sv.buscar_navegador({}, "linux", esta.get) == "/usr/bin/chromium")
    check("browser (Linux): returns None when there is none (falls back to the default browser)",
          sv.buscar_navegador({}, "linux", lambda n: None) is None)
    check("browser: config 'browser' takes priority", sv.buscar_navegador({"browser": sys.executable}, "linux", esta.get) == sys.executable)

    # --share and the remote admin rule
    cfg_c = {"host": "127.0.0.1", "api_key": ""}
    ent = {}
    k = sv.preparar_compartir(cfg_c, ent)
    check("--share: listens on 0.0.0.0 and creates a key when none is set",
          cfg_c["host"] == "0.0.0.0" and bool(k) and len(k) >= 10 and ent.get("LLM_API_KEY") == k, (cfg_c, k))
    check("--share: the key is not written into the config dict", cfg_c["api_key"] == "")
    ent2 = {}
    check("--share: keeps an existing key from config.json (returns None)", sv.preparar_compartir({"host": "127.0.0.1", "api_key": "mi-clave"}, ent2) is None and "LLM_API_KEY" not in ent2)
    ent3 = {"LLM_API_KEY": "del-entorno"}
    check("--share: keeps the key from the environment", sv.preparar_compartir({"host": "127.0.0.1"}, ent3) is None and ent3["LLM_API_KEY"] == "del-entorno")
    check("--share: two runs create different keys", sv.preparar_compartir({}, {}) != sv.preparar_compartir({}, {}))
    check("max_tokens_efectivo: nothing asked, no default -> until the context is full", sv.max_tokens_efectivo(None) == -1)
    check("max_tokens_efectivo: default used when nothing is asked", sv.max_tokens_efectivo(None, 64) == 64)
    check("max_tokens_efectivo: asked value kept below the limit", sv.max_tokens_efectivo(80, 64, 100) == 80)
    check("max_tokens_efectivo: capped by the limit", sv.max_tokens_efectivo(500, 64, 100) == 100 and sv.max_tokens_efectivo(0, 0, 100) == 100 and sv.max_tokens_efectivo(-1, 0, 100) == 100)
    check("max_tokens_efectivo: no limit keeps big values", sv.max_tokens_efectivo(10000, 0, 0) == 10000)
    check("panel: the server PC always sees it", sv.panel_permitido(True, {}) is True)
    check("panel: another PC does not see it by default", sv.panel_permitido(False, {}) is False)
    check("panel: another PC sees it only with remote_panel", sv.panel_permitido(False, {"remote_panel": True}) is True)
    check("config defaults: chat_models empty and remote_panel off", sv.CONFIG_DEFECTO["chat_models"] == [] and sv.CONFIG_DEFECTO["remote_panel"] is False)
    cambios, errs = sv.validar_ajustes({"chat_models": ["a", "b", "a"]})
    check("validar_ajustes: chat_models without duplicates", cambios.get("chat_models") == ["a", "b"] and not errs, (cambios, errs))
    check("admin from another PC without a key is refused", sv.operacion_admin_permitida(False, "") is False)
    check("admin from another PC with a key is allowed", sv.operacion_admin_permitida(False, "k") is True)
    check("admin from the server PC is always allowed", sv.operacion_admin_permitida(True, "") is True)

    # the chat page: no external resources (the suite promises no outgoing connections), no innerHTML on model output
    chat = open(os.path.join(AQUI, "web", "chat.html"), encoding="utf-8").read()
    externos = re.findall(r'(?:src|href)\s*=\s*["\']https?://|url\(\s*["\']?https?://|@import|fetch\(\s*["\']https?://', chat)
    check("chat.html loads nothing from the internet", not externos, externos)
    ej = open(os.path.join(AQUI, "examples", "agent-chat.html"), encoding="utf-8").read()
    check("examples/agent-chat.html loads nothing from the internet", not re.findall(r'(?:src|href)\s*=\s*["\']https?://|url\(\s*["\']?https?://|@import', ej))
    check("examples/agent-chat.html inserts the model's text as text (no .innerHTML = )", ".innerHTML" not in ej)
    check("chat.html does not use innerHTML (model output is inserted as text)", "innerHTML" not in chat)
    panel = open(os.path.join(AQUI, "web", "panel.html"), encoding="utf-8").read()
    claves = {}
    for lang in ("en", "es"):
        bloque = panel.split("  %s: {" % lang, 1)[1].split("\n},", 1)[0]
        claves[lang] = set(re.findall(r'^ "([A-Za-z0-9_ ]+)":', bloque, re.M))
    check("panel i18n: English and Spanish have the same keys", claves["en"] == claves["es"], sorted(claves["en"] ^ claves["es"]))
    usadas = set(re.findall(r'data-ih?="([A-Za-z0-9_]+)"', panel))
    check("panel i18n: every data-i / data-ih key exists", usadas <= claves["en"], sorted(usadas - claves["en"]))
    ct = open(os.path.join(AQUI, "web", "chat.html"), encoding="utf-8").read()
    tx = {}
    for lang in ("en", "es"):
        bloque = ct.split("  %s: {" % lang, 1)[1].split("\n  }", 1)[0]
        tx[lang] = set(re.findall(r'(?:^\s*|",\s*)([A-Za-z]+):\s*"', bloque, re.M))
    check("chat i18n: English and Spanish have the same keys", tx["en"] == tx["es"], sorted(tx["en"] ^ tx["es"]))

    js = open(os.path.join(AQUI, "web", "js", "engine.js"), encoding="utf-8").read()
    check("engine.js: AbortController and 35000 ms cutoff", "AbortController" in js and "35000" in js)
    m = re.search(r'VERSION_MOTOR\s*=\s*"([^"]+)"', js)
    check("engine.js VERSION_MOTOR equals the server VERSION", bool(m) and m.group(1) == sv.VERSION,
          (m.group(1) if m else None, sv.VERSION))


if __name__ == "__main__":
    sys.exit(main())
