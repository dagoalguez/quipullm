"""End-to-end test of decision models (Liquid d1): server.py + WebGPU engine (Chromium) vs llama-server /v1/systemone.
The references (referencias_d1.json) come from gen_d1.py; the synthetic model is regenerated here and checked by hash."""
import asyncio, json, os, socket, subprocess, sys, time, urllib.request
from playwright.async_api import async_playwright

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import gen_d1
PROY = os.path.dirname(os.path.dirname(AQUI))
PUERTO = 18310
BASE = "http://127.0.0.1:%d" % PUERTO
FLAGS = ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"]
TOL = 1e-3          # probabilities: f16 weights, f32 computation here and in llama.cpp

ok, fallas = 0, []
def check(nombre, cond, detalle=""):
    global ok
    if cond: ok += 1; print("  OK   ", nombre)
    else: fallas.append(nombre); print("  FALLA", nombre, detalle)

def api(ruta, datos=None, timeout=600):
    req = urllib.request.Request(BASE + ruta, data=None if datos is None else json.dumps(datos).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def comparar(nombre, nuestro, ref):
    """Compares one /v1/systemone answer with the one of llama.cpp. Returns the largest difference of any probability."""
    mx = 0.0
    check(f"{nombre}: input_tokens igual ({ref['usage']['input_tokens']})", nuestro["usage"]["input_tokens"] == ref["usage"]["input_tokens"],
          f"nuestro {nuestro['usage']} llama {ref['usage']}")
    check(f"{nombre}: output_tokens = 0", nuestro["usage"]["output_tokens"] == 0)
    for qid, rq in ref["answers"].items():
        nq = nuestro["answers"].get(qid)
        if not nq:
            check(f"{nombre}/{qid}: hay respuesta", False); continue
        check(f"{nombre}/{qid}: mismo tipo ({rq['type']})", nq["type"] == rq["type"])
        if rq["type"] == "noul":
            d = abs(nq["noul"] - rq["noul"]); mx = max(mx, d)
            check(f"{nombre}/{qid}: P(true) {nq['noul']:.5f} vs {rq['noul']:.5f}", d < TOL)
            continue
        ps = list(rq["probabilities"].values())
        d = max(abs(nq["probabilities"][k] - v) for k, v in rq["probabilities"].items()); mx = max(mx, d)
        check(f"{nombre}/{qid}: mismas opciones y probabilidades (dif máx {d:.1e})", list(nq["probabilities"]) == list(rq["probabilities"]) and d < TOL,
              f"{nq['probabilities']} vs {rq['probabilities']}")
        check(f"{nombre}/{qid}: suman 1", abs(sum(nq["probabilities"].values()) - 1) < 1e-9)
        check(f"{nombre}/{qid}: confidence {nq['confidence']:.4f} vs {rq['confidence']:.4f}", abs(nq["confidence"] - rq["confidence"]) < 5 * TOL)
        if rq["type"] == "choice":
            orden = sorted(ps, reverse=True)
            if len(orden) < 2 or orden[0] - orden[1] > 2 * TOL:
                check(f"{nombre}/{qid}: misma opción elegida ({rq['choice']})", nq["choice"] == rq["choice"], f"{nq['choice']} vs {rq['choice']}")
        else:
            check(f"{nombre}/{qid}: score {nq['score']:.4f} vs {rq['score']:.4f}", abs(nq["score"] - rq["score"]) < 5 * TOL)
            check(f"{nombre}/{qid}: legend igual", nq["legend"] == rq["legend"], f"{nq['legend']} vs {rq['legend']}")
    return mx


async def main():
    if not os.path.exists(gen_d1.REF):
        print("Missing %s: run gen_d1.py with LLAMA_SERVER pointing to a llama-server that includes lfm2-d1." % gen_d1.REF)
        return 1
    refs = json.load(open(gen_d1.REF, encoding="utf-8"))
    ruta = gen_d1.crear()
    check("el modelo sintético es el de las referencias (hash)", gen_d1.hash_modelo(ruta) == refs["sha256_modelo"],
          "regenerate the references with gen_d1.py (numpy may have changed the random stream)")
    srv = subprocess.Popen([sys.executable, os.path.join(PROY, "server.py"), "--no-engine", "--port", str(PUERTO),
                            "--models-dir", os.path.join(AQUI, "cache", "modelos_test")], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", PUERTO), timeout=0.5).close(); break
        except OSError:
            time.sleep(0.1)
    loop = asyncio.get_running_loop()
    try:
        async with async_playwright() as pw:
            nav = await pw.chromium.launch(args=FLAGS, headless=True, channel="chromium")
            motor = await nav.new_page()
            logs = []
            motor.on("console", lambda m: logs.append(m.text))
            motor.on("pageerror", lambda e: logs.append("PAGEERROR " + str(e)))
            await motor.goto(BASE + "/engine")
            for _ in range(60):
                st = api("/api/status")[1]
                if st["engine"]["connected"]:
                    break
                await asyncio.sleep(0.5)
            check("motor conectado al servidor", st["engine"]["connected"], logs[-5:])
            modelo = next((x for x in st["models"] if x["file"] == "d1-test-f16.gguf"), None)
            check("el servidor reconoce el modelo de decisión", bool(modelo) and modelo["decision"] is True and modelo["supported"], modelo)
            check("los demás modelos no son de decisión", all(not x["decision"] for x in st["models"] if x["file"] != "d1-test-f16.gguf"))
            cod, lista = api("/api/v0/models")
            tipos = {x["id"]: x.get("type") for x in lista.get("data", [])}
            check("/api/v0/models: tipo 'decision'", tipos.get(modelo["id"]) == "decision", tipos)
            mid = modelo["id"]
            print("== comparación con llama.cpp (%s)" % refs["llama_cpp"])
            respuestas = {}
            for nombre, cuerpo in refs["pedidos"].items():
                ref = refs["respuestas"][nombre]
                cuerpo = dict(cuerpo, model=mid)
                cod, r = await loop.run_in_executor(None, lambda c=cuerpo: api("/v1/systemone", c))
                if "error" in ref:
                    check(f"{nombre}: llama.cpp responde error {ref['http']} y nosotros {cod}", cod == ref["http"], r)
                    continue
                check(f"{nombre}: 200", cod == 200, r)
                if cod != 200:
                    continue
                respuestas[nombre] = r
                comparar(nombre, r, ref)
            # negative controls: the comparison can tell different prompts apart
            if "estado_nulo" in respuestas:
                d = abs(respuestas["estado_nulo"]["answers"]["q"]["noul"] - refs["respuestas"]["noul_simple"]["answers"]["q"]["noul"])
                check(f"control negativo: otro estado y otra pregunta dan otro P(true) (dif {d:.1e} > {5 * TOL:.0e})", d > 5 * TOL)
            print("== errores y límites")
            cod, r = api("/v1/systemone", {"model": mid, "questions": {"q": {"type": "noul", "instructions": "x"}}})
            check("sin state: 400", cod == 400, r)
            cod, r = api("/v1/systemone", {"model": mid, "state": "x", "questions": {}})
            check("sin preguntas: 400", cod == 400, r)
            cod, r = api("/v1/systemone", {"model": mid, "state": "x", "questions": {"q": {"type": "choice", "instructions": "x"}}})
            check("choice sin criteria: 400 con el id de la pregunta", cod == 400 and "questions.q" in json.dumps(r), r)
            cod, r = api("/v1/systemone", {"model": mid, "state": "x", "questions": {"q": {"type": "otra", "instructions": "x"}}})
            check("tipo desconocido: 400", cod == 400, r)
            cod, r = api("/v1/systemone", {"model": mid, "state": "x", "images": ["data:image/png;base64,AAAA"], "questions": {"q": {"type": "noul", "instructions": "x"}}})
            check("imágenes: 501 (solo texto)", cod == 501, r)
            cod, r = api("/v1/systemone", {"model": mid, "state": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}],
                                           "questions": {"q": {"type": "noul", "instructions": "x"}}})
            check("imagen dentro del state: 501", cod == 501, r)
            otro = next(x for x in st["models"] if not x["decision"] and not x["embedding"])["id"]
            cod, r = api("/v1/systemone", {"model": otro, "state": "x", "questions": {"q": {"type": "noul", "instructions": "x"}}})
            check("modelo que no es de decisión: 501", cod == 501, r)
            cod, r = api("/v1/chat/completions", {"model": mid, "messages": [{"role": "user", "content": "hola"}]})
            check("chat con un modelo de decisión: 400 que apunta a /v1/systemone", cod == 400 and "systemone" in json.dumps(r), r)
            cod, r = api("/v1/systemone", {"model": mid, "state": "x" * 12000, "questions": {"q": {"type": "noul", "instructions": "x"}}})
            check("estado más largo que el contexto: 400 claro", cod == 400 and "context" in json.dumps(r), (cod, str(r)[:200]))
            # the next text request is not contaminated by the decision state
            ref = None
            cod, r = api("/v1/systemone", {"model": mid, "state": "El servidor responde en el puerto 1234.", "questions": {"q": {"type": "noul", "instructions": "Es una frase sobre redes?"}}})
            if cod == 200 and "noul_simple" in respuestas:
                check("repetir la misma petición da el mismo resultado", abs(r["answers"]["q"]["noul"] - respuestas["noul_simple"]["answers"]["q"]["noul"]) < 1e-6, r)
            check("sin errores JS en el motor", not [l for l in logs if "PAGEERROR" in l], [l for l in logs if "PAGEERROR" in l][:3])
            print("  (logs motor)", [l for l in logs if "Decision" in l][-3:])
            await nav.close()
    finally:
        srv.terminate()
        salida = srv.communicate(timeout=5)[0]
        if fallas:
            print(salida[-3000:])
    print(f"\n{ok} OK, {len(fallas)} fallas")
    return 1 if fallas else 0

sys.exit(asyncio.run(main()))
