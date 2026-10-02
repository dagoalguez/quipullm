"""BERT engine (bert.js) + WPM tokenizer vs llama.cpp with synthetic nomic-bert models; and /v1/embeddings end to end."""
import socket
import asyncio, base64, json, math, os, struct, subprocess, sys, time, urllib.request
from playwright.async_api import async_playwright

AQUI = os.path.dirname(os.path.abspath(__file__))
PROY = os.path.dirname(os.path.dirname(AQUI))   # repository root
PUERTO = 18320
BASE = "http://127.0.0.1:%d" % PUERTO
REFS = json.load(open(os.path.join(AQUI, "cache", "referencias_bert.json"), encoding="utf-8"))
FLAGS = ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"]
SOLO = sys.argv[1:]
ok, fallas = 0, []
def check(nombre, cond, detalle=""):
    global ok
    if cond: ok += 1; print("  OK   ", nombre)
    else: fallas.append(nombre); print("  FALLA", nombre, detalle)

def api(ruta, datos=None, timeout=300):
    req = urllib.request.Request(BASE + ruta, data=None if datos is None else json.dumps(datos).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read())

def difmax(a, b): return max(abs(x - y) for x, y in zip(a, b))
def coseno(a, b): return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

JS = """async ([nombre, ref]) => {
  const {Tokenizador} = await import('/web/js/tokenizer.js');
  const {GPU} = await import('/web/js/gpu.js');
  const {ModeloBert} = await import('/web/js/bert.js');
  const estado = await (await fetch('/api/status')).json();
  window.__gpu = window.__gpu || await GPU.crear(() => {});
  const gpu = window.__gpu;
  const m = estado.models.find(x => x.file === nombre + '.gguf');
  const meta = await (await fetch('/engine/api/meta/' + encodeURIComponent(m.id))).json();
  meta.ctx = 512;
  if (ref.sinb) meta.options = {tiled_prefill: false};
  const tok = new Tokenizador(meta.kv);
  const r = {tokens: [], embeds: []};
  for (const [texto, esperado] of Object.entries(ref.q.tokens)) {
    const ids = tok.codificar(texto);
    r.tokens.push({texto, ok: JSON.stringify(ids) === JSON.stringify(esperado), ids, esperado});
  }
  const modelo = new ModeloBert(gpu, meta);
  await modelo.load('/engine/file/' + encodeURIComponent(m.id), () => {});
  for (const e of ref.q.embeds) {
    const ids = tok.codificar(e.texto);
    const v = await modelo.embed(ids);
    r.embeds.push({ids_ok: JSON.stringify(ids) === JSON.stringify(e.ids), n: ids.length, v: Array.from(v)});
  }
  modelo.free();
  return r;
}"""

async def main():
    srv = subprocess.Popen([sys.executable, os.path.join(PROY, "server.py"), "--no-engine", "--port", str(PUERTO),
                            "--models-dir", os.path.join(AQUI, "cache", "modelos_tf")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(100):   # wait until the server is listening
        try:
            socket.create_connection(("127.0.0.1", PUERTO), timeout=0.5).close(); break
        except OSError:
            time.sleep(0.1)
    try:
        async with async_playwright() as pw:
            nav = await pw.chromium.launch(args=FLAGS, headless=True, channel="chromium")
            pg = await nav.new_page()
            errores = []
            pg.on("pageerror", lambda e: errores.append(str(e)))
            pg.on("console", lambda m: m.type == "error" and errores.append(m.text))
            await pg.goto(BASE + "/api/status")
            resultados = {}
            for nombre, ref in REFS.items():
                if SOLO and nombre not in SOLO and "e2e" not in SOLO: continue
                if os.environ.get('SINB'): ref['sinb'] = True
                print("==", nombre, ref["tipos"])
                try:
                    r = await pg.evaluate(JS, [nombre, ref])
                except Exception as e:
                    check(f"{nombre} corre sin excepción", False, str(e)[:600]); continue
                resultados[nombre] = r
                bad = [t for t in r["tokens"] if not t["ok"]]
                check(f"{nombre} tokeniza (WPM) igual que llama.cpp {len(r['tokens'])-len(bad)}/{len(r['tokens'])}", not bad,
                      [(t['texto'][:24], t['ids'][:14], t['esperado'][:14]) for t in bad[:3]])
                for e, rq, rd in zip(r["embeds"], ref["q"]["embeds"], ref["deq"]["embeds"]):
                    etq = rq["texto"][:28].replace("\n", " ")
                    dd = difmax(e["v"], rd["vec"]); cq = coseno(e["v"], rq["vec"])
                    check(f"{nombre} embedding ({e['n']} tok) '{etq}' vs F32-deq: dif máx {dd:.1e}", e["ids_ok"] and dd < 3e-4, e["ids_ok"])
                    cb = coseno(rd["vec"], rq["vec"])   # llama.cpp's own noise: it quantizes activations to Q8 in the matmuls
                    check(f"{nombre} ... vs llama.cpp cuantizado: coseno {cq:.5f} (llama.cpp con pesos F32 vs cuantizado: {cb:.5f})", abs(cq - cb) < 1e-3)
            check("sin errores JS", not errores, errores[:3])
            if not SOLO or "e2e" in SOLO:
                motor = await nav.new_page()
                logs = []
                motor.on("console", lambda m: logs.append(m.text))
                motor.on("pageerror", lambda e: logs.append("PAGEERROR " + str(e)))
                await motor.goto(BASE + "/engine")
                for _ in range(60):
                    st = api("/api/status")[1]
                    if st["engine"]["connected"]: break
                    await asyncio.sleep(0.5)
                check("motor conectado", st["engine"]["connected"], logs[-5:])
                ids = {x["file"]: x["id"] for x in st["models"]}
                loop = asyncio.get_running_loop()
                print("== extremo a extremo: /v1/embeddings")
                for nombre in ("nomic-bert-q4km", "nomic-bert-q6k-cls"):
                    ref = REFS[nombre]
                    textos = [e["texto"] for e in ref["deq"]["embeds"][:4]]
                    cod, r = await loop.run_in_executor(None, lambda: api("/v1/embeddings", {"model": ids[nombre + ".gguf"], "input": textos}))
                    check(f"{nombre}: /v1/embeddings 200", cod == 200, r)
                    if cod != 200: continue
                    check(f"{nombre}: formato OpenAI (object list, index, model)", r["object"] == "list" and [d["index"] for d in r["data"]] == [0, 1, 2, 3]
                          and all(d["object"] == "embedding" for d in r["data"]) and r["model"] == ids[nombre + ".gguf"], r.get("model"))
                    dd = max(difmax(d["embedding"], e["vec"]) for d, e in zip(r["data"], ref["deq"]["embeds"][:4]))
                    check(f"{nombre}: vectores iguales a llama.cpp (dif máx {dd:.1e})", dd < 3e-4)
                    esp = sum(len(e["ids"]) for e in ref["deq"]["embeds"][:4])
                    check(f"{nombre}: usage.prompt_tokens = {esp}", r["usage"]["prompt_tokens"] == esp and r["usage"]["total_tokens"] == esp, r["usage"])
                    cod, r = await loop.run_in_executor(None, lambda: api("/v1/embeddings", {"model": nombre, "input": textos[1], "encoding_format": "base64"}))
                    if cod == 200:
                        v = struct.unpack("<%df" % 256, base64.b64decode(r["data"][0]["embedding"]))
                        check(f"{nombre}: base64 decodifica al mismo vector (dif {difmax(v, ref['deq']['embeds'][1]['vec']):.1e})", difmax(v, ref["deq"]["embeds"][1]["vec"]) < 3e-4)
                    else: check(f"{nombre}: base64 200", False, r)
                cod, r = await loop.run_in_executor(None, lambda: api("/v1/embeddings", {"model": "nomic-bert-q4km", "input": ""}))
                check("texto vacío -> 200 (solo [CLS] [SEP])", cod == 200 and len(r["data"][0]["embedding"]) == 256, (cod, r))
                cod, r = await loop.run_in_executor(None, lambda: api("/v1/embeddings", {"model": "nomic-bert-q4km", "input": "palabra " * 600}))
                check("texto más largo que el contexto -> 400 claro", cod == 400 and "tokens" in r["error"]["message"], (cod, r))
                cod, r = await loop.run_in_executor(None, lambda: api("/v1/embeddings", {"model": "nomic-bert-q4km", "input": 5}))
                check("input inválido -> 400", cod == 400, r)
                cod, r = await loop.run_in_executor(None, lambda: api("/v1/embeddings", {"model": "qwen2-q4km", "input": "x"}))
                check("modelo de chat usado para embeddings -> 400", cod == 400, (cod, r))
                cod, r = await loop.run_in_executor(None, lambda: api("/v1/chat/completions", {"model": "nomic-bert-q4km", "messages": [{"role": "user", "content": "x"}]}))
                check("modelo de embeddings usado para chat -> 400", cod == 400, (cod, r))
                cod, r = await loop.run_in_executor(None, lambda: api("/v1/chat/completions", {"model": "qwen2-q4km", "messages": [{"role": "user", "content": "La base de datos"}], "temperature": 0, "max_tokens": 4}))
                check("chat vuelve a funcionar tras embeddings (el motor cambia de modelo)", cod == 200 and r["usage"]["completion_tokens"] > 0, (cod, r))
                print("  (logs motor)", [l for l in logs if "Error" in l][-3:])
            await nav.close()
    finally:
        srv.terminate()
        salida = srv.communicate(timeout=5)[0]
        if fallas: print(salida[-2000:])
    print(f"\n{ok} OK, {len(fallas)} fallas")
    return 1 if fallas else 0

sys.exit(asyncio.run(main()))
