"""WebGPU engine (transformer.js + K-quants) vs llama.cpp with synthetic qwen2/llama/granite models."""
import socket
import asyncio, json, os, subprocess, sys, time, urllib.request
from playwright.async_api import async_playwright

AQUI = os.path.dirname(os.path.abspath(__file__))
PROY = os.path.dirname(os.path.dirname(AQUI))   # repository root
CACHE = os.path.join(AQUI, "cache")
PUERTO = 18310
BASE = "http://127.0.0.1:%d" % PUERTO
REFS = json.load(open(os.path.join(CACHE, "referencias_tf.json"), encoding="utf-8"))
REFS.update(json.load(open(os.path.join(CACHE, "referencias_gemma.json"), encoding="utf-8")))
FLAGS = ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"]
SOLO = sys.argv[1:]

EMPATES = {('nemo-q3kl', 1)}
ok, fallas = 0, []
def check(nombre, cond, detalle=""):
    global ok
    if cond: ok += 1; print("  OK   ", nombre)
    else: fallas.append(nombre); print("  FALLA", nombre, detalle)

def api(ruta, datos=None, timeout=300):
    req = urllib.request.Request(BASE + ruta, data=None if datos is None else json.dumps(datos).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read())

JS = """async ([nombre, ref]) => {
  const {Tokenizador} = await import('/web/js/tokenizer.js');
  const {GPU, TB} = await import('/web/js/gpu.js');
  const {ModeloTransformer} = await import('/web/js/transformer.js');
  const estado = await (await fetch('/api/status')).json();
  window.__gpu = window.__gpu || await GPU.crear(() => {});
  const gpu = window.__gpu;
  const m = estado.models.find(x => x.file === nombre + '.gguf');
  const meta = await (await fetch('/engine/api/meta/' + encodeURIComponent(m.id))).json();
  meta.ctx = 512;
  if (ref.sinb) meta.options = {tiled_prefill: false};
  const tok = new Tokenizador(meta.kv);
  const r = {tokens: [], prompts: [], pre: tok.pre};
  for (const [texto, esperado] of Object.entries(ref.q.tokens)) {
    const ids = tok.codificar(texto, {addBos: ref.q.add_bos});
    r.tokens.push({texto, ok: JSON.stringify(ids) === JSON.stringify(esperado), ids, esperado});
  }
  const modelo = new ModeloTransformer(gpu, meta);
  await modelo.load('/engine/file/' + encodeURIComponent(m.id), () => {});
  for (let k = 0; k < ref.q.prompts.length; k++) {
    const pq = ref.q.prompts[k], pd = ref.deq.prompts[k];
    modelo.reset();
    let logits;
    const TL = modelo.batch || TB;
    for (let i = 0; i < pq.ids.length; i += TL) logits = await modelo.forward(pq.ids.slice(i, i + TL), i + TL >= pq.ids.length);
    const cmp = (p) => {
      let maxd = 0, maxref = 0, am = 0, amr = 0;
      for (let i = 0; i < logits.length; i++) {
        maxd = Math.max(maxd, Math.abs(logits[i] - p.logits_ultimo[i]));
        maxref = Math.max(maxref, Math.abs(p.logits_ultimo[i]));
        if (logits[i] > logits[am]) am = i;
        if (p.logits_ultimo[i] > p.logits_ultimo[amr]) amr = i;
      }
      return {maxd, maxref, argmax_ok: am === amr};
    };
    const cq = cmp(pq), cd = cmp(pd);
    const gen = []; let lg = logits;
    for (let j = 0; j < pd.generados.length; j++) {
      let b = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[b]) b = i;
      gen.push(b); lg = await modelo.forward([b], true);
    }
    // decodificación de 1 token con historial (camino vec4), comparando con la referencia F32: se hizo arriba.
    r.prompts.push({n: pq.ids.length, q: cq, d: cd, gen, esp_d: pd.generados, esp_q: pq.generados});
  }
  modelo.free();
  return r;
}"""

async def main():
    srv = subprocess.Popen([sys.executable, os.path.join(PROY, "server.py"), "--no-engine", "--port", str(PUERTO),
                            "--models-dir", os.path.join(CACHE, "modelos_tf")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
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
            for nombre, ref in REFS.items():
                if SOLO and nombre not in SOLO: continue
                if os.environ.get('SINB'): ref['sinb'] = True
                print("==", nombre, ref["tipos"])
                try:
                    r = await pg.evaluate(JS, [nombre, ref])
                except Exception as e:
                    check(f"{nombre} corre sin excepción", False, str(e)[:600]); continue
                bad = [t for t in r["tokens"] if not t["ok"]]
                check(f"{nombre} tokeniza igual que llama.cpp (pre={r['pre']}) {len(r['tokens'])-len(bad)}/{len(r['tokens'])}", not bad,
                      [(t['texto'][:20], t['ids'][:12], t['esperado'][:12]) for t in bad[:2]])
                for i, p in enumerate(r["prompts"]):
                    rd = p["d"]["maxd"] / max(p["d"]["maxref"], 1e-6)
                    rq = p["q"]["maxd"] / max(p["q"]["maxref"], 1e-6)
                    check(f"{nombre} prompt{i} ({p['n']} tok) vs F32-deq: dif rel {rd:.1e}", rd < 2e-3)
                    check(f"{nombre} prompt{i} vs llama.cpp cuantizado: dif rel {rq:.1e} argmax={p['q']['argmax_ok']}", rq < 1e-1)
                    it = sum(1 for a, b in zip(p["gen"], p["esp_d"]) if a == b)
                    empate = (nombre, i) in EMPATES   # logit tie (gap 0.0016): llama.cpp itself is unstable there
                    check(f"{nombre} prompt{i} greedy vs F32-deq {it}/{len(p['esp_d'])}" + (" (empate conocido)" if empate else ""),
                          it == len(p["esp_d"]) or (empate and it >= 3),
                          f"\n     nuestro {p['gen']}\n     ref     {p['esp_d']}")
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
                from llama_cpp import Llama
                print("== extremo a extremo (chat con plantilla de respaldo, texto = llama.cpp)")
                for nombre, fam in (("qwen2-q4km", "qwen"), ("qwen3-q4km", "qwen3"), ("granite-q5km", "granite"), ("nemo-q3kl", "nemo"), ("gemma3-q4km", "gemma"), ("gemma3-q5km-sc", "gemma")):
                    msgs = [{"role": "user", "content": "La base de datos"}]
                    cod, r = await loop.run_in_executor(None, lambda: api("/v1/chat/completions", {
                        "model": ids[nombre + ".gguf"], "messages": msgs, "temperature": 0, "repeat_penalty": 1.0, "max_tokens": 13}))
                    check(f"{nombre}: /v1/chat/completions 200", cod == 200, r)
                    if cod != 200: continue
                    u = r["usage"]
                    # prompt rebuilt the same way as the server and evaluated with llama.cpp on the F32 weights
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("server", PROY + "/server.py"); sv = importlib.util.module_from_spec(spec); spec.loader.exec_module(sv); sv.cargar_manifiestos()
                    texto, incluye = sv.plantilla_chat({"qwen": "qwen2", "qwen3": "qwen3", "granite": "granite", "nemo": "llama", "gemma": "gemma3"}[fam], msgs)
                    if fam == "granite":   # the default system date is today's; it is reproduced identically
                        pass
                    llm = Llama(model_path=os.path.join(CACHE, "modelos_tf", nombre + "-deq.gguf"), n_ctx=512, logits_all=True, verbose=False, n_threads=2)
                    add_bos = REFS[nombre]["q"]["add_bos"] and not incluye
                    toks = llm.tokenize(texto.encode(), add_bos=add_bos, special=True)
                    check(f"{nombre}: prompt_tokens igual que llama.cpp ({len(toks)})", u["prompt_tokens"] == len(toks), (u, len(toks)))
                    llm.eval(toks); gen = []
                    for _ in range(12):
                        nx = int(llm.scores[llm.n_tokens - 1].argmax())
                        gen.append(nx); llm.eval([nx])
                    esp = llm.detokenize(gen).decode("utf-8", "replace")
                    igual = r["choices"][0]["message"]["content"] == esp
                    check(f"{nombre}: texto generado igual que llama.cpp", igual, f"\n     nuestro {r['choices'][0]['message']['content']!r}\n     llama   {esp!r}")
                print("  (logs motor)", [l for l in logs if "Error" in l][-3:])
            await nav.close()
    finally:
        srv.terminate()
        salida = srv.communicate(timeout=5)[0]
        if fallas: print(salida[-2000:])
    print(f"\n{ok} OK, {len(fallas)} fallas")
    return 1 if fallas else 0

sys.exit(asyncio.run(main()))
