"""End-to-end test: server.py + WebGPU engine (Chromium) vs llama.cpp."""
import socket
import asyncio, json, os, subprocess, sys, time, urllib.request
from playwright.async_api import async_playwright

AQUI = os.path.dirname(os.path.abspath(__file__))
PROY = os.path.dirname(os.path.dirname(AQUI))   # repository root
PUERTO = 18300
BASE = "http://127.0.0.1:%d" % PUERTO
REFS = json.load(open(os.path.join(AQUI, "cache", "referencias_llamacpp.json"), encoding="utf-8"))
FLAGS = ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"]

ok, fallas = 0, []
def check(nombre, cond, detalle=""):
    global ok
    if cond: ok += 1; print("  OK   ", nombre)
    else: fallas.append(nombre); print("  FALLA", nombre, detalle)

def api(ruta, datos=None, timeout=300):
    req = urllib.request.Request(BASE + ruta, data=None if datos is None else json.dumps(datos).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")

JS_MODULOS = """async (refs) => {
  const {Tokenizador} = await import('/web/js/tokenizer.js');
  const {GPU, TB} = await import('/web/js/gpu.js');
  const {ModeloLFM2} = await import('/web/js/lfm2.js');
  const estado = await (await fetch('/api/status')).json();
  const gpu = await GPU.crear(() => {});
  const res = {};
  for (const [nombre, ref] of Object.entries(refs)) {
    const m = estado.models.find(x => x.file === nombre + '.gguf');
    const meta = await (await fetch('/engine/api/meta/' + encodeURIComponent(m.id))).json();
    meta.ctx = 512;
    const tok = new Tokenizador(meta.kv);
    const r = {tokens: [], prompts: []};
    for (const [texto, esperado] of Object.entries(ref.tokens)) {
      const ids = tok.codificar(texto);
      r.tokens.push({texto, ok: JSON.stringify(ids) === JSON.stringify(esperado), ids, esperado,
                     ida_vuelta: tok.decodificar(ids) === texto});
    }
    const modelo = new ModeloLFM2(gpu, meta);
    await modelo.load('/engine/file/' + encodeURIComponent(m.id), () => {});
    for (const p of ref.prompts) {
      modelo.reset();
      let logits;
      for (let i = 0; i < p.ids.length; i += TB) logits = await modelo.forward(p.ids.slice(i, i + TB), i + TB >= p.ids.length);
      let maxd = 0, maxref = 0, am = 0, amr = 0;
      for (let i = 0; i < logits.length; i++) {
        maxd = Math.max(maxd, Math.abs(logits[i] - p.logits_ultimo[i]));
        maxref = Math.max(maxref, Math.abs(p.logits_ultimo[i]));
        if (logits[i] > logits[am]) am = i;
        if (p.logits_ultimo[i] > p.logits_ultimo[amr]) amr = i;
      }
      const gen = [];
      let lg = logits;
      for (let k = 0; k < p.generados.length; k++) {
        let b = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[b]) b = i;
        gen.push(b);
        lg = await modelo.forward([b], true);
      }
      r.prompts.push({n: p.ids.length, maxd, maxref, argmax_ok: am === amr, gen, esperado: p.generados,
                      iguales: gen.filter((g, i) => g === p.generados[i]).length});
    }
    modelo.free();
    res[nombre] = r;
  }
  return res;
}"""

async def main():
    env = dict(os.environ)
    srv = subprocess.Popen([sys.executable, os.path.join(PROY, "server.py"), "--no-engine", "--port", str(PUERTO),
                            "--models-dir", os.path.join(AQUI, "cache", "modelos_test")], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    for _ in range(100):   # wait until the server is listening
        try:
            socket.create_connection(("127.0.0.1", PUERTO), timeout=0.5).close(); break
        except OSError:
            time.sleep(0.1)
    try:
        async with async_playwright() as pw:
            nav = await pw.chromium.launch(args=FLAGS, headless=True, channel="chromium")
            # --- 1) modules: tokenizer and compute against llama.cpp
            pg = await nav.new_page()
            errores_js = []
            pg.on("pageerror", lambda e: errores_js.append(str(e)))
            pg.on("console", lambda m: m.type == "error" and errores_js.append(m.text))
            await pg.goto(BASE + "/api/status")
            res = await pg.evaluate(JS_MODULOS, {k: {"tokens": v["tokens"], "prompts": v["prompts"]} for k, v in REFS.items()})
            for nombre, r in res.items():
                print("==", nombre)
                for t in r["tokens"]:
                    check(f"{nombre} tokeniza {t['texto'][:30]!r}", t["ok"], f"\n     nuestro {t['ids']}\n     llama   {t['esperado']}")
                    check(f"{nombre} ida y vuelta {t['texto'][:30]!r}", t["ida_vuelta"])
                for i, p in enumerate(r["prompts"]):
                    rel = p["maxd"] / max(p["maxref"], 1e-6)
                    check(f"{nombre} prompt{i} ({p['n']} tok) logits dif máx {p['maxd']:.2e} (rel {rel:.1e})", rel < 2e-3)
                    check(f"{nombre} prompt{i} argmax igual", p["argmax_ok"])
                    check(f"{nombre} prompt{i} greedy {p['iguales']}/{len(p['esperado'])} tokens iguales",
                          p["iguales"] == len(p["esperado"]), f"\n     nuestro {p['gen']}\n     llama   {p['esperado']}")
            check("sin errores JS en módulos", not errores_js, errores_js[:3])
            # --- 2) real API with the engine running
            motor = await nav.new_page()
            logs_motor = []
            motor.on("console", lambda m: logs_motor.append(m.text))
            motor.on("pageerror", lambda e: logs_motor.append("PAGEERROR " + str(e)))
            await motor.goto(BASE + "/engine")
            for _ in range(60):
                st = api("/api/status")[1]
                if st["engine"]["connected"]:
                    break
                await asyncio.sleep(0.5)
            check("motor conectado al servidor", st["engine"]["connected"], logs_motor[-5:])
            ids = {x["file"]: x["id"] for x in st["models"]}
            ref = REFS["lfm2-test-q8"]["prompts"][0]
            loop = asyncio.get_running_loop()
            cod, r = await loop.run_in_executor(None, lambda: api("/v1/completions", {
                "model": ids["lfm2-test-q8.gguf"], "prompt": ref["prompt"], "temperature": 0,
                "repeat_penalty": 1.0, "max_tokens": len(ref["generados"]) + 1}))
            check("API /v1/completions responde 200", cod == 200, r)
            if cod == 200:
                check("API completion = texto de llama.cpp", r["choices"][0]["text"] == ref["texto"],
                      f"\n     nuestro {r['choices'][0]['text']!r}\n     llama   {ref['texto']!r}")
                check("max_tokens=N+1: texto de N tokens (como LM Studio) y completion_tokens=N+1",
                      len(ref["generados"]) <= r["usage"]["completion_tokens"] <= len(ref["generados"]) + 1, r["usage"])
                check("API usage.prompt_tokens correcto", r["usage"]["prompt_tokens"] == len(ref["ids"]), r["usage"])
            ref2 = REFS["lfm2-test-f16-swa"]["prompts"][2]
            cod, r = await loop.run_in_executor(None, lambda: api("/v1/completions", {
                "model": ids["lfm2-test-f16-swa.gguf"], "prompt": ref2["prompt"], "temperature": 0,
                "repeat_penalty": 1.0, "max_tokens": len(ref2["generados"]) + 1}))
            check("API cambia de modelo y coincide (F16 + SWA + qkv fusionado)",
                  cod == 200 and r["choices"][0]["text"] == ref2["texto"], (cod, r.get("choices", r)))
            print("  (logs motor)", [l for l in logs_motor if "Done" in l or "Error" in l][-4:])
            await nav.close()
    finally:
        srv.terminate()
        salida = srv.communicate(timeout=5)[0]
        if fallas:
            print(salida[-3000:])
    print(f"\n{ok} OK, {len(fallas)} fallas")
    return 1 if fallas else 0

sys.exit(asyncio.run(main()))
