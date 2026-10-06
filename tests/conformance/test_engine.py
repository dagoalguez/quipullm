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


# Prompt cache (LFM2): the same computation with and without reusing the previous request's state.
JS_CACHE = """async (refs) => {
  const {GPU, TB} = await import('/web/js/gpu.js');
  const {ModeloLFM2} = await import('/web/js/lfm2.js');
  const estado = await (await fetch('/api/status')).json();
  const gpu = await GPU.crear(() => {});
  const res = {};
  const argmax = (l) => { let b = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[b]) b = i; return b; };
  const rel = (a, b) => { let md = 0, mr = 0; for (let i = 0; i < a.length; i++) { md = Math.max(md, Math.abs(a[i] - b[i])); mr = Math.max(mr, Math.abs(b[i])); } return md / Math.max(mr, 1e-6); };
  // the same loop as engine.js (a checkpoint before the last token; the batch never crosses it)
  const prefill = async (m, ids, saltar, conPunto) => {
    const corte = ids.length - 1, P = m.intervaloCP; let i = saltar, lg = null;
    while (i < ids.length) {
      if (conPunto && i === corte) m.guardarPunto();
      let fin = Math.min(i + TB, ids.length);
      if (conPunto && i < corte && fin > corte) fin = corte;
      if (conPunto) { const sig = (Math.floor(i / P) + 1) * P; if (fin > sig) fin = sig; }
      lg = await m.forward(ids.slice(i, fin), fin >= ids.length);
      i = fin;
      if (conPunto && fin < ids.length && fin % P === 0) m.guardarPunto();
    }
    if (conPunto) m.guardarPunto();      // as the engine does after the whole prompt
    return lg;
  };
  // greedy decoding; like the engine, the last sampled token is not fed back
  const decodificar = async (m, lg, k, conPunto) => {
    const out = [];
    for (let j = 0; j < k; j++) {
      const b = argmax(lg); out.push(b);
      if (j < k - 1) { lg = await m.forward([b], true); if (conPunto && m.pos % m.intervaloCP === 0) m.guardarPunto(); }
    }
    return out;
  };
  for (const [nombre, ref] of Object.entries(refs)) {
    const mm = estado.models.find(x => x.file === nombre + '.gguf');
    const meta = await (await fetch('/engine/api/meta/' + encodeURIComponent(mm.id))).json();
    meta.ctx = 512;
    const m = new ModeloLFM2(gpu, meta);
    await m.load('/engine/file/' + encodeURIComponent(mm.id), () => {});
    m.intervaloCP = 4;      // small, so that the short test prompts have several checkpoints
    const A = ref.prompts[0].ids, E = (ref.prompts[1] || ref.prompts[0]).ids.slice(0, 5);
    const K = 6, r = {opcion: m.cacheOn, nA: A.length};
    const pedido1 = async () => { m.reset(); const l = await prefill(m, A, 0, true); return decodificar(m, l, K, true); };
    const completo = async (ids) => { m.reset(); const l = await prefill(m, ids, 0, false); return [l, await decodificar(m, l, K)]; };
    // 1) the chat continues: previous prompt + its answer + new text
    const g1 = await pedido1();
    r.punto = m.puntos.map((p) => p.L).includes(A.length - 1);
    const ids2 = A.concat(g1, E);
    const n1 = m.reutilizar(ids2);
    const lc = await prefill(m, ids2, n1, true); const sc = await decodificar(m, lc, K, true);
    const [lf, sf] = await completo(ids2);
    r.cont = {n: n1, esperado: A.length + g1.length - 1, rel: rel(lc, lf), argmax: argmax(lc) === argmax(lf), seq: JSON.stringify(sc) === JSON.stringify(sf)};
    // 2) regenerate: the same prompt again
    await pedido1();
    const n2 = m.reutilizar(A);
    const lc2 = await prefill(m, A, n2, true); const sc2 = await decodificar(m, lc2, K);
    const [lf2, sf2] = await completo(A);
    r.regen = {n: n2, esperado: A.length - 1, rel: rel(lc2, lf2), argmax: argmax(lc2) === argmax(lf2), seq: JSON.stringify(sc2) === JSON.stringify(sf2)};
    // 3) it diverges before every checkpoint: nothing may be reused
    await pedido1();
    r.diverge = m.reutilizar(A.slice(0, 2).concat((A[2] + 5) % m.vocab, E));
    // 3b) a difference late in the history (inside the generated answer, as when its text is tokenized differently):
    //     it must go back to the last checkpoint before it (not to the start of the answer) and give exactly the same result
    const g1b = await pedido1();
    const ids3 = A.concat(g1b.slice(0, 3), [(g1b[3] + 5) % m.vocab], g1b.slice(4), E);
    const n3 = m.reutilizar(ids3);
    const lc3 = await prefill(m, ids3, n3, true); const sc3 = await decodificar(m, lc3, K, true);
    const [lf4, sf4] = await completo(ids3);
    let esp3 = A.length; for (let L = A.length + 1; L <= A.length + 3; L++) if (L % 4 === 0) esp3 = L;
    r.tarde = {n: n3, esperado: esp3, minimo: A.length, maximo: A.length + 3, rel: rel(lc3, lf4), argmax: argmax(lc3) === argmax(lf4), seq: JSON.stringify(sc3) === JSON.stringify(sf4)};
    // 3c) after that the history is coherent and the older checkpoints are still there for a second divergence
    r.puntos_ok = m.puntos.every((p, i, a) => p.L <= m.pos && (i === 0 || a[i - 1].L < p.L));
    // 4) negative control: pretend to reuse a state that does not belong to the prompt; the test must see the damage
    await pedido1();
    const A2 = A.slice(); A2[1] = (A2[1] + 7) % m.vocab;
    m.pos = A.length - 1;
    const lmalo = await m.forward(A2.slice(-1), true);
    const [lf3] = await completo(A2);
    r.control_rel = rel(lmalo, lf3);
    // 5) it must not reuse with the option off, nor after an invalidation
    await pedido1(); m.invalidar(); r.invalidado = m.reutilizar(ids2);
    await pedido1(); m.cacheOn = false; r.apagado = m.reutilizar(ids2); m.cacheOn = true;
    // 6) the whole prompt processed in one go and the same prompt from a checkpoint must leave the same state (pos, history)
    m.reset(); await prefill(m, A, 0, true); r.hist_ok = m.hist.length === m.pos && m.pos === A.length;
    m.free();
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
            # --- 1b) prompt cache: with reuse == without reuse (and the test can see a wrong reuse)
            rc = await pg.evaluate(JS_CACHE, {k: {"prompts": v["prompts"]} for k, v in REFS.items()})
            for nombre, r in rc.items():
                print("== caché de prompts:", nombre)
                check(f"{nombre} caché: la opción está activa por defecto", r["opcion"] is True)
                check(f"{nombre} caché: queda un punto de control justo antes del último token del prompt", r["punto"] is True, r)
                for clave, texto in (("cont", "la conversación continúa"), ("regen", "mismo prompt otra vez (regenerar)")):
                    c = r[clave]
                    check(f"{nombre} caché, {texto}: reutiliza {c['n']} tokens (esperado {c['esperado']})", c["n"] == c["esperado"] and c["n"] > 0, c)
                    check(f"{nombre} caché, {texto}: logits con caché = sin caché (dif rel {c['rel']:.1e})", c["rel"] < 2e-3, c)
                    check(f"{nombre} caché, {texto}: argmax y continuación voraz idénticos", c["argmax"] and c["seq"], c)
                check(f"{nombre} caché: si el prompt diverge antes de todo punto de control no se reutiliza nada", r["diverge"] == 0, r)
                t = r["tarde"]
                check(f"{nombre} caché, diferencia tardía en la respuesta anterior: vuelve al último punto antes de ella ({t['n']}, esperado {t['esperado']}), no al principio de la respuesta", t["n"] == t["esperado"] and t["minimo"] <= t["n"] <= t["maximo"], t)
                check(f"{nombre} caché, diferencia tardía: logits con caché = sin caché (dif rel {t['rel']:.1e})", t["rel"] < 2e-3, t)
                check(f"{nombre} caché, diferencia tardía: argmax y continuación voraz idénticos", t["argmax"] and t["seq"], t)
                check(f"{nombre} caché: los puntos de control quedan ordenados y dentro de la posición", r["puntos_ok"], r)
                check(f"{nombre} caché: control negativo, reutilizar un estado ajeno se nota (dif rel {r['control_rel']:.1e})", r["control_rel"] > 1e-2, r)
                check(f"{nombre} caché: tras invalidar y con la opción apagada no se reutiliza", r["invalidado"] == 0 and r["apagado"] == 0, r)
                check(f"{nombre} caché: el historial coincide con la posición", r["hist_ok"], r)
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
            cod, r = await loop.run_in_executor(None, lambda: api("/v1/completions", {
                "model": ids["lfm2-test-q8.gguf"], "prompt": ref["prompt"], "temperature": 0,
                "repeat_penalty": 1.0, "max_tokens": len(ref["generados"]) + 1}))
            check("API: la misma petición otra vez da el mismo texto de llama.cpp (con caché)", cod == 200 and r["choices"][0]["text"] == ref["texto"], (cod, r))
            await asyncio.sleep(0.3)
            check("API: el motor reutilizó el prompt anterior (se ve en su registro)", any("reused from the previous request" in l for l in logs_motor), [l for l in logs_motor if "Done" in l][-3:])
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
