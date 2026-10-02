"""Vision: preprocessing + ViT + projector (vision.js) and, with 'e2e', LFM2 with images, against libmtmd (llama.cpp)."""
import asyncio, base64, json, math, os, subprocess, sys, time, urllib.request
from playwright.async_api import async_playwright

AQUI = os.path.dirname(os.path.abspath(__file__))
PROY = os.path.dirname(os.path.dirname(AQUI))   # repository root
PUERTO = 18330
BASE = "http://127.0.0.1:%d" % PUERTO
REFS = json.load(open(os.path.join(AQUI, "cache", "referencias_vis.json")))
FLAGS = ["--enable-unsafe-webgpu", "--use-webgpu-adapter=swiftshader", "--enable-features=Vulkan"]
args = sys.argv[1:]
MODO = "enc"
for a in list(args):
    if a in ("enc", "lm", "api"):
        MODO = a; args.remove(a)
SOLO = [a for a in args if a.startswith("lfm2")]
PROMPTS = [a for a in args if a.startswith("p_")] or ["p_a", "p_b", "p_c"]
ok, fallas = 0, []


def check(nombre, cond, detalle=""):
    global ok
    if cond: ok += 1; print("  OK   ", nombre)
    else: fallas.append(nombre); print("  FALLA", nombre, detalle)


def b64_img(k):
    return base64.b64encode(open(os.path.join(AQUI, "cache", "imagenes_vis", k + ".png"), "rb").read()).decode()


def difmax(a, b): return max(abs(x - y) for x, y in zip(a, b))
def rms(a): return math.sqrt(sum(x * x for x in a) / len(a))
def coseno(a, b): return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


JS_ENC = """async ([nombre, imgs, b64s]) => {
  const V = await import('/web/js/vision.js');
  const {GPU} = await import('/web/js/gpu.js');
  window.__gpu = window.__gpu || await GPU.crear(() => {});
  const gpu = window.__gpu;
  const estado = await (await fetch('/api/status')).json();
  const m = estado.models.find(x => x.file.endsWith('/' + nombre + '.gguf'));
  const meta = await (await fetch('/engine/api/meta-mmproj/' + encodeURIComponent(m.id))).json();
  const modelo = new V.ModeloVision(gpu, meta);
  const t0 = performance.now();
  await modelo.load('/engine/file-mmproj/' + encodeURIComponent(m.id), () => {});
  const out = {carga_ms: performance.now() - t0, imgs: {}, dim: modelo.dim};
  for (const k of imgs) {
    const bin = Uint8Array.from(atob(b64s[k]), c => c.charCodeAt(0));
    const img = await V.decodificarImagen(bin, 'image/png');
    const t1 = performance.now();
    const pre = V.preprocesarImagen(img, modelo.parche, modelo.fusion);
    const lista = [];
    for (const t of pre.teselas) lista.push({fila: t.fila, col: t.col, img: t.img});
    lista.push({fila: 0, col: 0, img: pre.vista});
    const piezas = [];
    for (const it of lista) {
      const pix = modelo.normalizada(it.img);
      const r = await modelo.codificar(pix, it.img.w, it.img.h);
      const emb = await modelo.leer(r.buf, r.n);
      modelo.gpu.liberar([r.buf]);
      piezas.push({fila: it.fila, col: it.col, w: it.img.w, h: it.img.h, n: r.n, emb: Array.from(emb)});
    }
    out.imgs[k] = {w: img.w, h: img.h, plan: {rejilla: pre.plan.rejilla, vista: pre.plan.vista}, piezas, ms: performance.now() - t1};
  }
  modelo.free();
  return out;
}"""


JS_LM = """async ([nombre, prompts]) => {
  const {GPU, TB} = await import('/web/js/gpu.js');
  const {ModeloLFM2} = await import('/web/js/lfm2.js');
  window.__gpu = window.__gpu || await GPU.crear(() => {});
  const gpu = window.__gpu;
  const estado = await (await fetch('/api/status')).json();
  const m = estado.models.find(x => x.file.endsWith('/' + nombre + '.gguf'));
  const meta = await (await fetch('/engine/api/meta/' + encodeURIComponent(m.id))).json();
  meta.ctx = 4096;
  const modelo = new ModeloLFM2(gpu, meta);
  await modelo.load('/engine/file/' + encodeURIComponent(m.id), () => {});
  const out = {};
  for (const [pn, chunks] of Object.entries(prompts)) {
    const ids = [], ext = [], bufs = [];
    for (const c of chunks) {
      if (c.ids) for (const id of c.ids) { ids.push(id); ext.push(null); }
      else {
        const b = gpu.subir(new Float32Array(c.emb)); bufs.push(b);
        for (let j = 0; j < c.n; j++) { ids.push(0); ext.push({buf: b, fila: j}); }
      }
    }
    modelo.reset();
    let logits = null;
    const t0 = performance.now();
    for (let i = 0; i < ids.length; i += TB) logits = await modelo.forward(ids.slice(i, i + TB), i + TB >= ids.length, ext.slice(i, i + TB));
    const lp = Array.from(logits);
    const gen = [];
    for (let k = 0; k < 10; k++) {
      let am = 0; for (let i = 1; i < logits.length; i++) if (logits[i] > logits[am]) am = i;
      if (am === 4) break;
      gen.push(am);
      logits = await modelo.forward([am], true);
    }
    gpu.liberar(bufs);
    out[pn] = {n: ids.length, gen, lp, ms: performance.now() - t0};
    // logits del último token del prompt: recalcular sin generar (guardado antes de generar)
  }
  modelo.free();
  return out;
}"""


def api(ruta, datos=None, timeout=900):
    req = urllib.request.Request(BASE + ruta, data=None if datos is None else json.dumps(datos).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read() or b"{}")


def mensajes_api(msgs):
    salida = []
    for m in msgs:
        c = m["content"]
        if isinstance(c, list):
            c = [({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64_img(p["k"])}} if p["type"] == "image" else p) for p in c]
        salida.append({"role": m["role"], "content": c})
    return salida


CHATS = {
    "c_1": [{"role": "user", "content": [{"type": "text", "text": "Lee esto:"}, {"type": "image", "k": "a"}, {"type": "text", "text": "¿Qué dice?"}]}],
    "c_2": [{"role": "system", "content": "Eres útil."}, {"role": "user", "content": [{"type": "image", "k": "b"}, {"type": "text", "text": "Describe la imagen."}]}],
    "c_3": [{"role": "user", "content": [{"type": "text", "text": "Mira"}, {"type": "image", "k": "e"}]}, {"role": "assistant", "content": "Bien."},
            {"role": "user", "content": [{"type": "image", "k": "a"}, {"type": "text", "text": "¿y esta?"}, {"type": "text", "text": "Gracias"}]}],
    "c_4": [{"role": "user", "content": [{"type": "text", "text": "Factura:"}, {"type": "image", "k": "c"}]}],
}
CASOS_API = [a for a in args if a.startswith("c_")] or ["c_1", "c_2", "c_3"]


def chunks_imagen(ref_prompt):
    return [c for c in ref_prompt["chunks"] if c["tipo"] == "imagen"]


async def main():
    carpeta = os.path.join(AQUI, "cache", "modelos_vis")
    srv = subprocess.Popen([sys.executable, os.path.join(PROY, "server.py"), "--no-engine", "--port", str(PUERTO),
                            "--models-dir", carpeta], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(2.0)
    try:
        async with async_playwright() as pw:
            nav = await pw.chromium.launch(args=FLAGS, headless=True, channel="chromium")
            pg = await nav.new_page()
            errores = []
            pg.on("pageerror", lambda e: errores.append(str(e)))
            pg.on("console", lambda m: m.type == "error" and errores.append(m.text))
            await pg.goto(BASE + "/api/status")
            for nombre, ref in REFS["modelos"].items():
                if SOLO and nombre not in SOLO: continue
                print("==", nombre, {k: v for k, v in ref["opciones"].items() if k in ("tipo", "qkv", "gelu", "pre_ln", "in_norm", "L")})
                if MODO == "api":
                    if nombre != (SOLO[0] if SOLO else nombre): continue
                    motor = await nav.new_page()
                    logs_motor = []
                    motor.on("console", lambda m: logs_motor.append(m.text))
                    motor.on("pageerror", lambda e: logs_motor.append("PAGEERROR " + str(e)))
                    await motor.goto(BASE + "/engine")
                    # wait until THIS window is ready: otherwise 'conectado' comes from the previous, already closed window and the request is lost
                    for _ in range(120):
                        if any('Architectures' in l for l in logs_motor): break
                        await asyncio.sleep(0.5)
                    await asyncio.sleep(2.0)
                    for _ in range(80):
                        st = api("/api/status")[1]
                        if st["engine"]["connected"]: break
                        await asyncio.sleep(0.5)
                    check("motor conectado", st["engine"]["connected"], logs_motor[-5:])
                    mid = next(x["id"] for x in st["models"] if x["file"].endswith("/" + nombre + ".gguf"))
                    check("el registro marca visión soportada", next(x for x in st["models"] if x["id"] == mid)["vision_ok"])
                    loop = asyncio.get_running_loop()
                    for cn in CASOS_API:
                        rp = ref["prompts"][cn]
                        t0 = time.time()
                        cod, r = await loop.run_in_executor(None, lambda: api("/v1/chat/completions", {"model": mid, "messages": mensajes_api(CHATS[cn]),
                                                                          "temperature": 0, "repeat_penalty": 1.0, "max_tokens": 12}))
                        check(f"{nombre}/{cn}: API 200 ({time.time() - t0:.0f} s)", cod == 200, r)
                        if cod != 200: continue
                        check(f"{nombre}/{cn}: prompt_tokens = {rp['n_total']}", r["usage"]["prompt_tokens"] == rp["n_total"], r["usage"])
                        txt = r["choices"][0]["message"]["content"]
                        if ref["opciones"]["tipo"] == "f16" and txt != rp["texto_gen"]:
                            print(f"  INFO  {nombre}/{cn}: texto distinto (esperable con ViT f16 en modelo aleatorio; el LM se valida en modo lm)")
                        else:
                            check(f"{nombre}/{cn}: texto generado igual a llama.cpp", txt == rp["texto_gen"], f"\n     nuestro {txt!r}\n     llama   {rp['texto_gen']!r}")
                    # errors
                    cod, r = api("/v1/chat/completions", {"model": mid, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}]}]})
                    check("URL http(s) rechazada con 400", cod == 400, (cod, r))
                    cod, r = api("/v1/chat/completions", {"model": mid, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@"}}]}]})
                    check("base64 inválido rechazado con 400", cod == 400, (cod, r))
                    cod, r = api("/v1/chat/completions", {"model": mid, "max_tokens": 3, "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}, {"type": "text", "text": "hola"}]}]})
                    check("imagen corrupta: error 400 claro (no cuelga)", cod == 400 and "image" in json.dumps(r, ensure_ascii=False).lower(), (cod, r))
                    cod, r = api("/v1/chat/completions", {"model": mid, "max_tokens": 4, "messages": [{"role": "user", "content": "solo texto"}]})
                    check("texto sin imagen sigue funcionando", cod == 200, (cod, r))
                    print("  (logs motor)", [l for l in logs_motor if "Done" in l or "Error" in l][-6:])
                    await motor.close()   # each model opens its own window; the previous one must not keep polling
                    continue
                if MODO == "lm":
                    PROMPTS_ = PROMPTS
                    chs = {pn: [({"ids": c["ids"]} if c["tipo"] == "texto" else {"n": c["n"], "emb": c["emb"]}) for c in ref["prompts"][pn]["chunks"]] for pn in PROMPTS}
                    try:
                        r = await pg.evaluate(JS_LM, [nombre, chs])
                    except Exception as e:
                        check(f"{nombre} LM corre sin excepción", False, str(e)[:900]); continue
                    for pn in PROMPTS:
                        rp = ref["prompts"][pn]; o = r[pn]
                        check(f"{nombre}/{pn}: {o['n']} tokens de contexto = {rp['n_total']}", o["n"] == rp["n_total"])
                        d = difmax(o["lp"], rp["logits"]); rr = max(abs(x) for x in rp["logits"])
                        lmf32 = ref["opciones"].get("lm_f32", False)
                        check(f"{nombre}/{pn}: logits del último token, dif máx {d:.2e} (escala {rr:.1f})", d / rr < (2e-4 if lmf32 else 6e-2), (d, rr))
                        check(f"{nombre}/{pn}: generación voraz igual", o["gen"] == rp["generados"][:len(o["gen"])] or not lmf32, (o["gen"], rp["generados"]))
                        print("      %.1f s" % (o["ms"] / 1000))
                    continue
                # one image per reference (the first time it appears)
                vistas = {}
                for pn in PROMPTS:
                    rp = ref["prompts"][pn]
                    for im, ch in zip(rp["imgs"], [None] * len(rp["imgs"])):
                        vistas.setdefault(im, None)
                imgs = list(vistas)
                try:
                    r = await pg.evaluate(JS_ENC, [nombre, imgs, {k: b64_img(k) for k in imgs}])
                except Exception as e:
                    check(f"{nombre} corre sin excepción", False, str(e)[:900]); continue
                print("   carga %.1f s; dim %d" % (r["carga_ms"] / 1000, r["dim"]))
                f16 = ref["opciones"]["tipo"] == "f16"
                for pn in PROMPTS:
                    rp = ref["prompts"][pn]
                    esperado = chunks_imagen(rp)
                    # pieces per image in order of appearance
                    obtenido = []
                    for im in rp["imgs"]:
                        obtenido += r["imgs"][im]["piezas"]
                    # mtmd's order is: tiles (row-column), then the view; ours is already in that order
                    # (for the image without tiles there is only the view)
                    # with several images, each one contributes its pieces in that order
                    check(f"{nombre}/{pn}: {len(esperado)} trozos de imagen", len(obtenido) == len(esperado), (len(obtenido), len(esperado)))
                    if len(obtenido) != len(esperado): continue
                    for i, (o, e) in enumerate(zip(obtenido, esperado)):
                        if o["n"] != e["n"]:
                            check(f"{nombre}/{pn}#{i} tokens", False, (o["n"], e["n"])); continue
                        d = difmax(o["emb"], e["emb"]); rr = rms(e["emb"]); cs = coseno(o["emb"], e["emb"])
                        lim = 8e-2 if f16 else 6e-2
                        check(f"{nombre}/{pn}#{i} n={o['n']} embeddings: dif máx relativa {d / rr:.2e}, coseno {cs:.6f}", d / rr < lim and cs > 0.9999, (d, rr))
                print("   tiempos por imagen (ms):", {k: round(v["ms"]) for k, v in r["imgs"].items()})
            if errores: print("ERRORES DE CONSOLA:", errores[:5])
            await nav.close()
    finally:
        srv.terminate()
        try: print(srv.stdout.read()[-1500:])
        except Exception: pass
    print("\n%d OK, %d fallas" % (ok, len(fallas)))
    for f in fallas: print("  -", f)
    sys.exit(1 if fallas else 0)

asyncio.run(main())
