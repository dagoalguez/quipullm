// Engine main loop: asks the server for jobs, loads models and generates text.
import { GPU, TB } from "./gpu.js";
import { Tokenizador } from "./tokenizer.js";
import { ModeloLFM2 } from "./lfm2.js";
import { ModeloTransformer } from "./transformer.js";
import { ModeloDeepseek2 } from "./deepseek2.js";
import { ModeloBert } from "./bert.js";
import { ModeloVision, decodificarImagen, preprocesarImagen, redimensionar } from "./vision.js";
import { Muestreador } from "./sampling.js";

// Core classes that a declarative manifest can name in "base". Architectures are discovered in web/arch/*.json
// (the server lists them at /api/architectures); a manifest with "modulo" loads its own class with import().
const BASES = { transformer: ModeloTransformer, lfm2: ModeloLFM2, bert: ModeloBert, deepseek2: ModeloDeepseek2 };
const REGISTRO = {};   // general.architecture -> { Clase, manifiesto }

async function cargarRegistro() {
  const lista = await (await fetch("/api/architectures")).json();
  for (const k of Object.keys(REGISTRO)) delete REGISTRO[k];
  for (const man of lista) {
    let Clase;
    try {
      if (man.base) Clase = BASES[man.base];
      else Clase = (await import(`/web/arch/${man.module}?v=${man._v || 0}`)).default;
      if (typeof Clase !== "function") throw new Error("the module does not default-export a class");
    } catch (e) {
      log(`Architecture '${man.id}' not available: ${e.message}`, false);
      continue;
    }
    for (const a of man.arch) REGISTRO[a] = { Clase, manifiesto: man };
  }
  globalThis.__registroArch = REGISTRO;   // for the tests
  return REGISTRO;
}
const VERSION_MOTOR = "4.1.0";
const MARCADOR_IMAGEN = "<__media__>";

const $ = (id) => document.getElementById(id);
const S = { gpu: null, modelo: null, tok: null, modeloId: null, vision: null, ocupado: false,
  id: "m" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36) };

function ui(id, txt) { const e = $(id); if (e) e.textContent = txt; }
function log(msg, enviar = true) {
  const li = document.createElement("div");
  li.textContent = `${new Date().toLocaleTimeString()}  ${msg}`;
  const box = $("log");
  if (box) { box.prepend(li); while (box.childNodes.length > 200) box.lastChild.remove(); }
  console.log("[engine]", msg);
  if (enviar) evento({ tipo: "log", mensaje: msg }).catch(() => {});
}

async function post(ruta, datos) {
  const r = await fetch(ruta, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(datos) });
  return r.json();
}
const evento = (d) => post("/engine/api/event", Object.assign({ motor: S.id }, d));
function conLimite(promesa, ms, mensaje) {
  let id;
  const limite = new Promise((_, rej) => { id = setTimeout(() => rej(new Error(mensaje)), ms); });
  return Promise.race([promesa, limite]).finally(() => clearTimeout(id));
}
const dormir = (ms) => new Promise((r) => setTimeout(r, ms));

function estado(txt) { ui("estado", txt); }

async function informarEstado() {
  await evento({ tipo: "estado", modelo_cargado: S.modeloId, estado: S.ocupado ? "ocupado" : "listo",
    info: { gpu: S.gpu ? S.gpu.info : null, memory_mb: S.gpu ? Math.round(S.gpu.bytesUsados / 1e6) : 0, version: VERSION_MOTOR } });
}

// ------------------------------------------------------------------ models
async function descargarModelo() {
  if (S.vision) { S.vision.free(); S.vision = null; }
  if (S.modelo) { S.modelo.free(); log(`Model ${S.modeloId} unloaded from memory`); }
  S.modelo = null; S.tok = null; S.modeloId = null;
  ui("modelo", "(none)");
}

async function asegurarModelo(m, idTrabajo) {
  if (S.modeloId === m.id && S.modelo) return;
  if (!REGISTRO[m.arch]) await cargarRegistro();   // a manifest may have been added after the engine was opened
  if (!REGISTRO[m.arch]) throw new Error(`architecture '${m.arch}' is not supported by this engine (see web/arch/ and docs/ARCH_GUIDE.md)`);
  const { Clase } = REGISTRO[m.arch];
  await descargarModelo();
  const t0 = performance.now();
  estado(`Loading ${m.id}...`);
  log(`Loading ${m.id} (${(m.bytes / 1e9).toFixed(2)} GB, context ${m.ctx})`);
  const meta = await (await fetch(m.meta_url)).json();
  meta.ctx = m.ctx;
  const tok = new Tokenizador(meta.kv);
  const modelo = new Clase(S.gpu, meta);
  let ultimo = 0;
  log(`Options: tiled prefill ${meta.options && meta.options.tiled_prefill === false ? "off" : "on"}`, false);
  await modelo.load(m.archivo_url, (frac) => {
    const pct = Math.round(frac * 100);
    $("barra").style.width = pct + "%";
    const ahora = performance.now();
    if (ahora - ultimo > 700) {
      ultimo = ahora;
      ui("progreso", `${pct}%`);
      if (idTrabajo) evento({ tipo: "progreso", id: idTrabajo, fase: `loading model ${pct}%` }).catch(() => {});
    }
  });
  S.modelo = modelo; S.tok = tok; S.modeloId = m.id;
  $("barra").style.width = "0%"; ui("progreso", "");
  const seg = ((performance.now() - t0) / 1000).toFixed(1);
  ui("modelo", `${m.id}  ·  ${modelo.nLayers} layers  ·  ${(S.gpu.bytesUsados / 1e9).toFixed(2)} GB on GPU`);
  log(`Model ${m.id} loaded in ${seg} s`);
  await informarEstado();
}

// ------------------------------------------------------------------ vision
async function asegurarVision(m, idTrabajo) {
  if (S.vision) return;
  if (!m.vision) { const e = new Error("this model has no usable vision projector (mmproj)"); e.codigo = 400; throw e; }
  if (!S.modelo.supportsImages) { const e = new Error(`architecture '${m.arch}' does not accept images yet`); e.codigo = 400; throw e; }
  const t0 = performance.now();
  estado("Loading the vision projector...");
  log(`Loading vision projector (${(m.vision.bytes / 1e6).toFixed(0)} MB)`);
  const meta = await (await fetch(m.vision.meta_url)).json();
  const v = new ModeloVision(S.gpu, meta);
  await v.load(m.vision.archivo_url, (frac) => {
    const pct = Math.round(frac * 100);
    $("barra").style.width = pct + "%";
    if (idTrabajo) evento({ tipo: "progreso", id: idTrabajo, fase: `loading vision ${pct}%` }).catch(() => {});
  });
  $("barra").style.width = "0%";
  if (v.dim !== S.modelo.D) { v.free(); const e = new Error(`the projector outputs vectors of ${v.dim} but the language model uses ${S.modelo.D}: they are not compatible`); e.codigo = 400; throw e; }
  S.vision = v;
  log(`Vision projector loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
}

function base64ABytes(b64) {
  const bin = atob(b64), u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}

// Builds the prompt token sequence with images, same as mtmd (llama.cpp) for LFM2-VL:
//   <|image_start|> [ <|img_row_F_col_C|> tile ]... <|img_thumbnail|> view <|image_end|>
// (with a single tile llama.cpp also prepends <|img_thumbnail|>; it is reproduced the same way).
// Returns { ids, ext, buffers }: ext[i] = null or {buf, fila} (the image embedding of that token).
async function armarSecuenciaConImagenes(job, p, tok, modelo) {
  const segs = (job.prompt || "").split(MARCADOR_IMAGEN);
  const imgs = p.imagenes;
  if (segs.length - 1 !== imgs.length) { const e = new Error(`the prompt has ${segs.length - 1} image markers and ${imgs.length} images`); e.codigo = 400; throw e; }
  const vis = S.vision;
  const esp = (t) => { const id = tok.vocab.get(t); if (id === undefined) { const e = new Error(`the model does not have the special token ${t}`); e.codigo = 400; throw e; } return id; };
  const ids = [], ext = [], buffers = [];
  const diagImg = [];
  const texto = (t, addBos) => { for (const id of tok.codificar(t, { addBos })) { ids.push(id); ext.push(null); } };
  const token = (t) => { ids.push(esp(t)); ext.push(null); };
  const bidir = vis.proy === "gemma3";   // Gemma 3: the tokens of an image attend to each other (non-causal attention)
  const pieza = (r) => { buffers.push(r.buf); for (let j = 0; j < r.n; j++) { ids.push(0); ext.push({ buf: r.buf, fila: j, bidir }); } };
  try {
    for (let i = 0; i < segs.length; i++) {
      if (i === 0) { if (segs[0] !== "" || p.add_bos !== false) texto(segs[0], p.add_bos === false ? false : null); }
      else if (segs[i] !== "") texto(segs[i], false);
      if (i === segs.length - 1) break;
      if (job.id) evento({ tipo: "progreso", id: job.id, fase: `codificando imagen ${i + 1}/${imgs.length}` }).catch(() => {});
      let crudo;
      try { crudo = await decodificarImagen(base64ABytes(imgs[i].b64), imgs[i].tipo); }
      catch (err) { const e = new Error(`could not read image ${i + 1} (unsupported format or damaged file): ${err.message}`); e.codigo = 400; throw e; }
      const tImg = performance.now();
      if (vis.proy === "gemma3") {
        // as in mtmd: <start_of_image> [embeddings] <end_of_image>; the image is resized to image_size x image_size preserving
        // the aspect ratio (black padding, bilinear)
        const sz = vis.tamImagen;
        const img = redimensionar(crudo, sz, sz, true);
        const t1 = performance.now();
        token("<start_of_image>");
        vis.onCapa = (n, L) => { if (job.id) evento({ tipo: "progreso", id: job.id, fase: `codificando imagen ${i + 1}/${imgs.length}: capa ${n}/${L}` }).catch(() => {}); };
        const r = await conLimite(vis.codificar(vis.normalizada(img), img.w, img.h), 300000, "the GPU did not respond within 300 s encoding the image");
        await S.gpu.device.queue.onSubmittedWorkDone();
        pieza(r);
        token("<end_of_image>");
        log(`Image ${i + 1}: ${crudo.w}x${crudo.h} -> ${sz}x${sz}, ${r.n} tokens, encoded in ${((performance.now() - t1) / 1000).toFixed(1)} s`);
        if (vis.ultimaPerf) diagImg.push({ imagen: i + 1, codificar_s: +((performance.now() - t1) / 1000).toFixed(1), qkv_s: +(vis.ultimaPerf.qkv / 1000).toFixed(1), atencion_s: +(vis.ultimaPerf.atencion / 1000).toFixed(1), mlp_s: +(vis.ultimaPerf.mlp / 1000).toFixed(1) });
        if (vis.ultimaPerf) log(`  encoder phases: q/k/v projections ${(vis.ultimaPerf.qkv / 1000).toFixed(1)} s, attention ${(vis.ultimaPerf.atencion / 1000).toFixed(1)} s, output+MLP ${(vis.ultimaPerf.mlp / 1000).toFixed(1)} s`);
        continue;
      }
      const pre = preprocesarImagen(crudo, vis.parche, vis.fusion);
      const cod = async (img) => {
        const t1 = performance.now();
        const r = await conLimite(vis.codificar(vis.normalizada(img), img.w, img.h), 120000, "the GPU did not respond within 120 s encoding the image");
        await S.gpu.device.queue.onSubmittedWorkDone();
        log(`Image ${i + 1}: piece ${img.w}x${img.h} -> ${r.n} tokens in ${((performance.now() - t1) / 1000).toFixed(1)} s`, false);
        return r;
      };
      token("<|image_start|>");
      for (const t of pre.teselas) { token(`<|img_row_${t.fila}_col_${t.col}|>`); pieza(await cod(t.img)); }
      token("<|img_thumbnail|>");
      pieza(await cod(pre.vista));
      token("<|image_end|>");
      log(`Image ${i + 1}: ${crudo.w}x${crudo.h}, ${pre.teselas.length ? pre.teselas.length + " tiles + thumbnail" : "a single view"}, encoded in ${((performance.now() - tImg) / 1000).toFixed(1)} s`);
    }
  } catch (e) {
    S.gpu.liberar(buffers);
    throw e;
  }
  return { ids, ext, buffers, diagImg };
}

// ------------------------------------------------------------------ generation
async function generar(job) {
  const p = job.params || {};
  const tJob = performance.now();
  await asegurarModelo(job.modelo, job.id);
  const tok = S.tok, modelo = S.modelo;
  const t0 = performance.now();
  const diag = { modelo_ms: Math.round(t0 - tJob), tok_ms: 0, lotes_ms: [] };
  let ids, ext = null, buffersImg = [];
  if (p.imagenes && p.imagenes.length) {
    await asegurarVision(job.modelo, job.id);
    ({ ids, ext, buffers: buffersImg, diagImg: diag.imagenes } = await armarSecuenciaConImagenes(job, p, tok, modelo));
  } else {
    ids = tok.codificar(job.prompt || "", { addBos: p.add_bos === false ? false : null });
  }
  if (!ids.length) throw new Error("the prompt is empty");
  diag.tok_ms = Math.round(performance.now() - t0);
  const tIni = performance.now();
  await evento({ tipo: "inicio", id: job.id, prompt_tokens: ids.length });
  diag.evento_inicio_ms = Math.round(performance.now() - tIni);
  const ctx = modelo.ctx;
  if (ids.length >= ctx) {
    if (buffersImg.length) S.gpu.liberar(buffersImg);
    const e = new Error(`the prompt has ${ids.length} tokens and the model context is ${ctx}. Shorten the text or, in config.json, raise "kv_max_mb" (memory for the context) or set "model_overrides": {"<model>": {"ctx": N}}.`);
    e.codigo = 400; throw e;
  }
  const maxTok = p.max_tokens > 0 ? Math.min(p.max_tokens, ctx - ids.length) : ctx - ids.length;
  estado(`Processing prompt (${ids.length} tokens)...`);
  modelo.reset();
  let logits = null;
  const TL = modelo.batch || TB;   // tokens per prompt pass (32 in the K-quant models, 8 in LFM2)
  try {
    let i = 0;
    while (i < ids.length) {
      let fin;
      if (ext && ext[i] && ext[i].bidir) {   // a Gemma 3 image goes whole in one batch (non-causal attention)
        fin = i;
        while (fin < ids.length && ext[fin] && ext[fin].bidir && ext[fin].buf === ext[i].buf) fin++;
      } else {
        fin = Math.min(i + TL, ids.length);
        if (ext) for (let j = i; j < fin; j++) if (ext[j] && ext[j].bidir) { fin = j; break; }
      }
      const lote = ids.slice(i, fin);
      const tl = performance.now();
      if (ids.length > TL) evento({ tipo: "progreso", id: job.id, fase: `procesando prompt ${fin}/${ids.length}` }).catch(() => {});
      logits = await conLimite(modelo.forward(lote, fin >= ids.length, ext ? ext.slice(i, fin) : null), 300000, "the GPU did not respond within 300 s processing the prompt (video driver reset?)");
      await conLimite(S.gpu.device.queue.onSubmittedWorkDone(), 300000, "the GPU did not finish the prompt batch within 300 s");   // real time per batch
      diag.lotes_ms.push(Math.round(performance.now() - tl));
      i = fin;
    }
  } finally {
    if (buffersImg.length) { S.gpu.liberar(buffersImg); buffersImg = []; }
  }
  const tPrompt = performance.now();
  const ttft = (tPrompt - t0) / 1000;
  const mu = new Muestreador(p);
  const hist = ids.slice();
  const dec = tok.decodificadorIncremental();
  let n = 0, razon = "length";
  estado("Generating...");
  while (n < maxTok) {
    const id = mu.elegir(logits, hist);
    if (tok.esFin(id)) { razon = "stop"; break; }
    hist.push(id); n++;
    if (n >= maxTok) break;   // LM Studio counts this last token but does not include its text when cutting by length
    const texto = dec.agregar(id);
    const siguiente = modelo.forward([id], true);   // the GPU works while we submit
    const r = await evento({ tipo: "token", id: job.id, texto, n: 1 });
    if (siguiente) logits = await siguiente;
    if (r && r.cancelar) { razon = "stop"; break; }
    if (n % 8 === 0) ui("velocidad", `${(n / ((performance.now() - tPrompt) / 1000)).toFixed(1)} tok/s`);
  }
  const cola = dec.terminar();
  if (cola) await evento({ tipo: "token", id: job.id, texto: cola, n: 0 });
  const seg = (performance.now() - tPrompt) / 1000;
  const tps = n > 0 && seg > 0 ? +(n / seg).toFixed(2) : null;
  const tpsPrompt = +(ids.length / Math.max(ttft, 1e-3)).toFixed(1);
  ui("velocidad", `${tps ?? "-"} tok/s  ·  prompt ${tpsPrompt} tok/s`);
  log(`Done: ${ids.length} prompt tokens (${tpsPrompt} tok/s), ${n} generated (${tps} tok/s). First token ${ttft.toFixed(2)} s; prompt batches (ms): ${diag.lotes_ms.join(", ")}; model ${diag.modelo_ms} ms; tokenize ${diag.tok_ms} ms`);
  await evento({ tipo: "fin", id: job.id, razon, prompt_tokens: ids.length, completion_tokens: n, tps,
    ttft: +ttft.toFixed(3), tps_prompt: tpsPrompt, diag });
}

// ------------------------------------------------------------------ embeddings
async function embeber(job) {
  const tJob = performance.now();
  await asegurarModelo(job.modelo, job.id);
  const tok = S.tok, modelo = S.modelo;
  if (!modelo.embed) { const e = new Error("this model is not an embeddings model"); e.codigo = 400; throw e; }
  const entradas = (job.params && job.params.entradas) || [];
  const vectores = [];
  let total = 0;
  estado(`Computing embeddings (${entradas.length} texts)...`);
  for (let i = 0; i < entradas.length; i++) {
    const ids = tok.codificar(entradas[i]);
    if (ids.length > modelo.ctx) {
      const e = new Error(`text ${i} has ${ids.length} tokens and the embeddings model context is ${modelo.ctx}`);
      e.codigo = 400; throw e;
    }
    total += ids.length;
    const v = await conLimite(modelo.embed(ids), 120000, "the GPU did not respond within 120 s computing embeddings");
    vectores.push(Array.from(v));
    if (entradas.length > 1) evento({ tipo: "progreso", id: job.id, fase: `embeddings ${i + 1}/${entradas.length}` }).catch(() => {});
  }
  const seg = (performance.now() - tJob) / 1000;
  log(`Embeddings: ${entradas.length} texts, ${total} tokens in ${seg.toFixed(2)} s`);
  await evento({ tipo: "fin", id: job.id, razon: "stop", prompt_tokens: total, completion_tokens: 0, embeddings: vectores });
}

async function atender(job) {
  S.ocupado = true;
  try {
    if (job.accion === "generar") await generar(job);
    else if (job.accion === "embeber") await embeber(job);
    else if (job.accion === "cargar") { await asegurarModelo(job.modelo, job.id); await evento({ tipo: "fin", id: job.id, razon: "stop" }); }
    else if (job.accion === "descargar") { await descargarModelo(); await informarEstado(); await evento({ tipo: "fin", id: job.id, razon: "stop" }); }
  } catch (e) {
    log(`Error: ${e.message}`);
    console.error(e);
    await evento({ tipo: "error", id: job.id, mensaje: e.message, codigo: e.codigo || 500 }).catch(() => {});
    if (S.gpu && S.gpu.perdido) { S.gpu = null; S.modelo = null; S.modeloId = null; }
  } finally {
    S.ocupado = false;
    estado("Ready, waiting for requests");
  }
}

// ------------------------------------------------------------------ startup
async function iniciarGPU() {
  S.gpu = await GPU.crear(log);
  await S.gpu.prepararPipelines();
  ui("gpu", S.gpu.info.summary);
  log(`WebGPU ready: ${S.gpu.info.summary} (max buffer ${(S.gpu.info.maxBuffer / 1e6).toFixed(0)} MB)`, false);
  try { await cargarRegistro(); log(`Architectures: ${Object.keys(REGISTRO).join(", ")}`, false); } catch (e) { log(`Could not read /api/architectures: ${e.message}`, false); }
}

async function bucle() {
  let avisado = false;
  for (;;) {
    try {
      if (!S.gpu) {
        await iniciarGPU();
        const h = await post("/engine/api/hello", { motor: S.id, gpu: S.gpu.info, version: VERSION_MOTOR, recarga: true, userAgent: navigator.userAgent });
        if (h && h.version && h.version !== VERSION_MOTOR) {
          // The server is a different version: reload this window to fetch the new JS (only once).
          let ya = null;
          try { ya = sessionStorage.getItem("recargado"); sessionStorage.setItem("recargado", h.version); } catch (e) { ya = "x"; }
          if (ya !== h.version) { log(`Server ${h.version}, engine ${VERSION_MOTOR}: reloading...`, false); location.reload(); return; }
        }
        await informarEstado();
        estado("Ready, waiting for requests");
      }
      // The long-poll has no timeout of its own: if the connection hangs (network, sleep, the browser freezing the
      // window) we would wait forever and the server would consider this engine disconnected. The server answers
      // within 20 s, so cut at 35 s and ask again.
      const ctl = new AbortController();
      const corte = setTimeout(() => ctl.abort(), 35000);
      let job;
      try {
        const r = await fetch("/engine/api/next?motor=" + S.id, { signal: ctl.signal });
        if (!r.ok) throw new Error("HTTP " + r.status);
        job = await r.json();
      } catch (e) {
        if (e && e.name === "AbortError") { log("The request to the server hung; retrying.", false); continue; }
        throw e;
      } finally {
        clearTimeout(corte);
      }
      avisado = false;
      $("conexion").className = "ok"; ui("conexion", "connected");
      if (job.accion === "cerrar") {
        estado("Inactive: another engine window was opened. You can close this window.");
        $("conexion").className = "mal"; ui("conexion", "inactive");
        log("Another engine window took over; this one is now inactive.", false);
        if (S.modelo) await descargarModelo();
        return;
      }
      if (job.accion && job.accion !== "nada") await atender(job);
    } catch (e) {
      $("conexion").className = "mal"; ui("conexion", "no connection");
      if (!avisado) { log(`No connection to the server, or error: ${e.message}`, false); avisado = true; }
      if (!S.gpu && /WebGPU|adapter/.test(e.message)) { estado("ERROR: " + e.message); await dormir(10000); }
      await dormir(2000);
    }
  }
}

window.motor = S;   // for debugging from the console
bucle();
