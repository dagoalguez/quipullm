// Vision (LFM2-VL): image preprocessing as in llama.cpp (mtmd), SigLIP-style ViT encoder in WebGPU and a
// projector (pixel-unshuffle + MLP) that delivers embeddings of the language model's size.
// Reference: llama.cpp tools/mtmd (clip.cpp, models/siglip.cpp, mtmd-image.cpp).
import { despachar, dim2, f16af32, reempaquetarQ8, registrarAtencionBi } from "./gpu.js";
import { CargadorPesos } from "./weights.js";

export const PIEZA_MAX = 1024;
const TRAMO_ATT = 512;   // attention queries per GPU submit (Gemma 3 ViT)   // max patches per image (512x512 tiles with patch 16)
const TESELA = 512, TOLERANCIA = 2.0, MIN_TESELAS = 2, MAX_TESELAS = 10;
const fr = Math.fround;

// ------------------------------------------------------------------ rounding to f16 (ggml converts the patches to f16 if the weight is F16)
const _f32 = new Float32Array(1), _u32 = new Uint32Array(_f32.buffer);
function parMasCercano(q) {
  const r = Math.round(q);
  return Math.abs(q - Math.trunc(q)) === 0.5 ? 2 * Math.round(q / 2) : r;
}
export function f16r(x) {
  if (x === 0 || !isFinite(x)) return x;
  const s = x < 0 ? -1 : 1, a = Math.abs(x);
  if (a >= 65520) return s * Infinity;
  if (a < 6.103515625e-5) return s * parMasCercano(a / 5.960464477539063e-8) * 5.960464477539063e-8;
  _f32[0] = a;
  const e = ((_u32[0] >>> 23) & 255) - 127;
  const paso = Math.pow(2, e - 10);
  return s * parMasCercano(a / paso) * paso;
}

// ------------------------------------------------------------------ preprocessing (same as mtmd_image_preprocessor_lfm2)
function pixel(img, x, y) { const o = (y * img.w + x) * 3; return [img.d[o], img.d[o + 1], img.d[o + 2]]; }

// Bilinear resize from mtmd (img_tool::resize_bilinear), f32 arithmetic
export function redimensionarBilineal(src, tw, th) {
  tw = Math.max(1, tw); th = Math.max(1, th);
  const dst = { w: tw, h: th, d: new Uint8Array(tw * th * 3) };
  const xr = tw > 1 ? fr(fr(src.w - 1) / (tw - 1)) : 0, yr = th > 1 ? fr(fr(src.h - 1) / (th - 1)) : 0;
  const lerp = (s, e, t) => fr(s + fr(fr(e - s) * t));
  for (let y = 0; y < th; y++) {
    const py = fr(y * yr);
    const y0 = Math.min(Math.trunc(py), src.h - 1), y1 = Math.min(y0 + 1, src.h - 1), yf = fr(py - y0);
    for (let x = 0; x < tw; x++) {
      const px = fr(x * xr);
      const x0 = Math.min(Math.trunc(px), src.w - 1), x1 = Math.min(x0 + 1, src.w - 1), xf = fr(px - x0);
      const o00 = (y0 * src.w + x0) * 3, o10 = (y0 * src.w + x1) * 3, o01 = (y1 * src.w + x0) * 3, o11 = (y1 * src.w + x1) * 3;
      const od = (y * tw + x) * 3;
      for (let c = 0; c < 3; c++) {
        const top = lerp(src.d[o00 + c], src.d[o10 + c], xf);
        const bot = lerp(src.d[o01 + c], src.d[o11 + c], xf);
        dst.d[od + c] = Math.trunc(lerp(top, bot, yf));
      }
    }
  }
  return dst;
}

// img_tool::resize with PAD_CEIL padding (tiles) or without padding (overview)
export function redimensionar(src, tw, th, relleno) {
  if (src.w === tw && src.h === th) return { w: src.w, h: src.h, d: src.d.slice() };
  if (!relleno) return redimensionarBilineal(src, tw, th);
  const sw = fr(tw / src.w), sh = fr(th / src.h), esc = Math.min(sw, sh);
  const nw = Math.min(Math.ceil(fr(src.w * esc)), tw), nh = Math.min(Math.ceil(fr(src.h * esc)), th);
  const r = redimensionarBilineal(src, nw, nh);
  const dst = { w: tw, h: th, d: new Uint8Array(tw * th * 3) };   // black padding
  const ox = Math.trunc((tw - nw) / 2), oy = Math.trunc((th - nh) / 2);
  for (let y = 0; y < nh; y++) {
    const dy = y + oy;
    if (dy < 0 || dy >= th) continue;
    for (let x = 0; x < nw; x++) {
      const dx = x + ox;
      if (dx < 0 || dx >= tw) continue;
      const o = (y * nw + x) * 3, od = (dy * tw + dx) * 3;
      dst.d[od] = r.d[o]; dst.d[od + 1] = r.d[o + 1]; dst.d[od + 2] = r.d[o + 2];
    }
  }
  return dst;
}

function recorte(src, x, y, w, h) {
  const dst = { w, h, d: new Uint8Array(w * h * 3) };
  for (let j = 0; j < h; j++) {
    const o = ((y + j) * src.w + x) * 3;
    dst.d.set(src.d.subarray(o, o + w * 3), j * w * 3);
  }
  return dst;
}

// calc_size_preserved_ratio (smart resize) with alignment, minimum and maximum pixel counts
export function tamanoConservandoRelacion(w, h, alinear, minPix, maxPix) {
  const redond = (x) => Math.round(fr(x / alinear)) * alinear;
  const techo = (x) => Math.ceil(fr(x / alinear)) * alinear;
  const piso = (x) => Math.floor(fr(x / alinear)) * alinear;
  let wb = Math.max(alinear, redond(w)), hb = Math.max(alinear, redond(h));
  if (maxPix > 0 && hb * wb > maxPix) {
    const beta = fr(Math.sqrt(fr(fr(fr(h) * w) / maxPix)));
    hb = Math.max(alinear, piso(fr(h / beta)));
    wb = Math.max(alinear, piso(fr(w / beta)));
  } else if (minPix > 0 && hb * wb < minPix) {
    const beta = fr(Math.sqrt(fr(fr(minPix) / fr(fr(h) * w))));
    hb = techo(fr(h * beta));
    wb = techo(fr(w * beta));
  }
  return { w: wb, h: hb };
}

function relacionesObjetivo() {
  const r = [];
  for (let n = MIN_TESELAS; n <= MAX_TESELAS; n++) {
    for (let w = 1; w <= n; w++) {
      for (let h = 1; h <= n; h++) {
        if (w * h >= MIN_TESELAS && w * h <= MAX_TESELAS && !r.some((q) => q.w === w && q.h === h)) r.push({ w, h });
      }
    }
  }
  return r.map((v, i) => [v, i]).sort((a, b) => (a[0].w * a[0].h - b[0].w * b[0].h) || (a[1] - b[1])).map((p) => p[0]);
}

export function rejillaDeTeselas(alto, ancho) {
  const aspecto = fr(ancho / alto);
  let mejor = { w: 1, h: 1 }, mejorDif = Infinity;
  const area = fr(ancho * alto);
  for (const r of relacionesObjetivo()) {
    const dif = Math.abs(fr(aspecto - fr(r.w / r.h)));
    if (dif < mejorDif) { mejorDif = dif; mejor = r; }
    else if (dif === mejorDif) {
      const areaObj = fr(TESELA * TESELA * r.w * r.h);
      if (area > fr(0.5 * areaObj)) mejor = r;
    }
  }
  return mejor;
}

// Slicing instructions: { vista: {w,h}, rejilla: {w,h}|null, refinada, cortes: [{x,y}] }
export function planearImagen(w, h, parche, fusion) {
  const alinear = parche * fusion, p2 = parche * parche * fusion * fusion;
  const vista = tamanoConservandoRelacion(w, h, alinear, 64 * p2, 256 * p2);
  const necesita = w > TESELA * TOLERANCIA || h > TESELA * TOLERANCIA;
  if (!necesita) return { vista, rejilla: null, cortes: [] };
  const rejilla = rejillaDeTeselas(h, w);
  const cortes = [];
  for (let f = 0; f < rejilla.h; f++) for (let c = 0; c < rejilla.w; c++) cortes.push({ x: c * TESELA, y: f * TESELA, fila: f + 1, col: c + 1 });
  return { vista, rejilla, refinada: { w: TESELA * rejilla.w, h: TESELA * rejilla.h }, cortes };
}

// img: {w,h,d:Uint8Array RGB}. Returns { teselas: [{fila,col,img}], vista: img }
export function preprocesarImagen(img, parche, fusion) {
  const plan = planearImagen(img.w, img.h, parche, fusion);
  const vista = redimensionar(img, plan.vista.w, plan.vista.h, false);
  const teselas = [];
  if (plan.cortes.length) {
    const ref = redimensionar(img, plan.refinada.w, plan.refinada.h, true);
    for (const c of plan.cortes) teselas.push({ fila: c.fila, col: c.col, img: recorte(ref, c.x, c.y, TESELA, TESELA) });
  }
  return { plan, teselas, vista };
}

// RGB u8 -> normalized planar f32 [3][h][w], ((v/255) - mean) / std
export function normalizar(img, media, desv) {
  const n = img.w * img.h, out = new Float32Array(n * 3);
  for (let c = 0; c < 3; c++) {
    const m = fr(media[c]), s = fr(desv[c]);
    for (let i = 0; i < n; i++) out[c * n + i] = fr(fr(fr(img.d[i * 3 + c] / 255) - m) / s);
  }
  return out;
}

// Decodes an image file (PNG/JPEG/WebP/GIF...) to RGB; transparency is composited over white.
export async function decodificarImagen(bytes, tipo) {
  const blob = new Blob([bytes], { type: tipo || "" });
  const bmp = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const cv = document.createElement("canvas");
  cv.width = bmp.width; cv.height = bmp.height;
  const cx = cv.getContext("2d", { willReadFrequently: true });
  cx.fillStyle = "#fff"; cx.fillRect(0, 0, cv.width, cv.height);
  cx.drawImage(bmp, 0, 0);
  const rgba = cx.getImageData(0, 0, cv.width, cv.height).data;
  const d = new Uint8Array(cv.width * cv.height * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) { d[j] = rgba[i]; d[j + 1] = rgba[i + 1]; d[j + 2] = rgba[i + 2]; }
  if (bmp.close) bmp.close();
  return { w: cv.width, h: cv.height, d };
}

// ------------------------------------------------------------------ position embeddings: bilinear interpolation with antialiasing (ggml_interpolate)
function pesos1D(n, nOrig, dim) {
  // for each output i: list of (source, weight) as in ggml_compute_forward_upscale_f32 (bilinear + antialias mode)
  const sf = fr(n / nOrig), apoyo = Math.max(1, fr(1 / sf)), inv = fr(1 / apoyo), desf = 0.5;
  const res = [];
  for (let i = 0; i < dim; i++) {
    const pos = fr(fr(i + desf) / fr(dim / nOrig));
    const mn = Math.max(Math.trunc(fr(fr(pos - apoyo) + desf)), 0), mx = Math.min(Math.trunc(fr(fr(pos + apoyo) + desf)), nOrig);
    const lista = []; let total = 0;
    for (let s = mn; s < mx; s++) {
      const w = Math.max(fr(1 - Math.abs(fr(fr(fr(s - pos) + desf) * inv))), 0);
      if (w <= 0) continue;
      lista.push([s, w]); total += w;
    }
    res.push({ lista, total });
  }
  return res;
}

export function interpolarPosiciones(tabla, nLado, D, gw, gh) {
  if (gw === nLado && gh === nLado) return tabla;
  const px = pesos1D(gw, nLado, gw), py = pesos1D(gh, nLado, gh);
  // horizontal: [nLado rows][gw][D]
  const h1 = new Float32Array(nLado * gw * D);
  for (let y = 0; y < nLado; y++) {
    for (let x = 0; x < gw; x++) {
      const { lista } = px[x], o = (y * gw + x) * D;
      for (const [s, w] of lista) { const b = (y * nLado + s) * D; for (let d = 0; d < D; d++) h1[o + d] += tabla[b + d] * w; }
    }
  }
  const out = new Float32Array(gh * gw * D);
  for (let y = 0; y < gh; y++) {
    const { lista, total: ty } = py[y];
    for (let x = 0; x < gw; x++) {
      const tx = px[x].total, o = (y * gw + x) * D, norm = 1 / (tx * ty);
      for (const [s, w] of lista) { const b = (s * gw + x) * D; for (let d = 0; d < D; d++) out[o + d] += h1[b + d] * w; }
      for (let d = 0; d < D; d++) out[o + d] *= norm;
    }
  }
  return out;
}

// ------------------------------------------------------------------ model
export class ModeloVision {
  constructor(gpu, meta) {
    this.gpu = gpu;
    this.meta = meta;
    const kv = meta.kv;
    const proyector = kv["clip.projector_type"] || kv["clip.vision.projector_type"];
    if (proyector !== "lfm2" && proyector !== "gemma3") throw new Error(`vision projector '${proyector}' not supported (only 'lfm2' and 'gemma3')`);
    this.proy = proyector;
    const n = (k, d) => (kv[`clip.vision.${k}`] !== undefined ? kv[`clip.vision.${k}`] : d);
    this.D = n("embedding_length"); this.nh = n("attention.head_count"); this.F = n("feed_forward_length");
    this.L = n("block_count"); this.parche = n("patch_size"); this.eps = n("attention.layer_norm_epsilon", 1e-6);
    this.fusion = n("projector.scale_factor", proyector === "gemma3" ? 4 : 1);
    this.tamImagen = n("image_size", 0);
    this.hd = kv["clip.vision.attention.head_dim"] || this.D / this.nh;
    this.media = kv["clip.vision.image_mean"]; this.desv = kv["clip.vision.image_std"];
    if (!this.media || !this.desv || this.media.length < 3) throw new Error("the mmproj has no image_mean/image_std");
    this.act = kv["clip.use_gelu"] ? 0 : (kv["clip.use_silu"] ? 1 : 2);
    if (this.D % this.nh) throw new Error("ViT dimension not divisible by the number of heads");
    if (this.hd > (proyector === "gemma3" ? 128 : 256)) throw new Error("attention head too large for the kernel");
    // max patches per image: LFM2 processes 512 px tiles; Gemma 3 a single image of image_size x image_size
    this.piezaMax = PIEZA_MAX;
    if (proyector === "gemma3") {
      if (!this.tamImagen || this.tamImagen % this.parche) throw new Error("the Gemma 3 mmproj has no valid clip.vision.image_size");
      this.piezaMax = (this.tamImagen / this.parche) ** 2;
      if (this.piezaMax > 8192) throw new Error(`image_size ${this.tamImagen} demasiado grande`);
      if ((this.tamImagen / this.parche) % this.fusion) throw new Error("the patch grid is not divisible by the reduction factor");
    }
    this.bufs = [];
    this.cachePos = new Map();
  }

  // ---- tensor loading
  async _f32cpu(nombre) {
    const c = this.cargador, t = c.info(nombre), u8 = await c.bytes(t);
    if (t.tipo === 0) return new Float32Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.length));
    if (t.tipo === 1) return f16af32(u8);
    if (t.tipo === 30) {
      const s = new Uint16Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.length)), f = new Float32Array(s.length), v = new Uint32Array(f.buffer);
      for (let i = 0; i < s.length; i++) v[i] = s[i] << 16;
      return f;
    }
    if (t.tipo === 8) {
      const { qs, ds } = reempaquetarQ8(u8), i8 = new Int8Array(qs.buffer), f = new Float32Array(i8.length);
      for (let i = 0; i < i8.length; i++) f[i] = i8[i] * ds[i >> 5];
      return f;
    }
    throw new Error(`${nombre}: type ${t.tipo_nombre} not supported in the vision encoder (use F16, F32, BF16 or Q8_0)`);
  }

  _sub(tipado, etiqueta) { const b = this.gpu.subir(tipado, etiqueta); this.cargador.buffers.push(b); return b; }

  async _vec(nombre, opcional = false) {
    if (!this.cargador.existe(nombre)) { if (opcional) return null; throw new Error(`missing tensor ${nombre}`); }
    return this._sub(await this._f32cpu(nombre), nombre);
  }

  // Weight matrix [N][K] (K = product of the first dimensions minus the last); rows [f0,f1) optional.
  async _mat(nombre, filas = null, u8Previo = null) {
    const c = this.cargador, t = c.info(nombre);
    const N0 = t.dims[t.dims.length - 1], K = t.dims.slice(0, -1).reduce((a, b) => a * b, 1);
    const [f0, f1] = filas || [0, N0];
    const N = f1 - f0;
    if (K % 4) throw new Error(`${nombre}: K=${K} is not a multiple of 4`);
    const u8 = u8Previo || await c.bytes(t);
    if (t.tipo === 1) {   // F16: two weights per u32
      const tr = u8.subarray(f0 * K * 2, f1 * K * 2);
      const copia = new Uint8Array(tr.length); copia.set(tr);
      return { tipo: "f16", N, K, w: this._sub(new Uint32Array(copia.buffer), nombre) };
    }
    let f;
    if (t.tipo === 0) { const tr = u8.subarray(f0 * K * 4, f1 * K * 4); const cp = new Uint8Array(tr.length); cp.set(tr); f = new Float32Array(cp.buffer); }
    else {
      const todo = await this._f32cpu(nombre);
      f = todo.subarray(f0 * K, f1 * K);
    }
    return { tipo: "f32", N, K, w: this._sub(f, nombre) };
  }

  async load(url, progreso) {
    const gpu = this.gpu, c = new CargadorPesos(gpu, this.meta, url, progreso);
    this.cargador = c;
    const gem = this.proy === "gemma3";
    // matrix-based attention (QK^T, softmax, PV) if it fits in vec4; otherwise, the one-thread-per-query kernel
    this.attTile = gem && this.hd % 4 === 0 && this.piezaMax % 4 === 0 && !globalThis.__SIN_TILE;
    await gpu.prepararPipelines(["matmul_vis_f16", "matmul_vis_f32", "layernorm_o", "activacion"].concat(gem ? ["pool_avg", "rmsnorm"].concat(this.attTile ? ["mm_estr", "softmax_filas", "trans_v"] : [(this.kAtt = registrarAtencionBi(this.hd))]) : ["desbaraja"]));
    const D = this.D, L = this.L;
    const orden = ["v.patch_embd.weight", "v.patch_embd.bias", "v.position_embd.weight", "v.pre_ln.weight", "v.pre_ln.bias"];
    for (let i = 0; i < L; i++) {
      const p = `v.blk.${i}.`;
      orden.push(p + "ln1.weight", p + "ln1.bias", p + "attn_qkv.weight", p + "attn_qkv.bias", p + "attn_q.weight", p + "attn_q.bias",
        p + "attn_k.weight", p + "attn_k.bias", p + "attn_v.weight", p + "attn_v.bias", p + "attn_out.weight", p + "attn_out.bias",
        p + "ln2.weight", p + "ln2.bias", p + "ffn_up.weight", p + "ffn_up.bias", p + "ffn_down.weight", p + "ffn_down.bias");
    }
    orden.push("v.post_ln.weight", "v.post_ln.bias");
    if (gem) orden.push("mm.soft_emb_norm.weight", "mm.input_projection.weight");
    else orden.push("mm.input_norm.weight", "mm.input_norm.bias", "mm.1.weight", "mm.1.bias", "mm.2.weight", "mm.2.bias");
    c.planificar(orden);
    if (!c.existe("v.patch_embd.weight") || !c.existe("v.position_embd.weight")) throw new Error("the mmproj has no v.patch_embd/v.position_embd");
    const tp = c.info("v.patch_embd.weight");
    this.patchF16 = tp.tipo === 1;
    this.K0 = tp.dims.slice(0, -1).reduce((a, b) => a * b, 1);
    if (this.K0 !== 3 * this.parche * this.parche) throw new Error("unexpected shape of v.patch_embd.weight");
    this.patchW = await this._mat("v.patch_embd.weight");
    this.patchB = await this._vec("v.patch_embd.bias", true);
    const tpos = c.info("v.position_embd.weight");
    this.nLado = Math.round(Math.sqrt(tpos.dims[1]));
    this.posTabla = await this._f32cpu("v.position_embd.weight");
    this.preLn = c.existe("v.pre_ln.weight") ? { w: await this._vec("v.pre_ln.weight"), b: await this._vec("v.pre_ln.bias") } : null;
    this.capas = [];
    for (let i = 0; i < L; i++) {
      const p = `v.blk.${i}.`;
      for (const n of ["ffn_gate.weight", "attn_q_norm.weight", "attn_k_norm.weight", "ls1.weight", "ls2.weight"]) {
        if (c.existe(p + n)) throw new Error(`the mmproj uses ${n}, which is not supported`);
      }
      const capa = { ln1: { w: await this._vec(p + "ln1.weight"), b: await this._vec(p + "ln1.bias") },
        ln2: { w: await this._vec(p + "ln2.weight"), b: await this._vec(p + "ln2.bias") } };
      if (c.existe(p + "attn_qkv.weight")) {
        const t = c.info(p + "attn_qkv.weight"), u8 = await c.bytes(t);
        capa.wq = await this._mat(p + "attn_qkv.weight", [0, D], u8);
        capa.wk = await this._mat(p + "attn_qkv.weight", [D, 2 * D], u8);
        capa.wv = await this._mat(p + "attn_qkv.weight", [2 * D, 3 * D], u8);
        if (c.existe(p + "attn_qkv.bias")) {
          const b = await this._f32cpu(p + "attn_qkv.bias");
          capa.bq = this._sub(b.slice(0, D), "bq"); capa.bk = this._sub(b.slice(D, 2 * D), "bk"); capa.bv = this._sub(b.slice(2 * D, 3 * D), "bv");
        }
      } else {
        capa.wq = await this._mat(p + "attn_q.weight"); capa.wk = await this._mat(p + "attn_k.weight"); capa.wv = await this._mat(p + "attn_v.weight");
        capa.bq = await this._vec(p + "attn_q.bias", true); capa.bk = await this._vec(p + "attn_k.bias", true); capa.bv = await this._vec(p + "attn_v.bias", true);
      }
      capa.wo = await this._mat(p + "attn_out.weight"); capa.bo = await this._vec(p + "attn_out.bias", true);
      // old Gemma 3 mmproj files (e.g. lmstudio-community): ffn_up and ffn_down come swapped; llama.cpp
      // detects this because 'ffn_down' receives D input values. It is reordered the same way.
      let nUp = p + "ffn_up", nDown = p + "ffn_down";
      if (gem && c.info(nDown + ".weight").dims[0] === D && c.info(nUp + ".weight").dims[0] !== D) [nUp, nDown] = [nDown, nUp];
      capa.up = await this._mat(nUp + ".weight"); capa.bup = await this._vec(nUp + ".bias", true);
      capa.down = await this._mat(nDown + ".weight"); capa.bdown = await this._vec(nDown + ".bias", true);
      this.capas.push(capa);
    }
    this.postLn = c.existe("v.post_ln.weight") ? { w: await this._vec("v.post_ln.weight"), b: await this._vec("v.post_ln.bias") } : null;
    const K1 = D * this.fusion * this.fusion;
    this.inNorm = null;
    if (gem) {
      // Gemma 3: k x k average -> RMSNorm(w) -> x @ W, with W stored as [D_vit][D_lm] (ggml: ne=[D_lm, D_vit]); transposed to [D_lm][D_vit]
      if (!c.existe("mm.soft_emb_norm.weight") || !c.existe("mm.input_projection.weight")) throw new Error("the Gemma 3 mmproj has no mm.soft_emb_norm / mm.input_projection");
      this.softNorm = await this._vec("mm.soft_emb_norm.weight");
      const tw = c.info("mm.input_projection.weight");
      if (tw.dims[1] !== D) throw new Error(`mm.input_projection expects ${tw.dims[1]} inputs but the ViT outputs ${D}`);
      const dl = tw.dims[0], W = await this._f32cpu("mm.input_projection.weight"), Wt = new Float32Array(dl * D);
      for (let i = 0; i < D; i++) for (let j = 0; j < dl; j++) Wt[j * D + i] = W[i * dl + j];
      this.proj = { tipo: "f32", N: dl, K: D, w: this._sub(Wt, "mm.proj_t") };
      this.dim = dl;
      this._activaciones();
      this._armarGrupos();
      c.cola = null;
      return;
    }
    if (c.existe("mm.input_norm.weight")) {
      const w = await this._vec("mm.input_norm.weight");
      const b = c.existe("mm.input_norm.bias") ? await this._vec("mm.input_norm.bias") : this._sub(new Float32Array(K1), "ceros");
      this.inNorm = { w, b };
    }
    this.mm1 = await this._mat("mm.1.weight"); this.mm1b = await this._vec("mm.1.bias", true);
    this.mm2 = await this._mat("mm.2.weight"); this.mm2b = await this._vec("mm.2.bias", true);
    if (this.mm1.K !== K1) throw new Error(`mm.1 expects K=${this.mm1.K} but the ViT outputs ${K1}`);
    this.dim = this.mm2.N;   // dimension of the output embeddings (= language model dimension)
    this._activaciones();
    this._armarGrupos();
    c.cola = null;
  }

  _activaciones() {
    const gpu = this.gpu, P = this.piezaMax, D = this.D;
    const B = (n, l) => { const b = gpu.buffer(n * 4, undefined, l); this.bufs.push(b); return b; };
    const uni = (n) => { const b = gpu.device.createBuffer({ size: n, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }); this.bufs.push(b); return b; };
    this.bPaso = uni(16); this.bPasoN = uni(16); this.bUn = uni(16);
    this.bPatch = B(P * this.K0, "parches"); this.bPos = B(P * D, "pos");
    this.x = B(P * D, "vx"); this.xn = B(P * D, "vxn");
    this.q = B(P * D, "vq"); this.k = B(P * D, "vk"); this.v = B(P * D, "vv"); this.att = B(P * D, "vatt");
    this.h = B(P * Math.max(this.F, this.proy === "gemma3" ? 1 : this.mm1.N), "vh");
    this.bSal = B((P / (this.fusion * this.fusion)) * this.dim, "vsal");
    if (this.attTile) {
      this.QC = Math.min(globalThis.__QC || 256, P);   // queries per block (__QC for tests only)
      this.bS = B(this.nh * this.QC * P, "vS");       // scores [head][query][key]
      this.bVt = B(P * D, "vVt");                      // V transposed [nh*hd][T]
      this.bPasoQ = uni(16); this.bPasoR = uni(16);
      gpu.device.queue.writeBuffer(this.bPasoQ, 0, new Uint32Array([this.QC, 0, 0, 0]));
      gpu.device.queue.writeBuffer(this.bPasoR, 0, new Uint32Array([P % this.QC, 0, 0, 0]));
    }
  }

  _armarGrupos() {
    const gpu = this.gpu, D = this.D, P = this.bPaso, PN = this.bPasoN;
    const u = (vals) => gpu.uniforme(vals);
    const mm = (W, x, y, acc, paso) => ({ g: gpu.grupo(W.tipo === "f16" ? "matmul_vis_f16" : "matmul_vis_f32", [u([["u", W.N], ["u", W.K], ["u", acc], ["u", 0]]), paso, W.w, x, y]), N: W.N });
    const bias = (N, y, b, paso) => (b ? gpu.grupo("bias", [u([["u", N]]), paso, y, b]) : null);
    const ln = (Dn, b, w, bb, paso, eps) => gpu.grupo("layernorm", [u([["u", Dn], ["u", 0], ["f", eps], ["u", 0]]), paso, b, w, bb]);
    const lno = (x, w, bb, y) => gpu.grupo("layernorm_o", [u([["u", D], ["u", 0], ["f", this.eps], ["u", 0]]), P, x, w, bb, y]);
    const G = {};
    G.patch = mm(this.patchW, this.bPatch, this.x, 0, P);
    G.patchB = bias(D, this.x, this.patchB, P);
    G.pos = gpu.grupo("suma_esc", [u([["u", D], ["f", 1]]), P, this.x, this.bPos]);
    G.pre = this.preLn ? ln(D, this.x, this.preLn.w, this.preLn.b, P, this.eps) : null;
    const hd = this.hd, nh = this.nh, gem = this.proy === "gemma3";
    G.capas = this.capas.map((c) => ({
      ln1: lno(this.x, c.ln1.w, c.ln1.b, this.xn),
      q: mm(c.wq, this.xn, this.q, 0, P), bq: bias(D, this.q, c.bq, P),
      k: mm(c.wk, this.xn, this.k, 0, P), bk: bias(D, this.k, c.bk, P),
      v: mm(c.wv, this.xn, this.v, 0, P), bv: bias(D, this.v, c.bv, P),
      att: gem ? null : gpu.grupo("atencion", [u([["u", hd], ["u", nh], ["u", nh], ["u", 0], ["f", 1 / Math.sqrt(hd)], ["u", 0], ["u", 0xFFFFFFFF], ["u", 0]]), P, this.q, this.k, this.v, this.att]),
      o: mm(c.wo, this.att, this.x, 1, P), bo: bias(D, this.x, c.bo, P),
      ln2: lno(this.x, c.ln2.w, c.ln2.b, this.xn),
      up: mm(c.up, this.xn, this.h, 0, P), bup: bias(c.up.N, this.h, c.bup, P),
      act: gpu.grupo("activacion", [u([["u", c.up.N], ["u", this.act]]), P, this.h]), F: c.up.N,
      down: mm(c.down, this.h, this.x, 1, P), bdown: bias(D, this.x, c.bdown, P),
    }));
    G.post = this.postLn ? ln(D, this.x, this.postLn.w, this.postLn.b, P, this.eps) : null;
    if (gem) {
      // attention is the same in all layers (same buffers): one group per query chunk
      if (this.attTile) {
        const T = this.piezaMax, QC = this.QC, kvd = nh * hd, PB = this.bPaso;
        G.vt = gpu.grupo("trans_v", [u([["u", kvd], ["u", 0], ["u", 0], ["u", 0]]), PB, this.v, this.bVt]);
        G.chunks = [];
        for (let q0 = 0; q0 < T; q0 += QC) {
          const n = Math.min(QC, T - q0), ps = n === QC ? this.bPasoQ : this.bPasoR;
          G.chunks.push({ n,
            qk: gpu.grupo("mm_estr", [u([["u", T], ["u", hd], ["u", hd / 4], ["u", kvd / 4], ["u", hd / 4], ["u", kvd / 4], ["u", q0 * kvd / 4],
              ["u", QC * T], ["u", T], ["u", 0], ["f", 1 / Math.sqrt(hd)], ["u", 0]]), ps, this.k, this.q, this.bS]),
            sm: gpu.grupo("softmax_filas", [u([["u", T], ["u", QC * T], ["u", 0], ["u", 0]]), PB, this.bS]),
            pv: gpu.grupo("mm_estr", [u([["u", hd], ["u", T], ["u", hd * T / 4], ["u", T / 4], ["u", QC * T / 4], ["u", T / 4], ["u", 0],
              ["u", hd], ["u", kvd], ["u", q0 * kvd], ["f", 1], ["u", 0]]), ps, this.bVt, this.bS, this.att]) });
        }
      }
      G.attBi = [];
      this.tramo = globalThis.__TRAMO_ATT || TRAMO_ATT;   // (__TRAMO_ATT for tests only)
      for (let t0 = 0; !this.attTile && t0 < this.piezaMax; t0 += this.tramo) {
        G.attBi.push(gpu.grupo(this.kAtt, [u([["u", nh], ["u", t0], ["f", 1 / Math.sqrt(hd)], ["u", 0]]), P, this.q, this.k, this.v, this.att]));
      }
      G.pool = gpu.grupo("pool_avg", [this.bUn, PN, this.x, this.xn]);
      G.soft = gpu.grupo("rmsnorm", [u([["u", D], ["u", 0], ["f", this.eps], ["u", 0]]), PN, this.xn, this.softNorm, this.q]);
      G.proj = { g: gpu.grupo("matmul_vis_f32", [u([["u", this.proj.N], ["u", this.proj.K], ["u", 0], ["u", 0]]), PN, this.proj.w, this.q, this.bSal]), N: this.proj.N };
      this.G = G;
      return;
    }
    G.unshuffle = gpu.grupo("desbaraja", [this.bUn, P, this.x, this.xn]);
    const K1 = D * this.fusion * this.fusion;
    G.inNorm = this.inNorm ? ln(K1, this.xn, this.inNorm.w, this.inNorm.b, PN, 1e-5) : null;
    G.mm1 = mm(this.mm1, this.xn, this.h, 0, PN); G.mm1b = bias(this.mm1.N, this.h, this.mm1b, PN);
    G.act1 = gpu.grupo("activacion", [u([["u", this.mm1.N], ["u", 0]]), PN, this.h]);
    G.mm2 = { g: gpu.grupo(this.mm2.tipo === "f16" ? "matmul_vis_f16" : "matmul_vis_f32", [u([["u", this.mm2.N], ["u", this.mm2.K], ["u", 0], ["u", 0]]), PN, this.mm2.w, this.h, this.bSal]), N: this.mm2.N };
    G.mm2b = bias(this.mm2.N, this.bSal, this.mm2b, PN);
    this.G = G;
  }

  // img RGB u8 -> normalized planar Float32Array
  normalizada(img) { return normalizar(img, this.media, this.desv); }

  _parches(pix, nx, ny) {
    const p = this.parche, gw = nx / p, gh = ny / p, K0 = this.K0, n = nx * ny;
    const out = new Float32Array(gw * gh * K0);
    const redondea = this.patchF16;
    let o = 0;
    for (let py = 0; py < gh; py++) {
      for (let px = 0; px < gw; px++) {
        for (let c = 0; c < 3; c++) {
          for (let ky = 0; ky < p; ky++) {
            const base = c * n + (py * p + ky) * nx + px * p;
            for (let kx = 0; kx < p; kx++) { const v = pix[base + kx]; out[o++] = redondea ? f16r(v) : v; }
          }
        }
      }
    }
    return out;
  }

  // pix: normalized planar data of an nx x ny image (multiples of patch*merge). Returns { buf, n } with n*dim floats on the GPU.
  async codificar(pix, nx, ny) {
    const p = this.parche, m = this.fusion, D = this.D;
    if (nx % (p * m) || ny % (p * m)) throw new Error(`image size ${nx}x${ny} is not aligned to ${p * m}`);
    const gw = nx / p, gh = ny / p, T = gw * gh;
    if (T > this.piezaMax) throw new Error(`the image produces ${T} patches (maximum ${this.piezaMax})`);
    const nTok = T / (m * m);
    const dev = this.gpu.device, G = this.G;
    if (this.proy === "gemma3") return this._codificarGemma(pix, nx, ny, gw, gh, T, nTok);
    const clave = gw + "x" + gh;
    let pos = this.cachePos.get(clave);
    if (!pos) {
      pos = interpolarPosiciones(this.posTabla, this.nLado, D, gw, gh);
      if (this.cachePos.size > 6) this.cachePos.delete(this.cachePos.keys().next().value);
      this.cachePos.set(clave, pos);
    }
    dev.queue.writeBuffer(this.bPatch, 0, this._parches(pix, nx, ny));
    dev.queue.writeBuffer(this.bPos, 0, pos.buffer, pos.byteOffset, pos.byteLength);
    dev.queue.writeBuffer(this.bPaso, 0, new Uint32Array([T, 0, 0, 0]));
    dev.queue.writeBuffer(this.bPasoN, 0, new Uint32Array([nTok, 0, 0, 0]));
    dev.queue.writeBuffer(this.bUn, 0, new Uint32Array([D, gw, m, 0]));
    const salida = this.gpu.buffer(nTok * this.dim * 4, undefined, "emb_imagen");
    const validar = !this._validado;
    if (validar) dev.pushErrorScope("validation");
    const filas = (n, T2) => Math.ceil(T2 * n / 256);
    const mat = (pass, g, T2) => {
      const [x, y] = dim2(Math.ceil(g.N / 64));
      despachar(pass, g.g, x, y, Math.ceil(T2 / 32));
    };
    const enviar = (f) => { const enc = dev.createCommandEncoder(), pass = enc.beginComputePass(); f(pass); pass.end(); dev.queue.submit([enc.finish()]); };
    enviar((pass) => {
      mat(pass, G.patch, T);
      if (G.patchB) despachar(pass, G.patchB, filas(D, T));
      despachar(pass, G.pos, filas(D, T));
      if (G.pre) despachar(pass, G.pre, T);
    });
    if (this._parar === 0) return { raw: await this._leerBuf(this.x, T * D) };
    let capaN = 0;
    for (const gc of G.capas) {
      enviar((pass) => {
        despachar(pass, gc.ln1, T);
        mat(pass, gc.q, T); if (gc.bq) despachar(pass, gc.bq, filas(D, T));
        mat(pass, gc.k, T); if (gc.bk) despachar(pass, gc.bk, filas(D, T));
        mat(pass, gc.v, T); if (gc.bv) despachar(pass, gc.bv, filas(D, T));
        despachar(pass, gc.att, this.nh, T);
        mat(pass, gc.o, T); if (gc.bo) despachar(pass, gc.bo, filas(D, T));
        despachar(pass, gc.ln2, T);
        mat(pass, gc.up, T); if (gc.bup) despachar(pass, gc.bup, filas(gc.F, T));
        despachar(pass, gc.act, filas(gc.F, T));
        mat(pass, gc.down, T); if (gc.bdown) despachar(pass, gc.bdown, filas(D, T));
      });
      capaN++;
      if (this._parar === capaN) return { raw: await this._leerBuf(this.x, T * D) };
    }
    // projector: (post-LN) -> pixel-unshuffle -> [LN] -> MLP
    {
      const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
      if (G.post) despachar(pass, G.post, T);
      despachar(pass, G.unshuffle, filas(D, T));
      if (G.inNorm) despachar(pass, G.inNorm, nTok);
      mat(pass, G.mm1, nTok); if (G.mm1b) despachar(pass, G.mm1b, filas(this.mm1.N, nTok));
      despachar(pass, G.act1, filas(this.mm1.N, nTok));
      mat(pass, G.mm2, nTok); if (G.mm2b) despachar(pass, G.mm2b, filas(this.mm2.N, nTok));
      pass.end();
      enc.copyBufferToBuffer(this.bSal, 0, salida, 0, nTok * this.dim * 4);
      dev.queue.submit([enc.finish()]);
    }
    if (validar) {
      const err = await dev.popErrorScope();
      if (err) { this.gpu.liberar([salida]); throw new Error("WebGPU rejected the vision computation: " + err.message); }
      this._validado = true;
    }
    return { buf: salida, n: nTok };
  }


  // Gemma 3: a single nx x ny image (= image_size). Several submits per layer to stay under the GPU's ~2 s limit.
  async _codificarGemma(pix, nx, ny, gw, gh, T, nTok) {
    const dev = this.gpu.device, G = this.G, D = this.D, m = this.fusion;
    if (this.attTile && T !== this.piezaMax) throw new Error(`the image must be ${this.tamImagen}x${this.tamImagen}`);
    const clave = gw + "x" + gh;
    let pos = this.cachePos.get(clave);
    if (!pos) {
      pos = interpolarPosiciones(this.posTabla, this.nLado, D, gw, gh);
      this.cachePos.set(clave, pos);
    }
    dev.queue.writeBuffer(this.bPatch, 0, this._parches(pix, nx, ny));
    dev.queue.writeBuffer(this.bPos, 0, pos.buffer, pos.byteOffset, pos.byteLength);
    dev.queue.writeBuffer(this.bPaso, 0, new Uint32Array([T, 0, 0, 0]));
    dev.queue.writeBuffer(this.bPasoN, 0, new Uint32Array([nTok, 0, 0, 0]));
    dev.queue.writeBuffer(this.bUn, 0, new Uint32Array([D, gw, m, 0]));
    const salida = this.gpu.buffer(nTok * this.dim * 4, undefined, "emb_imagen");
    const validar = !this._validado;
    if (validar) dev.pushErrorScope("validation");
    const elem = (pass, g, n) => { const [x, y] = dim2(Math.ceil(n / 256)); despachar(pass, g, x, y); };
    const mat = (pass, g, T2) => { const [x, y] = dim2(Math.ceil(g.N / 64)); despachar(pass, g.g, x, y, Math.ceil(T2 / 32)); };
    const enviar = (f) => { const enc = dev.createCommandEncoder(), pass = enc.beginComputePass(); f(pass); pass.end(); dev.queue.submit([enc.finish()]); };
    // per-phase timings (waiting on the GPU after each submit): where the encoder time goes
    const perf = { qkv: 0, atencion: 0, mlp: 0 };
    const medir = async (fase, f) => { const t = performance.now(); enviar(f); await dev.queue.onSubmittedWorkDone(); perf[fase] += performance.now() - t; };
    this.ultimaPerf = perf;
    enviar((pass) => {
      mat(pass, G.patch, T);
      if (G.patchB) elem(pass, G.patchB, T * D);
      elem(pass, G.pos, T * D);
      if (G.pre) despachar(pass, G.pre, T);
    });
    if (this._parar === 0) return { raw: await this._leerBuf(this.x, T * D) };
    let capaN = 0;
    for (const gc of G.capas) {
      await medir("qkv", (pass) => {
        despachar(pass, gc.ln1, T);
        mat(pass, gc.q, T); if (gc.bq) elem(pass, gc.bq, T * D);
        mat(pass, gc.k, T); if (gc.bk) elem(pass, gc.bk, T * D);
        mat(pass, gc.v, T); if (gc.bv) elem(pass, gc.bv, T * D);
      });
      if (this.attTile) {
        await medir("atencion", (pass) => elem(pass, G.vt, T * this.nh * this.hd));
        for (let c = 0; c < G.chunks.length; c += 4) {
          await medir("atencion", (pass) => {
            for (const ch of G.chunks.slice(c, c + 4)) {
              const [a, b] = dim2(Math.ceil(T / 64));
              despachar(pass, ch.qk, a, this.nh, Math.ceil(ch.n / 32));
              despachar(pass, ch.sm, ch.n, this.nh);
              despachar(pass, ch.pv, Math.ceil(this.hd / 64), this.nh, Math.ceil(ch.n / 32));
            }
          });
        }
      } else {
        for (let t0 = 0, i = 0; t0 < T; t0 += this.tramo, i++) {
          await medir("atencion", (pass) => despachar(pass, G.attBi[i], Math.ceil(Math.min(this.tramo, T - t0) / 64), this.nh));
        }
      }
      await medir("mlp", (pass) => {
        mat(pass, gc.o, T); if (gc.bo) elem(pass, gc.bo, T * D);
        despachar(pass, gc.ln2, T);
        mat(pass, gc.up, T); if (gc.bup) elem(pass, gc.bup, T * gc.F);
        elem(pass, gc.act, T * gc.F);
      });
      await medir("mlp", (pass) => { mat(pass, gc.down, T); if (gc.bdown) elem(pass, gc.bdown, T * D); });
      capaN++;
      if (this.onCapa) this.onCapa(capaN, this.L);
      if (this._parar === capaN) return { raw: await this._leerBuf(this.x, T * D) };
    }
    {
      const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
      if (G.post) despachar(pass, G.post, T);
      elem(pass, G.pool, nTok * D);
      despachar(pass, G.soft, nTok);
      mat(pass, G.proj, nTok);
      pass.end();
      enc.copyBufferToBuffer(this.bSal, 0, salida, 0, nTok * this.dim * 4);
      dev.queue.submit([enc.finish()]);
    }
    if (validar) {
      const err = await dev.popErrorScope();
      if (err) { this.gpu.liberar([salida]); throw new Error("WebGPU rejected the vision computation: " + err.message); }
      this._validado = true;
    }
    return { buf: salida, n: nTok };
  }

  async _leerBuf(buf, nfloats) {
    const dev = this.gpu.device, bytes = nfloats * 4;
    const lectura = dev.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = dev.createCommandEncoder(); enc.copyBufferToBuffer(buf, 0, lectura, 0, bytes); dev.queue.submit([enc.finish()]);
    await lectura.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(lectura.getMappedRange(0, bytes).slice(0)); lectura.unmap(); lectura.destroy();
    return Array.from(out);
  }

  async leer(buf, n) {
    const dev = this.gpu.device, bytes = n * this.dim * 4;
    const lectura = dev.createBuffer({ size: Math.max(16, Math.ceil(bytes / 4) * 4), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(buf, 0, lectura, 0, bytes);
    dev.queue.submit([enc.finish()]);
    await lectura.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(lectura.getMappedRange(0, bytes).slice(0));
    lectura.unmap(); lectura.destroy();
    return out;
  }

  free() {
    this.gpu.liberar([...this.bufs, ...(this.cargador ? this.cargador.buffers : [])]);
    this.bufs = [];
    this.cachePos.clear();
  }
}
