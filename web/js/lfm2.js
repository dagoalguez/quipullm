// LFM2 / LFM2.5 model (Liquid AI): short-convolution layers + GQA attention layers.
// Reference: llama.cpp src/models/lfm2.cpp
import { TB, despachar, dim2 } from "./gpu.js";
import { CargadorPesos } from "./weights.js";

export class ModeloLFM2 {
  constructor(gpu, meta) {
    this.gpu = gpu;
    this.meta = meta;
    const kv = meta.kv, a = kv["general.architecture"];
    const g = (k, d) => (kv[`${a}.${k}`] !== undefined ? kv[`${a}.${k}`] : d);
    this.nLayers = g("block_count");
    this.D = g("embedding_length");
    this.nh = g("attention.head_count");
    const hkv = g("attention.head_count_kv");
    this.nkvCapa = Array.isArray(hkv) ? hkv : new Array(this.nLayers).fill(hkv);
    this.hd = g("attention.key_length", this.D / this.nh);
    this.eps = g("attention.layer_norm_rms_epsilon", 1e-5);
    this.base = g("rope.freq_base", 10000);
    this.nrot = g("rope.dimension_count", this.hd);
    this.L = g("shortconv.l_cache", 3);
    this.swa = g("attention.sliding_window", 0) || 0;
    this.ctx = meta.ctx || 4096;
    this.vocab = kv["tokenizer.ggml.tokens"].length;
    this.bufs = [];
    this.supportsImages = true;   // forward() accepts external embeddings (vision)
  }

  async load(url, progreso) {
    const gpu = this.gpu, c = new CargadorPesos(gpu, this.meta, url, progreso);
    this.cargador = c;
    await gpu.prepararPipelines(["mezcla_emb"]);
    const D = this.D;
    const nombreNorm = c.existe("token_embd_norm.weight") ? "token_embd_norm.weight" : "output_norm.weight";
    // Order in which the tensors will be requested (to read them ahead of time)
    const orden = ["token_embd.weight", "output.weight", nombreNorm];
    for (let i = 0; i < this.nLayers; i++) {
      const p = `blk.${i}.`;
      orden.push(p + "attn_norm.weight", p + "ffn_norm.weight");
      if (this.nkvCapa[i] > 0) {
        orden.push(p + "attn_qkv.weight", p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight",
          p + "attn_output.weight", p + "attn_q_norm.weight", p + "attn_k_norm.weight");
      } else {
        orden.push(p + "shortconv.in_proj.weight", p + "shortconv.out_proj.weight", p + "shortconv.conv.weight");
      }
      orden.push(p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight");
    }
    c.planificar(orden);
    this.emb = await c.matriz("token_embd.weight");
    this.vocab = this.emb.N;
    this.salida = c.existe("output.weight") ? await c.matriz("output.weight") : this.emb;
    this.normFinal = await c.f32(nombreNorm);
    this.capas = [];
    let F = 0, kvdMax = 0;
    for (let i = 0; i < this.nLayers; i++) {
      const p = `blk.${i}.`;
      const capa = { attnNorm: await c.f32(p + "attn_norm.weight"), ffnNorm: await c.f32(p + "ffn_norm.weight") };
      const nkv = this.nkvCapa[i];
      if (nkv > 0) {
        capa.tipo = "atencion";
        capa.nkv = nkv;
        const qd = this.nh * this.hd, kvd = nkv * this.hd;
        kvdMax = Math.max(kvdMax, kvd);
        if (c.existe(p + "attn_qkv.weight")) {
          const t = c.info(p + "attn_qkv.weight");
          const u8 = await c.bytes(t);
          capa.wq = await c.matriz(p + "attn_qkv.weight", [0, qd], u8);
          capa.wk = await c.matriz(p + "attn_qkv.weight", [qd, qd + kvd], u8);
          capa.wv = await c.matriz(p + "attn_qkv.weight", [qd + kvd, qd + 2 * kvd], u8);
        } else {
          capa.wq = await c.matriz(p + "attn_q.weight");
          capa.wk = await c.matriz(p + "attn_k.weight");
          capa.wv = await c.matriz(p + "attn_v.weight");
        }
        capa.wo = await c.matriz(p + "attn_output.weight");
        capa.qNorm = c.existe(p + "attn_q_norm.weight") ? await c.f32(p + "attn_q_norm.weight") : null;
        capa.kNorm = c.existe(p + "attn_k_norm.weight") ? await c.f32(p + "attn_k_norm.weight") : null;
        capa.kc = gpu.buffer(this.ctx * kvd * 4, undefined, `kcache${i}`);
        capa.vc = gpu.buffer(this.ctx * kvd * 4, undefined, `vcache${i}`);
        this.bufs.push(capa.kc, capa.vc);
      } else {
        capa.tipo = "conv";
        capa.inProj = await c.matriz(p + "shortconv.in_proj.weight");
        capa.outProj = await c.matriz(p + "shortconv.out_proj.weight");
        capa.convW = await c.f32(p + "shortconv.conv.weight");
        capa.estado = gpu.buffer(Math.max(1, this.L - 1) * D * 4, undefined, `conv${i}`);
        this.bufs.push(capa.estado);
      }
      capa.gate = await c.matriz(p + "ffn_gate.weight");
      capa.up = await c.matriz(p + "ffn_up.weight");
      capa.down = await c.matriz(p + "ffn_down.weight");
      F = Math.max(F, capa.gate.N);
      this.capas.push(capa);
    }
    this.F = F;
    // Activations
    const B = (n, l) => { const b = gpu.buffer(n * 4, undefined, l); this.bufs.push(b); return b; };
    this.bPaso = gpu.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bIds = B(TB, "ids");
    this.x = B(TB * D, "x");
    this.xn = B(TB * D, "xn");
    this.bcx = B(TB * 3 * D, "bcx");
    this.y = B(TB * D, "y");
    this.q = B(TB * this.nh * this.hd, "q");
    this.k = B(TB * Math.max(kvdMax, 1), "k");
    this.v = B(TB * Math.max(kvdMax, 1), "v");
    this.att = B(TB * this.nh * this.hd, "att");
    this.g = B(TB * F, "g");
    this.u = B(TB * F, "u");
    this.xl = B(D, "xlast");
    this.xext = B(TB * D, "xext");        // external embeddings (images) of the batch
    this.bMask = B(TB, "mascara_ext");
    this.logits = B(this.vocab, "logits");
    this.lectura = gpu.device.createBuffer({ size: this.vocab * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this._armarGrupos();
    this.reset();
  }

  // ------------------------------------------------------------------ dispatch groups (fixed)
  _mm(W, x, y, acc = 0, tfijo = 0) {
    // Two variants: batched (prefill) and single-token (decoding, faster).
    const gpu = this.gpu;
    const u = gpu.uniforme([["u", W.N], ["u", W.K], ["u", acc], ["u", tfijo]]);
    const lote = W.tipo === "q8"
      ? gpu.grupo("matmul_q8", [u, this.bPaso, W.qs, W.ds, x, y])
      : gpu.grupo("matmul_f32", [u, this.bPaso, W.w, x, y]);
    let uno = lote, lote4 = lote, z4 = false;
    if (W.K % 4 === 0) {
      uno = W.tipo === "q8"
        ? gpu.grupo("matmul_q8_1", [u, this.bPaso, W.qs, W.ds, x, y])
        : gpu.grupo("matmul_f32_1", [u, this.bPaso, W.w, x, y]);
      lote4 = W.tipo === "q8"
        ? gpu.grupo("matmul_q8_4", [u, this.bPaso, W.qs, W.ds, x, y])
        : gpu.grupo("matmul_f32_4", [u, this.bPaso, W.w, x, y]);
      z4 = true;
    }
    return { lote: lote4, uno, filas: W.N, z4 };
  }

  _armarGrupos() {
    const gpu = this.gpu, D = this.D, P = this.bPaso;
    const G = {};
    G.embed = this.emb.tipo === "q8"
      ? gpu.grupo("embed_q8", [gpu.uniforme([["u", D]]), P, this.emb.qs, this.emb.ds, this.bIds, this.x])
      : gpu.grupo("embed_f32", [gpu.uniforme([["u", D]]), P, this.emb.w, this.bIds, this.x]);
    G.mezcla = gpu.grupo("mezcla_emb", [gpu.uniforme([["u", D]]), P, this.x, this.xext, this.bMask]);
    const uNorm = gpu.uniforme([["u", D], ["u", 0], ["f", this.eps], ["u", 0]]);
    G.capas = this.capas.map((c) => {
      const gc = {
        attnNorm: gpu.grupo("rmsnorm", [uNorm, P, this.x, c.attnNorm, this.xn]),
        ffnNorm: gpu.grupo("rmsnorm", [uNorm, P, this.x, c.ffnNorm, this.xn]),
        gate: this._mm(c.gate, this.xn, this.g),
        up: this._mm(c.up, this.xn, this.u),
        swiglu: gpu.grupo("swiglu", [gpu.uniforme([["u", c.gate.N]]), P, this.g, this.u]),
        down: this._mm(c.down, this.g, this.x, 1),
        F: c.gate.N,
      };
      if (c.tipo === "conv") {
        gc.inProj = this._mm(c.inProj, this.xn, this.bcx);
        gc.conv = gpu.grupo("conv_lfm2", [gpu.uniforme([["u", D], ["u", this.L]]), P, this.bcx, c.convW, c.estado, this.y]);
        gc.outProj = this._mm(c.outProj, this.y, this.x, 1);
      } else {
        const kvd = c.nkv * this.hd;
        const dummy = this.normFinal;
        gc.wq = this._mm(c.wq, this.xn, this.q);
        gc.wk = this._mm(c.wk, this.xn, this.k);
        gc.wv = this._mm(c.wv, this.xn, this.v);
        const ur = (nh, usa) => gpu.uniforme([["u", this.hd], ["u", nh], ["u", usa ? 1 : 0], ["u", 1],
          ["f", this.eps], ["f", this.base], ["u", this.nrot], ["u", 0]]);
        gc.ropeQ = gpu.grupo("norm_rope", [ur(this.nh, !!c.qNorm), P, this.q, c.qNorm || dummy]);
        gc.ropeK = gpu.grupo("norm_rope", [ur(c.nkv, !!c.kNorm), P, this.k, c.kNorm || dummy]);
        gc.kvw = gpu.grupo("kv_write", [gpu.uniforme([["u", kvd]]), P, this.k, this.v, c.kc, c.vc]);
        gc.att = gpu.grupo("atencion", [gpu.uniforme([["u", this.hd], ["u", this.nh], ["u", c.nkv], ["u", this.swa],
          ["f", 1 / Math.sqrt(this.hd)], ["u", 0], ["u", 0], ["u", 0]]), P, this.q, c.kc, c.vc, this.att]);
        gc.wo = this._mm(c.wo, this.att, this.x, 1);
        gc.kvd = kvd;
        gc.nkv = c.nkv;
      }
      return gc;
    });
    G.normFinal = gpu.grupo("rmsnorm", [gpu.uniforme([["u", D], ["u", 1], ["f", this.eps], ["u", 0]]), P, this.x, this.normFinal, this.xl]);
    G.salida = this._mm(this.salida, this.xl, this.logits, 0, 1);
    this.G = G;
  }

  reset() {
    // Zeroes the convolution states. The KV cache is overwritten by position.
    for (const c of this.capas) {
      if (c.estado) this.gpu.device.queue.writeBuffer(c.estado, 0, new Float32Array(c.estado._tam / 4));
    }
    this.pos = 0;
  }

  // ------------------------------------------------------------------ forward pass
  // Processes up to TB tokens from the current position. If conLogits, returns a Float32Array with the logits of the last one.
  // ext (optional): per token, null or {buf, fila}: instead of the token embedding, row 'fila' (D floats) of the
  // GPU buffer 'buf' (image embeddings) is used. The id of those tokens is only filler.
  async forward(ids, conLogits, ext = null) {
    const T = ids.length;
    if (T < 1 || T > TB) throw new Error("invalid batch");
    if (this.pos + T > this.ctx) throw new Error(`context exceeded (${this.ctx} tokens)`);
    const dev = this.gpu.device, G = this.G;
    dev.queue.writeBuffer(this.bPaso, 0, new Uint32Array([T, this.pos, 0, 0]));
    dev.queue.writeBuffer(this.bIds, 0, new Uint32Array(ids));
    const validar = !this._validado;
    if (validar) dev.pushErrorScope("validation");
    const enc = dev.createCommandEncoder();
    const hayExt = ext && ext.some((e) => e);
    if (hayExt) {
      const mascara = new Uint32Array(TB), D4 = this.D * 4;
      for (let t = 0; t < T; t++) {
        const e = ext[t];
        if (!e) continue;
        mascara[t] = 1;
        enc.copyBufferToBuffer(e.buf, e.fila * D4, this.xext, t * D4, D4);
      }
      dev.queue.writeBuffer(this.bMask, 0, mascara);
    }
    const pass = enc.beginComputePass();
    const mm = (g, uno = T === 1) => {
      const [x, y] = dim2(Math.ceil(g.filas / 8));
      // Without z4 (K not a multiple of 4) the old kernel handles all tokens in a single group.
      despachar(pass, uno ? g.uno : g.lote, x, y, uno || !g.z4 ? 1 : Math.ceil(T / 4));
    };
    despachar(pass, G.embed, Math.ceil(T * this.D / 256));
    if (hayExt) despachar(pass, G.mezcla, Math.ceil(T * this.D / 256));
    for (const gc of G.capas) {
      despachar(pass, gc.attnNorm, T);
      if (gc.conv) {
        mm(gc.inProj);
        despachar(pass, gc.conv, Math.ceil(this.D / 256));
        mm(gc.outProj);
      } else {
        mm(gc.wq); mm(gc.wk); mm(gc.wv);
        despachar(pass, gc.ropeQ, this.nh, T);
        despachar(pass, gc.ropeK, gc.nkv, T);
        despachar(pass, gc.kvw, Math.ceil(T * gc.kvd / 256));
        despachar(pass, gc.att, this.nh, T);
        mm(gc.wo);
      }
      despachar(pass, gc.ffnNorm, T);
      mm(gc.gate); mm(gc.up);
      despachar(pass, gc.swiglu, Math.ceil(T * gc.F / 256));
      mm(gc.down);
    }
    if (conLogits) {
      despachar(pass, G.normFinal, 1);
      mm(G.salida, true);
    }
    pass.end();
    if (conLogits) enc.copyBufferToBuffer(this.logits, 0, this.lectura, 0, this.vocab * 4);
    dev.queue.submit([enc.finish()]);
    if (validar) {
      const err = await dev.popErrorScope();
      if (err) throw new Error("WebGPU rejected the computation: " + err.message);
      if (T === 1 && conLogits) this._validado = true;
    }
    this.pos += T;
    if (!conLogits) return null;
    await this.lectura.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.lectura.getMappedRange().slice(0));
    this.lectura.unmap();
    return out;
  }

  free() {
    const todos = [...this.bufs, ...(this.cargador ? this.cargador.buffers : [])];
    this.gpu.liberar(todos);
    if (this.lectura) this.lectura.destroy();
    this.bufs = [];
  }
}
