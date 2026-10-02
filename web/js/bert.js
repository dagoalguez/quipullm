// BERT-type embedding model: nomic-bert (nomic-embed-text).
// Reference: llama.cpp src/models/{bert,nomic-bert}.cpp
//   x = LayerNorm(tok_embd[id] + type_embd[0])
//   layer: q,k,v = Wqkv x ; RoPE NEOX on q,k ; bidirectional attention ; x = LN(x + Wo att) ; x = LN(x + down(silu(gate x) * up x))
//   output: mean (or CLS) of all tokens, normalized to norm 1.
import { despachar, dim2 } from "./gpu.js";
import { CargadorPesos } from "./weights.js";

const K_TIPOS = ["q3k", "q4k", "q5k", "q6k"];

export class ModeloBert {
  constructor(gpu, meta) {
    this.gpu = gpu;
    this.meta = meta;
    const kv = meta.kv, a = kv["general.architecture"];
    this.arch = a;
    const g = (k, d) => (kv[`${a}.${k}`] !== undefined ? kv[`${a}.${k}`] : d);
    this.nLayers = g("block_count");
    this.D = g("embedding_length");
    this.nh = g("attention.head_count");
    this.hd = this.D / this.nh;
    this.eps = g("attention.layer_norm_epsilon", 1e-12);
    this.base = g("rope.freq_base", 1000);
    this.pooling = g("pooling_type", 1);   // 1 = mean, 2 = CLS
    this.ctxTrain = g("context_length", 2048);
    this.ctx = Math.min(meta.ctx || this.ctxTrain, this.ctxTrain);
    this.vocab = kv["tokenizer.ggml.tokens"].length;
    this.bufs = [];
    this.baldosa = !(meta.options && meta.options.tiled_prefill === false);
    if (this.hd > 256) throw new Error(`head_dim ${this.hd} > 256 not supported`);
    if (this.pooling !== 1 && this.pooling !== 2) throw new Error(`pooling ${this.pooling} not supported (only mean and CLS)`);
  }

  async load(url, progreso) {
    const gpu = this.gpu, c = new CargadorPesos(gpu, this.meta, url, progreso);
    this.cargador = c;
    const D = this.D;
    const ord = ["token_embd.weight", "token_types.weight", "token_embd_norm.weight", "token_embd_norm.bias"];
    for (let i = 0; i < this.nLayers; i++) {
      const p = `blk.${i}.`;
      ord.push(p + "attn_qkv.weight", p + "attn_qkv.bias", p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight",
        p + "attn_output.weight", p + "attn_output.bias", p + "attn_output_norm.weight", p + "attn_output_norm.bias",
        p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight", p + "layer_output_norm.weight", p + "layer_output_norm.bias");
    }
    c.planificar(ord);
    this.emb = await c.matriz("token_embd.weight");
    this.vocab = this.emb.N;
    this.tipoEmb = c.existe("token_types.weight") ? await c.f32("token_types.weight") : null;
    this.normEmbW = await c.f32("token_embd_norm.weight");
    this.normEmbB = await c.f32("token_embd_norm.bias");
    const tipos = new Set([this.emb.tipo]);
    const opc = async (n) => (c.existe(n) ? await c.f32(n) : null);
    this.capas = [];
    let F = 0;
    for (let i = 0; i < this.nLayers; i++) {
      const p = `blk.${i}.`;
      const capa = {};
      if (c.existe(p + "attn_qkv.weight")) {
        const t = c.info(p + "attn_qkv.weight");
        const u8 = await c.bytes(t);
        capa.wq = await c.matriz(p + "attn_qkv.weight", [0, D], u8);
        capa.wk = await c.matriz(p + "attn_qkv.weight", [D, 2 * D], u8);
        capa.wv = await c.matriz(p + "attn_qkv.weight", [2 * D, 3 * D], u8);
        if (c.existe(p + "attn_qkv.bias")) {
          const tb = c.info(p + "attn_qkv.bias");
          const b = new Float32Array(await this._f32cpu(c, tb));
          capa.bq = gpu.subir(b.subarray(0, D), p + "bq"); capa.bk = gpu.subir(b.subarray(D, 2 * D), p + "bk");
          capa.bv = gpu.subir(b.subarray(2 * D, 3 * D), p + "bv");
          c.buffers.push(capa.bq, capa.bk, capa.bv);
        }
      } else {
        capa.wq = await c.matriz(p + "attn_q.weight");
        capa.wk = await c.matriz(p + "attn_k.weight");
        capa.wv = await c.matriz(p + "attn_v.weight");
      }
      capa.wo = await c.matriz(p + "attn_output.weight");
      capa.bo = await opc(p + "attn_output.bias");
      capa.n1w = await c.f32(p + "attn_output_norm.weight"); capa.n1b = await c.f32(p + "attn_output_norm.bias");
      capa.gate = await c.matriz(p + "ffn_gate.weight");
      capa.up = await c.matriz(p + "ffn_up.weight");
      capa.down = await c.matriz(p + "ffn_down.weight");
      if (c.existe(p + "ffn_up.bias") || c.existe(p + "ffn_down.bias")) throw new Error("FFN with biases is not supported in this version");
      capa.n2w = await c.f32(p + "layer_output_norm.weight"); capa.n2b = await c.f32(p + "layer_output_norm.bias");
      for (const m of [capa.wq, capa.wk, capa.wv, capa.wo, capa.gate, capa.up, capa.down]) tipos.add(m.tipo);
      F = Math.max(F, capa.gate.N);
      this.capas.push(capa);
    }
    this.F = F;
    const extra = [];
    for (const t of tipos) {
      if (K_TIPOS.includes(t)) extra.push(`matmul_${t}_1`, `matmul_${t}_4`);
      if (K_TIPOS.includes(t) && this.baldosa) extra.push(`matmul_${t}_t`);
    }
    if (K_TIPOS.includes(this.emb.tipo)) extra.push(`embed_${this.emb.tipo}`);
    await gpu.prepararPipelines(extra);
    const T = this.ctx;
    const B = (n, l) => { const b = gpu.buffer(n * 4, undefined, l); this.bufs.push(b); return b; };
    this.bPaso = gpu.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bIds = B(T, "ids");
    this.x = B(T * D, "x");
    this.q = B(T * D, "q"); this.k = B(T * D, "k"); this.v = B(T * D, "v"); this.att = B(T * D, "att");
    this.tmp = B(T * D, "tmp");
    this.g = B(T * F, "g"); this.u = B(T * F, "u");
    this.lectura = gpu.device.createBuffer({ size: T * D * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this._armarGrupos();
  }

  async _f32cpu(c, t) {
    const u8 = await c.bytes(t);
    if (t.tipo === 0) return new Float32Array(u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.length));
    if (t.tipo === 1) { const { f16af32 } = await import("./gpu.js"); return f16af32(u8); }
    throw new Error(`${t.nombre}: type not supported`);
  }

  _mm(W, x, y, acc = 0) {
    const gpu = this.gpu;
    const u = gpu.uniforme([["u", W.N], ["u", W.K], ["u", acc], ["u", 0]]);
    if (K_TIPOS.includes(W.tipo)) {
      return {
        uno: gpu.grupo(`matmul_${W.tipo}_1`, [u, this.bPaso, W.w, x, y]),
        lote: gpu.grupo(`matmul_${W.tipo}_4`, [u, this.bPaso, W.w, x, y]), z4: true, filas: W.N,
        baldosa: this.baldosa ? gpu.grupo(`matmul_${W.tipo}_t`, [u, this.bPaso, W.w, x, y]) : null,
      };
    }
    if (W.tipo === "q8") {
      return {
        uno: gpu.grupo("matmul_q8_1", [u, this.bPaso, W.qs, W.ds, x, y]),
        lote: gpu.grupo("matmul_q8_4", [u, this.bPaso, W.qs, W.ds, x, y]), z4: true, filas: W.N,
      };
    }
    return {
      uno: gpu.grupo("matmul_f32_1", [u, this.bPaso, W.w, x, y]),
      lote: gpu.grupo("matmul_f32_4", [u, this.bPaso, W.w, x, y]), z4: true, filas: W.N,
    };
  }

  _armarGrupos() {
    const gpu = this.gpu, D = this.D, P = this.bPaso;
    const G = {};
    const et = this.emb.tipo;
    if (et === "q8") G.embed = gpu.grupo("embed_q8", [gpu.uniforme([["u", D]]), P, this.emb.qs, this.emb.ds, this.bIds, this.x]);
    else G.embed = gpu.grupo(et === "f32" ? "embed_f32" : `embed_${et}`, [gpu.uniforme([["u", D]]), P, this.emb.w, this.bIds, this.x]);
    if (this.tipoEmb) G.tipo = gpu.grupo("bias", [gpu.uniforme([["u", D]]), P, this.x, this.tipoEmb]);   // row 0 of token_types
    const uLN = gpu.uniforme([["u", D], ["u", 0], ["f", this.eps], ["u", 0]]);
    G.normEmb = gpu.grupo("layernorm", [uLN, P, this.x, this.normEmbW, this.normEmbB]);
    const sesgo = (N, b, y) => (b ? gpu.grupo("bias", [gpu.uniforme([["u", N]]), P, y, b]) : null);
    const uRope = gpu.uniforme([["u", this.hd], ["u", this.nh], ["u", 0], ["u", 1], ["f", this.eps], ["f", this.base], ["u", this.hd], ["u", 0]]);
    const uAtt = gpu.uniforme([["u", this.hd], ["u", this.nh], ["u", this.nh], ["u", 0], ["f", 1 / Math.sqrt(this.hd)], ["u", 0], ["u", 0xFFFFFFFF], ["u", 0]]);
    G.capas = this.capas.map((c) => {
      const gc = {
        wq: this._mm(c.wq, this.x, this.q), bq: sesgo(D, c.bq, this.q),
        wk: this._mm(c.wk, this.x, this.k), bk: sesgo(D, c.bk, this.k),
        wv: this._mm(c.wv, this.x, this.v), bv: sesgo(D, c.bv, this.v),
        ropeQ: gpu.grupo("norm_rope", [uRope, P, this.q, this.normEmbW]),
        ropeK: gpu.grupo("norm_rope", [uRope, P, this.k, this.normEmbW]),
        att: gpu.grupo("atencion", [uAtt, P, this.q, this.k, this.v, this.att]),
        n1: gpu.grupo("layernorm", [uLN, P, this.x, c.n1w, c.n1b]),
        gate: this._mm(c.gate, this.x, this.g), up: this._mm(c.up, this.x, this.u),
        swiglu: gpu.grupo("swiglu", [gpu.uniforme([["u", c.gate.N]]), P, this.g, this.u]), F: c.gate.N,
        n2: gpu.grupo("layernorm", [uLN, P, this.x, c.n2w, c.n2b]),
      };
      if (c.bo) {
        gc.wo = this._mm(c.wo, this.att, this.tmp, 0); gc.bo = sesgo(D, c.bo, this.tmp);
        gc.suma = gpu.grupo("suma_esc", [gpu.uniforme([["u", D], ["f", 1]]), P, this.x, this.tmp]);
      } else gc.wo = this._mm(c.wo, this.att, this.x, 1);
      gc.down = this._mm(c.down, this.g, this.x, 1);
      return gc;
    });
    this.G = G;
  }

  reset() {}

  // ids: tokens (with CLS and SEP already in place). Returns Float32Array(D): mean (or CLS) normalized to norm 1.
  async embed(ids) {
    const T = ids.length;
    if (T < 1) throw new Error("empty text");
    if (T > this.ctx) throw new Error(`the text has ${T} tokens and the model context is ${this.ctx}`);
    const dev = this.gpu.device, G = this.G, D = this.D;
    dev.queue.writeBuffer(this.bPaso, 0, new Uint32Array([T, 0, 0, 0]));
    dev.queue.writeBuffer(this.bIds, 0, new Uint32Array(ids));
    const validar = !this._validado;
    if (validar) dev.pushErrorScope("validation");
    const filas = (n) => Math.ceil(T * n / 256);
    const mm = (pass, g) => {
      if (this.baldosa && g.baldosa && T >= 8) {
        const [x, y] = dim2(Math.ceil(g.filas / 64));
        despachar(pass, g.baldosa, x, y, Math.ceil(T / 32));
        return;
      }
      const uno = T === 1;
      const [x, y] = dim2(Math.ceil(g.filas / 8));
      despachar(pass, uno ? g.uno : g.lote, x, y, uno ? 1 : Math.ceil(T / 4));
    };
    {
      const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
      despachar(pass, G.embed, filas(D));
      if (G.tipo) despachar(pass, G.tipo, filas(D));
      despachar(pass, G.normEmb, T);
      pass.end(); dev.queue.submit([enc.finish()]);
    }
    for (const gc of G.capas) {   // one submit per layer: short commands for the integrated GPU
      const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
      mm(pass, gc.wq); if (gc.bq) despachar(pass, gc.bq, filas(D));
      mm(pass, gc.wk); if (gc.bk) despachar(pass, gc.bk, filas(D));
      mm(pass, gc.wv); if (gc.bv) despachar(pass, gc.bv, filas(D));
      despachar(pass, gc.ropeQ, this.nh, T);
      despachar(pass, gc.ropeK, this.nh, T);
      despachar(pass, gc.att, this.nh, T);
      mm(pass, gc.wo);
      if (gc.suma) { if (gc.bo) despachar(pass, gc.bo, filas(D)); despachar(pass, gc.suma, filas(D)); }
      despachar(pass, gc.n1, T);
      mm(pass, gc.gate); mm(pass, gc.up);
      despachar(pass, gc.swiglu, filas(gc.F));
      mm(pass, gc.down);
      despachar(pass, gc.n2, T);
      pass.end(); dev.queue.submit([enc.finish()]);
    }
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.x, 0, this.lectura, 0, T * D * 4);
    dev.queue.submit([enc.finish()]);
    if (validar) {
      const err = await dev.popErrorScope();
      if (err) throw new Error("WebGPU rejected the computation: " + err.message);
      this._validado = true;
    }
    await this.lectura.mapAsync(GPUMapMode.READ, 0, T * D * 4);
    const h = new Float32Array(this.lectura.getMappedRange(0, T * D * 4).slice(0));
    this.lectura.unmap();
    const out = new Float32Array(D);
    if (this.pooling === 2) out.set(h.subarray(0, D));
    else {
      for (let t = 0; t < T; t++) for (let i = 0; i < D; i++) out[i] += h[t * D + i];
      for (let i = 0; i < D; i++) out[i] /= T;
    }
    let n = 0;
    for (let i = 0; i < D; i++) n += out[i] * out[i];
    n = Math.sqrt(n) || 1;
    for (let i = 0; i < D; i++) out[i] /= n;
    return out;
  }

  free() {
    this.gpu.liberar([...this.bufs, ...(this.cargador ? this.cargador.buffers : [])]);
    if (this.lectura) this.lectura.destroy();
    this.bufs = [];
  }
}
