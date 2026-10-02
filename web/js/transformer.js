// Dense Llama-style transformer: qwen2, llama (Mistral-Nemo...), granite, gemma3.
// Reference: llama.cpp src/models/{qwen2,llama,granite,gemma3}.cpp
//   layer: x = x + wo(attn(rmsnorm(x)))   ;   x = x + down(silu(gate(h)) * up(h)), h = rmsnorm(x)
//   qwen2: biases on q/k/v and NEOX RoPE; llama/granite: RoPE over consecutive pairs (NORM)
//   granite: embedding, residual, attention and logits scaling.
//   gemma3: embedding x sqrt(D), per-head q/k-norm, post-attention and post-FFN norms, GeGLU,
//           local layers (sliding window, RoPE base 10000) and global layers (linear RoPE x1/8) in a 5:1 pattern.
import { despachar, dim2 } from "./gpu.js";

const TB = 32;   // tokens per prefill pass (tiled matmul over 32-token tiles)
const TB_IMG = 256;   // gemma3 with vision: an image (256 tokens) is processed in one go, with non-causal attention
import { CargadorPesos } from "./weights.js";

const ROPE_NEOX = new Set(["qwen2", "qwen3", "phi3", "gemma", "gemma2", "gemma3", "stablelm", "olmo2"]);

export class ModeloTransformer {
  constructor(gpu, meta) {
    this.gpu = gpu;
    this.meta = meta;
    const kv = meta.kv, a = kv["general.architecture"];
    this.arch = a;
    const g = (k, d) => (kv[`${a}.${k}`] !== undefined ? kv[`${a}.${k}`] : d);
    this.nLayers = g("block_count");
    this.D = g("embedding_length");
    this.nh = g("attention.head_count");
    const hkv = g("attention.head_count_kv", this.nh);
    this.nkvCapa = Array.isArray(hkv) ? hkv : new Array(this.nLayers).fill(hkv);
    this.hd = g("attention.key_length", this.D / this.nh);
    this.eps = g("attention.layer_norm_rms_epsilon", 1e-5);
    this.base = g("rope.freq_base", 10000);
    this.nrot = g("rope.dimension_count", this.hd);
    const oa = meta.arch_options || {};   // options from the web/arch/<arch>.json manifest (declarative architectures)
    this.neox = oa.rope ? (oa.rope === "neox" ? 1 : 0) : (ROPE_NEOX.has(a) ? 1 : 0);
    // Granite scales (0 = undefined)
    this.escEmb = g("embedding_scale", 0) || 0;
    this.escRes = g("residual_scale", 0) || 0;
    this.escAtt = g("attention.scale", 0) || 1 / Math.sqrt(this.hd);
    this.escLogit = g("logit_scale", 0) || 0;
    if (oa.embedding_scale === "sqrt_d") this.escEmb = Math.fround(Math.sqrt(this.D));
    else if (typeof oa.embedding_scale === "number") this.escEmb = oa.embedding_scale;
    this.ctx = meta.ctx || 4096;
    this.vocab = kv["tokenizer.ggml.tokens"].length;
    this.gemma = a === "gemma3";
    this.ventana = 0;
    this.softcap = g("final_logit_softcapping", 0) || 0;
    if (this.gemma) {
      this.ventana = g("attention.sliding_window", 0) || 0;
      this.patron = g("attention.sliding_window_pattern", 6) || 6;
      this.baseSwa = g("rope.freq_base_swa", 10000);
      const fac = g("rope.scaling.factor", 1);
      this.escalaGlobal = g("rope.scaling.type", "") === "linear" && fac > 0 ? 1 / fac : 1;
      this.escEmb = Math.fround(Math.sqrt(this.D));
      this.escAtt = this.nLayers === 62 ? 1 / Math.sqrt(this.D / this.nh) : 1 / Math.sqrt(this.hd);
    }
    if (this.hd > 256) throw new Error(`head_dim ${this.hd} > 256 not supported`);
    this.bufs = [];
    this.batch = TB;
    this.tb = this.gemma ? TB_IMG : TB;              // activation capacity
    this.supportsImages = this.gemma;            // forward() accepts external embeddings (vision)
    this.baldosa = !(meta.options && meta.options.tiled_prefill === false);   // config.json: "tiled_prefill"
  }

  esSwa(i) { return this.ventana > 0 && (i % this.patron) < this.patron - 1; }

  async load(url, progreso) {
    const gpu = this.gpu, c = new CargadorPesos(gpu, this.meta, url, progreso);
    this.cargador = c;
    if (c.existe("rope_freqs.weight")) throw new Error("this model uses rope_freqs.weight (scaled RoPE), which is not supported yet");
    const D = this.D;
    const ord = ["token_embd.weight", "output.weight", "output_norm.weight"];
    for (let i = 0; i < this.nLayers; i++) {
      const p = `blk.${i}.`;
      ord.push(p + "attn_norm.weight", p + "attn_q.weight", p + "attn_q.bias", p + "attn_k.weight", p + "attn_k.bias",
        p + "attn_v.weight", p + "attn_v.bias", p + "attn_output.weight", p + "attn_output.bias",
        p + "attn_q_norm.weight", p + "attn_k_norm.weight", p + "post_attention_norm.weight", p + "post_ffw_norm.weight",
        p + "ffn_norm.weight", p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight");
    }
    c.planificar(ord);
    this.emb = await c.matriz("token_embd.weight");
    this.vocab = this.emb.N;
    this.salida = c.existe("output.weight") ? await c.matriz("output.weight") : this.emb;
    this.normFinal = await c.f32("output_norm.weight");
    this.capas = [];
    let F = 0;
    const tipos = new Set([this.emb.tipo, this.salida.tipo]);
    const bias = async (n) => (c.existe(n) ? await c.f32(n) : null);
    for (let i = 0; i < this.nLayers; i++) {
      const p = `blk.${i}.`;
      const nkv = this.nkvCapa[i];
      const kvd = nkv * this.hd;
      const capa = { nkv, kvd };
      capa.attnNorm = await c.f32(p + "attn_norm.weight");
      capa.wq = await c.matriz(p + "attn_q.weight"); capa.bq = await bias(p + "attn_q.bias");
      capa.wk = await c.matriz(p + "attn_k.weight"); capa.bk = await bias(p + "attn_k.bias");
      capa.wv = await c.matriz(p + "attn_v.weight"); capa.bv = await bias(p + "attn_v.bias");
      capa.wo = await c.matriz(p + "attn_output.weight"); capa.bo = await bias(p + "attn_output.bias");
      capa.qNorm = await bias(p + "attn_q_norm.weight");
      capa.kNorm = await bias(p + "attn_k_norm.weight");
      capa.postAtt = await bias(p + "post_attention_norm.weight");
      capa.postFfn = await bias(p + "post_ffw_norm.weight");
      capa.ffnNorm = await c.f32(p + "ffn_norm.weight");
      capa.gate = await c.matriz(p + "ffn_gate.weight");
      capa.up = await c.matriz(p + "ffn_up.weight");
      capa.down = await c.matriz(p + "ffn_down.weight");
      for (const m of [capa.wq, capa.wk, capa.wv, capa.wo, capa.gate, capa.up, capa.down]) tipos.add(m.tipo);
      // Local layers (window W): a ring of W + TB positions is enough (TB = tokens per batch, see forward)
      capa.swa = this.esSwa(i) ? this.ventana : 0;
      const anillo = capa.swa && this.ctx > capa.swa + this.tb ? capa.swa + this.tb : 0;
      capa.anillo = anillo;
      capa.kc = gpu.buffer((anillo || this.ctx) * kvd * 4, undefined, `kcache${i}`);
      capa.vc = gpu.buffer((anillo || this.ctx) * kvd * 4, undefined, `vcache${i}`);
      this.bufs.push(capa.kc, capa.vc);
      F = Math.max(F, capa.gate.N);
      this.capas.push(capa);
    }
    this.F = F;
    // K-quant kernels that this model needs
    const extra = [];
    if (this.gemma) extra.push("geglu");
    for (const t of tipos) {
      if (["q3k", "q4k", "q5k", "q6k"].includes(t)) {
        extra.push(`matmul_${t}_1`, `matmul_${t}_4`);
        if (this.baldosa) extra.push(`matmul_${t}_t`);
      }
    }
    if (["q3k", "q4k", "q5k", "q6k"].includes(this.emb.tipo)) extra.push(`embed_${this.emb.tipo}`);
    await gpu.prepararPipelines(extra);
    this.tiposPesos = [...tipos];
    // Activations
    const B = (n, l) => { const b = gpu.buffer(n * 4, undefined, l); this.bufs.push(b); return b; };
    const TB = this.tb;
    const qd = this.nh * this.hd;
    const kvdMax = Math.max(...this.capas.map((x) => x.kvd), 1);
    this.bPaso = gpu.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bIds = B(TB, "ids");
    this.x = B(TB * D, "x");
    this.xn = B(TB * D, "xn");
    this.tmp = B(TB * D, "tmp");
    this.q = B(TB * qd, "q");
    this.k = B(TB * kvdMax, "k");
    this.v = B(TB * kvdMax, "v");
    this.att = B(TB * qd, "att");
    this.g = B(TB * F, "g");
    this.u = B(TB * F, "u");
    if (this.gemma) {
      this.xext = B(TB * D, "xext");        // external embeddings (images) of the batch
      this.bMask = B(TB, "mascara_ext");
      await gpu.prepararPipelines(["mezcla_emb"]);
    }
    this.xl = B(D, "xlast");
    this.logits = B(this.vocab, "logits");
    this.lectura = gpu.device.createBuffer({ size: this.vocab * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this._armarGrupos();
    this.reset();
  }

  // ------------------------------------------------------------------ matmul (1 token / blocks of 4)
  _mm(W, x, y, acc = 0) {
    const gpu = this.gpu;
    const u = gpu.uniforme([["u", W.N], ["u", W.K], ["u", acc], ["u", 0]]);
    if (["q3k", "q4k", "q5k", "q6k"].includes(W.tipo)) {
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
    else if (et === "f32") G.embed = gpu.grupo("embed_f32", [gpu.uniforme([["u", D]]), P, this.emb.w, this.bIds, this.x]);
    else G.embed = gpu.grupo(`embed_${et}`, [gpu.uniforme([["u", D]]), P, this.emb.w, this.bIds, this.x]);
    if (this.escEmb) G.escEmb = gpu.grupo("escalar", [gpu.uniforme([["u", D], ["f", this.escEmb]]), P, this.x]);
    if (this.gemma) G.mezcla = gpu.grupo("mezcla_emb", [gpu.uniforme([["u", D]]), P, this.x, this.xext, this.bMask]);
    const uNorm = gpu.uniforme([["u", D], ["u", 0], ["f", this.eps], ["u", 0]]);
    const dummy = this.normFinal;
    const sesgo = (N, b, y) => (b ? gpu.grupo("bias", [gpu.uniforme([["u", N]]), P, y, b]) : null);
    const resid = this.escRes || 1;
    G.capas = this.capas.map((c) => {
      const gc = {
        attnNorm: gpu.grupo("rmsnorm", [uNorm, P, this.x, c.attnNorm, this.xn]),
        ffnNorm: gpu.grupo("rmsnorm", [uNorm, P, this.x, c.ffnNorm, this.xn]),
        wq: this._mm(c.wq, this.xn, this.q), bq: sesgo(c.wq.N, c.bq, this.q),
        wk: this._mm(c.wk, this.xn, this.k), bk: sesgo(c.wk.N, c.bk, this.k),
        wv: this._mm(c.wv, this.xn, this.v), bv: sesgo(c.wv.N, c.bv, this.v),
        kvd: c.kvd, nkv: c.nkv,
        gate: this._mm(c.gate, this.xn, this.g),
        up: this._mm(c.up, this.xn, this.u),
        swiglu: gpu.grupo(this.gemma ? "geglu" : "swiglu", [gpu.uniforme([["u", c.gate.N]]), P, this.g, this.u]),
        F: c.gate.N,
      };
      const baseCapa = this.gemma && c.swa ? this.baseSwa : this.base;
      const escRope = this.gemma && !c.swa ? this.escalaGlobal : 1;
      const ur = (nh, usa) => gpu.uniforme([["u", this.hd], ["u", nh], ["u", usa ? 1 : 0], ["u", this.neox],
        ["f", this.eps], ["f", baseCapa], ["u", this.nrot], ["f", escRope === 1 ? 0 : escRope]]);
      gc.ropeQ = gpu.grupo("norm_rope", [ur(this.nh, !!c.qNorm), P, this.q, c.qNorm || dummy]);
      gc.ropeK = gpu.grupo("norm_rope", [ur(c.nkv, !!c.kNorm), P, this.k, c.kNorm || dummy]);
      gc.kvw = gpu.grupo("kv_write", [gpu.uniforme([["u", c.kvd], ["u", c.anillo]]), P, this.k, this.v, c.kc, c.vc]);
      gc.att = gpu.grupo("atencion", [gpu.uniforme([["u", this.hd], ["u", this.nh], ["u", c.nkv], ["u", c.swa],
        ["f", this.escAtt], ["u", c.anillo], ["u", 0], ["u", 0]]), P, this.q, c.kc, c.vc, this.att]);
      if (this.gemma) {   // same attention but non-causal within the batch (image tokens): z2 = 1
        gc.attNC = gpu.grupo("atencion", [gpu.uniforme([["u", this.hd], ["u", this.nh], ["u", c.nkv], ["u", c.swa],
          ["f", this.escAtt], ["u", c.anillo], ["u", 0], ["u", 1]]), P, this.q, c.kc, c.vc, this.att]);
      }
      // Outputs added to the residual: directly (acc=1) or via tmp when there is a bias or residual scale.
      const directo = (W, x, b, N, post) => {
        if (!b && !post && resid === 1) return { mm: this._mm(W, x, this.x, 1) };
        if (post) {   // gemma3: x += rmsnorm(salida)   (the output is normalized before being added to the residual)
          return {
            mm: this._mm(W, x, this.tmp, 0), sesgo: sesgo(N, b, this.tmp),
            norma: gpu.grupo("rmsnorm", [uNorm, P, this.tmp, post, this.xn]),
            suma: gpu.grupo("suma_esc", [gpu.uniforme([["u", D], ["f", resid]]), P, this.x, this.xn]),
          };
        }
        return {
          mm: this._mm(W, x, this.tmp, 0), sesgo: sesgo(N, b, this.tmp),
          suma: gpu.grupo("suma_esc", [gpu.uniforme([["u", D], ["f", resid]]), P, this.x, this.tmp]),
        };
      };
      gc.wo = directo(c.wo, this.att, c.bo, c.wo.N, c.postAtt);
      gc.down = directo(c.down, this.g, null, c.down.N, c.postFfn);
      return gc;
    });
    G.normFinal = gpu.grupo("rmsnorm", [gpu.uniforme([["u", D], ["u", 1], ["f", this.eps], ["u", 0]]), P, this.x, this.normFinal, this.xl]);
    G.salida = this._mm(this.salida, this.xl, this.logits, 0);
    this.G = G;
  }

  reset() {
    this.pos = 0;   // the KV cache is overwritten by position; there are no recurrent states
  }

  // Processes up to TB tokens from the current position (up to 256 if they are image tokens). If conLogits, returns a
  // Float32Array with the logits of the last one.
  // ext (optional, gemma3 only): per token, null or {buf, fila, bidir}: instead of the token embedding, row 'fila' is used
  // (D floats) of GPU buffer 'buf' (image embedding). If ALL tokens in the batch are image tokens with bidir, the
  // attention within the batch is non-causal (like llama.cpp with Gemma 3 images).
  async forward(ids, conLogits, ext = null) {
    const T = ids.length, TBM = this.tb;
    const hayExt = !!(ext && this.xext && ext.some((e) => e));
    const noCausal = hayExt && ext.length === T && ext.every((e) => e && e.bidir);
    if (T < 1 || T > (hayExt ? TBM : TB)) throw new Error("invalid batch");
    if (this.pos + T > this.ctx) throw new Error(`context exceeded (${this.ctx} tokens)`);
    const dev = this.gpu.device, G = this.G, D = this.D;
    dev.queue.writeBuffer(this.bPaso, 0, new Uint32Array([T, this.pos, 0, 0]));
    dev.queue.writeBuffer(this.bIds, 0, new Uint32Array(ids));
    const validar = !this._validado;
    if (validar) dev.pushErrorScope("validation");
    let enc = dev.createCommandEncoder();
    if (hayExt) {
      const mascara = new Uint32Array(TBM), D4 = D * 4;
      for (let t = 0; t < T; t++) {
        const e = ext[t];
        if (!e) continue;
        mascara[t] = 1;
        enc.copyBufferToBuffer(e.buf, e.fila * D4, this.xext, t * D4, D4);
      }
      dev.queue.writeBuffer(this.bMask, 0, mascara);
    }
    let pass = enc.beginComputePass();
    const mm = (g, uno = T === 1) => {
      if (!uno && this.baldosa && g.baldosa && T >= 8) {
        const [x, y] = dim2(Math.ceil(g.filas / 64));
        despachar(pass, g.baldosa, x, y, Math.ceil(T / 32));
        return;
      }
      const [x, y] = dim2(Math.ceil(g.filas / 8));
      despachar(pass, uno ? g.uno : g.lote, x, y, uno || !g.z4 ? 1 : Math.ceil(T / 4));
    };
    const filas = (n) => Math.ceil(T * n / 256);
    // With more than 32 tokens a layer can take hundreds of ms: one submit per layer (Windows resets the GPU after ~2 s).
    const porCapa = T > TB;
    const cortar = () => { pass.end(); dev.queue.submit([enc.finish()]); enc = dev.createCommandEncoder(); pass = enc.beginComputePass(); };
    despachar(pass, G.embed, filas(D));
    if (G.escEmb) despachar(pass, G.escEmb, filas(D));
    if (hayExt) despachar(pass, G.mezcla, filas(D));
    for (const gc of G.capas) {
      despachar(pass, gc.attnNorm, T);
      mm(gc.wq); if (gc.bq) despachar(pass, gc.bq, filas(gc.wq.filas));
      mm(gc.wk); if (gc.bk) despachar(pass, gc.bk, filas(gc.wk.filas));
      mm(gc.wv); if (gc.bv) despachar(pass, gc.bv, filas(gc.wv.filas));
      despachar(pass, gc.ropeQ, this.nh, T);
      despachar(pass, gc.ropeK, gc.nkv, T);
      despachar(pass, gc.kvw, Math.ceil(T * gc.kvd / 256));
      despachar(pass, noCausal ? gc.attNC : gc.att, this.nh, T);
      mm(gc.wo.mm);
      if (gc.wo.suma) {
        if (gc.wo.sesgo) despachar(pass, gc.wo.sesgo, filas(D));
        if (gc.wo.norma) despachar(pass, gc.wo.norma, T);
        despachar(pass, gc.wo.suma, filas(D));
      }
      despachar(pass, gc.ffnNorm, T);
      mm(gc.gate); mm(gc.up);
      despachar(pass, gc.swiglu, filas(gc.F));
      mm(gc.down.mm);
      if (gc.down.suma) {
        if (gc.down.norma) despachar(pass, gc.down.norma, T);
        despachar(pass, gc.down.suma, filas(D));
      }
      if (porCapa) cortar();
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
    if (this.escLogit) { const k = 1 / this.escLogit; for (let i = 0; i < out.length; i++) out[i] *= k; }
    if (this.softcap) { const c = this.softcap; for (let i = 0; i < out.length; i++) out[i] = c * Math.tanh(out[i] / c); }
    return out;
  }

  free() {
    const todos = [...this.bufs, ...(this.cargador ? this.cargador.buffers : [])];
    this.gpu.liberar(todos);
    if (this.lectura) this.lectura.destroy();
    this.bufs = [];
  }
}
