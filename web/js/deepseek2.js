// DeepSeek-V2 / Coder-V2-Lite ("deepseek2" architecture): MLA attention + MoE with shared experts.
// Reference: llama.cpp src/models/deepseek2.cpp ("lite" path: wq without compression, merged wkv_b).
//
//   layer: x += wo( MLA(rmsnorm(x)) );  x += FFN(rmsnorm(x))   (dense FFN in the first layers, MoE afterwards)
//   MLA : q = wq h -> (q_nope | q_pe); kv = wkv_a h -> (c | k_pe); c = rmsnorm(c); RoPE (YaRN, consecutive pairs)
//         on q_pe and k_pe. llama.cpp expands K,V per head with wkv_b; here the "absorbed" form is used (equivalent):
//         the cache stores only (c | k_pe) = 576 values per token and layer, and the query is brought into c's space:
//             score = (W_k^T q_nope) . c + q_pe . k_pe ;   output = W_v (sum p c)
//         (the expanded cache with 16 heads would need 9 times more memory).
//   MoE : router softmax, k experts out of E, weights = probabilities (opt. normalized) x scale, + shared expert.
import { despachar, dim2 } from "./gpu.js";
import { CargadorPesos } from "./weights.js";

const TB = 128;   // tokens per prompt pass (more tokens per batch = more reuse of each expert's weights)
const TK = ["q2k", "q3k", "q4k", "q5k", "q6k"];

export class ModeloDeepseek2 {
  constructor(gpu, meta) {
    this.gpu = gpu;
    this.meta = meta;
    const kv = meta.kv, a = "deepseek2";
    this.arch = a;
    const g = (k, d) => (kv[`${a}.${k}`] !== undefined ? kv[`${a}.${k}`] : d);
    this.nLayers = g("block_count");
    this.D = g("embedding_length");
    this.nh = g("attention.head_count");
    this.R = g("attention.kv_lora_rank");
    this.hdk = g("attention.key_length");
    this.vd = g("attention.value_length");
    this.nr = g("rope.dimension_count");
    this.nope = this.hdk - this.nr;
    this.eps = g("attention.layer_norm_rms_epsilon", 1e-6);
    this.base = g("rope.freq_base", 10000);
    this.nExp = g("expert_count", 0);
    this.nUsados = g("expert_used_count", 0);
    this.nComp = g("expert_shared_count", 0);
    this.ffe = g("expert_feed_forward_length", 0);
    this.densas = g("leading_dense_block_count", 0);
    this.escExp = g("expert_weights_scale", 1) || 1;
    this.normExp = !!g("expert_weights_norm", false);
    const gating = g("expert_gating_func", 0);
    if (gating && gating !== 1) throw new Error(`deepseek2: expert gating function ${gating} not supported (softmax only)`);
    if (g("attention.q_lora_rank", 0)) throw new Error("deepseek2: this model compresses the query (q_lora_rank); only the 'Lite' variants are supported");
    if (g("attention.key_length_mla", 0)) throw new Error("deepseek2: GGUF with separate wk_b/wv_b is not supported yet");
    if (this.nExp > 256 || this.nUsados > this.nExp) throw new Error("deepseek2: number of experts not supported");
    this.ctx = meta.ctx || 4096;
    this.vocab = kv["tokenizer.ggml.tokens"].length;
    this._yarn(kv, g);
    this.bufs = [];
    this.batch = TB;
    this.baldosa = !(meta.options && meta.options.tiled_prefill === false);
  }

  // YaRN RoPE parameters and attention scale, computed as llama.cpp does (llama-context.cpp + deepseek2.cpp)
  _yarn(kv, g) {
    const f = Math.fround, ln = (x) => f(Math.log(x));
    const tipo = g("rope.scaling.type", "linear");
    const factorTr = g("rope.scaling.factor", 0);
    let fscale = factorTr ? f(1 / factorTr) : 1;
    if (tipo === "none") fscale = 1;
    const ext = tipo === "yarn" ? 1 : 0;
    const logMul = kv[`deepseek2.rope.scaling.yarn_log_multiplier`] !== undefined ? f(kv["deepseek2.rope.scaling.yarn_log_multiplier"] / f(0.1)) : 0;
    const attnRope = g("rope.scaling.attn_factor", 1);
    const gm = (s, m) => (s <= 1 ? 1 : f(f(f(0.1) * m * ln(s)) + 1));
    let attn = 1;
    if (ext) {
      const factor = f(1 / fscale);
      if (logMul !== 0) {
        let ms = 1;
        if (logMul !== 1) ms = logMul;
        attn = f(gm(factor, ms) / gm(factor, logMul));
      } else attn = gm(factor, 1);
      attn = f(attn * f(1 / f(1 + f(0.1) * ln(factor))));
    }
    attn = f(attn * attnRope);
    // scale of the RoPE sines/cosines (ggml: mscale = attn_factor * (1 + 0.1 ln(1/scale)) if YaRN)
    this.ropeMs = ext ? f(attn * f(1 + f(0.1) * ln(f(1 / fscale)))) : attn;
    this.fscale = fscale;
    this.ext = ext;
    // attention scale (deepseek2.cpp)
    const attnOrg = f(attn * f(1 + f(0.1) * ln(f(1 / fscale))));
    const mscale = f(attnOrg * f(1 + f(f(0.1) * logMul) * ln(f(1 / fscale))));
    this.escAtt = f(f(mscale * mscale) / f(Math.sqrt(this.hdk)));
    // range of correction dimensions (ggml_rope_yarn_corr_dims)
    const ctxOrig = g("rope.scaling.original_context_length", g("context_length", 4096));
    const bf = g("rope.scaling.yarn_beta_fast", 32), bs = g("rope.scaling.yarn_beta_slow", 1);
    const corr = (beta) => f(f(this.nr * f(Math.log(ctxOrig / (beta * 2 * Math.PI)))) / f(2 * ln(this.base)));
    this.low = Math.max(0, Math.floor(corr(bf)));
    this.high = Math.min(this.nr - 1, Math.ceil(corr(bs)));
  }

  esDensa(i) { return i < this.densas; }

  async load(url, progreso) {
    const gpu = this.gpu, c = new CargadorPesos(gpu, this.meta, url, progreso);
    this.cargador = c;
    const D = this.D;
    for (const n of ["attn_q_a", "attn_k_b", "attn_v_b"]) if (c.existe(`blk.0.${n}.weight`)) throw new Error(`deepseek2: tensor ${n} not supported (Lite variants only)`);
    if (c.existe("blk.1.exp_probs_b.bias")) throw new Error("deepseek2: expert selection bias (exp_probs_b) not supported");
    const ord = ["token_embd.weight", "output.weight", "output_norm.weight"];
    for (let i = 0; i < this.nLayers; i++) {
      const p = `blk.${i}.`;
      ord.push(p + "attn_norm.weight", p + "attn_q.weight", p + "attn_kv_a_mqa.weight", p + "attn_kv_a_norm.weight",
        p + "attn_kv_b.weight", p + "attn_output.weight", p + "ffn_norm.weight");
      if (this.esDensa(i)) ord.push(p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight");
      else ord.push(p + "ffn_gate_inp.weight", p + "ffn_gate_exps.weight", p + "ffn_up_exps.weight", p + "ffn_down_exps.weight",
        p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight", p + "ffn_down_shexp.weight");
    }
    c.planificar(ord);
    this.emb = await c.matriz("token_embd.weight");
    this.vocab = this.emb.N;
    this.salida = c.existe("output.weight") ? await c.matriz("output.weight") : this.emb;
    this.normFinal = await c.f32("output_norm.weight");
    const tipos = new Set([this.emb.tipo, this.salida.tipo]);
    const tiposE = new Set();
    let Fd = 1;
    this.capas = [];
    for (let i = 0; i < this.nLayers; i++) {
      const p = `blk.${i}.`;
      const capa = {};
      capa.attnNorm = await c.f32(p + "attn_norm.weight");
      capa.wq = await c.matriz(p + "attn_q.weight");
      capa.wkva = await c.matriz(p + "attn_kv_a_mqa.weight");
      capa.kvNorm = await c.f32(p + "attn_kv_a_norm.weight");
      capa.wkvb = await this._aF32(await c.matriz(p + "attn_kv_b.weight"));
      capa.wo = await c.matriz(p + "attn_output.weight");
      capa.ffnNorm = await c.f32(p + "ffn_norm.weight");
      for (const m of [capa.wq, capa.wkva, capa.wo]) tipos.add(m.tipo);
      if (this.esDensa(i)) {
        capa.gate = await c.matriz(p + "ffn_gate.weight");
        capa.up = await c.matriz(p + "ffn_up.weight");
        capa.down = await c.matriz(p + "ffn_down.weight");
        for (const m of [capa.gate, capa.up, capa.down]) tipos.add(m.tipo);
        Fd = Math.max(Fd, capa.gate.N);
      } else {
        capa.router = await c.matriz(p + "ffn_gate_inp.weight");
        capa.eg = await c.matriz(p + "ffn_gate_exps.weight");
        capa.eu = await c.matriz(p + "ffn_up_exps.weight");
        capa.ed = await c.matriz(p + "ffn_down_exps.weight");
        capa.sg = await c.matriz(p + "ffn_gate_shexp.weight");
        capa.su = await c.matriz(p + "ffn_up_shexp.weight");
        capa.sd = await c.matriz(p + "ffn_down_shexp.weight");
        tipos.add(capa.router.tipo);
        for (const m of [capa.sg, capa.su, capa.sd]) tipos.add(m.tipo);
        for (const m of [capa.eg, capa.eu, capa.ed]) { tiposE.add(m.tipo); }
        if (capa.eg.N !== this.nExp * this.ffe || capa.ed.N !== this.nExp * D) throw new Error(`layer ${i}: unexpected expert dimensions`);
      }
      capa.cc = gpu.buffer(this.ctx * (this.R + this.nr) * 4, undefined, `kvcomp${i}`);
      this.bufs.push(capa.cc);
      this.capas.push(capa);
    }
    this.F = Fd;
    const extra = ["mla_kv", "mla_q", "mla_att", "mla_v"];
    if (this.nExp) extra.push("moe_topk", "moe_comb", "moe_group");
    const pidePipeline = (t, exp) => {
      if (TK.includes(t)) {
        extra.push(`matmul_${t}_1`, `matmul_${t}_4`);
        if (this.baldosa) extra.push(`matmul_${t}_t`);
        if (exp) extra.push(`matmul_${t}_e`, `matmul_${t}_eg`);
      } else if (t === "iq4nl") extra.push("matmul_iq4nl_1", "matmul_iq4nl_4", ...(exp ? ["matmul_iq4nl_e", "matmul_iq4nl_eg"] : []));
      else if (t === "f32" && exp) extra.push("matmul_f32_e");
      else if (t === "q8" && exp) extra.push("matmul_q8_e");
    };
    for (const t of tipos) pidePipeline(t, false);
    for (const t of tiposE) pidePipeline(t, true);
    for (const t of tiposE) if (!TK.includes(t) && t !== "iq4nl" && t !== "f32" && t !== "q8") throw new Error(`deepseek2: expert quantization ${t} not supported`);
    if (this.emb.tipo === "iq4nl") throw new Error("deepseek2: token_embd in IQ4_NL not supported");
    if (TK.includes(this.emb.tipo)) extra.push(`embed_${this.emb.tipo}`);
    await gpu.prepararPipelines([...new Set(extra)]);
    this.tiposPesos = [...tipos, ...tiposE];
    const B = (n, l, u32 = false) => { const b = gpu.buffer(n * 4, undefined, l); this.bufs.push(b); return b; };
    const { nh, R, nr, vd, nUsados: K, ffe, nComp } = this;
    this.bPaso = gpu.device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bIds = B(TB, "ids");
    this.x = B(TB * D, "x");
    this.xn = B(TB * D, "xn");
    this.q = B(TB * nh * this.hdk, "q");
    this.kvc = B(TB * (R + nr), "kvc");
    this.qa = B(TB * nh * (R + nr), "qa");
    this.ol = B(TB * nh * R, "ol");
    this.att = B(TB * nh * vd, "att");
    this.g = B(TB * Fd, "g");
    this.u = B(TB * Fd, "u");
    if (this.nExp) {
      this.rl = B(TB * this.nExp, "router");
      this.sel = B(TB * K, "sel");
      this.wts = B(TB * K, "wts");
      this.lst = B(TB * K, "lst");
      this.off = B(this.nExp + 1, "off");
      this.ge = B(TB * K * ffe, "ge");
      this.ue = B(TB * K * ffe, "ue");
      this.de = B(TB * K * D, "de");
      this.sgb = B(TB * ffe * nComp, "sg");
      this.sub = B(TB * ffe * nComp, "su");
    }
    this.xl = B(D, "xlast");
    this.logits = B(this.vocab, "logits");
    this.lectura = gpu.device.createBuffer({ size: this.vocab * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this._armarGrupos();
    this.reset();
  }

  // Brings any matrix to f32 on the GPU (with the same "embed" kernel that decodes rows) and frees the original.
  async _aF32(W) {
    if (W.tipo === "f32") return W.w;
    const gpu = this.gpu, dev = gpu.device, N = W.N, K = W.K;
    const kern = W.tipo === "q8" ? "embed_q8" : `embed_${W.tipo}`;
    await gpu.prepararPipelines([kern]);
    const out = gpu.buffer(N * K * 4, undefined, "wkvb_f32");
    const ids = gpu.subir(Uint32Array.from({ length: N }, (_, i) => i), "ids_kvb");
    const paso = dev.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(paso, 0, new Uint32Array([N, 0, 0, 0]));
    const u = gpu.uniforme([["u", K]]);
    const g = W.tipo === "q8" ? gpu.grupo(kern, [u, paso, W.qs, W.ds, ids, out]) : gpu.grupo(kern, [u, paso, W.w, ids, out]);
    const nwg = Math.ceil(N * K / 256);
    if (nwg > 65535) throw new Error("wkv_b is too large to convert to f32");
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    despachar(pass, g, nwg);
    pass.end();
    dev.queue.submit([enc.finish()]);
    await dev.queue.onSubmittedWorkDone();
    const viejos = W.tipo === "q8" ? [W.qs, W.ds] : [W.w];
    this.cargador.buffers = this.cargador.buffers.filter((b) => !viejos.includes(b));
    gpu.liberar([...viejos, ids, paso]);
    this.bufs.push(out);
    return out;
  }

  // dense matmul: T=1 ("uno"), blocks of 4 tokens ("lote") and tiled for prefill
  _mm(W, x, y, acc = 0) {
    const gpu = this.gpu, P = this.bPaso;
    const u = gpu.uniforme([["u", W.N], ["u", W.K], ["u", acc], ["u", 0]]);
    if (TK.includes(W.tipo)) {
      return {
        uno: gpu.grupo(`matmul_${W.tipo}_1`, [u, P, W.w, x, y]),
        lote: gpu.grupo(`matmul_${W.tipo}_4`, [u, P, W.w, x, y]), z4: true, filas: W.N,
        baldosa: this.baldosa ? gpu.grupo(`matmul_${W.tipo}_t`, [u, P, W.w, x, y]) : null,
      };
    }
    if (W.tipo === "iq4nl") {
      return {
        uno: gpu.grupo("matmul_iq4nl_1", [u, P, W.qs, W.dh, x, y]),
        lote: gpu.grupo("matmul_iq4nl_4", [u, P, W.qs, W.dh, x, y]), z4: true, filas: W.N,
      };
    }
    if (W.tipo === "q8") {
      return {
        uno: gpu.grupo("matmul_q8_1", [u, P, W.qs, W.ds, x, y]),
        lote: gpu.grupo("matmul_q8_4", [u, P, W.qs, W.ds, x, y]), z4: true, filas: W.N,
      };
    }
    return {
      uno: gpu.grupo("matmul_f32_1", [u, P, W.w, x, y]),
      lote: gpu.grupo("matmul_f32_4", [u, P, W.w, x, y]), z4: true, filas: W.N,
    };
  }

  // expert matmul: W stacks nExp matrices; one job per (token, slot)
  _mmE(W, x, y, tfijo) {
    const gpu = this.gpu, P = this.bPaso, filas = W.N / this.nExp;
    const u = gpu.uniforme([["u", filas], ["u", W.K], ["u", 0], ["u", tfijo]]);
    let g, gg = null;
    if (TK.includes(W.tipo)) {
      g = gpu.grupo(`matmul_${W.tipo}_e`, [u, P, W.w, x, y, this.sel]);
      gg = gpu.grupo(`matmul_${W.tipo}_eg`, [u, P, W.w, x, y, this.lst, this.off]);
    } else if (W.tipo === "iq4nl") {
      g = gpu.grupo("matmul_iq4nl_e", [u, P, W.qs, W.dh, x, y, this.sel]);
      gg = gpu.grupo("matmul_iq4nl_eg", [u, P, W.qs, W.dh, x, y, this.lst, this.off]);
    } else if (W.tipo === "q8") g = gpu.grupo("matmul_q8_e", [u, P, W.qs, W.ds, x, y, this.sel]);
    else g = gpu.grupo("matmul_f32_e", [u, P, W.w, x, y, this.sel]);
    return { g, gg, filas };
  }

  _armarGrupos() {
    const gpu = this.gpu, D = this.D, P = this.bPaso;
    const { nh, R, nr, nope, vd, nUsados: K, ffe } = this;
    const G = {};
    const et = this.emb.tipo;
    if (et === "q8") G.embed = gpu.grupo("embed_q8", [gpu.uniforme([["u", D]]), P, this.emb.qs, this.emb.ds, this.bIds, this.x]);
    else G.embed = gpu.grupo(`embed_${et}`, [gpu.uniforme([["u", D]]), P, this.emb.w, this.bIds, this.x]);
    const uNorm = gpu.uniforme([["u", D], ["u", 0], ["f", this.eps], ["u", 0]]);
    const rope = [["f", this.base], ["f", this.fscale], ["f", this.ext], ["f", this.ropeMs], ["f", this.low], ["f", this.high]];
    const uKv = gpu.uniforme([["u", R], ["u", nr], ["f", this.eps], ...rope, ["u", 0], ["u", 0], ["u", 0]]);
    const uQ = gpu.uniforme([["u", nh], ["u", nope], ["u", nr], ["u", R], ["u", vd], ...rope, ["u", 0]]);
    const uAtt = gpu.uniforme([["u", R + nr], ["u", R], ["u", nh], ["u", 0], ["f", this.escAtt], ["u", 0], ["u", 0], ["u", 0]]);
    const uV = gpu.uniforme([["u", nh], ["u", nope], ["u", R], ["u", vd]]);
    G.capas = this.capas.map((c) => {
      const gc = {
        attnNorm: gpu.grupo("rmsnorm", [uNorm, P, this.x, c.attnNorm, this.xn]),
        ffnNorm: gpu.grupo("rmsnorm", [uNorm, P, this.x, c.ffnNorm, this.xn]),
        wq: this._mm(c.wq, this.xn, this.q),
        wkva: this._mm(c.wkva, this.xn, this.kvc),
        kv: gpu.grupo("mla_kv", [uKv, P, this.kvc, c.kvNorm, c.cc]),
        qa: gpu.grupo("mla_q", [uQ, P, this.q, c.wkvb, this.qa]),
        att: gpu.grupo("mla_att", [uAtt, P, this.qa, c.cc, this.ol]),
        v: gpu.grupo("mla_v", [uV, P, this.ol, c.wkvb, this.att]),
        wo: this._mm(c.wo, this.att, this.x, 1),
      };
      if (c.gate) {
        gc.gate = this._mm(c.gate, this.xn, this.g);
        gc.up = this._mm(c.up, this.xn, this.u);
        gc.act = gpu.grupo("swiglu", [gpu.uniforme([["u", c.gate.N]]), P, this.g, this.u]);
        gc.F = c.gate.N;
        gc.down = this._mm(c.down, this.g, this.x, 1);
      } else {
        gc.router = this._mm(c.router, this.xn, this.rl);
        gc.topk = gpu.grupo("moe_topk", [gpu.uniforme([["u", this.nExp], ["u", K], ["u", this.normExp ? 1 : 0], ["f", this.escExp]]), P, this.rl, this.sel, this.wts]);
        gc.eg = this._mmE(c.eg, this.xn, this.ge, K);
        gc.eu = this._mmE(c.eu, this.xn, this.ue, K);
        gc.eact = gpu.grupo("swiglu", [gpu.uniforme([["u", K * ffe]]), P, this.ge, this.ue]);
        gc.ed = this._mmE(c.ed, this.ge, this.de, 1);
        gc.grupar = gpu.grupo("moe_group", [gpu.uniforme([["u", this.nExp], ["u", K]]), P, this.sel, this.lst, this.off]);
        gc.agrupado = !!(gc.eg.gg && gc.eu.gg && gc.ed.gg);
        gc.comb = gpu.grupo("moe_comb", [gpu.uniforme([["u", D], ["u", K]]), P, this.x, this.de, this.wts]);
        gc.sg = this._mm(c.sg, this.xn, this.sgb);
        gc.su = this._mm(c.su, this.xn, this.sub);
        gc.sact = gpu.grupo("swiglu", [gpu.uniforme([["u", c.sg.N]]), P, this.sgb, this.sub]);
        gc.sd = this._mm(c.sd, this.sgb, this.x, 1);
      }
      return gc;
    });
    G.normFinal = gpu.grupo("rmsnorm", [gpu.uniforme([["u", D], ["u", 1], ["f", this.eps], ["u", 0]]), P, this.x, this.normFinal, this.xl]);
    G.salida = this._mm(this.salida, this.xl, this.logits, 0);
    this.G = G;
  }

  reset() { this.pos = 0; }

  async forward(ids, conLogits) {
    const T = ids.length;
    if (T < 1 || T > TB) throw new Error("invalid batch");
    if (this.pos + T > this.ctx) throw new Error(`context exceeded (${this.ctx} tokens)`);
    const dev = this.gpu.device, G = this.G, D = this.D;
    dev.queue.writeBuffer(this.bPaso, 0, new Uint32Array([T, this.pos, 0, 0]));
    dev.queue.writeBuffer(this.bIds, 0, new Uint32Array(ids));
    const validar = !this._validado;
    if (validar) dev.pushErrorScope("validation");
    // With several tokens the work is long: it is submitted layer by layer so that no batch exceeds the time limit
    // of the video driver (Windows resets the GPU if a batch goes over ~2 s).
    const porCapa = T > 1;
    let enc = dev.createCommandEncoder();
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
    const agrupar = T >= 8;
    const mmE = (e, gr) => {
      if (gr) { despachar(pass, e.gg, Math.ceil(e.filas / 64), this.nExp, Math.ceil(T / 32)); return; }
      const [x, y] = dim2(Math.ceil(e.filas / 8));
      despachar(pass, e.g, x, y, T * this.nUsados);
    };
    const filas = (n) => Math.ceil(T * n / 256);
    despachar(pass, G.embed, filas(D));
    for (const gc of G.capas) {
      despachar(pass, gc.attnNorm, T);
      mm(gc.wq); mm(gc.wkva);
      despachar(pass, gc.kv, T);
      despachar(pass, gc.qa, this.nh, T);
      despachar(pass, gc.att, this.nh, T);
      despachar(pass, gc.v, this.nh, T);
      mm(gc.wo);
      despachar(pass, gc.ffnNorm, T);
      if (gc.gate) {
        mm(gc.gate); mm(gc.up);
        despachar(pass, gc.act, filas(gc.F));
        mm(gc.down);
      } else {
        mm(gc.router);
        despachar(pass, gc.topk, Math.ceil(T / 64));
        const gr = agrupar && gc.agrupado && this.agrupado !== false;
        if (gr) despachar(pass, gc.grupar, 1);
        mmE(gc.eg, gr); mmE(gc.eu, gr);
        despachar(pass, gc.eact, filas(this.nUsados * this.ffe));
        mmE(gc.ed, gr);
        despachar(pass, gc.comb, filas(D));
        mm(gc.sg); mm(gc.su);
        despachar(pass, gc.sact, filas(this.ffe * this.nComp));
        mm(gc.sd);
      }
      if (porCapa) { pass.end(); dev.queue.submit([enc.finish()]); enc = dev.createCommandEncoder(); pass = enc.beginComputePass(); }
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
