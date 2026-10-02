// WGSL kernels for deepseek2 (DeepSeek-V2/Coder-V2-Lite): compressed MLA attention, YaRN, experts (MoE)
// and IQ4_NL quantization. Same conventions as gpu.js (rows [T][D]; weights [N][K]).

const PASO = `struct Paso { T: u32, pos0: u32, a: u32, b: u32 };`;

// custom sine/cosine (see gpu.js): the native ones on some GPUs only give ~2e-4
const SINCOS = `fn sincos2(x: f32) -> vec2<f32> {
  let k = i32(round(x * 0.6366197723675814));
  let kf = f32(k);
  let r = ((x - kf * 1.5703125) - kf * 4.837512969970703125e-4) - kf * 7.549789948768648e-8;
  let r2 = r * r;
  let sn = r + r * r2 * (-1.6666654611e-1 + r2 * (8.3321608736e-3 + r2 * -1.9515295891e-4));
  let cs = 1.0 - 0.5 * r2 + r2 * r2 * (4.166664568298827e-2 + r2 * (-1.388731625493765e-3 + r2 * 2.443315711809948e-5));
  let q = u32(k) & 3u;
  if (q == 0u) { return vec2<f32>(sn, cs); }
  if (q == 1u) { return vec2<f32>(cs, -sn); }
  if (q == 2u) { return vec2<f32>(-sn, -cs); }
  return vec2<f32>(-cs, sn);
}`;

// ggml RoPE YaRN (rope_yarn): returns (cosine, sine) of pair j. `ms` already includes the factor 1 + 0.1 ln(1/scale).
const YARN = `${SINCOS}
fn yarn_cs(pos: f32, j: u32, nd: f32, base: f32, fscale: f32, ext: f32, ms: f32, low: f32, high: f32) -> vec2<f32> {
  let ext_t = pos * pow(base, -2.0 * f32(j) / nd);
  let interp = fscale * ext_t;
  var th = interp;
  if (ext != 0.0) {
    let y = (f32(j) - low) / max(0.001, high - low);
    let ramp = (1.0 - min(1.0, max(0.0, y))) * ext;
    th = interp * (1.0 - ramp) + ext_t * ramp;
  }
  let sc = sincos2(th);
  return vec2<f32>(sc.y * ms, sc.x * ms);
}`;

// ------------------------------------------------------------------------------------ IQ4_NL
// Block of 32 weights, 18 bytes: d f16 + 16 bytes (low nibble = weights 0..15, high = 16..31), value = d * kvalues[nibble].
// On the GPU: qs (4 u32 per block) and dh (one f16 per block, two per u32).
const KVALUES = "-127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0";

function matmulIQ4(nt, exp) {
  const tok = [...Array(nt).keys()];
  const T = nt === 1 ? "_ = paso.T;" : "let T = paso.T; let t0 = wg.z * " + nt + "u;";
  const job = exp ? "let job = wg.z; let ex = sel[job];" : "";
  const offs = exp ? "let o0 = (job / p.tfijo) * kv;"
    : nt === 1 ? "let o0 = 0u;" : tok.map((j) => `let o${j} = min(t0 + ${j}u, T - 1u) * kv;`).join("\n  ");
  return `${PASO}
struct P { N: u32, K: u32, acc: u32, tfijo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> qs: array<u32>;
@group(0) @binding(3) var<storage, read> dh: array<u32>;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
${exp ? "@group(0) @binding(6) var<storage, read> sel: array<u32>;" : ""}
var<private> kvt: array<f32, 16> = array<f32, 16>(${KVALUES});
var<workgroup> red: array<f32, ${nt * 256}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  ${T}
  ${job}
  let r = (wg.x + wg.y * nwg.x) * 8u + li / 32u;
  let lane = li % 32u;
  let kv = p.K / 4u;
  ${offs}
  ${tok.map((j) => `var a${j} = 0.0;`).join(" ")}
  let nb = p.K / 32u;
  if (r < p.N) {
    let rr = ${exp ? "ex * p.N + r" : "r"};
    for (var b = lane; b < nb; b += 32u) {
      let bi = rr * nb + b;
      let dd = unpack2x16float(dh[bi >> 1u]);
      let d = select(dd.x, dd.y, (bi & 1u) == 1u);
      let qb = bi * 4u;
      ${tok.map((k) => `var s${k} = 0.0;`).join(" ")}
      for (var wi = 0u; wi < 4u; wi++) {
        let word = qs[qb + wi];
        let lo = vec4<f32>(kvt[word & 15u], kvt[(word >> 8u) & 15u], kvt[(word >> 16u) & 15u], kvt[(word >> 24u) & 15u]);
        let hi = vec4<f32>(kvt[(word >> 4u) & 15u], kvt[(word >> 12u) & 15u], kvt[(word >> 20u) & 15u], kvt[(word >> 28u) & 15u]);
        ${tok.map((k) => `s${k} += dot(lo, x[o${k} + b * 8u + wi]) + dot(hi, x[o${k} + b * 8u + 4u + wi]);`).join("\n        ")}
      }
      ${tok.map((k) => `a${k} += d * s${k};`).join(" ")}
    }
  }
  ${tok.map((k) => `red[${k * 256}u + li] = a${k};`).join(" ")}
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) {
      ${tok.map((k) => `red[${k * 256}u + li] += red[${k * 256}u + li + s];`).join(" ")}
    }
    workgroupBarrier();
  }
  if (lane == 0u && r < p.N) {
    ${exp ? "y[job * p.N + r] = red[li];"
    : nt === 1 ? "let v = red[li]; if (p.acc == 1u) { y[r] = y[r] + v; } else { y[r] = v; }"
      : `for (var j = 0u; j < ${nt}u; j++) {
      let t = t0 + j;
      if (t < T) {
        let o = t * p.N + r;
        let v = red[j * 256u + li];
        if (p.acc == 1u) { y[o] = y[o] + v; } else { y[o] = v; }
      }
    }`}
  }
}`;
}

// Expert matmul with Q8_0 weights (qs = int8 packed 4 per u32, ds = f32 scale per block of 32)
const MATMUL_Q8_E = `${PASO}
struct P { N: u32, K: u32, acc: u32, tfijo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> qs: array<u32>;
@group(0) @binding(3) var<storage, read> ds: array<f32>;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<storage, read> sel: array<u32>;
var<workgroup> red: array<f32, 256>;
fn i8x4(w: u32) -> vec4<f32> {
  return vec4<f32>(f32(bitcast<i32>(w << 24u) >> 24u), f32(bitcast<i32>(w << 16u) >> 24u),
                   f32(bitcast<i32>(w << 8u) >> 24u), f32(bitcast<i32>(w) >> 24u));
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  _ = paso.T;
  let job = wg.z;
  let ex = sel[job];
  let r = (wg.x + wg.y * nwg.x) * 8u + li / 32u;
  let lane = li % 32u;
  let kv = p.K / 4u;
  let xo = (job / p.tfijo) * kv;
  let nb = p.K / 32u;
  var a = 0.0;
  if (r < p.N) {
    let rr = ex * p.N + r;
    for (var b = lane; b < nb; b += 32u) {
      let bi = rr * nb + b;
      var s = 0.0;
      for (var wi = 0u; wi < 8u; wi++) { s += dot(i8x4(qs[bi * 8u + wi]), x[xo + b * 8u + wi]); }
      a += ds[bi] * s;
    }
  }
  red[li] = a;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && r < p.N) { y[job * p.N + r] = red[li]; }
}`;

// Expert matmul with f32 weights (unquantized models, used in the tests)
const MATMUL_F32_E = `${PASO}
struct P { N: u32, K: u32, acc: u32, tfijo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> sel: array<u32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  _ = paso.T;
  let job = wg.z;
  let ex = sel[job];
  let r = (wg.x + wg.y * nwg.x) * 8u + li / 32u;
  let lane = li % 32u;
  let kv = p.K / 4u;
  let xo = (job / p.tfijo) * kv;
  var a = 0.0;
  if (r < p.N) {
    let base = (ex * p.N + r) * kv;
    for (var i = lane; i < kv; i += 32u) { a += dot(w[base + i], x[xo + i]); }
  }
  red[li] = a;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && r < p.N) { y[job * p.N + r] = red[li]; }
}`;

// ------------------------------------------------------------------------------------ MoE
// Softmax over the token's E logits and selection of the k largest (ties: lower index), as in llama.cpp.
// Weights = selected probabilities (optionally normalized to sum 1) times `scale`.
const MOE_TOPK = `${PASO}
struct P { E: u32, k: u32, norm: u32, escala: f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> lg: array<f32>;
@group(0) @binding(3) var<storage, read_write> sel: array<u32>;
@group(0) @binding(4) var<storage, read_write> wts: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x;
  if (t >= paso.T) { return; }
  var pr = array<f32, 256>();
  var m = -3.0e38;
  for (var e = 0u; e < p.E; e++) { let v = lg[t * p.E + e]; pr[e] = v; m = max(m, v); }
  var s = 0.0;
  for (var e = 0u; e < p.E; e++) { let v = exp(pr[e] - m); pr[e] = v; s += v; }
  for (var e = 0u; e < p.E; e++) { pr[e] = pr[e] / s; }
  var tot = 0.0;
  for (var i = 0u; i < p.k; i++) {
    var bi = 0u;
    var bv = -1.0;
    for (var e = 0u; e < p.E; e++) { if (pr[e] > bv) { bv = pr[e]; bi = e; } }
    sel[t * p.k + i] = bi;
    wts[t * p.k + i] = bv;
    tot += bv;
    pr[bi] = -2.0;
  }
  var f = p.escala;
  if (p.norm == 1u) { f = f / max(tot, 6.103515625e-5); }
  for (var i = 0u; i < p.k; i++) { wts[t * p.k + i] = wts[t * p.k + i] * f; }
}`;

// x[t][i] += sum_s wts[t*k+s] * de[(t*k+s)*D + i]
const MOE_COMB = `${PASO}
struct P { D: u32, k: u32, z0: u32, z1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read> de: array<f32>;
@group(0) @binding(4) var<storage, read> wts: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= paso.T * p.D) { return; }
  let t = i / p.D;
  let c = i % p.D;
  var a = 0.0;
  for (var s = 0u; s < p.k; s++) { a += wts[t * p.k + s] * de[(t * p.k + s) * p.D + c]; }
  x[i] = x[i] + a;
}`;

// ------------------------------------------------------------------------------------ MLA
// Compressed cache: per position R=kv_lora_rank values (normalized kv) + nr values (k_pe with RoPE).
// One token per group: normalizes kv[0..R), stores it in the cache and applies RoPE to k_pe (consecutive pairs).
const MLA_KV = `${PASO}
${YARN}
struct P { R: u32, nr: u32, eps: f32, base: f32, fscale: f32, ext: f32, ms: f32, low: f32, high: f32, z0: u32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> kvc: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> cc: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = wg.x;
  let rw = p.R + p.nr;
  var s = 0.0;
  for (var i = li; i < p.R; i += 256u) { let v = kvc[t * rw + i]; s += v * v; }
  red[li] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li < k) { red[li] += red[li + k]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(p.R) + p.eps);
  let slot = paso.pos0 + t;
  for (var i = li; i < p.R; i += 256u) { cc[slot * rw + i] = kvc[t * rw + i] * inv * w[i]; }
  let pos = f32(slot);
  let nd = f32(p.nr);
  for (var j = li; j < p.nr / 2u; j += 256u) {
    let a = kvc[t * rw + p.R + 2u * j];
    let b = kvc[t * rw + p.R + 2u * j + 1u];
    let cs = yarn_cs(pos, j, nd, p.base, p.fscale, p.ext, p.ms, p.low, p.high);
    cc[slot * rw + p.R + 2u * j] = a * cs.x - b * cs.y;
    cc[slot * rw + p.R + 2u * j + 1u] = a * cs.y + b * cs.x;
  }
}`;

// Absorbed query: qa[t][h] = ( W_k_h^T q_nope (R values) | q_pe with RoPE (nr values) ).
// wkvb: [nh*(nope+vd)][R] in f32; the first nope rows of each head are the k ones.
const MLA_Q = `${PASO}
${YARN}
struct P { nh: u32, nope: u32, nr: u32, R: u32, vd: u32, base: f32, fscale: f32, ext: f32, ms: f32, low: f32, high: f32, z0: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> wkvb: array<f32>;
@group(0) @binding(4) var<storage, read_write> qa: array<f32>;
var<workgroup> qn: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wg.x;
  let t = wg.y;
  let hd = p.nope + p.nr;
  let qb = (t * p.nh + h) * hd;
  let ob = (t * p.nh + h) * (p.R + p.nr);
  for (var i = li; i < p.nope; i += 256u) { qn[i] = q[qb + i]; }
  workgroupBarrier();
  let pos = f32(paso.pos0 + t);
  let nd = f32(p.nr);
  for (var j = li; j < p.nr / 2u; j += 256u) {
    let a = q[qb + p.nope + 2u * j];
    let b = q[qb + p.nope + 2u * j + 1u];
    let cs = yarn_cs(pos, j, nd, p.base, p.fscale, p.ext, p.ms, p.low, p.high);
    qa[ob + p.R + 2u * j] = a * cs.x - b * cs.y;
    qa[ob + p.R + 2u * j + 1u] = a * cs.y + b * cs.x;
  }
  let fila0 = h * (p.nope + p.vd);
  for (var k = li; k < p.R; k += 256u) {
    var a = 0.0;
    for (var i = 0u; i < p.nope; i++) { a += qn[i] * wkvb[(fila0 + i) * p.R + k]; }
    qa[ob + k] = a;
  }
}`;

// Attention over the compressed cache (one query head per group; K = 576 values, V = the first R)
const MLA_ATT = `${PASO}
struct P { hdk: u32, R: u32, nh: u32, z0: u32, escala: f32, z1: u32, z2: u32, z3: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> cc: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
var<workgroup> qs: array<f32, 640>;
var<workgroup> sc: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wg.x;
  let t = wg.y;
  let nk = paso.pos0 + t + 1u;
  let qb = (t * p.nh + h) * p.hdk;
  for (var i = li; i < p.hdk; i += 64u) { qs[i] = q[qb + i]; }
  workgroupBarrier();
  var m = -3.0e38;
  var l = 0.0;
  var acc = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var tile = 0u; tile < nk; tile += 64u) {
    let pk = tile + li;
    var s = -3.0e38;
    if (pk < nk) {
      let kb = pk * p.hdk;
      var dot = 0.0;
      for (var i = 0u; i < p.hdk; i++) { dot += qs[i] * cc[kb + i]; }
      s = dot * p.escala;
    }
    sc[li] = s;
    workgroupBarrier();
    var mt = m;
    for (var j = 0u; j < 64u; j++) { mt = max(mt, sc[j]); }
    workgroupBarrier();
    var e = 0.0;
    if (pk < nk) { e = exp(s - mt); }
    sc[li] = e;
    workgroupBarrier();
    let corr = exp(m - mt);
    var suma = 0.0;
    for (var j = 0u; j < 64u; j++) { suma += sc[j]; }
    l = l * corr + suma;
    let nt = min(64u, nk - tile);
    for (var c = 0u; c < 8u; c++) {
      let d = li + c * 64u;
      if (d < p.R) {
        var a = acc[c] * corr;
        for (var j = 0u; j < nt; j++) { a += sc[j] * cc[(tile + j) * p.hdk + d]; }
        acc[c] = a;
      }
    }
    m = mt;
    workgroupBarrier();
  }
  let ob = (t * p.nh + h) * p.R;
  for (var c = 0u; c < 8u; c++) {
    let d = li + c * 64u;
    if (d < p.R) { o[ob + d] = acc[c] / l; }
  }
}`;

// Per-head output: att[t][h][j] = sum_k wkvb[h*(nope+vd) + nope + j][k] * olat[t][h][k]
const MLA_V = `${PASO}
struct P { nh: u32, nope: u32, R: u32, vd: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> ol: array<f32>;
@group(0) @binding(3) var<storage, read> wkvb: array<f32>;
@group(0) @binding(4) var<storage, read_write> att: array<f32>;
var<workgroup> os: array<f32, 512>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = paso.T;
  let h = wg.x;
  let t = wg.y;
  for (var k = li; k < p.R; k += 128u) { os[k] = ol[(t * p.nh + h) * p.R + k]; }
  workgroupBarrier();
  for (var j = li; j < p.vd; j += 128u) {
    let rb = (h * (p.nope + p.vd) + p.nope + j) * p.R;
    var a = 0.0;
    for (var k = 0u; k < p.R; k++) { a += wkvb[rb + k] * os[k]; }
    att[(t * p.nh + h) * p.vd + j] = a;
  }
}`;

// Tiled, expert-grouped version of IQ4_NL (one step of 32 weights = one block)
const MATMUL_IQ4_EG = `${PASO}
struct P { N: u32, K: u32, acc: u32, tfijo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> qs: array<u32>;
@group(0) @binding(3) var<storage, read> dh: array<u32>;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@group(0) @binding(6) var<storage, read> lst: array<u32>;
@group(0) @binding(7) var<storage, read> off: array<u32>;
var<private> kvt: array<f32, 16> = array<f32, 16>(${KVALUES});
var<workgroup> wt: array<f32, 2080>;
var<workgroup> xt: array<f32, 1024>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = paso.T;
  let ex = wg.y;
  let c0 = off[ex];
  let cn = off[ex + 1u] - c0;
  let t0 = wg.z * 32u;
  if (t0 >= cn) { return; }
  let r0 = wg.x * 64u;
  let kv = p.K / 4u;
  let nbk = p.K / 32u;
  let lrow = li >> 2u;
  let part = li & 3u;
  let grow = ex * p.N + min(r0 + lrow, p.N - 1u);
  let xtok = li >> 3u;
  let xk = li & 7u;
  let xrow = (lst[c0 + min(t0 + xtok, cn - 1u)] / p.tfijo) * kv;
  let tx = li & 15u;
  let ty = li >> 4u;
  var c00 = 0.0; var c01 = 0.0; var c10 = 0.0; var c11 = 0.0;
  var c20 = 0.0; var c21 = 0.0; var c30 = 0.0; var c31 = 0.0;
  for (var s = 0u; s < nbk; s++) {
    let bi = grow * nbk + s;
    let dd = unpack2x16float(dh[bi >> 1u]);
    let d = select(dd.x, dd.y, (bi & 1u) == 1u);
    for (var h = 0u; h < 2u; h++) {
      let v = part * 2u + h;
      let word = qs[bi * 4u + (v & 3u)];
      let sh = select(0u, 4u, v >= 4u);
      let dq = d * vec4<f32>(kvt[(word >> sh) & 15u], kvt[(word >> (sh + 8u)) & 15u],
                             kvt[(word >> (sh + 16u)) & 15u], kvt[(word >> (sh + 24u)) & 15u]);
      let kb = v * 4u;
      wt[(kb + 0u) * 65u + lrow] = dq.x;
      wt[(kb + 1u) * 65u + lrow] = dq.y;
      wt[(kb + 2u) * 65u + lrow] = dq.z;
      wt[(kb + 3u) * 65u + lrow] = dq.w;
    }
    let xv = x[xrow + s * 8u + xk];
    xt[(xk * 4u + 0u) * 32u + xtok] = xv.x;
    xt[(xk * 4u + 1u) * 32u + xtok] = xv.y;
    xt[(xk * 4u + 2u) * 32u + xtok] = xv.z;
    xt[(xk * 4u + 3u) * 32u + xtok] = xv.w;
    workgroupBarrier();
    for (var kk = 0u; kk < 32u; kk++) {
      let wb = kk * 65u + tx * 4u;
      let a0 = wt[wb]; let a1 = wt[wb + 1u]; let a2 = wt[wb + 2u]; let a3 = wt[wb + 3u];
      let b0 = xt[kk * 32u + ty * 2u]; let b1 = xt[kk * 32u + ty * 2u + 1u];
      c00 += a0 * b0; c01 += a0 * b1;
      c10 += a1 * b0; c11 += a1 * b1;
      c20 += a2 * b0; c21 += a2 * b1;
      c30 += a3 * b0; c31 += a3 * b1;
    }
    workgroupBarrier();
  }
  let ta = t0 + ty * 2u;
  if (ta < cn) {
    let ja = lst[c0 + ta] * p.N;
    let rb = r0 + tx * 4u;
    if (rb < p.N) { y[ja + rb] = c00; }
    if (rb + 1u < p.N) { y[ja + rb + 1u] = c10; }
    if (rb + 2u < p.N) { y[ja + rb + 2u] = c20; }
    if (rb + 3u < p.N) { y[ja + rb + 3u] = c30; }
  }
  if (ta + 1u < cn) {
    let jb = lst[c0 + ta + 1u] * p.N;
    let rb = r0 + tx * 4u;
    if (rb < p.N) { y[jb + rb] = c01; }
    if (rb + 1u < p.N) { y[jb + rb + 1u] = c11; }
    if (rb + 2u < p.N) { y[jb + rb + 2u] = c21; }
    if (rb + 3u < p.N) { y[jb + rb + 3u] = c31; }
  }
}`;

// Groups the (token, slot) jobs by expert: lst = jobs sorted by expert, off = start offsets (E+1 values)
const MOE_GROUP = `${PASO}
struct P { E: u32, n: u32, z0: u32, z1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> sel: array<u32>;
@group(0) @binding(3) var<storage, read_write> lst: array<u32>;
@group(0) @binding(4) var<storage, read_write> off: array<u32>;
var<workgroup> cnt: array<u32, 257>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  let n = paso.T * p.n;
  var c = 0u;
  if (li < p.E) { for (var i = 0u; i < n; i++) { if (sel[i] == li) { c++; } } }
  cnt[li] = c;
  workgroupBarrier();
  if (li == 0u) {
    var a = 0u;
    for (var e = 0u; e < p.E; e++) { let v = cnt[e]; cnt[e] = a; off[e] = a; a += v; }
    off[p.E] = a;
  }
  workgroupBarrier();
  if (li < p.E) {
    var o = off[li];
    for (var i = 0u; i < n; i++) { if (sel[i] == li) { lst[o] = i; o++; } }
  }
}`;

export const WGSL_DS = {
  matmul_iq4nl_1: matmulIQ4(1, false),
  matmul_iq4nl_4: matmulIQ4(4, false),
  matmul_iq4nl_e: matmulIQ4(1, true),
  matmul_iq4nl_eg: MATMUL_IQ4_EG,
  moe_group: MOE_GROUP,
  matmul_f32_e: MATMUL_F32_E,
  matmul_q8_e: MATMUL_Q8_E,
  moe_topk: MOE_TOPK,
  moe_comb: MOE_COMB,
  mla_kv: MLA_KV,
  mla_q: MLA_Q,
  mla_att: MLA_ATT,
  mla_v: MLA_V,
};

// Uploads an IQ4_NL tensor (18-byte blocks) as qs (u32) + dh (f16 in pairs), in batches to avoid duplicating RAM.
export function subirIQ4(gpu, u8, etiqueta) {
  const nb = u8.length / 18;
  if (!Number.isInteger(nb)) throw new Error("invalid IQ4_NL size");
  const bq = gpu.buffer(nb * 16, undefined, etiqueta + ".qs");
  const bd = gpu.buffer(Math.ceil(nb / 2) * 4, undefined, etiqueta + ".dh");
  const porTanda = 1 << 20;   // blocks per batch (even)
  for (let b0 = 0; b0 < nb; b0 += porTanda) {
    const n = Math.min(porTanda, nb - b0);
    const qs = new Uint8Array(n * 16);
    const dh = new Uint16Array(Math.ceil(n / 2) * 2);
    for (let b = 0; b < n; b++) {
      const o = (b0 + b) * 18;
      dh[b] = u8[o] | (u8[o + 1] << 8);
      const q0 = b * 16, o2 = o + 2;
      for (let k = 0; k < 16; k++) qs[q0 + k] = u8[o2 + k];
    }
    gpu.device.queue.writeBuffer(bq, b0 * 16, qs);
    gpu.device.queue.writeBuffer(bd, (b0 >> 1) * 4, dh.buffer, 0, dh.byteLength);
  }
  return { qs: bq, dh: bd };
}
