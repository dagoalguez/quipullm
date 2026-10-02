// K-quants from llama.cpp (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) for WebGPU.
// Weights are stored on the GPU exactly as they come in the GGUF (only padded to a multiple of 4 bytes per
// block) and each kernel decodes them on the fly. An "item" is a sub-block of 32 (Q4_K/Q5_K) or
// 16 (Q3_K/Q6_K) weights; the 32 threads of a row share the items among them.
//
// Formats (block of 256 weights), taken from ggml-common.h:
//   Q4_K 144 B: d f16, dmin f16, scales[12], qs[128]
//   Q5_K 176 B: d f16, dmin f16, scales[12], qh[32], qs[128]
//   Q6_K 210 B: ql[128], qh[64], scales[16] int8, d f16
//   Q3_K 110 B: hmask[32], qs[64], scales[12], d f16
//   Q2_K  84 B: scales[16] (low nibble = scale, high nibble = min), qs[64], d f16, dmin f16

const F16 = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) t[h] = s * Math.pow(2, -14) * (f / 1024);
    else if (e === 31) t[h] = f ? NaN : s * Infinity;
    else t[h] = s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return t;
})();
void F16;

// GGML type id -> description
export const TIPOS_K = {
  10: { nombre: "q2k", bytes: 84, stride: 84 },
  11: { nombre: "q3k", bytes: 110, stride: 112 },
  12: { nombre: "q4k", bytes: 144, stride: 144 },
  13: { nombre: "q5k", bytes: 176, stride: 176 },
  14: { nombre: "q6k", bytes: 210, stride: 212 },
};

// Pads each block up to `stride` bytes (a multiple of 4) so they can be read as u32.
export function reempaquetarK(u8, info) {
  const nb = u8.length / info.bytes;
  if (!Number.isInteger(nb)) throw new Error(`invalid size ${info.nombre}`);
  if (info.stride === info.bytes) return u8;
  const out = new Uint8Array(nb * info.stride);
  for (let b = 0; b < nb; b++) out.set(u8.subarray(b * info.bytes, (b + 1) * info.bytes), b * info.stride);
  return out;
}

// Uploads to the GPU without duplicating a large tensor in memory: pads and sends in batches of ~32 MB
// (the host machine can be tight on RAM and the output tensor of a 7B weighs about 450 MB).
export function subirK(gpu, u8, info, etiqueta) {
  const nb = u8.length / info.bytes;
  if (!Number.isInteger(nb)) throw new Error(`invalid size ${info.nombre}`);
  if (info.stride === info.bytes) return gpu.subir(u8, etiqueta);
  const buf = gpu.buffer(nb * info.stride, undefined, etiqueta);
  const porTanda = Math.max(1, Math.floor(32e6 / info.stride));
  for (let b0 = 0; b0 < nb; b0 += porTanda) {
    const n = Math.min(porTanda, nb - b0);
    const out = new Uint8Array(n * info.stride);
    for (let b = 0; b < n; b++) out.set(u8.subarray((b0 + b) * info.bytes, (b0 + b + 1) * info.bytes), b * info.stride);
    gpu.device.queue.writeBuffer(buf, b0 * info.stride, out);
  }
  return buf;
}

// --------------------------------------------------------------------- WGSL code per type
const BYTE = `fn b8(base: u32, off: u32) -> u32 { return (w[base + (off >> 2u)] >> ((off & 3u) * 8u)) & 0xFFu; }`;

// Scale and min (already multiplied by d and dmin) of sub-block j of Q4_K/Q5_K
const SCALE_K45 = `
fn scmin(bb: u32, j: u32) -> vec2<f32> {
  let dm = unpack2x16float(w[bb]);
  let sb = bb + 1u;
  var sc: u32; var mn: u32;
  if (j < 4u) { sc = b8(sb, j) & 63u; mn = b8(sb, j + 4u) & 63u; }
  else {
    sc = (b8(sb, j + 4u) & 15u) | ((b8(sb, j - 4u) >> 6u) << 4u);
    mn = (b8(sb, j + 4u) >> 4u) | ((b8(sb, j) >> 6u) << 4u);
  }
  return vec2<f32>(dm.x * f32(sc), dm.y * f32(mn));
}`;

const SCALE_Q3K = `
fn scq3(bb: u32, s: u32) -> f32 {
  let a0 = w[bb + 24u]; let a1 = w[bb + 25u]; let tmp = w[bb + 26u];
  let k1 = 0x03030303u; let k2 = 0x0f0f0f0fu;
  var wd: u32;
  let g = s >> 2u;
  if (g == 0u) { wd = (a0 & k2) | (((tmp >> 0u) & k1) << 4u); }
  else if (g == 1u) { wd = (a1 & k2) | (((tmp >> 2u) & k1) << 4u); }
  else if (g == 2u) { wd = ((a0 >> 4u) & k2) | (((tmp >> 4u) & k1) << 4u); }
  else { wd = ((a1 >> 4u) & k2) | (((tmp >> 6u) & k1) << 4u); }
  let sc = f32((wd >> ((s & 3u) * 8u)) & 0xFFu) - 32.0;
  return unpack2x16float(w[bb + 27u]).x * sc;
}`;

const SCALE_Q6K = `
fn scq6(bb: u32, s: u32) -> f32 {
  let v = f32(bitcast<i32>(b8(bb + 48u, s) << 24u) >> 24u);
  return unpack2x16float(w[bb + 52u]).x * v;
}`;

const VEC4 = (a, b, c, d) => `vec4<f32>(f32(${a}), f32(${b}), f32(${c}), f32(${d}))`;
const nib4 = (word, sh) => VEC4(`(${word} >> ${sh}) & 15u`, `(${word} >> (${sh} + 8u)) & 15u`,
  `(${word} >> (${sh} + 16u)) & 15u`, `(${word} >> (${sh} + 24u)) & 15u`);

// For each type: prelude (functions), ipb (items per block), nw (u32 per item), stride (u32 per block),
// setup (computes fd, fm, xb), q (vec4 of weights for word wi), usaMin, deq (element e of the block)
const TIPOS = {
  q2k: {
    stride: 21, ipb: 16, nw: 4, usaMin: true,
    prelude: BYTE,
    setup: `let n = j >> 3u; let jj = (j & 7u) >> 1u; let l0 = (j & 1u) * 16u;
      let scb = b8(bb, j); let dmn = unpack2x16float(w[bb + 20u]);
      let fd = dmn.x * f32(scb & 15u); let fm = dmn.y * f32(scb >> 4u);
      let xb = b * 64u + n * 32u + jj * 8u + (l0 >> 2u);
      let qsb = bb + 4u + n * 8u + (l0 >> 2u); let sq2 = jj * 2u;`,
    q: `${VEC4("(w[qsb + wi] >> sq2) & 3u", "(w[qsb + wi] >> (8u + sq2)) & 3u", "(w[qsb + wi] >> (16u + sq2)) & 3u", "(w[qsb + wi] >> (24u + sq2)) & 3u")}`,
    deq: "",
  },
  q4k: {
    stride: 36, ipb: 8, nw: 8, usaMin: true,
    prelude: BYTE + SCALE_K45,
    setup: `let sm = scmin(bb, j); let fd = sm.x; let fm = sm.y; let xb = b * 64u + j * 8u;
      let qb = bb + 4u + (j >> 1u) * 8u; let sh = (j & 1u) * 4u;`,
    q: `${nib4("w[qb + wi]", "sh")}`,
    deq: `let j = e >> 5u; let l = e & 31u; let sm = scmin(bb, j);
      let by = b8(bb + 4u + (j >> 1u) * 8u, l); let q = (by >> ((j & 1u) * 4u)) & 15u;
      return sm.x * f32(q) - sm.y;`,
  },
  q5k: {
    stride: 44, ipb: 8, nw: 8, usaMin: true,
    prelude: BYTE + SCALE_K45,
    setup: `let sm = scmin(bb, j); let fd = sm.x; let fm = sm.y; let xb = b * 64u + j * 8u;
      let qb = bb + 12u + (j >> 1u) * 8u; let sh = (j & 1u) * 4u;`,
    q: `${nib4("w[qb + wi]", "sh")} + 16.0 * ${VEC4("(w[bb + 4u + wi] >> j) & 1u", "(w[bb + 4u + wi] >> (8u + j)) & 1u",
      "(w[bb + 4u + wi] >> (16u + j)) & 1u", "(w[bb + 4u + wi] >> (24u + j)) & 1u")}`,
    deq: "",
  },
  q6k: {
    stride: 53, ipb: 16, nw: 4, usaMin: false,
    prelude: BYTE + SCALE_Q6K,
    setup: `let h = j >> 3u; let qd = (j & 7u) >> 1u; let l0 = (j & 1u) * 16u;
      let fd = scq6(bb, j); let fm = 0.0;
      let xb = b * 64u + h * 32u + qd * 8u + (l0 >> 2u);
      let qlo = bb + h * 16u + (qd & 1u) * 8u + (l0 >> 2u); let qhi = bb + 32u + h * 8u + (l0 >> 2u);
      let nsh = (qd >> 1u) * 4u; let hsh = qd * 2u;`,
    q: `${VEC4("((w[qlo + wi] >> nsh) & 15u) | (((w[qhi + wi] >> hsh) & 3u) << 4u)",
      "((w[qlo + wi] >> (8u + nsh)) & 15u) | (((w[qhi + wi] >> (8u + hsh)) & 3u) << 4u)",
      "((w[qlo + wi] >> (16u + nsh)) & 15u) | (((w[qhi + wi] >> (16u + hsh)) & 3u) << 4u)",
      "((w[qlo + wi] >> (24u + nsh)) & 15u) | (((w[qhi + wi] >> (24u + hsh)) & 3u) << 4u)")} - vec4<f32>(32.0)`,
    deq: "",
  },
  q3k: {
    stride: 28, ipb: 16, nw: 4, usaMin: false,
    prelude: BYTE + SCALE_Q3K,
    setup: `let n = j >> 3u; let jj = (j & 7u) >> 1u; let l0 = (j & 1u) * 16u;
      let fd = scq3(bb, j); let fm = 0.0;
      let xb = b * 64u + n * 32u + jj * 8u + (l0 >> 2u);
      let qsb = bb + 8u + n * 8u + (l0 >> 2u); let hmb = bb + (l0 >> 2u);
      let sq2 = jj * 2u; let hb = n * 4u + jj;`,
    q: `${VEC4("((w[qsb + wi] >> sq2) & 3u) + 4u * ((w[hmb + wi] >> hb) & 1u)",
      "((w[qsb + wi] >> (8u + sq2)) & 3u) + 4u * ((w[hmb + wi] >> (8u + hb)) & 1u)",
      "((w[qsb + wi] >> (16u + sq2)) & 3u) + 4u * ((w[hmb + wi] >> (16u + hb)) & 1u)",
      "((w[qsb + wi] >> (24u + sq2)) & 3u) + 4u * ((w[hmb + wi] >> (24u + hb)) & 1u)")} - vec4<f32>(4.0)`,
    deq: "",
  },
};

// Decoding of a single element e (0..255) of the block starting at bb (for the embedding)
TIPOS.q4k.deq = `let j = e >> 5u; let l = e & 31u; let sm = scmin(bb, j);
      let by = b8(bb + 4u + (j >> 1u) * 8u, l); let q = (by >> ((j & 1u) * 4u)) & 15u;
      return sm.x * f32(q) - sm.y;`;
TIPOS.q5k.deq = `let j = e >> 5u; let l = e & 31u; let sm = scmin(bb, j);
      let by = b8(bb + 12u + (j >> 1u) * 8u, l); let q = ((by >> ((j & 1u) * 4u)) & 15u) + (((b8(bb + 4u, l) >> j) & 1u) << 4u);
      return sm.x * f32(q) - sm.y;`;
TIPOS.q6k.deq = `let s = e >> 4u; let h = e >> 7u; let r = e & 127u; let qd = r >> 5u; let l = r & 31u;
      let lo = (b8(bb + h * 16u + (qd & 1u) * 8u, l) >> ((qd >> 1u) * 4u)) & 15u;
      let hi = (b8(bb + 32u + h * 8u, l) >> (qd * 2u)) & 3u;
      return scq6(bb, s) * (f32(lo | (hi << 4u)) - 32.0);`;
TIPOS.q3k.deq = `let s = e >> 4u; let n = e >> 7u; let r = e & 127u; let jj = r >> 5u; let l = r & 31u;
      let q2 = (b8(bb + 8u + n * 8u, l) >> (jj * 2u)) & 3u;
      let hm = (b8(bb, l) >> (n * 4u + jj)) & 1u;
      return scq3(bb, s) * (f32(q2) - 4.0 * f32(1u - hm));`;

TIPOS.q2k.deq = `let s = e >> 4u; let n = e >> 7u; let r = e & 127u; let jj = r >> 5u; let l = r & 31u;
      let q2 = (b8(bb + 4u + n * 8u, l) >> (jj * 2u)) & 3u;
      let scb = b8(bb, s); let dm = unpack2x16float(w[bb + 20u]);
      return dm.x * f32(scb & 15u) * f32(q2) - dm.y * f32(scb >> 4u);`;

const PASO = `struct Paso { T: u32, pos0: u32, a: u32, b: u32 };`;

function matmul(nombre, nt) {
  const t = TIPOS[nombre];
  const tok = [...Array(nt).keys()];
  const T = nt === 1 ? "_ = paso.T;" : `let T = paso.T; let t0 = wg.z * ${nt}u;`;
  const offs = nt === 1 ? "" : tok.map((j) => `let o${j} = min(t0 + ${j}u, T - 1u) * kv;`).join("\n  ");
  const xr = (j) => (nt === 1 ? "x[xb + wi]" : `x[o${j} + xb + wi]`);
  const oj = (j) => (nt === 1 ? "" : "");
  void oj;
  return `${PASO}
struct P { N: u32, K: u32, acc: u32, tfijo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, ${nt * 256}>;
${t.prelude}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  ${T}
  let r = (wg.x + wg.y * nwg.x) * 8u + li / 32u;
  let lane = li % 32u;
  let kv = p.K / 4u;
  ${offs}
  ${tok.map((j) => `var a${j} = 0.0;`).join(" ")}
  let nb = p.K / 256u;
  let ni = nb * ${t.ipb}u;
  if (r < p.N) {
    for (var it = lane; it < ni; it += 32u) {
      let b = it / ${t.ipb}u;
      let j = it % ${t.ipb}u;
      let bb = (r * nb + b) * ${t.stride}u;
      ${t.setup}
      ${tok.map((k) => `var s${k} = 0.0;${t.usaMin ? ` var m${k} = 0.0;` : ""}`).join(" ")}
      for (var wi = 0u; wi < ${t.nw}u; wi++) {
        let q = ${t.q};
        ${tok.map((k) => `let xv${k} = ${xr(k)}; s${k} += dot(q, xv${k});${t.usaMin ? ` m${k} += dot(xv${k}, vec4<f32>(1.0));` : ""}`).join("\n        ")}
      }
      ${tok.map((k) => `a${k} += fd * s${k}${t.usaMin ? ` - fm * m${k}` : ""};`).join(" ")}
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
    ${nt === 1
      ? `let v = red[li]; if (p.acc == 1u) { y[r] = y[r] + v; } else { y[r] = v; }`
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

// "Expert" matmul (MoE): stacked tensor [E][N][K]. Each job z = (token, slot) uses expert sel[z]:
//   y[z][r] = W[sel[z]][r] . x[z / tfijo]      (tfijo = number of experts per token for the shared input, 1 if each
//   job has its own input). N = rows per expert.
function matmulExp(nombre) {
  const t = TIPOS[nombre];
  return `${PASO}
struct P { N: u32, K: u32, acc: u32, tfijo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> sel: array<u32>;
var<workgroup> red: array<f32, 256>;
${t.prelude}
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
  var a0 = 0.0;
  let nb = p.K / 256u;
  let ni = nb * ${t.ipb}u;
  if (r < p.N) {
    let rr = ex * p.N + r;
    for (var it = lane; it < ni; it += 32u) {
      let b = it / ${t.ipb}u;
      let j = it % ${t.ipb}u;
      let bb = (rr * nb + b) * ${t.stride}u;
      ${t.setup}
      var s0 = 0.0;${t.usaMin ? " var m0 = 0.0;" : ""}
      for (var wi = 0u; wi < ${t.nw}u; wi++) {
        let q = ${t.q};
        let xv0 = x[xo + xb + wi]; s0 += dot(q, xv0);${t.usaMin ? " m0 += dot(xv0, vec4<f32>(1.0));" : ""}
      }
      a0 += fd * s0${t.usaMin ? " - fm * m0" : ""};
    }
  }
  red[li] = a0;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && r < p.N) { y[job * p.N + r] = red[li]; }
}`;
}

// Tiled matmul for prefill: each group computes 64 rows x 32 tokens. It decodes each weight ONCE
// per 32-token tile (shared memory) instead of once per 4 tokens, and spreads the reads of
// the activations among the threads. K step = 32 weights.
function matmulTile(nombre) {
  const t = TIPOS[nombre];
  const dosItems = t.ipb === 16;   // Q3_K/Q6_K: a step of 32 is 2 items of 16
  return `${PASO}
struct P { N: u32, K: u32, acc: u32, tfijo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> wt: array<f32, 2080>;
var<workgroup> xt: array<f32, 1024>;
${t.prelude}
fn salida(r: u32, tk: u32, val: f32, T: u32) {
  if (r < p.N && tk < T) {
    let o = tk * p.N + r;
    if (p.acc == 1u) { y[o] = y[o] + val; } else { y[o] = val; }
  }
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let T = paso.T;
  let r0 = (wg.x + wg.y * nwg.x) * 64u;
  let t0 = wg.z * 32u;
  let kv = p.K / 4u;
  let nb = p.K / 256u;
  let lrow = li >> 2u;
  let part = li & 3u;
  let grow = min(r0 + lrow, p.N - 1u);
  let xtok = li >> 3u;
  let xk = li & 7u;
  let xrow = min(t0 + xtok, T - 1u) * kv;
  let tx = li & 15u;
  let ty = li >> 4u;
  var c00 = 0.0; var c01 = 0.0; var c10 = 0.0; var c11 = 0.0;
  var c20 = 0.0; var c21 = 0.0; var c30 = 0.0; var c31 = 0.0;
  let ns = p.K / 32u;
  for (var s = 0u; s < ns; s++) {
    let b = s >> 3u;
    let bb = (grow * nb + b) * ${t.stride}u;
    for (var h = 0u; h < 2u; h++) {
      let v = part + h * 4u;
      ${dosItems ? "let j = (s & 7u) * 2u + (v >> 2u); let wi = v & 3u;" : "let j = s & 7u; let wi = v;"}
      ${t.setup}
      let q = ${t.q};
      let dq = fd * q - vec4<f32>(fm);
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
  let ra = r0 + tx * 4u;
  let ta = t0 + ty * 2u;
  salida(r0 + tx * 4u + 0u, t0 + ty * 2u + 0u, c00, T); salida(r0 + tx * 4u + 0u, t0 + ty * 2u + 1u, c01, T);
  salida(r0 + tx * 4u + 1u, t0 + ty * 2u + 0u, c10, T); salida(r0 + tx * 4u + 1u, t0 + ty * 2u + 1u, c11, T);
  salida(r0 + tx * 4u + 2u, t0 + ty * 2u + 0u, c20, T); salida(r0 + tx * 4u + 2u, t0 + ty * 2u + 1u, c21, T);
  salida(r0 + tx * 4u + 3u, t0 + ty * 2u + 0u, c30, T); salida(r0 + tx * 4u + 3u, t0 + ty * 2u + 1u, c31, T);
}`;
}

// Tiled matmul "grouped by expert" (MoE prefill): each group computes 64 rows of ONE expert for up to 32 of
// the jobs (token, slot) that selected it, so that the expert's weights are read and decoded once per
// tile instead of once per token. lst[off[e] .. off[e+1]) = jobs of expert e (see moe_group).
//   y[trabajo][r] = W[e][r] . x[trabajo / tfijo]     group: x = 64-row blocks, y = expert, z = 32-job tile
function matmulTileGrp(nombre) {
  const t = TIPOS[nombre];
  const dosItems = t.ipb === 16;
  return `${PASO}
struct P { N: u32, K: u32, acc: u32, tfijo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@group(0) @binding(5) var<storage, read> lst: array<u32>;
@group(0) @binding(6) var<storage, read> off: array<u32>;
var<workgroup> wt: array<f32, 2080>;
var<workgroup> xt: array<f32, 1024>;
${t.prelude}
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
  let nb = p.K / 256u;
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
  let ns = p.K / 32u;
  for (var s = 0u; s < ns; s++) {
    let b = s >> 3u;
    let bb = (grow * nb + b) * ${t.stride}u;
    for (var h = 0u; h < 2u; h++) {
      let v = part + h * 4u;
      ${dosItems ? "let j = (s & 7u) * 2u + (v >> 2u); let wi = v & 3u;" : "let j = s & 7u; let wi = v;"}
      ${t.setup}
      let q = ${t.q};
      let dq = fd * q - vec4<f32>(fm);
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
}

function embed(nombre) {
  const t = TIPOS[nombre];
  return `${PASO}
struct P { D: u32, z0: u32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
${t.prelude}
fn deq(bb: u32, e: u32) -> f32 {
  ${t.deq}
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= paso.T * p.D) { return; }
  let tk = i / p.D;
  let k = i % p.D;
  let nb = p.D / 256u;
  let bb = (ids[tk] * nb + k / 256u) * ${t.stride}u;
  y[i] = deq(bb, k & 255u);
}`;
}

export const WGSL_K = {};
for (const n of Object.keys(TIPOS)) {
  WGSL_K[`matmul_${n}_1`] = matmul(n, 1);
  WGSL_K[`matmul_${n}_4`] = matmul(n, 4);
  WGSL_K[`matmul_${n}_t`] = matmulTile(n);
  WGSL_K[`matmul_${n}_e`] = matmulExp(n);
  WGSL_K[`matmul_${n}_eg`] = matmulTileGrp(n);
  WGSL_K[`embed_${n}`] = embed(n);
}
