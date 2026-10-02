// WebGPU layer: device, buffers and WGSL kernels (everything in f32).
// Conventions: activations in rows [T][D]; weights in rows [N][K] (N outputs, K inputs).

import { WGSL_K } from "./kquants.js";
import { WGSL_DS } from "./kds.js";

export const TB = 8;   // max tokens per pass (batched prefill)

const PASO = `struct Paso { T: u32, pos0: u32, a: u32, b: u32 };`;

// custom sine and cosine (Cody-Waite range reduction + cephes polynomials): the native sin/cos of some
// GPUs/compilers only guarantee ~2e-4 error, and in RoPE that error accumulates layer after layer.
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

const WGSL = {
  // ---------------------------------------------------------------- matmul Q8_0
  matmul_q8: `${PASO}
struct P { N: u32, K: u32, acc: u32, tfijo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> qs: array<u32>;
@group(0) @binding(3) var<storage, read> ds: array<f32>;
@group(0) @binding(4) var<storage, read> x: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 2048>;
fn i8(v: u32) -> f32 { return f32(bitcast<i32>(v) >> 24u); }
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let r = (wg.x + wg.y * nwg.x) * 8u + li / 32u;
  let lane = li % 32u;
  var T = paso.T;
  if (p.tfijo > 0u) { T = p.tfijo; }
  var acc = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  let nb = p.K / 32u;
  if (r < p.N) {
    for (var b = lane; b < nb; b += 32u) {
      let blk = r * nb + b;
      let d = ds[blk];
      var part = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
      for (var w = 0u; w < 8u; w++) {
        let q = qs[blk * 8u + w];
        let q0 = i8(q << 24u); let q1 = i8(q << 16u); let q2 = i8(q << 8u); let q3 = i8(q);
        let xo = b * 32u + w * 4u;
        for (var t = 0u; t < T; t++) {
          let xi = t * p.K + xo;
          part[t] += q0 * x[xi] + q1 * x[xi + 1u] + q2 * x[xi + 2u] + q3 * x[xi + 3u];
        }
      }
      for (var t = 0u; t < T; t++) { acc[t] += d * part[t]; }
    }
  }
  for (var t = 0u; t < T; t++) { red[t * 256u + li] = acc[t]; }
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) {
      for (var t = 0u; t < T; t++) { red[t * 256u + li] += red[t * 256u + li + s]; }
    }
    workgroupBarrier();
  }
  if (lane == 0u && r < p.N) {
    for (var t = 0u; t < T; t++) {
      let o = t * p.N + r;
      let v = red[t * 256u + li];
      if (p.acc == 1u) { y[o] = y[o] + v; } else { y[o] = v; }
    }
  }
}`,

  // ---------------------------------------------------------------- matmul f32
  matmul_f32: `${PASO}
struct P { N: u32, K: u32, acc: u32, tfijo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 2048>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let r = (wg.x + wg.y * nwg.x) * 8u + li / 32u;
  let lane = li % 32u;
  var T = paso.T;
  if (p.tfijo > 0u) { T = p.tfijo; }
  var acc = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  if (r < p.N) {
    for (var k = lane; k < p.K; k += 32u) {
      let wv = w[r * p.K + k];
      for (var t = 0u; t < T; t++) { acc[t] += wv * x[t * p.K + k]; }
    }
  }
  for (var t = 0u; t < T; t++) { red[t * 256u + li] = acc[t]; }
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) {
      for (var t = 0u; t < T; t++) { red[t * 256u + li] += red[t * 256u + li + s]; }
    }
    workgroupBarrier();
  }
  if (lane == 0u && r < p.N) {
    for (var t = 0u; t < T; t++) {
      let o = t * p.N + r;
      let v = red[t * 256u + li];
      if (p.acc == 1u) { y[o] = y[o] + v; } else { y[o] = v; }
    }
  }
}`,

  // ---------------------------------------------------------------- matmul Q8_0, one token (decoding)
  matmul_q8_1: `${PASO}
struct P { N: u32, K: u32, acc: u32, tfijo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> qs: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> ds: array<f32>;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;
fn d4(q: u32, xv: vec4<f32>) -> f32 {
  let v = vec4<f32>(f32(bitcast<i32>(q << 24u) >> 24u), f32(bitcast<i32>(q << 16u) >> 24u),
                    f32(bitcast<i32>(q << 8u) >> 24u), f32(bitcast<i32>(q) >> 24u));
  return dot(v, xv);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  _ = paso.T;
  let r = (wg.x + wg.y * nwg.x) * 8u + li / 32u;
  let lane = li % 32u;
  var acc = 0.0;
  let nb = p.K / 32u;
  if (r < p.N) {
    for (var b = lane; b < nb; b += 32u) {
      let blk = r * nb + b;
      let q0 = qs[blk * 2u];
      let q1 = qs[blk * 2u + 1u];
      let xb = b * 8u;
      var s = d4(q0.x, x[xb]) + d4(q0.y, x[xb + 1u]) + d4(q0.z, x[xb + 2u]) + d4(q0.w, x[xb + 3u]);
      s += d4(q1.x, x[xb + 4u]) + d4(q1.y, x[xb + 5u]) + d4(q1.z, x[xb + 6u]) + d4(q1.w, x[xb + 7u]);
      acc += ds[blk] * s;
    }
  }
  red[li] = acc;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && r < p.N) {
    let v = red[li];
    if (p.acc == 1u) { y[r] = y[r] + v; } else { y[r] = v; }
  }
}`,

  // ---------------------------------------------------------------- matmul f32, one token
  matmul_f32_1: `${PASO}
struct P { N: u32, K: u32, acc: u32, tfijo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  _ = paso.T;
  let r = (wg.x + wg.y * nwg.x) * 8u + li / 32u;
  let lane = li % 32u;
  var acc = 0.0;
  let k4 = p.K / 4u;
  if (r < p.N) {
    for (var k = lane; k < k4; k += 32u) { acc += dot(w[r * k4 + k], x[k]); }
  }
  red[li] = acc;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) { red[li] += red[li + s]; }
    workgroupBarrier();
  }
  if (lane == 0u && r < p.N) {
    let v = red[li];
    if (p.acc == 1u) { y[r] = y[r] + v; } else { y[r] = v; }
  }
}`,

  // ---------------------------------------------------------------- matmul Q8_0, 4 tokens per group (prefill)
  // The z axis of the dispatch selects the block of 4 tokens. Named accumulators (no dynamically indexed
  // arrays, which the GPU would spill to private memory). Weights are read once per 4 tokens.
  matmul_q8_4: `${PASO}
struct P { N: u32, K: u32, acc: u32, tfijo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> qs: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> ds: array<f32>;
@group(0) @binding(4) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 1024>;
fn d4(q: u32, xv: vec4<f32>) -> f32 {
  let v = vec4<f32>(f32(bitcast<i32>(q << 24u) >> 24u), f32(bitcast<i32>(q << 16u) >> 24u),
                    f32(bitcast<i32>(q << 8u) >> 24u), f32(bitcast<i32>(q) >> 24u));
  return dot(v, xv);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let T = paso.T;
  let t0 = wg.z * 4u;
  let r = (wg.x + wg.y * nwg.x) * 8u + li / 32u;
  let lane = li % 32u;
  let kv = p.K / 4u;
  let o0 = min(t0, T - 1u) * kv;
  let o1 = min(t0 + 1u, T - 1u) * kv;
  let o2 = min(t0 + 2u, T - 1u) * kv;
  let o3 = min(t0 + 3u, T - 1u) * kv;
  var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
  let nb = p.K / 32u;
  if (r < p.N) {
    for (var b = lane; b < nb; b += 32u) {
      let blk = r * nb + b;
      let d = ds[blk];
      var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0;
      for (var h = 0u; h < 2u; h++) {
        let q = qs[blk * 2u + h];
        let xb = b * 8u + h * 4u;
        s0 += d4(q.x, x[o0 + xb]) + d4(q.y, x[o0 + xb + 1u]) + d4(q.z, x[o0 + xb + 2u]) + d4(q.w, x[o0 + xb + 3u]);
        s1 += d4(q.x, x[o1 + xb]) + d4(q.y, x[o1 + xb + 1u]) + d4(q.z, x[o1 + xb + 2u]) + d4(q.w, x[o1 + xb + 3u]);
        s2 += d4(q.x, x[o2 + xb]) + d4(q.y, x[o2 + xb + 1u]) + d4(q.z, x[o2 + xb + 2u]) + d4(q.w, x[o2 + xb + 3u]);
        s3 += d4(q.x, x[o3 + xb]) + d4(q.y, x[o3 + xb + 1u]) + d4(q.z, x[o3 + xb + 2u]) + d4(q.w, x[o3 + xb + 3u]);
      }
      a0 += d * s0; a1 += d * s1; a2 += d * s2; a3 += d * s3;
    }
  }
  red[li] = a0; red[256u + li] = a1; red[512u + li] = a2; red[768u + li] = a3;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) {
      red[li] += red[li + s]; red[256u + li] += red[256u + li + s];
      red[512u + li] += red[512u + li + s]; red[768u + li] += red[768u + li + s];
    }
    workgroupBarrier();
  }
  if (lane == 0u && r < p.N) {
    for (var j = 0u; j < 4u; j++) {
      let t = t0 + j;
      if (t < T) {
        let o = t * p.N + r;
        let v = red[j * 256u + li];
        if (p.acc == 1u) { y[o] = y[o] + v; } else { y[o] = v; }
      }
    }
  }
}`,

  // ---------------------------------------------------------------- matmul f32, 4 tokens per group (prefill)
  matmul_f32_4: `${PASO}
struct P { N: u32, K: u32, acc: u32, tfijo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 1024>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let T = paso.T;
  let t0 = wg.z * 4u;
  let r = (wg.x + wg.y * nwg.x) * 8u + li / 32u;
  let lane = li % 32u;
  let k4 = p.K / 4u;
  let o0 = min(t0, T - 1u) * k4;
  let o1 = min(t0 + 1u, T - 1u) * k4;
  let o2 = min(t0 + 2u, T - 1u) * k4;
  let o3 = min(t0 + 3u, T - 1u) * k4;
  var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
  if (r < p.N) {
    for (var k = lane; k < k4; k += 32u) {
      let wv = w[r * k4 + k];
      a0 += dot(wv, x[o0 + k]); a1 += dot(wv, x[o1 + k]); a2 += dot(wv, x[o2 + k]); a3 += dot(wv, x[o3 + k]);
    }
  }
  red[li] = a0; red[256u + li] = a1; red[512u + li] = a2; red[768u + li] = a3;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if (lane < s) {
      red[li] += red[li + s]; red[256u + li] += red[256u + li + s];
      red[512u + li] += red[512u + li + s]; red[768u + li] += red[768u + li + s];
    }
    workgroupBarrier();
  }
  if (lane == 0u && r < p.N) {
    for (var j = 0u; j < 4u; j++) {
      let t = t0 + j;
      if (t < T) {
        let o = t * p.N + r;
        let v = red[j * 256u + li];
        if (p.acc == 1u) { y[o] = y[o] + v; } else { y[o] = v; }
      }
    }
  }
}`,

  // ---------------------------------------------------------------- RMSNorm per row
  geglu: `${PASO}
struct P { F: u32, z0: u32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read_write> g: array<f32>;
@group(0) @binding(3) var<storage, read> u: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3<u32>) {
  let i = gi.x;
  if (i >= paso.T * p.F) { return; }
  let a = g[i];
  let z = clamp(0.79788456 * (a + 0.044715 * a * a * a), -20.0, 20.0);
  g[i] = 0.5 * a * (1.0 + tanh(z)) * u[i];
}`,
  rmsnorm: `${PASO}
struct P { D: u32, ultima: u32, eps: f32, z: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  var fila = wg.x;
  var filaOut = wg.x;
  if (p.ultima == 1u) { fila = paso.T - 1u; filaOut = 0u; }
  let base = fila * p.D;
  var s = 0.0;
  for (var i = li; i < p.D; i += 256u) { let v = x[base + i]; s += v * v; }
  red[li] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li < k) { red[li] += red[li + k]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(p.D) + p.eps);
  for (var i = li; i < p.D; i += 256u) { y[filaOut * p.D + i] = x[base + i] * inv * w[i]; }
}`,

  // Classic LayerNorm (BERT): y = (x - mean) / sqrt(var + eps) * w + b, in place
  layernorm: `${PASO}
struct P { D: u32, z0: u32, eps: f32, z1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> bb: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let base = wg.x * p.D;
  if (wg.x >= paso.T) { return; }
  var s = 0.0;
  for (var i = li; i < p.D; i += 256u) { s += x[base + i]; }
  red[li] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li < k) { red[li] += red[li + k]; }
    workgroupBarrier();
  }
  let media = red[0] / f32(p.D);
  workgroupBarrier();
  var v = 0.0;
  for (var i = li; i < p.D; i += 256u) { let d = x[base + i] - media; v += d * d; }
  red[li] = v;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li < k) { red[li] += red[li + k]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(p.D) + p.eps);
  for (var i = li; i < p.D; i += 256u) { x[base + i] = (x[base + i] - media) * inv * w[i] + bb[i]; }
}`,

  // ---------------------------------------------------------------- per-head norm + RoPE (in place)
  norm_rope: `${PASO}
${SINCOS}
struct P { hd: u32, nh: u32, usaNorm: u32, neox: u32, eps: f32, base: f32, nrot: u32, z: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read_write> v: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
var<workgroup> red: array<f32, 256>;
var<workgroup> val: array<f32, 512>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wg.x;
  let t = wg.y;
  let base = (t * p.nh + h) * p.hd;
  var s = 0.0;
  for (var i = li; i < p.hd; i += 256u) { let a = v[base + i]; val[i] = a; s += a * a; }
  red[li] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li < k) { red[li] += red[li + k]; }
    workgroupBarrier();
  }
  if (p.usaNorm == 1u) {
    let inv = inverseSqrt(red[0] / f32(p.hd) + p.eps);
    for (var i = li; i < p.hd; i += 256u) { val[i] = val[i] * inv * w[i]; }
  }
  workgroupBarrier();
  let pos = f32(paso.pos0 + t);
  let mitad = p.nrot / 2u;
  for (var i = li; i < p.hd; i += 256u) {
    var r = val[i];
    if (i < p.nrot) {
      var j: u32; var par: u32; var a: f32; var b: f32;
      if (p.neox == 1u) {
        j = i % mitad;
        if (i < mitad) { a = val[i]; b = val[i + mitad]; } else { a = val[i - mitad]; b = val[i]; }
      } else {
        j = i / 2u;
        let i0 = i - (i % 2u);
        a = val[i0]; b = val[i0 + 1u];
      }
      let esc = select(bitcast<f32>(p.z), 1.0, p.z == 0u);
      let theta = pos * esc * pow(p.base, -2.0 * f32(j) / f32(p.nrot));
      let scv = sincos2(theta); let c = scv.y; let sn = scv.x;
      var primero: bool;
      if (p.neox == 1u) { primero = i < mitad; } else { primero = (i % 2u) == 0u; }
      if (primero) { r = a * c - b * sn; } else { r = a * sn + b * c; }
    }
    v[base + i] = r;
  }
}`,

  // ---------------------------------------------------------------- write K/V into the cache
  kv_write: `${PASO}
struct P { kvd: u32, cap: u32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> kc: array<f32>;
@group(0) @binding(5) var<storage, read_write> vc: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= paso.T * p.kvd) { return; }
  let t = i / p.kvd;
  let c = i % p.kvd;
  var slot = paso.pos0 + t;
  if (p.cap > 0u) { slot = slot % p.cap; }
  let o = slot * p.kvd + c;
  kc[o] = k[i];
  vc[o] = v[i];
}`,
  atencion: `${PASO}
struct P { hd: u32, nh: u32, nkv: u32, swa: u32, escala: f32, cap: u32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kc: array<f32>;
@group(0) @binding(4) var<storage, read> vc: array<f32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
var<workgroup> qs: array<f32, 256>;
var<workgroup> sc: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wg.x;
  let t = wg.y;
  let kvh = h / (p.nh / p.nkv);
  let kvd = p.nkv * p.hd;
  let nkc = paso.pos0 + t + 1u;
  var nk = nkc;
  if (p.z1 == 0xFFFFFFFFu) { nk = paso.T; }   // bidirectional attention (BERT): all tokens see the T tokens
  else if (p.z2 == 1u) { nk = paso.pos0 + paso.T; }   // non-causal within the batch (Gemma 3 image): they see up to the last token of the batch
  var ini = 0u;
  if (p.swa > 0u && nkc > p.swa) { ini = nkc - p.swa; }   // sliding window measured from the query position
  let qb = (t * p.nh + h) * p.hd;
  for (var i = li; i < p.hd; i += 64u) { qs[i] = q[qb + i]; }
  workgroupBarrier();
  var m = -3.0e38;
  var l = 0.0;
  var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  for (var tile = ini; tile < nk; tile += 64u) {
    let pk = tile + li;
    var s = -3.0e38;
    if (pk < nk) {
      var sl = pk;
      if (p.cap > 0u) { sl = pk % p.cap; }
      let kb = sl * kvd + kvh * p.hd;
      var dot = 0.0;
      for (var i = 0u; i < p.hd; i++) { dot += qs[i] * kc[kb + i]; }
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
    for (var c = 0u; c < 4u; c++) {
      let d = li + c * 64u;
      if (d < p.hd) {
        var a = acc[c] * corr;
        for (var j = 0u; j < nt; j++) {
          var sv = tile + j;
          if (p.cap > 0u) { sv = sv % p.cap; }
          a += sc[j] * vc[sv * kvd + kvh * p.hd + d];
        }
        acc[c] = a;
      }
    }
    m = mt;
    workgroupBarrier();
  }
  for (var c = 0u; c < 4u; c++) {
    let d = li + c * 64u;
    if (d < p.hd) { o[qb + d] = acc[c] / l; }
  }
}`,

  // ---------------------------------------------------------------- LFM2 short convolution
  conv_lfm2: `${PASO}
struct P { D: u32, L: u32, z0: u32, z1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> bcx: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read_write> st: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let c = g.x;
  if (c >= p.D) { return; }
  let nh = p.L - 1u;
  var hist = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  for (var j = 0u; j < nh; j++) { hist[j] = st[j * p.D + c]; }
  for (var t = 0u; t < paso.T; t++) {
    let fb = t * 3u * p.D;
    let bx = bcx[fb + c] * bcx[fb + 2u * p.D + c];
    var s = w[c * p.L + nh] * bx;
    for (var j = 0u; j < nh; j++) { s += w[c * p.L + j] * hist[j]; }
    y[t * p.D + c] = bcx[fb + p.D + c] * s;
    for (var j = 0u; j + 1u < nh; j++) { hist[j] = hist[j + 1u]; }
    if (nh > 0u) { hist[nh - 1u] = bx; }
  }
  for (var j = 0u; j < nh; j++) { st[j * p.D + c] = hist[j]; }
}`,

  // ---------------------------------------------------------------- SwiGLU: g = silu(g) * u
  swiglu: `${PASO}
struct P { F: u32, z0: u32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read_write> g: array<f32>;
@group(0) @binding(3) var<storage, read> u: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gi: vec3<u32>) {
  let i = gi.x;
  if (i >= paso.T * p.F) { return; }
  let a = g[i];
  g[i] = a / (1.0 + exp(-a)) * u[i];
}`,

  // ---------------------------------------------------------------- embeddings
  embed_q8: `${PASO}
struct P { D: u32, z0: u32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> qs: array<u32>;
@group(0) @binding(3) var<storage, read> ds: array<f32>;
@group(0) @binding(4) var<storage, read> ids: array<u32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= paso.T * p.D) { return; }
  let t = i / p.D;
  let k = i % p.D;
  let e = ids[t] * p.D + k;
  let q = qs[e / 4u];
  let sh = (3u - (e % 4u)) * 8u;
  let v = f32(bitcast<i32>(q << sh) >> 24u);
  y[i] = v * ds[e / 32u];
}`,
  embed_f32: `${PASO}
struct P { D: u32, z0: u32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= paso.T * p.D) { return; }
  let t = i / p.D;
  y[i] = w[ids[t] * p.D + (i % p.D)];
}`,

  // ---------------------------------------------------------------- per-row bias addition: y[t][r] += b[r]
  bias: `${PASO}
struct P { N: u32, z0: u32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = g.x + g.y * nwg.x * 256u;
  if (i >= paso.T * p.N) { return; }
  y[i] = y[i] + b[i % p.N];
}`,

  // ---------------------------------------------------------------- x = x * scale (T rows of D)
  escalar: `${PASO}
struct P { D: u32, escala: f32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= paso.T * p.D) { return; }
  x[i] = x[i] * p.escala;
}`,

  // ---------------------------------------------------------------- x = x + scale * t (scaled residual, Granite)
  suma_esc: `${PASO}
struct P { D: u32, escala: f32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read> t: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = g.x + g.y * nwg.x * 256u;
  if (i >= paso.T * p.D) { return; }
  x[i] = x[i] + p.escala * t[i];
}`,
};


// ---------------------------------------------------------------- vision kernels (ViT)
// Tiled matmul (64 rows x 32 tokens) with F16 weights (two per u32) or F32. K a multiple of 4 (a multiple of 32 is not needed:
// the last step is padded with zeros). x: [T][K] in f32.
function matmulVision(f16) {
  const wbind = f16 ? "array<u32>" : "array<vec4<f32>>";
  const wload = f16
    ? `let b2 = grow * (p.K / 2u) + s * 16u + v * 2u;
      let a = unpack2x16float(w[b2]); let b = unpack2x16float(w[b2 + 1u]);
      wv = vec4<f32>(a.x, a.y, b.x, b.y);`
    : `wv = w[grow * (p.K / 4u) + s * 8u + v];`;
  return `${PASO}
struct P { N: u32, K: u32, acc: u32, tfijo: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> w: ${wbind};
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> wt: array<f32, 2080>;
var<workgroup> xt: array<f32, 1024>;
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
  let ns = (p.K + 31u) / 32u;
  for (var s = 0u; s < ns; s++) {
    for (var h = 0u; h < 2u; h++) {
      let v = part + h * 4u;
      var wv = vec4<f32>(0.0);
      if (s * 8u + v < kv) {
        ${wload}
      }
      let kb = v * 4u;
      wt[(kb + 0u) * 65u + lrow] = wv.x;
      wt[(kb + 1u) * 65u + lrow] = wv.y;
      wt[(kb + 2u) * 65u + lrow] = wv.z;
      wt[(kb + 3u) * 65u + lrow] = wv.w;
    }
    var xv = vec4<f32>(0.0);
    if (s * 8u + xk < kv) { xv = x[xrow + s * 8u + xk]; }
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
  salida(r0 + tx * 4u + 0u, t0 + ty * 2u + 0u, c00, T); salida(r0 + tx * 4u + 0u, t0 + ty * 2u + 1u, c01, T);
  salida(r0 + tx * 4u + 1u, t0 + ty * 2u + 0u, c10, T); salida(r0 + tx * 4u + 1u, t0 + ty * 2u + 1u, c11, T);
  salida(r0 + tx * 4u + 2u, t0 + ty * 2u + 0u, c20, T); salida(r0 + tx * 4u + 2u, t0 + ty * 2u + 1u, c21, T);
  salida(r0 + tx * 4u + 3u, t0 + ty * 2u + 0u, c30, T); salida(r0 + tx * 4u + 3u, t0 + ty * 2u + 1u, c31, T);
}`;
}
WGSL.matmul_vis_f16 = matmulVision(true);
WGSL.matmul_vis_f32 = matmulVision(false);

// Out-of-place LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b  (x untouched: it is the residual)
WGSL.layernorm_o = `${PASO}
struct P { D: u32, z0: u32, eps: f32, z1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<f32>;
@group(0) @binding(4) var<storage, read> bb: array<f32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let base = wg.x * p.D;
  if (wg.x >= paso.T) { return; }
  var s = 0.0;
  for (var i = li; i < p.D; i += 256u) { s += x[base + i]; }
  red[li] = s;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li < k) { red[li] += red[li + k]; }
    workgroupBarrier();
  }
  let media = red[0] / f32(p.D);
  workgroupBarrier();
  var v = 0.0;
  for (var i = li; i < p.D; i += 256u) { let d = x[base + i] - media; v += d * d; }
  red[li] = v;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li < k) { red[li] += red[li + k]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(p.D) + p.eps);
  for (var i = li; i < p.D; i += 256u) { y[base + i] = (x[base + i] - media) * inv * w[i] + bb[i]; }
}`;

// In-place activation over T*N values: mode 0 = GELU (tanh), 1 = SiLU, 2 = fast GELU (x * sigmoid(1.702 x))
WGSL.activacion = `${PASO}
struct P { N: u32, modo: u32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = g.x + g.y * nwg.x * 256u;
  if (i >= paso.T * p.N) { return; }
  let a = y[i];
  if (p.modo == 0u) {
    let z = clamp(0.79788456 * (a + 0.044715 * a * a * a), -20.0, 20.0);
    y[i] = 0.5 * a * (1.0 + tanh(z));
  } else if (p.modo == 1u) {
    y[i] = a / (1.0 + exp(-a));
  } else {
    y[i] = a / (1.0 + exp(clamp(-1.702 * a, -80.0, 80.0)));
  }
}`;

// Pixel-unshuffle (LFM2-VL): input [gh*gw][E] in row order; output [(gh/m)*(gw/m)][m*m*E] with
// feature = (y%m)*m*E + (x%m)*E + e, tokens in row order.
WGSL.desbaraja = `${PASO}
struct P { E: u32, gw: u32, m: u32, z: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= paso.T * p.E) { return; }
  let F = p.m * p.m * p.E;
  let r = i / F;
  let f = i % F;
  let j = f / p.E;
  let e = f % p.E;
  let wo = p.gw / p.m;
  let yo = r / wo;
  let xo = r % wo;
  let yy = yo * p.m + j / p.m;
  let xx = xo * p.m + j % p.m;
  y[i] = x[(yy * p.gw + xx) * p.E + e];
}`;

// Injection of external embeddings (images): x[t] = ext[t] where mascara[t] != 0
WGSL.mezcla_emb = `${PASO}
struct P { D: u32, z0: u32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read_write> x: array<f32>;
@group(0) @binding(3) var<storage, read> ext: array<f32>;
@group(0) @binding(4) var<storage, read> mascara: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= paso.T * p.D) { return; }
  if (mascara[i / p.D] != 0u) { x[i] = ext[i]; }
}`;


// Bidirectional ViT attention without a cache: q,k,v,o = [T][nh*hd]; queries [toff, toff+despachadas). The chunks allow
// splitting an image of 4096 patches across several submits (~2 s per-submit limit on Windows).
WGSL.atencion_bi = `${PASO}
struct P { hd: u32, nh: u32, toff: u32, z0: u32, escala: f32, z1: u32, z2: u32, z3: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kc: array<f32>;
@group(0) @binding(4) var<storage, read> vc: array<f32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
var<workgroup> qs: array<f32, 256>;
var<workgroup> sc: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wg.x;
  let t = wg.y + p.toff;
  let kvd = p.nh * p.hd;
  let nk = paso.T;
  let qb = t * kvd + h * p.hd;
  for (var i = li; i < p.hd; i += 64u) { qs[i] = q[qb + i]; }
  workgroupBarrier();
  var m = -3.0e38;
  var l = 0.0;
  var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
  for (var tile = 0u; tile < nk; tile += 64u) {
    let pk = tile + li;
    var s = -3.0e38;
    if (pk < nk) {
      let kb = pk * kvd + h * p.hd;
      var dot = 0.0;
      for (var i = 0u; i < p.hd; i++) { dot += qs[i] * kc[kb + i]; }
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
    for (var c = 0u; c < 4u; c++) {
      let d = li + c * 64u;
      if (d < p.hd) {
        var a = acc[c] * corr;
        for (var j = 0u; j < nt; j++) { a += sc[j] * vc[(tile + j) * kvd + h * p.hd + d]; }
        acc[c] = a;
      }
    }
    m = mt;
    workgroupBarrier();
  }
  for (var c = 0u; c < 4u; c++) {
    let d = li + c * 64u;
    if (d < p.hd) { o[qb + d] = acc[c] / l; }
  }
}`;

// Average over k x k windows on the patch grid (Gemma 3): input [gh*gw][E], output [(gh/k)*(gw/k)][E]
WGSL.pool_avg = `${PASO}
struct P { E: u32, gw: u32, k: u32, z: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = g.x + g.y * nwg.x * 256u;
  if (i >= paso.T * p.E) { return; }
  let r = i / p.E;
  let e = i % p.E;
  let wo = p.gw / p.k;
  let yo = r / wo;
  let xo = r % wo;
  var s = 0.0;
  for (var a = 0u; a < p.k; a++) {
    for (var b = 0u; b < p.k; b++) {
      s += x[((yo * p.k + a) * p.gw + xo * p.k + b) * p.E + e];
    }
  }
  y[i] = s / f32(p.k * p.k);
}`;

Object.assign(WGSL, WGSL_K, WGSL_DS);
const KERNELS_K = new Set([...Object.keys(WGSL_K), ...Object.keys(WGSL_DS), "matmul_vis_f16", "matmul_vis_f32", "layernorm_o", "activacion", "desbaraja", "mezcla_emb", "atencion_bi", "pool_avg", "mm_estr", "softmax_filas", "trans_v"]);   // compiled on request (vision)

// ====================================================================
//  GGUF data conversion
// ====================================================================
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

// ---------------------------------------------------------------- ViT attention via matrices (Gemma 3)
// S[h] = scale * Q_h K_h^T  ->  row-wise softmax  ->  O_h = S[h] V_h, with the same tiled matmul (64 rows x 32
// queries) as the rest of the encoder. "Strided" matmul: w, x, y with a per-head offset (wh, xh, yh) and row strides
// (ws, xs, ys); everything in vec4 units except y (floats). K and the strides must be multiples of 4.
WGSL.mm_estr = `${PASO}
struct P { N: u32, K: u32, wh: u32, ws: u32, xh: u32, xs: u32, xbase: u32, yh: u32, ys: u32, ybase: u32, escala: f32, z: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> w: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;
var<workgroup> wt: array<f32, 2080>;
var<workgroup> xt: array<f32, 1024>;
fn salida(head: u32, r: u32, tk: u32, val: f32, T: u32) {
  if (r < p.N && tk < T) { y[p.ybase + head * p.yh + tk * p.ys + r] = val * p.escala; }
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let T = paso.T;
  let head = wg.y;
  let r0 = wg.x * 64u;
  let t0 = wg.z * 32u;
  let kv = p.K / 4u;
  let lrow = li >> 2u;
  let part = li & 3u;
  let grow = min(r0 + lrow, p.N - 1u);
  let wrow = p.wh * head + grow * p.ws;
  let xtok = li >> 3u;
  let xk = li & 7u;
  let xrow = p.xbase + p.xh * head + min(t0 + xtok, T - 1u) * p.xs;
  let tx = li & 15u;
  let ty = li >> 4u;
  var c00 = 0.0; var c01 = 0.0; var c10 = 0.0; var c11 = 0.0;
  var c20 = 0.0; var c21 = 0.0; var c30 = 0.0; var c31 = 0.0;
  let ns = (p.K + 31u) / 32u;
  for (var s = 0u; s < ns; s++) {
    for (var h = 0u; h < 2u; h++) {
      let v = part + h * 4u;
      var wv = vec4<f32>(0.0);
      if (s * 8u + v < kv) { wv = w[wrow + s * 8u + v]; }
      let kb = v * 4u;
      wt[(kb + 0u) * 65u + lrow] = wv.x;
      wt[(kb + 1u) * 65u + lrow] = wv.y;
      wt[(kb + 2u) * 65u + lrow] = wv.z;
      wt[(kb + 3u) * 65u + lrow] = wv.w;
    }
    var xv = vec4<f32>(0.0);
    if (s * 8u + xk < kv) { xv = x[xrow + s * 8u + xk]; }
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
  salida(head, r0 + tx * 4u + 0u, t0 + ty * 2u + 0u, c00, T); salida(head, r0 + tx * 4u + 0u, t0 + ty * 2u + 1u, c01, T);
  salida(head, r0 + tx * 4u + 1u, t0 + ty * 2u + 0u, c10, T); salida(head, r0 + tx * 4u + 1u, t0 + ty * 2u + 1u, c11, T);
  salida(head, r0 + tx * 4u + 2u, t0 + ty * 2u + 0u, c20, T); salida(head, r0 + tx * 4u + 2u, t0 + ty * 2u + 1u, c21, T);
  salida(head, r0 + tx * 4u + 3u, t0 + ty * 2u + 0u, c30, T); salida(head, r0 + tx * 4u + 3u, t0 + ty * 2u + 1u, c31, T);
}`;

// In-place row-wise softmax: row = (head wg.y, query wg.x) of length L, with the head at yh floats
WGSL.softmax_filas = `${PASO}
struct P { L: u32, yh: u32, z0: u32, z1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read_write> s: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  _ = paso.T;
  let base = wg.y * p.yh + wg.x * p.L;
  var m = -3.0e38;
  for (var i = li; i < p.L; i += 256u) { m = max(m, s[base + i]); }
  red[li] = m;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li < k) { red[li] = max(red[li], red[li + k]); }
    workgroupBarrier();
  }
  let mx = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var i = li; i < p.L; i += 256u) { let e = exp(s[base + i] - mx); s[base + i] = e; sum += e; }
  red[li] = sum;
  workgroupBarrier();
  for (var k = 128u; k > 0u; k >>= 1u) {
    if (li < k) { red[li] += red[li + k]; }
    workgroupBarrier();
  }
  let inv = 1.0 / red[0];
  for (var i = li; i < p.L; i += 256u) { s[base + i] = s[base + i] * inv; }
}`;

// V [T][kvd] -> Vt [kvd][T] (so that P·V reads the "weights" contiguously)
WGSL.trans_v = `${PASO}
struct P { kvd: u32, z0: u32, z1: u32, z2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> vt: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = g.x + g.y * nwg.x * 256u;
  if (i >= paso.T * p.kvd) { return; }
  let t = i / p.kvd;
  let c = i % p.kvd;
  vt[c * paso.T + t] = v[i];
}`;

// Bidirectional ViT attention with the head dimension fixed in the code (hd <= 128): each thread carries ONE query
// and walks over all the keys with online softmax; keys and values are loaded in blocks of 16 into shared memory
// (coalesced reads and no per-key barriers). Queries [toff, ...) per submit. q,k,v,o = [T][nh*hd].
export function registrarAtencionBi(hd) {
  const nombre = `atencion_bi_${hd}`;
  if (WGSL[nombre]) return nombre;
  WGSL[nombre] = `${PASO}
struct P { nh: u32, toff: u32, escala: f32, z0: u32 };
const HD: u32 = ${hd}u;
const BK: u32 = 16u;
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<uniform> paso: Paso;
@group(0) @binding(2) var<storage, read> q: array<f32>;
@group(0) @binding(3) var<storage, read> kc: array<f32>;
@group(0) @binding(4) var<storage, read> vc: array<f32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
var<workgroup> ks: array<f32, ${16 * hd}>;
var<workgroup> vs: array<f32, ${16 * hd}>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let h = wg.y;
  let kvd = p.nh * HD;
  let tq = p.toff + wg.x * 64u + li;
  let valida = tq < paso.T;
  let tl = select(0u, tq, valida);
  var qv: array<f32, ${hd}>;
  var acc: array<f32, ${hd}>;
  let qb = tl * kvd + h * HD;
  for (var i = 0u; i < HD; i++) { qv[i] = q[qb + i]; acc[i] = 0.0; }
  var m = -3.0e38;
  var l = 0.0;
  let nk = paso.T;
  for (var tile = 0u; tile < nk; tile += BK) {
    let nt = min(BK, nk - tile);
    for (var idx = li; idx < nt * HD; idx += 64u) {
      let j = idx / HD;
      let d = idx % HD;
      let src = (tile + j) * kvd + h * HD + d;
      ks[idx] = kc[src];
      vs[idx] = vc[src];
    }
    workgroupBarrier();
    var s: array<f32, 16>;
    var mt = m;
    for (var j = 0u; j < nt; j++) {
      var dot = 0.0;
      let kb = j * HD;
      for (var i = 0u; i < HD; i++) { dot += qv[i] * ks[kb + i]; }
      dot = dot * p.escala;
      s[j] = dot;
      mt = max(mt, dot);
    }
    let corr = exp(m - mt);
    l = l * corr;
    for (var i = 0u; i < HD; i++) { acc[i] = acc[i] * corr; }
    for (var j = 0u; j < nt; j++) {
      let e = exp(s[j] - mt);
      l += e;
      let vb = j * HD;
      for (var i = 0u; i < HD; i++) { acc[i] += e * vs[vb + i]; }
    }
    m = mt;
    workgroupBarrier();
  }
  if (valida) {
    for (var i = 0u; i < HD; i++) { o[qb + i] = acc[i] / l; }
  }
}`;
  KERNELS_K.add(nombre);
  return nombre;
}

export function f16af32(u8) {
  const n = u8.length >> 1;
  const src = new Uint16Array(u8.buffer, u8.byteOffset, n);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = F16[src[i]];
  return out;
}

// Q8_0: 34-byte blocks (f16 scale + 32 int8) -> qs (packed int8) + ds (f32 per block)
export function reempaquetarQ8(u8) {
  const nb = u8.length / 34;
  if (!Number.isInteger(nb)) throw new Error("invalid Q8_0 size");
  const qs = new Uint8Array(nb * 32);
  const ds = new Float32Array(nb);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.length);
  for (let b = 0; b < nb; b++) {
    const o = b * 34;
    ds[b] = F16[dv.getUint16(o, true)];
    qs.set(u8.subarray(o + 2, o + 34), b * 32);
  }
  return { qs, ds };
}

// ====================================================================
//  Device
// ====================================================================
export class GPU {
  static async crear(log) {
    if (!navigator.gpu) throw new Error("This browser has no WebGPU (navigator.gpu does not exist)");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter was found");
    const lim = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
        maxBufferSize: lim.maxBufferSize,
      },
    });
    const g = new GPU(adapter, device, log);
    return g;
  }

  constructor(adapter, device, log) {
    this.adapter = adapter;
    this.device = device;
    this.log = log || (() => {});
    const info = adapter.info || {};
    this.info = {
      summary: [info.vendor, info.architecture, info.description].filter(Boolean).join(" ") || "unknown",
      maxBuffer: device.limits.maxStorageBufferBindingSize,
      maxBufferSize: device.limits.maxBufferSize,
      vendor: info.vendor || "", arquitectura: info.architecture || "",
    };
    this.pipelines = {};
    this.perdido = false;
    device.lost.then((i) => { this.perdido = true; this.log("Dispositivo WebGPU perdido: " + i.message); });
    this.bytesUsados = 0;
  }

  async pipeline(nombre) {
    if (this.pipelines[nombre]) return this.pipelines[nombre];
    const mod = this.device.createShaderModule({ code: WGSL[nombre], label: nombre });
    const info = await mod.getCompilationInfo();
    const errores = info.messages.filter((m) => m.type === "error");
    if (errores.length) throw new Error(`WGSL ${nombre}: ` + errores.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join("; "));
    const t0 = performance.now();
    const p = await this.device.createComputePipelineAsync({ layout: "auto", compute: { module: mod, entryPoint: "main" }, label: nombre });
    const seg = (performance.now() - t0) / 1000;
    if (seg > 1.5) this.log(`Kernel ${nombre} compiled in ${seg.toFixed(1)} s`);
    this.pipelines[nombre] = p;
    return p;
  }

  // At startup only the basic kernels are compiled; the K-quant ones are requested when the model is loaded.
  async prepararPipelines(extra = []) {
    for (const n of Object.keys(WGSL)) if (!KERNELS_K.has(n)) await this.pipeline(n);
    for (const n of extra) await this.pipeline(n);
  }

  buffer(bytes, uso = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, label = "") {
    const tam = Math.max(16, Math.ceil(bytes / 4) * 4);
    if (tam > this.info.maxBuffer) throw new Error(`buffer '${label}' of ${(tam / 1e6).toFixed(0)} MB exceeds the GPU maximum (${(this.info.maxBuffer / 1e6).toFixed(0)} MB)`);
    const b = this.device.createBuffer({ size: tam, usage: uso, label });
    this.bytesUsados += tam;
    b._tam = tam;
    return b;
  }

  subir(typed, label = "") {
    const b = this.buffer(typed.byteLength, undefined, label);
    // writeBuffer requires multiples of 4 bytes
    if (typed.byteLength % 4) {
      const pad = new Uint8Array(Math.ceil(typed.byteLength / 4) * 4);
      pad.set(new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength));
      this.device.queue.writeBuffer(b, 0, pad);
    } else {
      this.device.queue.writeBuffer(b, 0, typed.buffer, typed.byteOffset, typed.byteLength);
    }
    return b;
  }

  uniforme(valores) {
    // valores: array of [type, value] with type 'u' or 'f'; padded to a multiple of 16 bytes
    const n = Math.max(4, Math.ceil(valores.length / 4) * 4);
    const ab = new ArrayBuffer(n * 4);
    const u = new Uint32Array(ab), f = new Float32Array(ab);
    valores.forEach(([t, v], i) => { if (t === "f") f[i] = v; else u[i] = v >>> 0; });
    const b = this.device.createBuffer({ size: ab.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(b, 0, ab);
    return b;
  }

  grupo(nombre, recursos) {
    const p = this.pipelines[nombre];
    return {
      pipeline: p,
      bg: this.device.createBindGroup({
        layout: p.getBindGroupLayout(0),
        entries: recursos.map((r, i) => ({ binding: i, resource: { buffer: r } })),
      }),
    };
  }

  liberar(buffers) {
    for (const b of buffers) {
      if (b && b.destroy) { this.bytesUsados -= b._tam || 0; b.destroy(); }
    }
  }
}

export function despachar(pass, g, x, y = 1, z = 1) {
  pass.setPipeline(g.pipeline);
  pass.setBindGroup(0, g.bg);
  pass.dispatchWorkgroups(x, y, z);
}

// Splits a large number of workgroups into 2D (65535 limit per dimension)
export function dim2(n) {
  if (n <= 65535) return [n, 1];
  const y = Math.ceil(n / 65535);
  return [Math.ceil(n / y), y];
}
