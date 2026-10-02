// Loading of GGUF tensors from the server (by ranges) into the GPU.
import { reempaquetarQ8, f16af32 } from "./gpu.js";
import { TIPOS_K, subirK } from "./kquants.js";
import { subirIQ4 } from "./kds.js";

const TIPOS = { 0: "F32", 1: "F16", 8: "Q8_0", 30: "BF16" };

export class CargadorPesos {
  constructor(gpu, meta, urlArchivo, progreso) {
    this.gpu = gpu;
    this.meta = meta;
    this.url = urlArchivo;
    this.progreso = progreso || (() => {});
    this.tensores = new Map(meta.tensors.map((t) => [t.nombre, t]));
    this.totalBytes = meta.tensors.reduce((a, t) => a + (t.bytes || 0), 0);
    this.leidos = 0;
    this.buffers = [];
  }

  existe(nombre) { return this.tensores.has(nombre); }

  info(nombre) {
    const t = this.tensores.get(nombre);
    if (!t) throw new Error(`missing tensor ${nombre}`);
    return t;
  }

  // Read-ahead: while the CPU repacks one tensor, the next ones are already being read.
  // Limited window (4 reads and ~192 MB in memory) because the PC's RAM is tight.
  planificar(nombres) {
    this.cola = nombres.filter((n) => this.tensores.has(n)).map((n) => this.tensores.get(n))
      .filter((t) => t.bytes !== null && t.bytes !== undefined);
    this.previos = new Map();
    this.enVuelo = 0;
    this.retenidos = 0;
    this._avanzar();
  }

  _avanzar() {
    while (this.cola && this.cola.length && this.enVuelo < 4 && (this.retenidos < 192e6 || this.retenidos === 0)) {
      const t = this.cola.shift();
      this.enVuelo++;
      this.retenidos += t.bytes;
      const p = this._leer(t).finally(() => { this.enVuelo--; this._avanzar(); });
      p.catch(() => {});
      this.previos.set(t.nombre, p);
    }
  }

  async _leer(t) {
    const r = await fetch(this.url, { headers: { Range: `bytes=${t.abs}-${t.abs + t.bytes - 1}` } });
    if (!r.ok) throw new Error(`could not read ${t.nombre}: HTTP ${r.status}`);
    const ab = await r.arrayBuffer();
    if (ab.byteLength !== t.bytes) throw new Error(`incomplete read of ${t.nombre}`);
    this.leidos += t.bytes;
    this.progreso(this.leidos / this.totalBytes, t.nombre);
    return new Uint8Array(ab);
  }

  async bytes(t) {
    if (t.bytes === null || t.bytes === undefined) throw new Error(`unsupported tensor type (${t.tipo_nombre}) in ${t.nombre}`);
    const p = this.previos && this.previos.get(t.nombre);
    if (p) {
      this.previos.delete(t.nombre);
      try { return await p; } finally { this.retenidos -= t.bytes; this._avanzar(); }
    }
    if (this.cola) { const i = this.cola.indexOf(t); if (i >= 0) this.cola.splice(i, 1); }
    return this._leer(t);
  }

  // Vector or matrix as f32 on the GPU
  async f32(nombre) {
    const t = this.info(nombre);
    const u8 = await this.bytes(t);
    let f;
    if (t.tipo === 0) f = new Float32Array(u8.buffer, u8.byteOffset, u8.length / 4);
    else if (t.tipo === 1) f = f16af32(u8);
    else if (t.tipo === 30) {
      const s = new Uint16Array(u8.buffer, u8.byteOffset, u8.length / 2);
      f = new Float32Array(s.length);
      const v = new Uint32Array(f.buffer);
      for (let i = 0; i < s.length; i++) v[i] = s[i] << 16;
    } else if (t.tipo === 8) {
      const { qs, ds } = reempaquetarQ8(u8);
      const i8 = new Int8Array(qs.buffer);
      f = new Float32Array(i8.length);
      for (let i = 0; i < i8.length; i++) f[i] = i8[i] * ds[i >> 5];
    } else throw new Error(`${nombre}: type ${t.tipo_nombre} not supported as f32`);
    const b = this.gpu.subir(f, nombre);
    this.buffers.push(b);
    return b;
  }

  // Weight matrix [N][K]. Returns {tipo:'q8'|'f32', N, K, buffers}
  // filas: optional [from, to) to split fused tensors (qkv).
  async matriz(nombre, filas = null, u8Previo = null) {
    const t = this.info(nombre);
    const K = t.dims[0], Ntot = t.dims.slice(1).reduce((a, b) => a * b, 1);
    const [f0, f1] = filas || [0, Ntot];
    const N = f1 - f0;
    const u8 = u8Previo || await this.bytes(t);
    if (TIPOS_K[t.tipo]) {
      const info = TIPOS_K[t.tipo];
      if (K % 256) throw new Error(`${nombre}: K=${K} is not a multiple of 256 (${info.nombre})`);
      const bpr = (K / 256) * info.bytes;   // bytes per row: rows are whole blocks, so they can be split
      const b = subirK(this.gpu, filas ? u8.subarray(f0 * bpr, f1 * bpr) : u8, info, nombre);
      this.buffers.push(b);
      return { tipo: info.nombre, N, K, w: b };
    }
    if (t.tipo === 20) {   // IQ4_NL: blocks of 32 (18 bytes)
      if (K % 32) throw new Error(`${nombre}: K=${K} is not a multiple of 32`);
      if (filas) throw new Error(`${nombre}: IQ4_NL does not support splitting by rows`);
      const { qs, dh } = subirIQ4(this.gpu, u8, nombre);
      this.buffers.push(qs, dh);
      return { tipo: "iq4nl", N, K, qs, dh };
    }
    if (t.tipo === 8) {
      if (K % 32) throw new Error(`${nombre}: K=${K} is not a multiple of 32`);
      const bpr = (K / 32) * 34;
      const { qs, ds } = reempaquetarQ8(u8.subarray(f0 * bpr, f1 * bpr));
      const bq = this.gpu.subir(qs, nombre + ".qs"), bd = this.gpu.subir(ds, nombre + ".ds");
      this.buffers.push(bq, bd);
      return { tipo: "q8", N, K, qs: bq, ds: bd };
    }
    let f;
    if (t.tipo === 0) f = new Float32Array(u8.buffer, u8.byteOffset + f0 * K * 4, N * K);
    else if (t.tipo === 1) f = f16af32(u8.subarray(f0 * K * 2, f1 * K * 2));
    else if (t.tipo === 30) {
      const s = new Uint16Array(u8.buffer, u8.byteOffset + f0 * K * 2, N * K);
      f = new Float32Array(N * K);
      const v = new Uint32Array(f.buffer);
      for (let i = 0; i < s.length; i++) v[i] = s[i] << 16;
    } else {
      throw new Error(`${nombre}: quantization ${t.tipo_nombre} not supported yet (supported: Q8_0, IQ4_NL, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, F16, BF16, F32)`);
    }
    const b = this.gpu.subir(f, nombre);
    this.buffers.push(b);
    return { tipo: "f32", N, K, w: b };
  }

  tipoNombre(nombre) { return TIPOS[this.info(nombre).tipo] || this.info(nombre).tipo_nombre; }
}
