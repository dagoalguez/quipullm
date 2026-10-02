// Token sampling (llama.cpp order: penalties -> top_k -> top_p -> min_p -> temperature).

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class Muestreador {
  constructor(p = {}) {
    this.temp = p.temperature ?? 0.8;
    this.topK = p.top_k ?? 40;
    this.topP = p.top_p ?? 0.95;
    this.minP = p.min_p ?? 0.05;
    this.rep = p.repeat_penalty ?? 1.0;
    this.freq = p.frequency_penalty ?? 0;
    this.pres = p.presence_penalty ?? 0;
    this.ventana = 64;
    this.rand = p.seed !== null && p.seed !== undefined ? mulberry32(p.seed) : Math.random;
  }

  elegir(logits, historial) {
    const V = logits.length;
    if (historial.length && (this.rep !== 1 || this.freq || this.pres)) {
      const cuenta = new Map();
      for (const t of historial.slice(-this.ventana)) cuenta.set(t, (cuenta.get(t) || 0) + 1);
      for (const [t, n] of cuenta) {
        if (t >= V) continue;
        let l = logits[t];
        if (this.rep !== 1) l = l > 0 ? l / this.rep : l * this.rep;
        l -= n * this.freq + this.pres;
        logits[t] = l;
      }
    }
    if (!(this.temp > 0)) {
      let mejor = 0, mv = -Infinity;
      for (let i = 0; i < V; i++) if (logits[i] > mv) { mv = logits[i]; mejor = i; }
      return mejor;
    }
    // candidates (top-k with a heap; if top_k<=0 all are used)
    let cand;
    const k = this.topK > 0 ? Math.min(this.topK, V) : V;
    if (k < V) {
      const heap = [];   // min-heap by logit
      const sube = (i) => { while (i > 0) { const p = (i - 1) >> 1; if (logits[heap[p]] <= logits[heap[i]]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
      const baja = (i) => { for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < heap.length && logits[heap[l]] < logits[heap[m]]) m = l; if (r < heap.length && logits[heap[r]] < logits[heap[m]]) m = r; if (m === i) break; [heap[m], heap[i]] = [heap[i], heap[m]]; i = m; } };
      for (let i = 0; i < V; i++) {
        if (heap.length < k) { heap.push(i); sube(heap.length - 1); }
        else if (logits[i] > logits[heap[0]]) { heap[0] = i; baja(0); }
      }
      cand = heap;
    } else {
      cand = Array.from({ length: V }, (_, i) => i);
    }
    cand.sort((a, b) => logits[b] - logits[a]);
    // probabilities without temperature for top_p / min_p
    const max = logits[cand[0]];
    let probs = cand.map((i) => Math.exp(logits[i] - max));
    let s = probs.reduce((a, b) => a + b, 0);
    probs = probs.map((p) => p / s);
    let n = cand.length;
    if (this.topP < 1) {
      let acum = 0;
      for (let i = 0; i < n; i++) { acum += probs[i]; if (acum >= this.topP) { n = i + 1; break; } }
    }
    if (this.minP > 0) {
      const lim = probs[0] * this.minP;
      let m = 1;
      while (m < n && probs[m] >= lim) m++;
      n = m;
    }
    cand = cand.slice(0, n);
    // temperature and sampling
    const pt = cand.map((i) => Math.exp((logits[i] - max) / this.temp));
    const tot = pt.reduce((a, b) => a + b, 0);
    let r = this.rand() * tot;
    for (let i = 0; i < cand.length; i++) { r -= pt[i]; if (r <= 0) return cand[i]; }
    return cand[cand.length - 1];
  }
}
