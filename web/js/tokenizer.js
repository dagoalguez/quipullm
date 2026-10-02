// Byte-level BPE tokenizer (GPT-2 / llama.cpp style), read from the GGUF.
// Supports the pre-tokenizers used by our model families.

// llama.cpp pre-tokenizers (src/llama-vocab.cpp). Each type is a list of expressions applied in
// cascade: the 1st splits the text and each following one refines the resulting pieces (including the
// gaps the previous one did not recognize), same as llama.cpp's unicode_regex_split.
const APOS = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";
const GPT2 = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)";
const LLAMA3 = APOS + "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const QWEN2 = APOS + "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const QWEN35 = APOS + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const TEKKEN = "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const DEEPSEEK3 = ["\\p{N}{1,3}", "[一-龥぀-ゟ゠-ヿ]+",
  "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"];
const DEEPSEEK_LLM = ["[\\r\\n]", "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+", "\\s?[!-/:-~！-／：-～‘-‟　-。]+", "\\s+$", "[一-龥ࠀ-一가-퟿]+", "\\p{N}+"];
const DEEPSEEK_CODER = ["[\\r\\n]", "\\s?\\p{L}+", "\\s?\\p{P}+", "[一-龥ࠀ-一가-퟿]+", "\\p{N}"];

const PRE_EXPRS = {
  llama3: [LLAMA3], llama3_sin_merges: [LLAMA3], qwen2: [QWEN2], qwen35: [QWEN35], tekken: [TEKKEN],
  gpt2: [GPT2], digitos_gpt2: ["\\p{N}", GPT2],
  deepseek3: DEEPSEEK3, deepseek_llm: DEEPSEEK_LLM, deepseek_coder: DEEPSEEK_CODER,
  por_defecto: ["[\\p{P}\\$\\+<=>\\^~\\|]+", GPT2, "\\p{N}+", "[0-9][0-9][0-9]"],
  falcon: ["[\\p{P}\\$\\+<=>\\^~\\|`]+", GPT2, "[0-9][0-9][0-9]"],
};
const PRE_REGEX = {};
for (const [k, v] of Object.entries(PRE_EXPRS)) PRE_REGEX[k] = v.map((s) => new RegExp(s, "gu"));

// value of tokenizer.ggml.pre -> type
const PRE_TIPO = {};
const registrar = (tipo, nombres) => nombres.forEach((n) => { PRE_TIPO[n] = tipo; });
registrar("llama3", ["llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral", "midm-2.0", "lfm2", "jina-v5-nano"]);
registrar("llama3_sin_merges", ["dbrx", "smaug-bpe", "glm4", "chatglm-bpe"]);
registrar("qwen2", ["qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2", "stablelm2", "hunyuan", "solar-open"]);
registrar("qwen35", ["qwen35"]);
registrar("tekken", ["tekken"]);
registrar("por_defecto", ["default"]);
registrar("gpt2", ["gpt-2", "phi-2", "jina-es", "jina-de", "gigachat", "jina-v2-es", "jina-v2-de", "a.x-4.0",
  "mellum", "modern-bert", "mpt", "olmo", "jais", "trillion", "granite-docling"]);
registrar("digitos_gpt2", ["refact", "starcoder", "command-r", "smollm", "codeshell", "exaone", "minerva-7b", "mellum2"]);
registrar("deepseek3", ["deepseek-v3", "hunyuan-dense", "joyai-llm"]);
registrar("deepseek_llm", ["deepseek-llm"]);
registrar("deepseek_coder", ["deepseek-coder"]);
registrar("falcon", ["falcon"]);
// llama.cpp: these use ignore_merges (look up the whole word in the vocabulary before applying merges)
// Types whose only expression ends in \\s+ (covers the whole text): only the matches can be iterated.
const COBERTURA_TOTAL = new Set(["llama3", "llama3_sin_merges", "qwen2", "qwen35", "tekken"]);
const IGNORA_MERGES = new Set(["llama3", "tekken"]);
// llama.cpp: only these enable BOS by default when the GGUF says nothing
const BOS_POR_DEFECTO = new Set(["llama3"]);

// Splits the text with a cascade of expressions (unicode_regex_split semantics from llama.cpp).
function partirCascada(texto, regexes) {
  let piezas = [texto];
  for (const re of regexes) {
    const nuevas = [];
    for (const p of piezas) {
      let ult = 0;
      for (const m of p.matchAll(re)) {
        if (m.index > ult) nuevas.push(p.slice(ult, m.index));
        if (m[0].length) nuevas.push(m[0]);
        ult = m.index + m[0].length;
      }
      if (ult < p.length) nuevas.push(p.slice(ult));
    }
    piezas = nuevas;
  }
  return piezas;
}

// Text of end-of-turn tokens recognized even if the GGUF does not mark them as EOS.
const TEXTOS_FIN = ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|return|>", "<|end_of_text|>"];

function bytesAUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  }
  const b2u = new Array(256), u2b = new Map();
  bs.forEach((b, i) => { b2u[b] = String.fromCodePoint(cs[i]); u2b.set(String.fromCodePoint(cs[i]), b); });
  return { b2u, u2b };
}

const ESPACIO_WPM = new Set([9, 10, 11, 12, 13, 32, 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
  0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A]);
const RE_CONTROL = /^\p{C}$/u, RE_NO_ASIGNADO = /^\p{Cn}$/u, RE_MARCA = /^\p{M}$/u, RE_PUNT = /^\p{P}$/u, RE_SIMB = /^\p{S}$/u;
const esChino = (c) => (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) || (c >= 0x20000 && c <= 0x2A6DF) ||
  (c >= 0x2A700 && c <= 0x2B73F) || (c >= 0x2B740 && c <= 0x2B81F) || (c >= 0x2B920 && c <= 0x2CEAF) ||
  (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F);

export class Tokenizador {
  constructor(kv) {
    const modelo = kv["tokenizer.ggml.model"];
    if (modelo !== "gpt2" && modelo !== "llama" && modelo !== "bert") throw new Error(`tokenizer '${modelo}' not supported in this version (only gpt2 BPE, llama SentencePiece and bert WordPiece)`);
    this.spm = modelo === "llama";
    this.wpm = modelo === "bert";
    this.tokens = kv["tokenizer.ggml.tokens"];
    this.tipos = kv["tokenizer.ggml.token_type"] || [];
    const addBos = kv["tokenizer.ggml.add_bos_token"];
    if (this.wpm) {
      this.pre = "wpm";
      this.ignoraMerges = false;
      this.addBos = addBos === undefined || addBos === null ? true : !!addBos;
      const sep = kv["tokenizer.ggml.add_sep_token"];
      this.addSep = sep === undefined || sep === null ? true : !!sep;
      this.sepId = kv["tokenizer.ggml.seperator_token_id"] ?? kv["tokenizer.ggml.separator_token_id"] ?? 102;
      this.unk = kv["tokenizer.ggml.unknown_token_id"] ?? 100;
      const bos = kv["tokenizer.ggml.bos_token_id"];
      if (bos === undefined || bos === null) kv = { ...kv, "tokenizer.ggml.bos_token_id": 101 };
      const minus = kv["tokenizer.ggml.normalizer.lowercase"];
      this.minuscula = minus === undefined || minus === null ? true : !!minus;
      const acc = kv["tokenizer.ggml.normalizer.strip_accents"];
      this.sinAcentos = acc === undefined || acc === null ? this.minuscula : !!acc;
    } else if (this.spm) {
      this.pre = "spm";
      this.ignoraMerges = false;
      this.scores = kv["tokenizer.ggml.scores"] || [];
      this.addEspacio = kv["tokenizer.ggml.add_space_prefix"] !== false;   // llama.cpp: true if the GGUF says nothing otherwise
      this.addBos = addBos === undefined || addBos === null ? true : !!addBos;
    } else {
      const pre = kv["tokenizer.ggml.pre"] || "default";
      if (!PRE_TIPO[pre]) throw new Error(`pre-tokenizer '${pre}' not supported in this version`);
      this.pre = PRE_TIPO[pre];
      this.regexes = PRE_REGEX[this.pre];
      this.ignoraMerges = IGNORA_MERGES.has(this.pre);
      this.cobertura = COBERTURA_TOTAL.has(this.pre);
      this.addBos = addBos === undefined || addBos === null ? BOS_POR_DEFECTO.has(this.pre) : !!addBos;
    }
    this.bos = kv["tokenizer.ggml.bos_token_id"];
    this.eos = kv["tokenizer.ggml.eos_token_id"];
    if (this.wpm) { this.maxLen = 0; }
    this.vocab = new Map();
    this.tokens.forEach((t, i) => { if (!this.vocab.has(t)) this.vocab.set(t, i); });
    this.rangos = new Map();
    (kv["tokenizer.ggml.merges"] || []).forEach((m, i) => this.rangos.set(m, i));
    const { b2u, u2b } = bytesAUnicode();
    this.b2u = b2u; this.u2b = u2b;
    this.enc = new TextEncoder();
    // Special tokens (control=3, user-defined=4) used to split the text.
    this.especiales = [];
    this.tipos.forEach((t, i) => {
      if ((t === 3 || t === 4 || (this.spm && t === 2)) && this.tokens[i]) this.especiales.push([this.tokens[i], i]);
    });
    this.especiales.sort((a, b) => b[0].length - a[0].length);
    // End of generation
    this.fin = new Set();
    for (const k of ["tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"]) {
      if (kv[k] !== undefined && kv[k] !== null) this.fin.add(kv[k]);
    }
    for (const t of TEXTOS_FIN) if (this.vocab.has(t)) this.fin.add(this.vocab.get(t));
    this.cache = new Map();
    if (this.wpm) for (const t of this.tokens) this.maxLen = Math.max(this.maxLen, Array.from(t).length);
    if (this.spm) {
      this.bytesTok = new Array(256).fill(-1);
      for (let b = 0; b < 256; b++) {
        const id = this.vocab.get("<0x" + b.toString(16).toUpperCase().padStart(2, "0") + ">");
        if (id !== undefined) this.bytesTok[b] = id;
      }
    }
  }

  // ---------------------------------------------------------------- encode
  codificar(texto, { addBos = null, especiales = true } = {}) {
    const ids = [];
    const usarBos = addBos === null ? this.addBos : addBos;
    if (usarBos && this.bos !== undefined && this.bos !== null) ids.push(this.bos);
    const trozos = especiales ? this._partirEspeciales(texto) : [texto];
    if (this.spm) { this._codificarSpm(trozos, ids); return ids; }
    if (this.wpm) {
      for (const tr of trozos) {
        if (typeof tr === "number") ids.push(tr);
        else if (tr) this._wpm(tr, ids);
      }
      if (this.addSep && this.sepId !== null && this.sepId !== undefined) ids.push(this.sepId);
      return ids;
    }
    for (const tr of trozos) {
      if (typeof tr === "number") { ids.push(tr); continue; }
      if (!tr) continue;
      if (this.cobertura) { for (const m of tr.matchAll(this.regexes[0])) this._bpePalabra(m[0], ids); }
      else for (const w of partirCascada(tr, this.regexes)) this._bpePalabra(w, ids);
    }
    return ids;
  }

  // WordPiece as in llama.cpp (llm_tokenizer_wpm_session): normalizes, splits into words and looks for the longest piece
  // (words carry the ▁ prefix in the GGUF vocabulary). A word with no complete match -> [UNK].
  _wpmPalabras(texto) {
    let cps = Array.from(texto);
    if (this.sinAcentos) cps = Array.from(texto.normalize("NFD"));
    const palabras = [""];
    for (const ch of cps) {
      const c = ch.codePointAt(0);
      if (ESPACIO_WPM.has(c)) { if (palabras[palabras.length - 1].length) palabras.push(""); continue; }
      if (c === 0 || c === 0xFFFD || (RE_CONTROL.test(ch) && !RE_NO_ASIGNADO.test(ch))) continue;
      if (this.sinAcentos && RE_MARCA.test(ch)) continue;
      let s = ch;
      if (this.minuscula) { const l = ch.toLowerCase(); if (Array.from(l).length === 1) s = l; }
      if (RE_PUNT.test(ch) || (c < 0x7F && RE_SIMB.test(ch)) || esChino(c)) {
        if (palabras[palabras.length - 1].length) palabras.push("");
        palabras[palabras.length - 1] = s;
        palabras.push("");
      } else palabras[palabras.length - 1] += s;
    }
    if (!palabras[palabras.length - 1].length) palabras.pop();
    return palabras;
  }

  _wpm(texto, ids) {
    for (const palabra of this._wpmPalabras(texto)) {
      if (!palabra.length) continue;
      const cps = Array.from("\u2581" + palabra), n = cps.length;
      const ini = ids.length;
      for (let i = 0; i < n; i++) {
        let hallado = false;
        for (let j = Math.min(n, i + this.maxLen + 1); j > i; j--) {
          const id = this.vocab.get(cps.slice(i, j).join(""));
          if (id !== undefined) { ids.push(id); hallado = true; i = j - 1; break; }
        }
        if (!hallado) { ids.length = ini; break; }
      }
      if (ids.length === ini) ids.push(this.unk);
    }
  }

  // SentencePiece as in llama.cpp (llm_tokenizer_spm_session): split into characters and merge pairs by
  // vocabulary score (highest score first; on ties, the leftmost).
  _codificarSpm(trozos, ids) {
    let previoEspecial = true;
    for (const tr of trozos) {
      if (typeof tr === "number") { ids.push(tr); previoEspecial = true; continue; }
      if (!tr) continue;
      let texto = (this.addEspacio && previoEspecial) ? " " + tr : tr;
      texto = texto.replace(/ /g, "\u2581");
      this._spm(texto, ids);
      previoEspecial = false;
    }
  }

  _spm(texto, salida) {
    const sim = [];
    for (const ch of texto) sim.push({ t: ch, prev: sim.length - 1, next: sim.length + 1, vivo: true });
    if (!sim.length) return;
    sim[sim.length - 1].next = -1;
    const heap = [];
    const menor = (a, b) => a.score > b.score || (a.score === b.score && a.left < b.left);   // a goes before b
    const subir = (i) => { while (i > 0) { const p = (i - 1) >> 1; if (menor(heap[i], heap[p])) { [heap[i], heap[p]] = [heap[p], heap[i]]; i = p; } else break; } };
    const bajar = (i) => {
      for (;;) {
        let m = i; const l = 2 * i + 1, r = l + 1;
        if (l < heap.length && menor(heap[l], heap[m])) m = l;
        if (r < heap.length && menor(heap[r], heap[m])) m = r;
        if (m === i) break;
        [heap[i], heap[m]] = [heap[m], heap[i]]; i = m;
      }
    };
    const agregar = (l, r) => {
      if (l === -1 || r === -1) return;
      const txt = sim[l].t + sim[r].t;
      const id = this.vocab.get(txt);
      if (id === undefined) return;
      heap.push({ left: l, right: r, score: this.scores[id] ?? 0, len: txt.length });
      subir(heap.length - 1);
    };
    for (let i = 1; i < sim.length; i++) agregar(i - 1, i);
    while (heap.length) {
      const top = heap[0];
      const ult = heap.pop();
      if (heap.length) { heap[0] = ult; bajar(0); }
      const L = sim[top.left], R = sim[top.right];
      if (!L.vivo || !R.vivo || L.t.length + R.t.length !== top.len) continue;   // already merged
      L.t += R.t; R.vivo = false;
      L.next = R.next;
      if (R.next >= 0) sim[R.next].prev = top.left;
      agregar(L.prev, top.left);
      agregar(top.left, L.next);
    }
    for (let i = 0; i !== -1; i = sim[i].next) {
      const s = sim[i].t;
      const id = this.vocab.get(s);
      if (id !== undefined) { salida.push(id); continue; }
      for (const b of this.enc.encode(s)) {   // no token: one byte token <0xXX> per byte
        if (this.bytesTok[b] >= 0) salida.push(this.bytesTok[b]);
        else { const c = this.vocab.get(String.fromCharCode(b)); if (c !== undefined) salida.push(c); }
      }
    }
  }

  _partirEspeciales(texto) {
    let partes = [texto];
    for (const [tok, id] of this.especiales) {
      const nuevas = [];
      for (const p of partes) {
        if (typeof p !== "string" || !p.includes(tok)) { nuevas.push(p); continue; }
        const seg = p.split(tok);
        seg.forEach((s, i) => { if (i) nuevas.push(id); if (s) nuevas.push(s); });
      }
      partes = nuevas;
    }
    return partes;
  }

  _bpePalabra(palabra, salida) {
    const c = this.cache.get(palabra);
    if (c) { for (const x of c) salida.push(x); return; }
    const bytes = this.enc.encode(palabra);
    let mapeada = "";
    for (const b of bytes) mapeada += this.b2u[b];
    let res;
    if (this.ignoraMerges && this.vocab.has(mapeada)) {
      res = [this.vocab.get(mapeada)];
    } else {
      let sim = Array.from(mapeada);
      while (sim.length > 1) {
        let mejor = Infinity, pos = -1;
        for (let i = 0; i < sim.length - 1; i++) {
          const r = this.rangos.get(sim[i] + " " + sim[i + 1]);
          if (r !== undefined && r < mejor) { mejor = r; pos = i; }
        }
        if (pos < 0) break;
        sim.splice(pos, 2, sim[pos] + sim[pos + 1]);
      }
      res = [];
      for (const s of sim) {
        const id = this.vocab.get(s);
        if (id !== undefined) res.push(id);
        else for (const ch of s) { const b = this.vocab.get(ch); if (b !== undefined) res.push(b); }
      }
    }
    if (this.cache.size < 50000) this.cache.set(palabra, res);
    for (const x of res) salida.push(x);
  }

  // ---------------------------------------------------------------- decode
  esFin(id) { return this.fin.has(id); }

  bytesDe(id) {
    const t = this.tokens[id];
    if (t === undefined) return new Uint8Array(0);
    if (id === this.bos) return new Uint8Array(0);
    const tipo = this.tipos[id];
    if (tipo === 3 || tipo === 4) return this.enc.encode(t);   // special: text as is
    if (this.spm) {
      if (tipo === 6) {
        const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(t);
        if (m) return new Uint8Array([parseInt(m[1], 16)]);
      }
      if (tipo === 5) return new Uint8Array(0);
      return this.enc.encode(t.replace(/\u2581/g, " "));
    }
    if (tipo === 6) {                                          // <0xXX>
      const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(t);
      if (m) return new Uint8Array([parseInt(m[1], 16)]);
    }
    const out = [];
    for (const ch of t) {
      const b = this.u2b.get(ch);
      if (b !== undefined) out.push(b); else for (const x of this.enc.encode(ch)) out.push(x);
    }
    return new Uint8Array(out);
  }

  decodificar(ids) {
    const partes = ids.map((i) => this.bytesDe(i));
    const n = partes.reduce((a, p) => a + p.length, 0);
    const buf = new Uint8Array(n);
    let o = 0;
    for (const p of partes) { buf.set(p, o); o += p.length; }
    return new TextDecoder("utf-8").decode(buf);
  }

  // Incremental decoder: handles UTF-8 characters split across tokens.
  decodificadorIncremental() {
    const dec = new TextDecoder("utf-8");
    return {
      agregar: (id) => dec.decode(this.bytesDe(id), { stream: true }),
      terminar: () => dec.decode(),
    };
  }
}
