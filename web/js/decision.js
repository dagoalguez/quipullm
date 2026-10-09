// Decision models (Liquid AI d1): typed questions answered in ONE forward pass, no text is generated.
// The answer is read from the logits of the tokens that stand for each option.
// Follows llama.cpp (tools/server/server-decision.cpp, type lfm2-d1) and the "systemone" prompt of the model.
// Pure functions: the server does the parsing and the JSON text of the values (see decision_* in server.py),
// the engine only needs the tokenizer to choose labels and the GPU to run the prompt.

// Single-token forms of a list of strings, without duplicates (llama.cpp: get_single_tokens).
function tokensUnicos(tok, formas) {
  const out = [];
  for (const f of formas) {
    const t = tok.codificar(f, { addBos: false, especiales: false });
    if (t.length === 1 && !out.includes(t[0])) out.push(t[0]);
  }
  return out;
}

// Labels of the options of a question, in the order of q.opciones.
// Returns { textos: [string], grupos: [[token ids]] }. The text is what the prompt shows; the group, the tokens
// whose logits stand for the option (the largest logit of the group is used).
export function etiquetas(tok, q) {
  const n = q.opciones.length;
  const textos = [], grupos = [];
  if (q.tipo !== "choice") {
    for (const o of q.opciones) {
      let g;
      if (q.tipo === "score") g = tokensUnicos(tok, [o.key]);
      else if (o.key === "true") g = tokensUnicos(tok, ["yes", "Yes", "YES"]);
      else g = tokensUnicos(tok, ["no", "No", "NO"]);
      if (!g.length) { const e = new Error("decision label is not a single token: " + o.key); e.codigo = 400; throw e; }
      textos.push(o.key); grupos.push(g);
    }
    return { textos, grupos };
  }
  const todasLetras = q.opciones.every((o) => o.key.length === 1 && /^[A-Za-z]$/.test(o.key));
  const codigos = q.opciones.map((o, i) => todasLetras ? o.key : n <= 26 ? String.fromCharCode(65 + i) : String(i).padStart(2, "0"));
  const pool = [];
  for (let c = 65; c <= 90; c++) pool.push(String.fromCharCode(c));
  for (let i = 0; i < 100; i++) pool.push(String(i).padStart(2, "0"));
  for (let c = 97; c <= 122; c++) pool.push(String.fromCharCode(c));
  for (let i = 0; i < 200; i++) pool.push("#" + i);
  for (let a = 65; a <= 90; a++) for (let b = 65; b <= 90; b++) pool.push(String.fromCharCode(a, b));
  const usados = [];
  const tomar = (codigo) => {
    const t = tok.codificar(codigo, { addBos: false, especiales: false });
    if (t.length !== 1 || usados.includes(t[0])) return false;
    usados.push(t[0]);
    const grupo = [t[0]];
    for (const x of tokensUnicos(tok, [" " + codigo])) if (x !== t[0]) grupo.push(x);
    textos.push(codigo); grupos.push(grupo);
    return true;
  };
  for (const codigo of codigos) {
    let ok = tomar(codigo);
    for (let i = 0; !ok && i < pool.length; i++) ok = tomar(pool[i]);
    if (!ok) { const e = new Error(`no single-token label left for ${n} options`); e.codigo = 400; throw e; }
  }
  return { textos, grupos };
}

// The "systemone" prompt of d1 (conversion/lfm2.py of llama.cpp, follows prompt.py of the model repo).
// q.estado: text of the state or null; q.instrucciones: text; q.opciones[i]:
//   { key, desc (text of the description: the string itself, or its JSON), es_nulo, verdad (description is truthy) }
export function armarPrompt(q, textos) {
  let s = "<|startoftext|><|im_start|>user\n";
  if (q.estado !== null && q.estado !== undefined) s += q.estado + "\n\n\nQUESTION:\n";
  s += q.instrucciones;
  const ops = q.opciones;
  if (q.tipo === "choice") {
    s += "\n\nOptions:\n";
    s += ops.map((o, i) => textos[i] + " " + (o.verdad ? o.desc : o.key.split("_").join(" "))).join("\n");
    s += "\n\nReply with the option code only.";
  } else if (q.tipo === "noul") {
    if (ops.some((o) => !o.es_nulo)) {
      for (const o of ops) s += (o.key === "true" ? "\nYes: " : "\nNo: ") + (o.es_nulo ? "None" : o.desc);
    }
    s += "\n\nReply with yes or no only.";
  } else {
    s += "\n\n";
    for (const o of ops) s += o.key + " " + o.desc + "\n";
    s += "\nReply with a single digit 0-" + (ops.length - 1) + " only.";
  }
  s += "<|im_end|>\n<|im_start|>assistant\n";
  return s;
}

// Score of each option: the largest logit among the tokens of its group.
export function puntajes(logits, grupos) {
  return grupos.map((g) => { let m = -Infinity; for (const t of g) if (logits[t] > m) m = logits[t]; return m; });
}
