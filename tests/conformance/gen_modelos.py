"""Generates synthetic LFM2 models (GGUF) and their llama.cpp references."""
import json, os, sys
import numpy as np
import regex
import gguf
from gguf.quants import quantize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "modelos_test")
os.makedirs(OUT, exist_ok=True)

CORPUS = """La base de datos guarda la información de la institución de forma ordenada.
Una base de datos permite consultar, actualizar y respaldar los registros.
El servidor responde a las peticiones de la red local en el puerto 1234.
¿Cuánto es 12 por 11? Son 132. La capital de Francia es París.
Los modelos de lenguaje generan texto token por token con la tarjeta gráfica.
def fibonacci(n):\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n
Juan tiene 34 años y trabaja en el área de datos. Él usa Python y Spark todos los días.
""" * 3

PAT = regex.compile(r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


def b2u():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def entrenar_bpe(n_merges=260):
    m = b2u()
    palabras = {}
    for w in PAT.findall(CORPUS):
        s = tuple(m[b] for b in w.encode("utf-8"))
        palabras[s] = palabras.get(s, 0) + 1
    merges = []
    for _ in range(n_merges):
        pares = {}
        for w, c in palabras.items():
            for a, b in zip(w, w[1:]):
                pares[(a, b)] = pares.get((a, b), 0) + c
        if not pares:
            break
        (a, b), _ = max(pares.items(), key=lambda kv: (kv[1], kv[0]))
        merges.append(a + " " + b)
        nuevas = {}
        for w, c in palabras.items():
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(w[i]); i += 1
            nuevas[tuple(out)] = nuevas.get(tuple(out), 0) + c
        palabras = nuevas
    especiales = [("<|pad|>", 3), ("<|startoftext|>", 3), ("<|endoftext|>", 3), ("<|im_start|>", 3),
                  ("<|im_end|>", 3), ("<think>", 4), ("</think>", 4)]
    tokens = [t for t, _ in especiales]
    tipos = [ty for _, ty in especiales]
    for b in range(256):
        tokens.append(m[b]); tipos.append(1)
    for mg in merges:
        t = mg.replace(" ", "")
        if t not in tokens:
            tokens.append(t); tipos.append(1)
    return tokens, tipos, merges


def crear(nombre, tipo_mat="Q8_0", swa=0, fusion_qkv=False, add_bos=True, semilla=0, extra=None, escala_salida=1.0):
    rng = np.random.default_rng(semilla)
    tokens, tipos, merges = entrenar_bpe()
    V, D, F, nh, hd, L = len(tokens), 64, 128, 4, 16, 3
    capas_kv = [0, 0, 2, 0, 2, 0]
    ruta = os.path.join(OUT, nombre + ".gguf")
    w = gguf.GGUFWriter(ruta, "lfm2")
    w.add_name(nombre)
    w.add_block_count(len(capas_kv))
    w.add_context_length(4096)
    w.add_embedding_length(D)
    w.add_feed_forward_length(F)
    w.add_head_count(nh)
    w.add_head_count_kv(capas_kv)
    w.add_rope_freq_base(1000000.0)
    w.add_layer_norm_rms_eps(1e-5)
    w.add_uint32("lfm2.shortconv.l_cache", L)
    w.add_vocab_size(V)
    if swa:
        w.add_sliding_window(swa)
    w.add_file_type(7 if tipo_mat == "Q8_0" else 1)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("lfm2")
    w.add_token_list(tokens)
    w.add_token_types(tipos)
    w.add_token_merges(merges)
    w.add_bos_token_id(1)
    w.add_eos_token_id(4)
    w.add_pad_token_id(0)
    w.add_add_bos_token(add_bos)
    w.add_chat_template("{{bos_token}}{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}")

    def mat(nombre_t, N, K, escala=1.0):
        a = (rng.standard_normal((N, K)) * escala / np.sqrt(K)).astype(np.float32)
        if tipo_mat == "Q8_0":
            q = quantize(a, gguf.GGMLQuantizationType.Q8_0)
            w.add_tensor(nombre_t, q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            w.add_tensor(nombre_t, a.astype(np.float16))

    def vec(nombre_t, n, centro=1.0, ruido=0.2):
        w.add_tensor(nombre_t, (centro + ruido * rng.standard_normal(n)).astype(np.float32))

    emb = (rng.standard_normal((V, D))).astype(np.float32)
    if tipo_mat == "Q8_0":
        w.add_tensor("token_embd.weight", quantize(emb, gguf.GGMLQuantizationType.Q8_0),
                     raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    else:
        w.add_tensor("token_embd.weight", emb.astype(np.float16))
    vec("token_embd_norm.weight", D, centro=escala_salida, ruido=0.2 * escala_salida)   # smaller = flatter output probabilities
    for i, nkv in enumerate(capas_kv):
        p = "blk.%d." % i
        vec(p + "attn_norm.weight", D)
        vec(p + "ffn_norm.weight", D)
        mat(p + "ffn_gate.weight", F, D, 2.0)
        mat(p + "ffn_up.weight", F, D, 2.0)
        mat(p + "ffn_down.weight", D, F, 1.0)
        if nkv == 0:
            mat(p + "shortconv.in_proj.weight", 3 * D, D, 2.0)
            mat(p + "shortconv.out_proj.weight", D, D, 1.0)
            w.add_tensor(p + "shortconv.conv.weight", (rng.standard_normal((D, L)) * 0.6).astype(np.float32))
        else:
            if fusion_qkv:
                mat(p + "attn_qkv.weight", nh * hd + 2 * nkv * hd, D, 2.0)
            else:
                mat(p + "attn_q.weight", nh * hd, D, 2.0)
                mat(p + "attn_k.weight", nkv * hd, D, 2.0)
                mat(p + "attn_v.weight", nkv * hd, D, 1.5)
            mat(p + "attn_output.weight", D, nh * hd, 1.0)
            vec(p + "attn_q_norm.weight", hd)
            vec(p + "attn_k_norm.weight", hd)
    if extra:
        extra(w)      # more metadata (for example, the decision type of a d1 model)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return ruta


TEXTOS_TOK = [
    "La capital de Francia es",
    "¿Cuánto es 12 por 11? Son 132.",
    "def fibonacci(n):\n    if n < 2:\n        return n",
    "<|im_start|>user\nHola, ¿qué tal?<|im_end|>\n<|im_start|>assistant\n",
    "Ñandú, pingüino y camión: 2026-09-29 😀  espacios   múltiples\n\n\nfin",
    "La base de datos de la institución",
]
PROMPTS = [
    "La base de datos",
    "<|im_start|>user\nExplica qué es una base de datos en tres frases.<|im_end|>\n<|im_start|>assistant\n",
    "El servidor responde a las peticiones de la red local en el puerto 1234. Los modelos de lenguaje generan texto",
]


def referencia(ruta):
    from llama_cpp import Llama
    llm = Llama(model_path=ruta, n_ctx=512, n_batch=512, n_ubatch=512, logits_all=True, verbose=False, n_threads=2, seed=0)
    ref = {"tokens": {}, "prompts": []}
    for t in TEXTOS_TOK:
        ref["tokens"][t] = llm.tokenize(t.encode("utf-8"), add_bos=True, special=True)
    for p in PROMPTS:
        llm.reset()
        toks = llm.tokenize(p.encode("utf-8"), add_bos=True, special=True)
        llm.eval(toks)
        logits = np.array(llm.scores[len(toks) - 1], dtype=np.float32)
        gen = []
        for _ in range(40):
            nxt = int(np.argmax(llm.scores[llm.n_tokens - 1]))
            if nxt == llm.token_eos():
                break
            gen.append(nxt)
            llm.eval([nxt])
        texto = llm.detokenize(gen).decode("utf-8", "replace")
        ref["prompts"].append({"prompt": p, "ids": toks, "logits_ultimo": logits.tolist(), "generados": gen, "texto": texto})
    return ref


if __name__ == "__main__":
    variantes = [("lfm2-test-q8", dict(tipo_mat="Q8_0")),
                 ("lfm2-test-f16-swa", dict(tipo_mat="F16", swa=5, fusion_qkv=True, semilla=1)),
                 ("lfm2-test-q8-nobos", dict(tipo_mat="Q8_0", add_bos=False, semilla=2))]
    refs = {}
    for nombre, kw in variantes:
        ruta = crear(nombre, **kw)
        refs[nombre] = referencia(ruta)
        print(nombre, os.path.getsize(ruta), "bytes;", refs[nombre]["prompts"][0]["texto"][:60].replace("\n", "\\n"))
    with open(os.path.join(OUT, "..", "referencias_llamacpp.json"), "w", encoding="utf-8") as f:
        json.dump(refs, f, ensure_ascii=False)
    # Q8 is compared against the SAME weights dequantized to F32 (llama.cpp quantizes activations to Q8 on CPU and deviates by ~3-5e-2).
    import gen_deq  # noqa: F401  (runs on import and rewrites the references of the Q8 variants)
