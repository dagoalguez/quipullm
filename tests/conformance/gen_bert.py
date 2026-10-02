"""Synthetic nomic-bert model (GGUF-style WordPiece vocabulary) quantized by llama.cpp + embedding references."""
import json, os, sys, unicodedata
import numpy as np
import gguf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_modelos as G
import gen_transformer as T

OUT = T.OUT
FT = gguf.LlamaFileType
SP = "▁"


def vocab_wpm():
    txt = unicodedata.normalize("NFD", G.CORPUS.replace("\\n", " ")).lower()
    txt = "".join(c for c in txt if unicodedata.category(c) != "Mn")
    import re
    palabras = re.findall(r"[a-z]+|[0-9]|[^\sa-z0-9]", txt)
    cuenta = {}
    for p in palabras: cuenta[p] = cuenta.get(p, 0) + 1
    tokens, tipos = [], []
    def add(t, ty): tokens.append(t); tipos.append(ty)
    add("[PAD]", 3)
    for i in range(1, 100): add("[unused%d]" % i, 5)
    add("[UNK]", 2); add("[CLS]", 3); add("[SEP]", 3); add("[MASK]", 3)
    chars = sorted({c for p in palabras for c in p})
    vistos = set(tokens)
    for c in chars:
        for t in (SP + c, c):
            if t not in vistos: add(t, 1); vistos.add(t)
    # complete frequent words and continuation pieces
    for p, n in sorted(cuenta.items(), key=lambda kv: (-kv[1], kv[0])):
        if len(p) > 1 and n >= 2 and (SP + p) not in vistos: add(SP + p, 1); vistos.add(SP + p)
    sub = {}
    for p in cuenta:
        if p.isalpha():
            for n in (2, 3, 4):
                for i in range(1, len(p) - n + 1):
                    sub[p[i:i + n]] = sub.get(p[i:i + n], 0) + cuenta[p]
    for s, n in sorted(sub.items(), key=lambda kv: (-kv[1], kv[0]))[:150]:
        if s not in vistos: add(s, 1); vistos.add(s)
    return tokens, tipos


VARIANTES = {
    "nomic-bert-q4km": (FT.MOSTLY_Q4_K_M, dict(semilla=21, nh=4, L=3, F=512, pooling=1, rope=1000.0)),
    "nomic-bert-q6k-cls": (FT.MOSTLY_Q6_K, dict(semilla=22, nh=2, L=2, F=768, pooling=2, rope=10000.0)),
}

TEXTOS = [
    "La base de datos", "¿Cuánto es 12 por 11? Son 132.", "def fibonacci(n):\n    if n < 2:\n        return n",
    "Ñandú, PINGÜINO y camión: 2026-09-29 😀  espacios   múltiples\n\n\nfin", "", "a", "  hola\tmundo feliz  ",
    "日本語のテキスト y más", "texto con [SEP] dentro y [CLS] tambien", "Hello, World! It's a test-case (really).",
    "xyzzyqq palabra desconocida", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
]
LARGO = ("El servidor responde a las peticiones de la red local en el puerto 1234. Los modelos de lenguaje generan texto "
         "token por token con la tarjeta gráfica. Una base de datos permite consultar, actualizar y respaldar los registros. ") * 4
EMBEDS = ["La base de datos", "search_query: ¿Qué es una base de datos?", "El servidor responde a las peticiones de la red local.", "a", LARGO,
          "Ñandú, PINGÜINO y camión: 2026-09-29 😀", "日本語のテキスト y más", "texto con [SEP] dentro"]


def crear_f32(nombre, o):
    rng = np.random.default_rng(o["semilla"])
    tokens, tipos = vocab_wpm()
    V, D, F, L, nh = len(tokens), 256, o["F"], o["L"], o["nh"]
    ruta = os.path.join(OUT, nombre + ".f32.gguf")
    w = gguf.GGUFWriter(ruta, "nomic-bert")
    w.add_name(nombre); w.add_block_count(L); w.add_context_length(2048); w.add_embedding_length(D)
    w.add_feed_forward_length(F); w.add_head_count(nh); w.add_layer_norm_eps(1e-12)
    w.add_causal_attention(False); w.add_token_type_count(2); w.add_pooling_type(gguf.PoolingType(o["pooling"]))
    w.add_rope_freq_base(o["rope"]); w.add_file_type(0)
    w.add_tokenizer_model("bert")
    w.add_token_list(tokens); w.add_token_types(tipos)
    w.add_unk_token_id(100); w.add_bos_token_id(101); w.add_sep_token_id(102); w.add_pad_token_id(0); w.add_mask_token_id(103)
    w.add_add_bos_token(True)

    def mat(n, N, K, esc=2.0):
        w.add_tensor(n, (rng.standard_normal((N, K)) * esc / np.sqrt(K)).astype(np.float32))
    def vec(n, k, c=1.0, r=0.2):
        w.add_tensor(n, (c + r * rng.standard_normal(k)).astype(np.float32))
    w.add_tensor("token_embd.weight", (0.5 * rng.standard_normal((V, D))).astype(np.float32))
    w.add_tensor("token_types.weight", (0.3 * rng.standard_normal((2, D))).astype(np.float32))
    vec("token_embd_norm.weight", D); w.add_tensor("token_embd_norm.bias", (0.1 * rng.standard_normal(D)).astype(np.float32))
    for i in range(L):
        p = "blk.%d." % i
        mat(p + "attn_qkv.weight", 3 * D, D, 2.5); mat(p + "attn_output.weight", D, D, 1.0)
        vec(p + "attn_output_norm.weight", D); w.add_tensor(p + "attn_output_norm.bias", (0.1 * rng.standard_normal(D)).astype(np.float32))
        mat(p + "ffn_gate.weight", F, D); mat(p + "ffn_up.weight", F, D); mat(p + "ffn_down.weight", D, F, 1.0)
        vec(p + "layer_output_norm.weight", D); w.add_tensor(p + "layer_output_norm.bias", (0.1 * rng.standard_normal(D)).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return ruta


def referencia(ruta, pooling):
    from llama_cpp import Llama
    import llama_cpp
    llm = Llama(model_path=ruta, n_ctx=512, n_batch=512, n_ubatch=512, embedding=True, verbose=False, n_threads=2, seed=0, flash_attn=False,
                pooling_type=llama_cpp.LLAMA_POOLING_TYPE_MEAN if pooling == 1 else llama_cpp.LLAMA_POOLING_TYPE_CLS)
    ref = {"tokens": {}, "embeds": []}
    for t in TEXTOS:
        ref["tokens"][t] = llm.tokenize(t.encode("utf-8"), add_bos=True, special=True)
    for t in EMBEDS:
        ids = llm.tokenize(t.encode("utf-8"), add_bos=True, special=True)
        b = llama_cpp.llama_batch_init(len(ids), 0, 1)
        b.n_tokens = len(ids)
        for i, t_ in enumerate(ids):
            b.token[i] = t_; b.pos[i] = i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 1
        assert llama_cpp.llama_decode(llm._ctx.ctx, b) == 0
        llama_cpp.llama_batch_free(b)
        n = llama_cpp.llama_model_n_embd(llm._model.model)
        ptr = llama_cpp.llama_get_embeddings_seq(llm._ctx.ctx, 0)
        v = np.ctypeslib.as_array(ptr, shape=(n,)).astype(np.float64)
        v = v / np.linalg.norm(v)
        ref["embeds"].append({"texto": t, "ids": ids, "vec": [float(x) for x in v]})
    return ref


if __name__ == "__main__":
    ruta_refs = os.path.join(OUT, "..", "referencias_bert.json")
    refs = {}
    for nombre, (ft, o) in VARIANTES.items():
        f32 = crear_f32(nombre, o)
        q = os.path.join(OUT, nombre + ".gguf")
        T.cuantizar(f32, q, ft); os.remove(f32)
        d = os.path.join(OUT, nombre + "-deq.gguf")
        tipos = T.deq(q, d)
        refs[nombre] = {"q": referencia(q, o["pooling"]), "deq": referencia(d, o["pooling"]), "tipos": tipos}
        assert refs[nombre]["q"]["tokens"] == refs[nombre]["deq"]["tokens"]
        print(nombre, os.path.getsize(q), tipos, [len(e["ids"]) for e in refs[nombre]["q"]["embeds"]])
    json.dump(refs, open(ruta_refs, "w", encoding="utf-8"), ensure_ascii=False)
