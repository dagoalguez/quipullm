"""Synthetic gemma3 models (SentencePiece vocabulary, sliding window, post-norms) + llama.cpp references."""
import json, os, sys
import numpy as np
import gguf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_modelos as G
import gen_transformer as T

OUT = T.OUT
FT = gguf.LlamaFileType
SOLO_NO = {"ñ", "Ñ", "¿", "ü"}   # without a piece: force the byte fallback


def vocab_spm(n_piezas=300):
    texto = G.CORPUS.replace("\\n", "\n")
    palabras = {}
    for p in texto.split(" "):
        w = "▁" + p
        palabras[w] = palabras.get(w, 0) + 1
    chars = sorted({c for w in palabras for c in w if c not in SOLO_NO and c != "\n"})
    cuenta = {}
    for w, c in palabras.items():
        for n in range(2, 9):
            for i in range(len(w) - n + 1):
                s = w[i:i + n]
                if "\n" in s or any(x in SOLO_NO for x in s): continue
                cuenta[s] = cuenta.get(s, 0) + c
    cand = sorted(cuenta.items(), key=lambda kv: (-(kv[1] * (len(kv[0]) - 1)), kv[0]))[:n_piezas]
    tokens, scores, tipos = [], [], []
    def add(t, s, ty): tokens.append(t); scores.append(float(s)); tipos.append(int(ty))
    add("<pad>", 0, 3); add("<eos>", 0, 3); add("<bos>", 0, 3); add("<unk>", 0, 2)
    add("<start_of_turn>", 0, 4); add("<end_of_turn>", 0, 4)
    for b in range(256): add("<0x%02X>" % b, 0, 6)
    add("\n", -1, 1); add("\n\n", -2, 1)
    for k, c in enumerate(chars): add(c, -1000 - k, 1)
    for extra in ("▁▁", "▁▁▁▁"):
        if extra not in tokens: add(extra, -5, 1)
    vistos = set(tokens)
    for rank, (s, _) in enumerate(cand):
        if s in vistos: continue
        add(s, -(3 + rank // 2), 1)      # scores tied in pairs: tests the tie-break by position
        vistos.add(s)
    return tokens, scores, tipos


VARIANTES = {
    "gemma3-q4km": (FT.MOSTLY_Q4_K_M, dict(semilla=11, ventana=8, patron=3, rope_escala=8.0, softcap=0.0, add_espacio=False, nkv=2, nh=4, hd=64, L=6)),
    "gemma3-q5km-sc": (FT.MOSTLY_Q5_K_M, dict(semilla=12, ventana=16, patron=None, rope_escala=None, softcap=30.0, add_espacio=True, nkv=1, nh=4, hd=64, L=6)),
    "gemma3-q4ks-hd256": (FT.MOSTLY_Q4_K_S, dict(semilla=13, ventana=8, patron=2, rope_escala=8.0, softcap=0.0, add_espacio=False, nkv=2, nh=4, hd=256, L=4)),
}

TEXTOS = [
    "La capital de Francia es",
    "¿Cuánto es 12 por 11? Son 132.",
    "def fibonacci(n):\n    if n < 2:\n        return n",
    "<start_of_turn>user\nHola, ¿qué tal?<end_of_turn>\n<start_of_turn>model\n",
    "Ñandú, pingüino y camión: 2026-09-29 😀  espacios   múltiples\n\n\nfin",
    "La base de datos de la institución",
    "  inicio con espacios",
    "",
    "<bos>ya con bos<eos>",
    "a\tb\r\nc",
    "日本語のテキスト",
    "\n\nlinea",
    "<unk> y <pad>",
    "aaaaaaaaaaaaaaaaaaaaaaaa",
    "La base de datos guarda la información",
]
PROMPTS = [
    "La base de datos",
    "<start_of_turn>user\nExplica qué es una base de datos en tres frases.<end_of_turn>\n<start_of_turn>model\n",
    ("El servidor responde a las peticiones de la red local en el puerto 1234. Los modelos de lenguaje generan texto "
     "token por token con la tarjeta gráfica. Una base de datos permite consultar, actualizar y respaldar los registros. "
     "La capital de Francia es París. Juan tiene 34 años y trabaja en el área de datos. Él usa Python y Spark todos los días."),
    ("La base de datos guarda la información de la institución de forma ordenada. " * 2 + "El servidor responde a las peticiones"),
]


def crear_f32(nombre, o):
    rng = np.random.default_rng(o["semilla"])
    tokens, scores, tipos = vocab_spm()
    V, D, F, L = len(tokens), 256, 512, o["L"]
    nh, nkv, hd = o["nh"], o["nkv"], o["hd"]
    ruta = os.path.join(OUT, nombre + ".f32.gguf")
    w = gguf.GGUFWriter(ruta, "gemma3")
    w.add_name(nombre); w.add_block_count(L); w.add_context_length(4096); w.add_embedding_length(D)
    w.add_feed_forward_length(F); w.add_head_count(nh); w.add_head_count_kv(nkv)
    w.add_key_length(hd); w.add_value_length(hd)
    w.add_rope_freq_base(1000000.0); w.add_rope_freq_base_swa(10000.0)
    if o["rope_escala"]:
        w.add_rope_scaling_type(gguf.RopeScalingType.LINEAR); w.add_rope_scaling_factor(o["rope_escala"])
    w.add_layer_norm_rms_eps(1e-6)
    w.add_sliding_window(o["ventana"])
    if o["patron"]: w.add_sliding_window_pattern(o["patron"])
    if o["softcap"]: w.add_final_logit_softcapping(o["softcap"])
    w.add_vocab_size(V)
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens); w.add_token_scores(scores); w.add_token_types(tipos)
    w.add_bos_token_id(2); w.add_eos_token_id(1); w.add_pad_token_id(0); w.add_unk_token_id(3)
    w.add_add_bos_token(True); w.add_add_eos_token(False)
    w.add_bool("tokenizer.ggml.add_space_prefix", bool(o["add_espacio"]))

    def mat(n, N, K, esc=2.0):
        w.add_tensor(n, (rng.standard_normal((N, K)) * esc / np.sqrt(K)).astype(np.float32))
    def vec(n, k, c=1.0, r=0.2):
        w.add_tensor(n, (c + r * rng.standard_normal(k)).astype(np.float32))
    w.add_tensor("token_embd.weight", (0.3 * rng.standard_normal((V, D))).astype(np.float32))
    vec("output_norm.weight", D)
    for i in range(L):
        p = "blk.%d." % i
        vec(p + "attn_norm.weight", D); vec(p + "ffn_norm.weight", D)
        vec(p + "post_attention_norm.weight", D, 0.8, 0.3); vec(p + "post_ffw_norm.weight", D, 0.8, 0.3)
        vec(p + "attn_q_norm.weight", hd); vec(p + "attn_k_norm.weight", hd)
        mat(p + "attn_q.weight", nh * hd, D); mat(p + "attn_k.weight", nkv * hd, D); mat(p + "attn_v.weight", nkv * hd, D, 1.5)
        mat(p + "attn_output.weight", D, nh * hd, 1.0)
        mat(p + "ffn_gate.weight", F, D); mat(p + "ffn_up.weight", F, D); mat(p + "ffn_down.weight", D, F, 1.0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return ruta


def referencia(ruta):
    from llama_cpp import Llama
    llm = Llama(model_path=ruta, n_ctx=512, n_batch=512, n_ubatch=512, logits_all=True, verbose=False, n_threads=2, seed=0)
    ref = {"tokens": {}, "prompts": [], "add_bos": True}
    for t in TEXTOS:
        ref["tokens"][t] = llm.tokenize(t.encode("utf-8"), add_bos=True, special=True)
    for p in PROMPTS:
        llm.reset()
        toks = llm.tokenize(p.encode("utf-8"), add_bos=True, special=True)
        llm.eval(toks)
        logits = np.array(llm.scores[len(toks) - 1], dtype=np.float32)
        gen = []
        for _ in range(40):
            nxt = int(np.argmax(llm.scores[llm.n_tokens - 1]))
            if nxt == llm.token_eos(): break
            gen.append(nxt); llm.eval([nxt])
        ref["prompts"].append({"prompt": p, "ids": toks, "logits_ultimo": logits.tolist(), "generados": gen})
    return ref


if __name__ == "__main__":
    ruta_refs = os.path.join(OUT, "..", "referencias_gemma.json")
    refs = json.load(open(ruta_refs, encoding="utf-8")) if os.path.exists(ruta_refs) and sys.argv[1:] else {}
    for nombre, (ft, o) in VARIANTES.items():
        if sys.argv[1:] and nombre not in sys.argv[1:]: continue
        f32 = crear_f32(nombre, o)
        q = os.path.join(OUT, nombre + ".gguf")
        T.cuantizar(f32, q, ft); os.remove(f32)
        d = os.path.join(OUT, nombre + "-deq.gguf")
        tipos = T.deq(q, d)
        refs[nombre] = {"q": referencia(q), "deq": referencia(d), "tipos": tipos}
        assert refs[nombre]["q"]["tokens"] == refs[nombre]["deq"]["tokens"]
        print(nombre, os.path.getsize(q), tipos, [len(p["ids"]) for p in refs[nombre]["q"]["prompts"]],
              [len(p["generados"]) for p in refs[nombre]["deq"]["prompts"]])
    json.dump(refs, open(os.path.join(OUT, "..", "referencias_gemma.json"), "w", encoding="utf-8"), ensure_ascii=False)
