"""Synthetic deepseek2 models (MLA + MoE + YaRN) quantized by llama.cpp + references."""
import json, os, sys
import numpy as np
import gguf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_modelos as G
import gen_transformer as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "modelos_ds")
os.makedirs(OUT, exist_ok=True)
FT = gguf.LlamaFileType

VARIANTES = {
    # name: (ftype, options)
    "ds2-q2k": (FT.MOSTLY_Q2_K, dict(semilla=21, L=26, E=8, k=3, norm=False, escala=1.0, factor=8.0, ctx_orig=64, log_mul=0.0707, dense=1)),
    "ds2-q4km-norm": (FT.MOSTLY_Q4_K_M, dict(semilla=22, L=27, E=8, k=2, norm=True, escala=2.5, factor=4.0, ctx_orig=32, log_mul=0.0, dense=1)),
    "ds2-q3ks": (FT.MOSTLY_Q3_K_S, dict(semilla=23, L=26, E=16, k=4, norm=False, escala=1.0, factor=1.0, ctx_orig=4096, log_mul=0.0, dense=2)),
}

D, NH, NOPE, NR, VD, R = 256, 4, 64, 32, 64, 256
FF, FFE, NSH = 352, 128, 2


def crear_f32(nombre, o):
    rng = np.random.default_rng(o["semilla"])
    tokens, tipos, merges = G.entrenar_bpe()
    V, L, E = len(tokens), o["L"], o["E"]
    ruta = os.path.join(OUT, nombre + ".f32.gguf")
    w = gguf.GGUFWriter(ruta, "deepseek2")
    a = "deepseek2."
    w.add_name(nombre)
    w.add_uint32(a + "block_count", L); w.add_uint32(a + "context_length", 4096); w.add_uint32(a + "embedding_length", D)
    w.add_uint32(a + "feed_forward_length", FF)
    w.add_uint32(a + "attention.head_count", NH); w.add_uint32(a + "attention.head_count_kv", NH)
    w.add_float32(a + "rope.freq_base", 10000.0)
    w.add_float32(a + "attention.layer_norm_rms_epsilon", 1e-6)
    w.add_uint32(a + "expert_used_count", o["k"]); w.add_uint32(a + "leading_dense_block_count", o["dense"])
    w.add_uint32(a + "vocab_size", V)
    w.add_uint32(a + "attention.kv_lora_rank", R)
    w.add_uint32(a + "attention.key_length", NOPE + NR); w.add_uint32(a + "attention.value_length", VD)
    w.add_uint32(a + "expert_feed_forward_length", FFE)
    w.add_uint32(a + "expert_count", E); w.add_uint32(a + "expert_shared_count", NSH)
    w.add_float32(a + "expert_weights_scale", o["escala"])
    if o["norm"]:
        w.add_bool(a + "expert_weights_norm", True)
    w.add_uint32(a + "rope.dimension_count", NR)
    if o["factor"] != 1.0:
        w.add_string(a + "rope.scaling.type", "yarn"); w.add_float32(a + "rope.scaling.factor", o["factor"])
        w.add_uint32(a + "rope.scaling.original_context_length", o["ctx_orig"])
        if o["log_mul"]: w.add_float32(a + "rope.scaling.yarn_log_multiplier", o["log_mul"])
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("deepseek-llm")
    w.add_token_list(tokens); w.add_token_types(tipos); w.add_token_merges(merges)
    w.add_bos_token_id(1); w.add_eos_token_id(2); w.add_pad_token_id(0); w.add_add_bos_token(True)

    def mat(n, *shape, esc=2.0):
        K = shape[-1]
        w.add_tensor(n, (rng.standard_normal(shape) * esc / np.sqrt(K)).astype(np.float32))
    def vec(n, k, c=1.0, r=0.2):
        w.add_tensor(n, (c + r * rng.standard_normal(k)).astype(np.float32))
    w.add_tensor("token_embd.weight", rng.standard_normal((V, D)).astype(np.float32))
    vec("output_norm.weight", D)
    mat("output.weight", V, D, esc=1.0)
    for i in range(L):
        p = "blk.%d." % i
        vec(p + "attn_norm.weight", D); vec(p + "ffn_norm.weight", D); vec(p + "attn_kv_a_norm.weight", R)
        mat(p + "attn_q.weight", NH * (NOPE + NR), D)
        mat(p + "attn_kv_a_mqa.weight", R + NR, D)
        mat(p + "attn_kv_b.weight", NH * (NOPE + VD), R, esc=1.5)
        mat(p + "attn_output.weight", D, NH * VD, esc=1.0)
        if i < o["dense"]:
            mat(p + "ffn_gate.weight", FF, D); mat(p + "ffn_up.weight", FF, D); mat(p + "ffn_down.weight", D, FF, esc=1.0)
        else:
            mat(p + "ffn_gate_inp.weight", E, D, esc=6.0)
            mat(p + "ffn_gate_exps.weight", E, FFE, D); mat(p + "ffn_up_exps.weight", E, FFE, D)
            mat(p + "ffn_down_exps.weight", E, D, FFE, esc=1.0)
            mat(p + "ffn_gate_shexp.weight", FFE * NSH, D); mat(p + "ffn_up_shexp.weight", FFE * NSH, D)
            mat(p + "ffn_down_shexp.weight", D, FFE * NSH, esc=1.0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return ruta


def referencia(ruta, add_bos):
    """Like T.referencia but with an f32 KV cache: with deep MoE, llama.cpp's f16 noise (1e-2) masks the differences."""
    from llama_cpp import Llama
    llm = Llama(model_path=ruta, n_ctx=512, n_batch=512, n_ubatch=512, logits_all=True, verbose=False, n_threads=2, seed=0,
                type_k=0, type_v=0, flash_attn=False)
    ref = {"tokens": {}, "prompts": [], "add_bos": add_bos}
    for t in G.TEXTOS_TOK:
        ref["tokens"][t] = llm.tokenize(t.encode("utf-8"), add_bos=add_bos, special=True)
    for p in G.PROMPTS:
        llm.reset()
        toks = llm.tokenize(p.encode("utf-8"), add_bos=add_bos, special=True)
        llm.eval(toks)
        logits = np.array(llm.scores[len(toks) - 1], dtype=np.float32)
        gen = []
        for _ in range(24):
            nxt = int(np.argmax(llm.scores[llm.n_tokens - 1]))
            if nxt == llm.token_eos(): break
            gen.append(nxt); llm.eval([nxt])
        ref["prompts"].append({"prompt": p, "ids": toks, "logits_ultimo": logits.tolist(), "generados": gen})
    return ref


KV = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


def cuantizar_iq4nl(x):
    """x: [..., K] f32 (K a multiple of 32) -> IQ4_NL bytes [..., K/32*18]. Simple per-block scale search."""
    sh = x.shape
    b = x.reshape(-1, 32).astype(np.float32)
    n = b.shape[0]
    amax = np.abs(b).max(axis=1)
    mejor_err = np.full(n, np.inf, dtype=np.float32)
    mejor_d = np.zeros(n, dtype=np.float32)
    mejor_i = np.zeros((n, 32), dtype=np.uint8)
    imax = b[np.arange(n), np.abs(b).argmax(axis=1)]
    for f in np.linspace(-127.0, -100.0, 28):
        d = imax / f
        d = np.where(d == 0, 1.0, d).astype(np.float16).astype(np.float32)
        dd = np.where(d == 0, 1.0, d)
        t = b / dd[:, None]
        idx = np.abs(t[:, :, None] - KV[None, None, :]).argmin(axis=2)
        err = ((dd[:, None] * KV[idx] - b) ** 2).sum(axis=1)
        m = err < mejor_err
        mejor_err = np.where(m, err, mejor_err); mejor_d = np.where(m, dd, mejor_d); mejor_i[m] = idx[m].astype(np.uint8)
    out = np.zeros((n, 18), dtype=np.uint8)
    out[:, :2] = mejor_d.astype(np.float16).view(np.uint8).reshape(n, 2)
    out[:, 2:] = mejor_i[:, :16] | (mejor_i[:, 16:] << 4)
    return out.reshape(*sh[:-1], sh[-1] // 32 * 18)


def a_iq4nl(ruta, salida):
    """Copies the GGUF, converting the Q4_0/Q5_0/Q4_1/Q5_1 tensors (K padding, K not a multiple of 256) to IQ4_NL."""
    from gguf.quants import dequantize
    r = gguf.GGUFReader(ruta)
    arch = bytes(r.fields["general.architecture"].parts[-1]).decode()
    w = gguf.GGUFWriter(salida, arch)
    for f in r.fields.values():
        if f.name.startswith("GGUF.") or f.name == "general.architecture": continue
        tipo = f.types[0]
        if tipo == gguf.GGUFValueType.ARRAY:
            sub = f.types[1]
            val = [bytes(f.parts[i]).decode("utf-8") for i in f.data] if sub == gguf.GGUFValueType.STRING else [f.parts[i].tolist()[0] for i in f.data]
            w.add_key_value(f.name, val, tipo, sub_type=sub)
        elif tipo == gguf.GGUFValueType.STRING:
            w.add_key_value(f.name, bytes(f.parts[f.data[0]]).decode("utf-8"), tipo)
        else:
            w.add_key_value(f.name, f.parts[f.data[0]].tolist()[0], tipo)
    Q = gguf.GGMLQuantizationType
    for t in r.tensors:
        if t.tensor_type in (Q.Q4_0, Q.Q5_0, Q.Q4_1, Q.Q5_1):
            a = dequantize(t.data, t.tensor_type).astype(np.float32)
            w.add_tensor(t.name, cuantizar_iq4nl(a), raw_dtype=Q.IQ4_NL)
        elif t.tensor_type == Q.F32:
            shape = [int(x) for x in reversed(t.shape.tolist())]
            w.add_tensor(t.name, np.array(t.data, dtype=np.float32).reshape(shape))
        else:
            w.add_tensor(t.name, np.array(t.data), raw_dtype=t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


if __name__ == "__main__":
    ruta_refs = os.path.join(OUT, "..", "referencias_ds.json")
    refs = json.load(open(ruta_refs, encoding="utf-8")) if os.path.exists(ruta_refs) and sys.argv[1:] else {}
    for nombre, (ft, o) in VARIANTES.items():
        if sys.argv[1:] and nombre not in sys.argv[1:]: continue
        f32 = crear_f32(nombre, o)
        q = os.path.join(OUT, nombre + ".gguf")
        q0 = q + ".tmp"
        T.cuantizar(f32, q0, ft)
        a_iq4nl(q0, q); os.remove(q0)
        d = os.path.join(OUT, nombre + "-deq.gguf")
        tipos = T.deq(q, d)
        os.remove(f32)
        refs[nombre] = {"q": referencia(q, True), "deq": referencia(d, True), "tipos": tipos}
        assert refs[nombre]["q"]["tokens"] == refs[nombre]["deq"]["tokens"]
        print(nombre, os.path.getsize(q), tipos, [len(p["ids"]) for p in refs[nombre]["q"]["prompts"]],
              [len(p["generados"]) for p in refs[nombre]["deq"]["prompts"]])
    json.dump(refs, open(ruta_refs, "w", encoding="utf-8"), ensure_ascii=False)
