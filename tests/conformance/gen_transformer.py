"""Synthetic qwen2 / llama(Nemo) / granite models quantized by llama.cpp (K-quants) + references."""
import json, os, sys
import numpy as np
import gguf
from gguf.quants import dequantize
import llama_cpp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_modelos as G

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "modelos_tf")
os.makedirs(OUT, exist_ok=True)
FT = gguf.LlamaFileType

VARIANTES = {
    # name: (arch, ftype, options)
    "qwen2-q4km": ("qwen2", FT.MOSTLY_Q4_K_M, dict(sesgos=True, nkv=2, hd=64, pre="qwen2", add_bos=False, tied=False, semilla=1)),
    "qwen2-q6k-tied": ("qwen2", FT.MOSTLY_Q6_K, dict(sesgos=True, nkv=4, hd=64, pre="deepseek-r1-qwen", add_bos=False, tied=True, semilla=2)),
    "nemo-q3kl": ("llama", FT.MOSTLY_Q3_K_L, dict(sesgos=False, nkv=2, hd=128, nh=4, pre="tekken", add_bos=True, tied=False, semilla=3)),
    "granite-q5km": ("granite", FT.MOSTLY_Q5_K_M, dict(sesgos=False, nkv=2, hd=64, pre="refact", add_bos=False, tied=False, semilla=4,
                     emb_scale=12.0, res_scale=0.22, att_scale=0.0625, logit_scale=8.0)),
    # Architecture added ONLY via web/arch/qwen3.json (without touching the engine): per-head q/k-norm, no biases, head_dim 128 != D/nh
    "qwen3-q4km": ("qwen3", FT.MOSTLY_Q4_K_M, dict(sesgos=False, nkv=2, hd=128, nh=4, pre="qwen2", add_bos=False, tied=False, semilla=6)),
    "qwen2-q4ks-nokv": ("qwen2", FT.MOSTLY_Q4_K_S, dict(sesgos=True, nkv=2, hd=64, pre="qwen2", add_bos=False, tied=False, semilla=5)),
}


def crear_f32(nombre, arch, o):
    rng = np.random.default_rng(o["semilla"])
    tokens, tipos, merges = G.entrenar_bpe()
    V, D, F, L = len(tokens), 256, 512, 3
    nh = o.get("nh", 4); nkv = o["nkv"]; hd = o["hd"]
    ruta = os.path.join(OUT, nombre + ".f32.gguf")
    w = gguf.GGUFWriter(ruta, arch)
    w.add_name(nombre); w.add_block_count(L); w.add_context_length(4096); w.add_embedding_length(D)
    w.add_feed_forward_length(F); w.add_head_count(nh); w.add_head_count_kv(nkv)
    if hd != D // nh:
        w.add_key_length(hd); w.add_value_length(hd)
    w.add_rope_dimension_count(hd)
    w.add_rope_freq_base(1000000.0 if arch in ("qwen2", "qwen3") else 500000.0)
    w.add_layer_norm_rms_eps(1e-6 if arch in ("qwen2", "qwen3") else 1e-5)
    w.add_vocab_size(V)
    if arch == "granite":
        w.add_embedding_scale(o["emb_scale"]); w.add_residual_scale(o["res_scale"])
        w.add_attention_scale(o["att_scale"]); w.add_logit_scale(o["logit_scale"])
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre(o["pre"])
    w.add_token_list(tokens); w.add_token_types(tipos); w.add_token_merges(merges)
    w.add_bos_token_id(1); w.add_eos_token_id(4); w.add_pad_token_id(0); w.add_add_bos_token(o["add_bos"])

    def mat(n, N, K, esc=2.0):
        w.add_tensor(n, (rng.standard_normal((N, K)) * esc / np.sqrt(K)).astype(np.float32))
    def vec(n, k, c=1.0, r=0.2):
        w.add_tensor(n, (c + r * rng.standard_normal(k)).astype(np.float32))
    w.add_tensor("token_embd.weight", rng.standard_normal((V, D)).astype(np.float32))
    vec("output_norm.weight", D)
    if not o["tied"]:
        mat("output.weight", V, D, 1.0)
    for i in range(L):
        p = "blk.%d." % i
        vec(p + "attn_norm.weight", D); vec(p + "ffn_norm.weight", D)
        mat(p + "attn_q.weight", nh * hd, D); mat(p + "attn_k.weight", nkv * hd, D); mat(p + "attn_v.weight", nkv * hd, D, 1.5)
        mat(p + "attn_output.weight", D, nh * hd, 1.0)
        if arch == "qwen3":
            vec(p + "attn_q_norm.weight", hd); vec(p + "attn_k_norm.weight", hd)
        if o["sesgos"]:
            for n, k in (("q", nh * hd), ("k", nkv * hd), ("v", nkv * hd)):
                w.add_tensor(p + "attn_%s.bias" % n, (0.3 * rng.standard_normal(k)).astype(np.float32))
        mat(p + "ffn_gate.weight", F, D); mat(p + "ffn_up.weight", F, D); mat(p + "ffn_down.weight", D, F, 1.0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return ruta


def cuantizar(fin, fout, ftype):
    p = llama_cpp.llama_model_quantize_default_params()
    p.ftype = int(ftype); p.nthread = 2
    r = llama_cpp.llama_model_quantize(fin.encode(), fout.encode(), p)
    assert r == 0, "cuantización falló"


def deq(ruta, salida):
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
    tipos = {}
    for t in r.tensors:
        shape = [int(x) for x in reversed(t.shape.tolist())]
        tipos[t.tensor_type.name] = tipos.get(t.tensor_type.name, 0) + 1
        w.add_tensor(t.name, dequantize(t.data, t.tensor_type).astype(np.float32).reshape(shape))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return tipos


def referencia(ruta, add_bos):
    from llama_cpp import Llama
    llm = Llama(model_path=ruta, n_ctx=512, n_batch=512, n_ubatch=512, logits_all=True, verbose=False, n_threads=2, seed=0)
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
            if nxt == llm.token_eos():
                break
            gen.append(nxt); llm.eval([nxt])
        ref["prompts"].append({"prompt": p, "ids": toks, "logits_ultimo": logits.tolist(), "generados": gen})
    return ref


if __name__ == "__main__":
    refs = {}
    for nombre, (arch, ft, o) in VARIANTES.items():
        f32 = crear_f32(nombre, arch, o)
        q = os.path.join(OUT, nombre + ".gguf")
        cuantizar(f32, q, ft)
        os.remove(f32)
        d = os.path.join(OUT, nombre + "-deq.gguf")
        tipos = deq(q, d)
        refs[nombre] = {"q": referencia(q, o["add_bos"]), "deq": referencia(d, o["add_bos"]), "tipos": tipos}
        assert refs[nombre]["q"]["tokens"] == refs[nombre]["deq"]["tokens"]
        print(nombre, os.path.getsize(q), tipos)
    json.dump(refs, open(os.path.join(OUT, "..", "referencias_tf.json"), "w", encoding="utf-8"), ensure_ascii=False)
