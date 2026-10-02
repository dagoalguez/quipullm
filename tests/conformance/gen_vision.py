"""Synthetic LFM2 models + mmproj ('lfm2' projector, SigLIP-style ViT) and references from libmtmd (llama.cpp)."""
import ctypes, json, os, sys
import numpy as np
import gguf
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_modelos as G

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "cache", "modelos_vis")
IMG = os.path.join(AQUI, "cache", "imagenes_vis")
os.makedirs(OUT, exist_ok=True); os.makedirs(IMG, exist_ok=True)

MARCADOR = "<__media__>"

# ---------------------------------------------------------------- images
def hacer_imagen(nombre, w, h, semilla):
    rng = np.random.default_rng(semilla)
    x = np.linspace(0, 1, w)[None, :, None]
    y = np.linspace(0, 1, h)[:, None, None]
    base = np.concatenate([x * 0.8 + y * 0.1, y * 0.7 + 0.2 * np.sin(6 * x), 0.5 + 0.4 * np.cos(5 * (x + y))], axis=2)
    a = (np.clip(base, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(a, "RGB")
    d = ImageDraw.Draw(im)
    for _ in range(14):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = x0 + int(rng.integers(5, max(6, w // 3))), y0 + int(rng.integers(5, max(6, h // 3)))
        d.rectangle([x0, y0, x1, y1], fill=tuple(int(v) for v in rng.integers(0, 256, 3)))
    d.text((max(2, w // 10), max(2, h // 3)), "Factura 0042 Total 132.50", fill=(255, 255, 255))
    ruido = rng.integers(-12, 13, (h, w, 3))
    a = np.clip(np.asarray(im).astype(np.int32) + ruido, 0, 255).astype(np.uint8)
    ruta = os.path.join(IMG, nombre + ".png")
    Image.fromarray(a, "RGB").save(ruta)
    return ruta, a


IMAGENES = {"a": (100, 60, 1), "b": (500, 300, 2), "c": (1300, 700, 3), "d": (600, 1400, 4), "e": (333, 251, 5)}
PROMPTS = [
    ("p_a", "Mira esto: " + MARCADOR + " ¿Qué dice?", ["a"]),
    ("p_b", MARCADOR + "\nDescribe la imagen.", ["b"]),
    ("p_c", "Texto antes. " + MARCADOR + " Texto después.", ["c"]),
    ("p_d", MARCADOR + " Lee el total.", ["d"]),
    ("p_dos", "Primera " + MARCADOR + " segunda " + MARCADOR + " fin", ["e", "a"]),
]


def T(t): return {"type": "text", "text": t}
def I(k): return {"type": "image", "k": k}
PRE = "<|startoftext|>"
CHATS = [
    ("c_1", [{"role": "user", "content": [T("Lee esto:"), I("a"), T("¿Qué dice?")]}],
     PRE + "<|im_start|>user\nLee esto:" + MARCADOR + "¿Qué dice?<|im_end|>\n<|im_start|>assistant\n", ["a"]),
    ("c_2", [{"role": "system", "content": "Eres útil."}, {"role": "user", "content": [I("b"), T("Describe la imagen.")]}],
     PRE + "<|im_start|>system\nEres útil.<|im_end|>\n<|im_start|>user\n" + MARCADOR + "Describe la imagen.<|im_end|>\n<|im_start|>assistant\n", ["b"]),
    ("c_3", [{"role": "user", "content": [T("Mira"), I("e")]}, {"role": "assistant", "content": "Bien."},
             {"role": "user", "content": [I("a"), T("¿y esta?"), T("Gracias")]}],
     PRE + "<|im_start|>user\nMira" + MARCADOR + "<|im_end|>\n<|im_start|>assistant\nBien.<|im_end|>\n<|im_start|>user\n" + MARCADOR + "¿y esta?\nGracias<|im_end|>\n<|im_start|>assistant\n", ["e", "a"]),
    ("c_4", [{"role": "user", "content": [T("Factura:"), I("c")]}],
     PRE + "<|im_start|>user\nFactura:" + MARCADOR + "<|im_end|>\n<|im_start|>assistant\n", ["c"]),
]


# ---------------------------------------------------------------- language model
def tokens_especiales():
    t = ["<|image_start|>", "<|image_end|>", "<|img_thumbnail|>"]
    for r in range(1, 11):
        for c in range(1, 11):
            t.append("<|img_row_%d_col_%d|>" % (r, c))
    return t


def crear_lm(nombre, semilla=0, f32=False):
    """Like gen_modelos.crear (LFM2 Q8_0) but with the image tokens at the end of the vocabulary."""
    rng = np.random.default_rng(semilla)
    tokens, tipos, merges = G.entrenar_bpe()
    for t in tokens_especiales():
        tokens.append(t); tipos.append(4)
    V, D, F, nh, hd, L = len(tokens), 64, 128, 4, 16, 3
    capas_kv = [0, 0, 2, 0, 2, 0]
    ruta = os.path.join(OUT, nombre, nombre + ".gguf")
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    w = gguf.GGUFWriter(ruta, "lfm2")
    w.add_name(nombre); w.add_block_count(len(capas_kv)); w.add_context_length(4096); w.add_embedding_length(D)
    w.add_feed_forward_length(F); w.add_head_count(nh); w.add_head_count_kv(capas_kv)
    w.add_rope_freq_base(1000000.0); w.add_layer_norm_rms_eps(1e-5)
    w.add_uint32("lfm2.shortconv.l_cache", L); w.add_vocab_size(V); w.add_file_type(0 if f32 else 7)
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("lfm2")
    w.add_token_list(tokens); w.add_token_types(tipos); w.add_token_merges(merges)
    w.add_bos_token_id(1); w.add_eos_token_id(4); w.add_pad_token_id(0); w.add_add_bos_token(True)
    w.add_chat_template("{{bos_token}}{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}")
    Q8 = gguf.GGMLQuantizationType.Q8_0
    from gguf.quants import quantize

    def mat(n, N, K, e=1.0):
        a = (rng.standard_normal((N, K)) * e / np.sqrt(K)).astype(np.float32)
        if f32: w.add_tensor(n, a)
        else: w.add_tensor(n, quantize(a, Q8), raw_dtype=Q8)

    def vec(n, k, c=1.0, r=0.2):
        w.add_tensor(n, (c + r * rng.standard_normal(k)).astype(np.float32))

    emb = rng.standard_normal((V, D)).astype(np.float32)
    if f32: w.add_tensor("token_embd.weight", emb)
    else: w.add_tensor("token_embd.weight", quantize(emb, Q8), raw_dtype=Q8)
    vec("token_embd_norm.weight", D)
    for i, nkv in enumerate(capas_kv):
        p = "blk.%d." % i
        vec(p + "attn_norm.weight", D); vec(p + "ffn_norm.weight", D)
        mat(p + "ffn_gate.weight", F, D, 2.0); mat(p + "ffn_up.weight", F, D, 2.0); mat(p + "ffn_down.weight", D, F, 1.0)
        if nkv == 0:
            mat(p + "shortconv.in_proj.weight", 3 * D, D, 2.0); mat(p + "shortconv.out_proj.weight", D, D, 1.0)
            w.add_tensor(p + "shortconv.conv.weight", (rng.standard_normal((D, L)) * 0.6).astype(np.float32))
        else:
            mat(p + "attn_q.weight", nh * hd, D, 2.0); mat(p + "attn_k.weight", nkv * hd, D, 2.0)
            mat(p + "attn_v.weight", nkv * hd, D, 1.5); mat(p + "attn_output.weight", D, nh * hd, 1.0)
            vec(p + "attn_q_norm.weight", hd); vec(p + "attn_k_norm.weight", hd)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return ruta, D


# ---------------------------------------------------------------- mmproj
def crear_mmproj(nombre, o, d_lm):
    rng = np.random.default_rng(o["semilla"])
    ruta = os.path.join(OUT, nombre, "mmproj-" + nombre + ".gguf")
    D, nh, F, L, P, NP, M = o["D"], o["nh"], o["F"], o["L"], 16, 16, 2
    w = gguf.GGUFWriter(ruta, "clip")
    w.add_name("mmproj-" + nombre)
    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "lfm2")
    if o["gelu"]:
        w.add_bool("clip.use_gelu", True)
    w.add_uint32("clip.vision.image_size", 256)
    w.add_uint32("clip.vision.patch_size", P)
    w.add_uint32("clip.vision.embedding_length", D)
    w.add_uint32("clip.vision.feed_forward_length", F)
    w.add_uint32("clip.vision.block_count", L)
    w.add_uint32("clip.vision.projection_dim", d_lm)
    w.add_uint32("clip.vision.attention.head_count", nh)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", o["eps"])
    w.add_array("clip.vision.image_mean", o["media"])
    w.add_array("clip.vision.image_std", o["std"])
    w.add_uint32("clip.vision.projector.scale_factor", M)
    tw = np.float32 if o["tipo"] == "f32" else np.float16

    def mat(n, N, K, e=1.0, forma=None):
        a = (rng.standard_normal((N, K)) * e / np.sqrt(K)).astype(np.float32)
        if forma: a = a.reshape(forma)
        w.add_tensor(n, a.astype(tw))

    def vec(n, k, c=0.0, r=0.1):
        w.add_tensor(n, (c + r * rng.standard_normal(k)).astype(np.float32))

    mat("v.patch_embd.weight", D, 3 * P * P, 2.0, (D, 3, P, P))
    vec("v.patch_embd.bias", D)
    w.add_tensor("v.position_embd.weight", (0.5 * rng.standard_normal((NP * NP, D))).astype(np.float32))
    if o["pre_ln"]:
        vec("v.pre_ln.weight", D, 1.0, 0.1); vec("v.pre_ln.bias", D)
    for i in range(L):
        p = "v.blk.%d." % i
        vec(p + "ln1.weight", D, 1.0, 0.1); vec(p + "ln1.bias", D)
        vec(p + "ln2.weight", D, 1.0, 0.1); vec(p + "ln2.bias", D)
        if o["qkv"]:
            mat(p + "attn_qkv.weight", 3 * D, D, 1.6); vec(p + "attn_qkv.bias", 3 * D)
        else:
            for n in ("q", "k", "v"):
                mat(p + "attn_%s.weight" % n, D, D, 1.6); vec(p + "attn_%s.bias" % n, D)
        mat(p + "attn_out.weight", D, D, 1.0); vec(p + "attn_out.bias", D)
        mat(p + "ffn_up.weight", F, D, 2.0); vec(p + "ffn_up.bias", F)
        mat(p + "ffn_down.weight", D, F, 1.0); vec(p + "ffn_down.bias", D)
    vec("v.post_ln.weight", D, 1.0, 0.1); vec("v.post_ln.bias", D)
    K1 = D * M * M
    if o["in_norm"]:
        vec("mm.input_norm.weight", K1, 1.0, 0.1); vec("mm.input_norm.bias", K1)
    mat("mm.1.weight", o["H"], K1, 2.0); vec("mm.1.bias", o["H"])
    mat("mm.2.weight", d_lm, o["H"], 1.0); vec("mm.2.bias", d_lm)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return ruta


VARIANTES = {
    "lfm2-vl-f32": (dict(lm_f32=True, semilla=21, D=72, nh=4, F=144, L=2, H=96, tipo="f32", qkv=False, gelu=False, pre_ln=False, in_norm=True,
                         eps=1e-6, media=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 0),
    "lfm2-vl-f16": (dict(semilla=22, D=72, nh=4, F=144, L=2, H=96, tipo="f16", qkv=True, gelu=True, pre_ln=True, in_norm=True,
                         eps=1e-6, media=[0.48, 0.45, 0.40], std=[0.27, 0.26, 0.28]), 1),
    "lfm2-vl-f16b": (dict(semilla=23, D=72, nh=4, F=144, L=3, H=96, tipo="f16", qkv=False, gelu=True, pre_ln=False, in_norm=False,
                          eps=1e-6, media=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 2),
}


# ---------------------------------------------------------------- reference with libmtmd
def referencia(ruta_lm, ruta_mm, rgbs, f32=False):
    import llama_cpp
    from llama_cpp import Llama, mtmd_cpp as M
    llm = Llama(model_path=ruta_lm, n_ctx=4096, n_batch=512, n_ubatch=512, verbose=False, n_threads=2, seed=0, **(dict(type_k=0, type_v=0) if f32 else {}))
    lctx = llm._ctx.ctx
    pr = M.mtmd_context_params_default()
    pr.use_gpu = False; pr.n_threads = 2; pr.print_timings = False; pr.warmup = False
    mctx = M.mtmd_init_from_file(ruta_mm.encode(), llm._model.model, pr)
    assert mctx, "mtmd no pudo cargar el mmproj"
    marcador = M.mtmd_default_marker().decode()
    assert marcador == MARCADOR, marcador
    V = llama_cpp.llama_vocab_n_tokens(llama_cpp.llama_model_get_vocab(llm._model.model))
    out = {"prompts": {}, "vocab": V}
    casos = [(n, t, i, True) for n, t, i in PROMPTS] + [(n, t, i, False) for n, m, t, i in CHATS]
    for nombre, texto, imgs, especial in casos:
        bms = []
        for im in imgs:
            a = rgbs[im]
            h, w, _ = a.shape
            buf = (ctypes.c_uint8 * a.size).from_buffer_copy(a.tobytes())
            bms.append(M.mtmd_bitmap_init(w, h, buf))
        arr = (M.mtmd_bitmap_p_ctypes * len(bms))(*bms)
        chunks = M.mtmd_input_chunks_init()
        it = M.mtmd_input_text(); it.text = texto.encode(); it.text_len = len(texto.encode()); it.add_special = especial; it.parse_special = True
        rc = M.mtmd_tokenize(mctx, chunks, ctypes.byref(it), arr, len(bms))
        assert rc == 0, ("tokenize", rc)
        lista = []
        for i in range(M.mtmd_input_chunks_size(chunks)):
            ch = M.mtmd_input_chunks_get(chunks, i)
            ty = M.mtmd_input_chunk_get_type(ch)
            if ty == M.MTMD_INPUT_CHUNK_TYPE_TEXT:
                n = ctypes.c_size_t(0)
                p = M.mtmd_input_chunk_get_tokens_text(ch, ctypes.byref(n))
                lista.append({"tipo": "texto", "ids": [int(p[j]) for j in range(n.value)]})
            else:
                n = M.mtmd_input_chunk_get_n_tokens(ch)
                assert M.mtmd_encode_chunk(mctx, ch) == 0
                e = M.mtmd_get_output_embd(mctx)
                emb = np.ctypeslib.as_array(e, shape=(n * 64,)).copy()
                lista.append({"tipo": "imagen", "n": int(n), "emb": emb.tolist()})
        # evaluates everything and extracts the last token's logits + greedy generation
        llama_cpp.llama_memory_clear(llama_cpp.llama_get_memory(lctx), True)
        npast = llama_cpp.llama_pos(0)
        rc = M.mtmd_helper_eval_chunks(mctx, lctx, chunks, 0, 0, 512, True, ctypes.byref(npast))
        assert rc == 0, ("eval", rc)
        lg = llama_cpp.llama_get_logits_ith(lctx, -1)
        logits = np.ctypeslib.as_array(lg, shape=(V,)).copy()
        gen, pos = [], npast.value
        batch = llama_cpp.llama_batch_init(1, 0, 1)
        for _ in range(12):
            nxt = int(np.argmax(logits))
            if nxt == 4: break
            gen.append(nxt)
            batch.n_tokens = 1
            batch.token[0] = nxt; batch.pos[0] = pos; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1
            assert llama_cpp.llama_decode(lctx, batch) == 0
            pos += 1
            logits = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(lctx, -1), shape=(V,)).copy()
        llama_cpp.llama_batch_free(batch)
        out["prompts"][nombre] = {"texto": texto, "imgs": imgs, "chunks": lista, "n_total": int(npast.value),
                                  "logits": None, "generados": gen,
                                  "texto_gen": llm.detokenize(gen[:11]).decode("utf-8", "replace")}
        # the logits of the prompt's last token (before generating)
        llama_cpp.llama_memory_clear(llama_cpp.llama_get_memory(lctx), True)
        npast2 = llama_cpp.llama_pos(0)
        assert M.mtmd_helper_eval_chunks(mctx, lctx, chunks, 0, 0, 512, True, ctypes.byref(npast2)) == 0
        out["prompts"][nombre]["logits"] = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(lctx, -1), shape=(V,)).copy().tolist()
        M.mtmd_input_chunks_free(chunks)
        for b in bms: M.mtmd_bitmap_free(b)
    M.mtmd_free(mctx)
    return out


if __name__ == "__main__":
    rgbs = {}
    for k, (w, h, s) in IMAGENES.items():
        _, a = hacer_imagen(k, w, h, s)
        rgbs[k] = a
    refs = {}
    for nombre, (o, sem) in VARIANTES.items():
        if sys.argv[1:] and nombre not in sys.argv[1:]: continue
        lm, d = crear_lm(nombre, sem, f32=o.get("lm_f32", False))
        mm = crear_mmproj(nombre, o, d)
        r = referencia(lm, mm, rgbs, f32=o.get("lm_f32", False))
        r["opciones"] = {k: v for k, v in o.items()}
        refs[nombre] = r
        print(nombre, {p: [(c["tipo"][0], len(c["ids"]) if c["tipo"] == "texto" else c["n"]) for c in v["chunks"]] for p, v in r["prompts"].items()})
    ruta = os.path.join(AQUI, "cache", "referencias_vis.json")
    if sys.argv[1:] and os.path.exists(ruta):
        viejo = json.load(open(ruta))["modelos"]; viejo.update(refs); refs = viejo
    json.dump({"imagenes": {k: {"w": v.shape[1], "h": v.shape[0]} for k, v in rgbs.items()}, "modelos": refs}, open(ruta, "w"))
