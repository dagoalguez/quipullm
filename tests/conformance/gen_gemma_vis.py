"""Synthetic Gemma 3 models (LM + 'gemma3' mmproj: SigLIP + pool + RMSNorm + projection) and references from libmtmd."""
import ctypes, json, os, sys
import numpy as np
import gguf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_gemma as GG
import gen_vision as GV

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "cache", "modelos_visg")
os.makedirs(OUT, exist_ok=True)
MARCADOR = GV.MARCADOR
D_LM = 256

_vocab_orig = GG.vocab_spm
def vocab_ext(n_piezas=300):
    t, s, ty = _vocab_orig(n_piezas)
    for x in ("<start_of_image>", "<end_of_image>", "<image_soft_token>"):
        t.append(x); s.append(0.0); ty.append(4)
    return t, s, ty
GG.vocab_spm = vocab_ext

PROMPTS = [
    ("p_a", "<bos>Mira esto: " + MARCADOR + " ¿Qué dice?", ["a"]),
    ("p_b", "<bos>" + MARCADOR + "\nDescribe la imagen.", ["b"]),
    ("p_c", "<bos>Texto antes. " + MARCADOR + " Texto después.", ["c"]),
    ("p_d", "<bos>" + MARCADOR + " Lee el total.", ["d"]),
    ("p_dos", "<bos>Primera " + MARCADOR + " segunda " + MARCADOR + " fin", ["e", "a"]),
]
def T(t): return {"type": "text", "text": t}
def I(k): return {"type": "image", "k": k}
SOT, EOT, NL = "<start_of_turn>", "<end_of_turn>", "\n"
CHATS = [
    ("c_1", [{"role": "user", "content": [T("Lee esto:"), I("a"), T("¿Qué dice?")]}],
     "<bos>" + SOT + "user\nLee esto:" + MARCADOR + "¿Qué dice?" + EOT + NL + SOT + "model\n", ["a"]),
    ("c_2", [{"role": "system", "content": "Eres útil."}, {"role": "user", "content": [I("b"), T("Describe la imagen.")]}],
     "<bos>" + SOT + "user\nEres útil.\n\n" + MARCADOR + "Describe la imagen." + EOT + NL + SOT + "model\n", ["b"]),
    ("c_3", [{"role": "user", "content": [T("Mira"), I("e")]}, {"role": "assistant", "content": "Bien."},
             {"role": "user", "content": [I("a"), T("¿y esta?"), T("Gracias")]}],
     "<bos>" + SOT + "user\nMira" + MARCADOR + EOT + NL + SOT + "model\nBien." + EOT + NL + SOT + "user\n" + MARCADOR + "¿y esta?\nGracias" + EOT + NL + SOT + "model\n", ["e", "a"]),
    ("c_4", [{"role": "user", "content": [T("Factura:"), I("c")]}],
     "<bos>" + SOT + "user\nFactura:" + MARCADOR + EOT + NL + SOT + "model\n", ["c"]),
]

VARIANTES = {
    # LM F32 and mmproj F32: exact comparison
    "gemma3-vl-f32": dict(semilla=31, tipo="f32", img=224, parche=14, merge=2, D=96, nh=4, F=192, L=2, eps=1e-6, gelu=True,
                          media=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                          lm=dict(semilla=41, ventana=8, patron=3, rope_escala=8.0, softcap=0.0, add_espacio=False, nkv=2, nh=4, hd=64, L=6)),
    # mmproj F16, 12x12 grid with reduction 3 (16 tokens), LM with window 16 and no pattern
    "gemma3-vl-f16": dict(semilla=32, tipo="f16", swap=True, img=168, parche=14, merge=3, D=72, nh=4, F=144, L=3, eps=1e-6, gelu=True,
                          media=[0.48, 0.45, 0.40], std=[0.27, 0.26, 0.28],
                          lm=dict(semilla=42, ventana=16, patron=None, rope_escala=None, softcap=0.0, add_espacio=False, nkv=1, nh=4, hd=64, L=4)),
}


def crear_mmproj(nombre, o):
    rng = np.random.default_rng(o["semilla"])
    ruta = os.path.join(OUT, nombre, "mmproj-" + nombre + ".gguf")
    D, nh, F, L, P = o["D"], o["nh"], o["F"], o["L"], o["parche"]
    NP = (o["img"] // P) ** 2
    w = gguf.GGUFWriter(ruta, "clip")
    w.add_name("mmproj-" + nombre)
    w.add_bool("clip.has_vision_encoder", True)
    w.add_string("clip.projector_type", "gemma3")
    if o["gelu"]: w.add_bool("clip.use_gelu", True)
    w.add_uint32("clip.vision.image_size", o["img"])
    w.add_uint32("clip.vision.patch_size", P)
    w.add_uint32("clip.vision.embedding_length", D)
    w.add_uint32("clip.vision.feed_forward_length", F)
    w.add_uint32("clip.vision.block_count", L)
    w.add_uint32("clip.vision.projection_dim", D_LM)
    w.add_uint32("clip.vision.attention.head_count", nh)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", o["eps"])
    w.add_array("clip.vision.image_mean", o["media"])
    w.add_array("clip.vision.image_std", o["std"])
    w.add_uint32("clip.vision.projector.scale_factor", o["merge"])
    tw = np.float32 if o["tipo"] == "f32" else np.float16

    def mat(n, N, K, e=1.0, forma=None):
        a = (rng.standard_normal((N, K)) * e / np.sqrt(K)).astype(np.float32)
        if forma: a = a.reshape(forma)
        w.add_tensor(n, a.astype(tw))
    def vec(n, k, c=0.0, r=0.1):
        w.add_tensor(n, (c + r * rng.standard_normal(k)).astype(np.float32))

    mat("v.patch_embd.weight", D, 3 * P * P, 2.0, (D, 3, P, P))
    vec("v.patch_embd.bias", D)
    w.add_tensor("v.position_embd.weight", (0.5 * rng.standard_normal((NP, D))).astype(np.float32))
    for i in range(L):
        p = "v.blk.%d." % i
        vec(p + "ln1.weight", D, 1.0, 0.1); vec(p + "ln1.bias", D)
        vec(p + "ln2.weight", D, 1.0, 0.1); vec(p + "ln2.bias", D)
        for n in ("q", "k", "v"):
            mat(p + "attn_%s.weight" % n, D, D, 1.6); vec(p + "attn_%s.bias" % n, D)
        mat(p + "attn_out.weight", D, D, 1.0); vec(p + "attn_out.bias", D)
        # o["swap"]: swapped names as in the old Gemma 3 mmproj files (fc1 -> ffn_down, fc2 -> ffn_up)
        nu, nd = ("ffn_down", "ffn_up") if o.get("swap") else ("ffn_up", "ffn_down")
        mat(p + nu + ".weight", F, D, 2.0); vec(p + nu + ".bias", F)
        mat(p + nd + ".weight", D, F, 1.0); vec(p + nd + ".bias", D)
    vec("v.post_ln.weight", D, 1.0, 0.1); vec("v.post_ln.bias", D)
    vec("mm.soft_emb_norm.weight", D, 1.0, 0.2)
    # W stored as [D_vit][D_lm] (ggml ne=[D_lm, D_vit]): output = x @ W
    w.add_tensor("mm.input_projection.weight", (rng.standard_normal((D, D_LM)) * 2.0 / np.sqrt(D)).astype(tw))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return ruta


def referencia(ruta_lm, ruta_mm, rgbs):
    import llama_cpp
    from llama_cpp import Llama, mtmd_cpp as M
    llm = Llama(model_path=ruta_lm, n_ctx=4096, n_batch=512, n_ubatch=512, verbose=False, n_threads=2, seed=0, type_k=0, type_v=0, flash_attn=False)
    lctx = llm._ctx.ctx
    pr = M.mtmd_context_params_default()
    pr.use_gpu = False; pr.n_threads = 2; pr.print_timings = False; pr.warmup = False
    mctx = M.mtmd_init_from_file(ruta_mm.encode(), llm._model.model, pr)
    assert mctx, "mtmd no pudo cargar el mmproj"
    assert M.mtmd_default_marker().decode() == MARCADOR
    V = llama_cpp.llama_vocab_n_tokens(llama_cpp.llama_model_get_vocab(llm._model.model))
    eos = llm.token_eos()
    out = {"prompts": {}, "vocab": V}
    casos = [(n, t, i, True) for n, t, i in PROMPTS] + [(n, t, i, False) for n, m, t, i in CHATS]
    for nombre, texto, imgs, especial in casos:
        bms = []
        for im in imgs:
            a = rgbs[im]; h, w, _ = a.shape
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
                emb = np.ctypeslib.as_array(e, shape=(n * D_LM,)).copy()
                lista.append({"tipo": "imagen", "n": int(n), "emb": emb.tolist()})
        def evaluar():
            llama_cpp.llama_memory_clear(llama_cpp.llama_get_memory(lctx), True)
            npast = llama_cpp.llama_pos(0)
            rc = M.mtmd_helper_eval_chunks(mctx, lctx, chunks, 0, 0, 512, True, ctypes.byref(npast))
            assert rc == 0, ("eval", rc)
            return npast.value
        npast = evaluar()
        logits = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(lctx, -1), shape=(V,)).copy()
        lp = logits.copy()
        gen, pos = [], npast
        batch = llama_cpp.llama_batch_init(1, 0, 1)
        for _ in range(12):
            nxt = int(np.argmax(logits))
            if nxt == eos: break
            gen.append(nxt)
            batch.n_tokens = 1
            batch.token[0] = nxt; batch.pos[0] = pos; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = 1
            assert llama_cpp.llama_decode(lctx, batch) == 0
            pos += 1
            logits = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(lctx, -1), shape=(V,)).copy()
        llama_cpp.llama_batch_free(batch)
        out["prompts"][nombre] = {"texto": texto, "imgs": imgs, "chunks": lista, "n_total": int(npast), "logits": lp.tolist(), "generados": gen,
                                  "texto_gen": llm.detokenize(gen[:11]).decode("utf-8", "replace")}
        M.mtmd_input_chunks_free(chunks)
        for b in bms: M.mtmd_bitmap_free(b)
    M.mtmd_free(mctx)
    return out


if __name__ == "__main__":
    rgbs = {}
    for k, (w, h, s) in GV.IMAGENES.items():
        _, a = GV.hacer_imagen(k, w, h, s)
        rgbs[k] = a
    refs = {}
    for nombre, o in VARIANTES.items():
        if sys.argv[1:] and nombre not in sys.argv[1:]: continue
        carpeta = os.path.join(OUT, nombre); os.makedirs(carpeta, exist_ok=True)
        f32 = GG.crear_f32(nombre, o["lm"])
        lm = os.path.join(carpeta, nombre + ".gguf")
        os.replace(f32, lm)
        mm = crear_mmproj(nombre, o)
        r = referencia(lm, mm, rgbs)
        r["opciones"] = {k: v for k, v in o.items() if k != "lm"}
        refs[nombre] = r
        print(nombre, {p: [(c["tipo"][0], len(c["ids"]) if c["tipo"] == "texto" else c["n"]) for c in v["chunks"]] for p, v in r["prompts"].items()})
        print("  generados:", {p: v["texto_gen"][:30] for p, v in r["prompts"].items()})
    ruta = os.path.join(AQUI, "cache", "referencias_visg.json")
    if sys.argv[1:] and os.path.exists(ruta):
        viejo = json.load(open(ruta))["modelos"]; viejo.update(refs); refs = viejo
    json.dump({"modelos": refs}, open(ruta, "w"))
