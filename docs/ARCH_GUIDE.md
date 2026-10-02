# Architecture guide (for engineers and for AI coding assistants)

How to teach the engine a new model architecture **without editing the core**, and how to prove it is correct.
If you are an AI assistant asked to add an architecture: read this whole file, then follow [the workflow](#workflow).
Never declare success from "the output looks plausible"; LLM bugs are silent. Only the conformance tests decide.

## 1. How architectures are discovered

The architecture name is the `general.architecture` string of the GGUF file (the same names llama.cpp uses).
Every file `web/arch/<id>.json` is a manifest. The server reads them at start and each time you press *Rescan*;
the engine window reads them from `/api/architectures` before loading a model. **Dropping a manifest and rescanning is enough.**

```json
{
  "id": "qwen3",
  "version": 1,
  "arch": ["qwen3"],
  "base": "transformer",
  "description": "free text",
  "options": { "rope": "neox" },
  "fallback_template": "chatml",
  "embedding": false,
  "vision": []
}
```

| field | meaning |
|---|---|
| `id` | unique manifest id |
| `arch` | list of `general.architecture` values it handles |
| `base` | **declarative**: reuse a core class: `transformer`, `lfm2`, `bert`, `deepseek2` |
| `module` | **code**: a file in `web/arch/` (e.g. `"my_arch.js"`) whose **default export** is the model class. Exactly one of `base`/`module` |
| `options` | options passed to the class as `meta.arch_options` (for `transformer`: `rope`: `"neox"` or `"norm"`; `embedding_scale`: a number or `"sqrt_d"`) |
| `fallback_template` | chat template used if the GGUF has none or the Jinja interpreter cannot run it: `chatml` or the name of a built-in family (`qwen2`, `llama`, `gemma3`, `granite`, `deepseek2`, `lfm2`) |
| `embedding` | `true` for embedding models (served by `/v1/embeddings`) |
| `vision` | list of `clip.projector_type` values of supported `mmproj` files (needs a model class that accepts external embeddings) |

An invalid manifest is ignored with a warning in the log; it never stops the server.
Security note: a `module` is JavaScript executed in the engine window on the server PC. Only install modules you have read.

### What the `transformer` base already handles (detected from tensors/KV, no option needed)
Grouped-query attention; explicit `attention.key_length` (head dim different from `embedding_length/head_count`);
Q/K/V/O biases; per-head Q/K RMSNorm (`attn_q_norm`/`attn_k_norm`); post-attention and post-FFN norms; sliding-window
layers in a pattern (Gemma 3); tied or separate output; SwiGLU and GeGLU; Granite scales; final logit softcapping; K-quant, Q8_0,
F16/BF16/F32 weights. If your model only differs in those knobs, a manifest is all you need (this is how `qwen3` was added).

### Not handled (needs a code module or a core change)
Fused QKV or fused gate/up tensors (Phi-3), attention softcapping (Gemma 2), recurrent/SSM layers (Mamba), `rope_freqs.weight`
(scaled RoPE), head dimension above 256, new tokenizer types (`web/js/tokenizer.js` supports 15 `tokenizer.ggml.pre` values),
and new chat templates that the Jinja interpreter in `templates.py` cannot run.

## 2. The model class interface (for `module` plugins)

```js
export default class MyModel {
  constructor(gpu, meta)            // meta: { kv, tensors, data_start, ctx, overrides, options, arch_options }
  async load(url, progress)         // stream weights from `url` (HTTP ranges), progress(0..1)
  reset()                           // new conversation: position back to 0
  async forward(ids, wantLogits, ext) // process up to `this.batch` tokens; return Float32Array logits of the last token or null
  free()                            // destroy every GPU buffer
  // properties used by the engine: nLayers, D (embedding size), ctx, batch (tokens per prefill pass), vocab
  // optional: embed(ids) -> Float32Array for embedding models; supportsImages = true if forward accepts `ext`
}
// `meta.tensors[i]` records still use the field names nombre, dims, tipo, tipo_nombre, bytes, abs (to be renamed in a later pass).
```
Start from `web/js/transformer.js` (the best-documented example) and use the helpers of `web/js/gpu.js`
(`GPU`, `despachar`, `dim2`; helper names are still Spanish in this release), `web/js/weights.js` (`CargadorPesos`: reads GGUF tensors, keeps K-quants raw on the GPU) and `web/js/kquants.js`.

## 3. Workflow

1. **Read the reference.** Open llama.cpp `src/models/<arch>.cpp` (or `llama-model.cpp`/`llama-graph.cpp` in older trees) and write down
   tensor names, KV keys, norm placement, activation, RoPE type, attention mask, scales. Compare with `transformer.js`.
2. **Write the manifest** (declarative) or the module (code).
3. **Add a generator variant** in `tests/conformance/gen_transformer.py` (`VARIANTES`): it writes a tiny random model
   of that architecture with the `gguf` package, quantizes it with llama.cpp, dequantizes a copy to F32, and records llama.cpp's
   token ids, last-token logits and greedy continuation. Exercise the odd cases (explicit head dim, bias, tied output, window).
4. **Run the conformance test** (`python tests/conformance/test_transformer.py <variant>`), then the end-to-end one
   (`... e2e`) that goes through the real HTTP API, the engine and the chat template.
5. **Pass criteria**: identical token ids; relative logit difference vs the F32-dequantized reference `< 2e-3`; greedy continuation identical
   (a documented tie with logit gap `< 0.002` may differ); engine output text identical to llama.cpp; no JS console errors.
   The difference vs llama.cpp on the *quantized* file is larger (`~3e-2`) because llama.cpp's CPU path quantizes activations to Q8; that is expected.
6. **Mutation test**: temporarily disable the feature you implemented (window, norm, bias...). The tests must fail. If they still pass, the
   test does not cover it; fix the generator.
7. **Real model**: run `python tools/verify.py` against a real GGUF and a reference (LM Studio or llama.cpp server) on the target GPU.
   Synthetic tests prove correctness, not speed or memory.

## 4. Pitfalls we already paid for

- **Silent plausibility.** Wrong attention masks and swapped tensors still produce fluent text. Compare numbers, not prose.
- **Gemma 3 vision:** image tokens use *non-causal* attention inside the image block, while the sliding window is measured from the
  query position (llama.cpp mask: `p1 - p0 >= n_swa`). Some older `mmproj` files have `ffn_up`/`ffn_down` swapped (llama.cpp reorders them).
- **`cos`/`sin` in WebGPU can have ~2e-4 error** on some drivers. RoPE uses its own sine/cosine (error ~5e-6).
- **Windows GPU watchdog (TDR) resets the driver after ~2 s of one submission.** Submit per layer when prefilling more than 32 tokens and split large attention.
- **Dispatch limit:** 65535 workgroups per dimension. Use `dim2()` and index with `num_workgroups` (the `bias`, `suma_esc`, `activacion` kernels do).
- **Reference precision:** llama.cpp with an f16 KV cache drifts ~1e-2 from its own exact result on deep MoE models. Generate references with an f32 KV cache and compare
  against the dequantized F32 file, not against llama.cpp's quantized one.
- **BOS handling:** if the chat template already emits BOS, tokenize with `add_bos=false`.
- **WGSL:** dynamic indexing of local arrays is very slow on integrated GPUs; prefer shared memory tiles. Check shader compile time on the real GPU (a 2.1 release hung on first compile).
- **Tied embeddings** (no `output.weight`), `rope_freqs.weight` and head dim `> 256` are the classic reasons a new model fails to load.
- **Memory:** WebGPU does not report free VRAM. Weights stay compressed on the GPU (K-quants decoded inside the kernel); the KV cache is f32.

## 5. Contributing an architecture

Open a pull request with: the manifest (and module), the generator variant, the passing `test_transformer.py` output (including the mutation
check), and one real-model measurement (model, GPU, tokens/s, parity result). Keep modules text-only (no `.wasm`, no binaries).
