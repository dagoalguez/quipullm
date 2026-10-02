# Architecture

```
 client (OpenAI SDK, n8n, curl, notebooks)           browser tab on the server PC
        │  HTTP :1234                                  (Edge --app window, WebGPU)
        ▼                                                     ▲   │
 ┌────────────────────────── server.py ───────────────────────┴───┴───────────┐
 │  ThreadingHTTPServer (one thread per request)                                  │
 │   /v1/*  OpenAI API ──► chat templates, sampling params ──► FIFO job queue     │
 │   /api/* panel, status, config, folder browser, architectures                  │
 │   /engine/*  (127.0.0.1 only) long-poll for jobs, event stream back,            │
 │             GGUF served by byte ranges + parsed metadata as JSON               │
 └────────────────────────────────────────────────────────────────────────────────┘
```

**One process, one engine, one job at a time.** The server parses GGUF headers (pure Python), applies the chat template (a
small Jinja interpreter in `templates.py`), puts a job in the queue and streams the engine's token events back as SSE.
The engine page (`web/engine.html` + `web/js/*`) long-polls `/engine/api/next`, loads the model into GPU buffers by streaming the
file from the server, tokenizes (BPE/SentencePiece/WordPiece reimplemented to match llama.cpp), runs the forward pass in WGSL,
samples, and posts tokens back.

## Why the engine lives in a browser tab
WebGPU gives portable GPU compute with **no native code to install**. The tab is launched by the server (`--app` window with its own profile).
That is what makes the repository text-only and runnable behind a proxy that blocks binaries. The cost is performance and
control (see [LIMITATIONS.md](LIMITATIONS.md)).

## Components
| file | role |
|---|---|
| `server.py` | HTTP server, API, queue, config, memory estimation, GGUF reader, panel API |
| `templates.py` | Jinja subset interpreter for `tokenizer.chat_template` |
| `web/arch/*.json` | architecture manifests ([ARCH_GUIDE](ARCH_GUIDE.md)) |
| `web/js/engine.js` | engine loop, job handling, model lifecycle, architecture registry |
| `web/js/gpu.js` | WebGPU device, all WGSL kernels, helpers |
| `web/js/weights.js`, `kquants.js`, `kds.js` | GGUF tensor loading; K-quant / IQ4_NL decoding inside the kernels |
| `web/js/transformer.js` | dense decoder (qwen2/qwen3/llama/granite/gemma3) |
| `web/js/lfm2.js`, `deepseek2.js`, `bert.js`, `vision.js` | LFM2, DeepSeek MLA+MoE, embeddings, ViT encoders |
| `web/js/tokenizer.js`, `sampling.js` | tokenizers, sampler (temperature, top-k/p, min-p, repetition penalty, seed) |
| `web/panel.html` | control panel |
| `tests/` | API tests (simulated engine) and conformance tests vs llama.cpp |

## Key design decisions
- **Weights stay quantized on the GPU.** K-quants are decoded inside the matmul kernels; the KV cache is f32 (halving it with f16 is a TODO).
- **Prefill uses tiled matmul kernels**; decoding uses 1-token kernels. One submission per layer when prefilling >32 tokens to avoid Windows TDR resets.
- **Sliding-window layers use a ring buffer** (`window + tokens-per-batch` positions).
- **Context is limited by a memory estimate** (`kv_max_mb`, or automatic from the hardware budget).
- **The engine is replaceable per model:** every architecture is a class with `load/forward/reset/free`, registered by manifest.

## Correctness methodology
Each architecture is verified with synthetic models (random weights, tiny size) written with the `gguf` package, quantized by llama.cpp, and compared to
llama.cpp on token ids, last-token logits (vs a dequantized F32 copy) and greedy text, then end-to-end through the HTTP API; negative controls must fail.
See `tests/conformance/` and [ARCH_GUIDE.md](ARCH_GUIDE.md).
