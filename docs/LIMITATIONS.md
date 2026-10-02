# Limitations (honest list)

**Hardware and platform**
- The comparison against LM Studio was done on one machine only: Intel Core i7-1270P, Intel UHD (Gen12 LP) integrated GPU, 15.6 GB RAM, Windows, Edge. Two more machines were measured later without LM Studio (NVIDIA GTX 1050 Ti on Windows 11 + Edge; Intel HD 4000 on Linux Mint + Chrome; see [BENCHMARKS.md](BENCHMARKS.md)) with only two runs per model. AMD and Apple GPUs *should* work through WebGPU but are **unmeasured**.
- WebGPU is slower than native CUDA/Metal/Vulkan inference for the same card. Expect a ceiling below llama.cpp's native backends.
- The engine needs a browser window open on the server PC in a user session (no headless service mode yet).
- WebGPU does not report free VRAM. The "fits / tight / does not fit" labels are **estimates** calibrated on the models we tested.
- **Linux:** on an old Intel HD 4000 laptop (Linux Mint 22.3, Chrome 154) the engine only worked after starting Chrome with `--enable-unsafe-webgpu --enable-features=Vulkan --ignore-gpu-blocklist`; without them WebGPU returned no adapter even though `chrome://gpu` listed the GPU. Which flag is required was not tested, and recent Linux GPUs were not tested. Firefox is not supported for the engine on Linux (no WebGPU by default there). From 4.0.1 the server prefers `google-chrome`, `chromium` or `brave-browser` on `PATH` for the engine window, and falls back to the default browser. Set `browser` in `config.json` to a full path to choose one, or use `--no-engine` and open `http://localhost:1234/engine` yourself in a browser started with the flags you need.
- macOS launch of the engine window is untested.

**Serving**
- **One request at a time.** Others wait in a FIFO queue (limit `max_queue`, default 64, then HTTP 429). Switching between models reloads the model.
- No continuous batching, no prompt/KV cache reuse between requests (long multi-turn prompts are recomputed).
- KV cache is f32.
- No tool/function calling and no grammar-constrained or JSON-schema-enforced output. `response_format: json_object` returns 400
  (as LM Studio did); `json_schema` is accepted but not enforced.
- Not implemented: logprobs, `n > 1`, multiple models loaded at once.

**Models**
- Supported architectures are listed in the README. DeepSeek is the *Lite* variant only. `rope_freqs.weight` models and head dimension > 256 are unsupported.
- Vision: LFM2-VL and Gemma 3 only (base64 images, 25 MB max). Gemma 3 encodes an image in ~30 s on the tested iGPU.
- **Vision follows llama.cpp as of `llama-cpp-python` 0.3.35 (ggml 0.20.0) and not newer.** With synthetic models the three image suites
  (LFM2-VL encoder 24/24 and API 41/41, Gemma 3 encoder 12/12) pass against 0.3.35 and fail against 0.3.36 (ggml 0.25.3: 9/24, 34/43, 6/12),
  because 0.3.36 changed (1) the *bilinear* resize, now a Pillow-style triangle filter that widens when downscaling instead of the old
  4-point interpolation (affects Gemma 3 and every LFM2 resize), (2) LFM2 single-tile images, which no longer get the `<|img_thumbnail|>`
  token (one token less per image; the engine still emits it), and (3) the LFM2 tiling rule, now based on the rounded pixel area against
  `image_max_pixels` x 2 instead of each side against 2 x 512 px (the synthetic images do not exercise this one). **Not measured:** how much these
  differences change the output of the real models, and which llama.cpp build LM Studio 0.4.25 used when the reference outputs were recorded.
- 7–8B models run at roughly 2–3 tokens/s on the tested iGPU; 1B-class models at ~15 tokens/s.

**Project**
- Many internal identifiers (function and variable names) are still in Spanish; interface, messages, config keys, HTTP API, comments and docstrings are English. The server listens on 127.0.0.1 by default; on the LAN it is open unless a key is set ([SECURITY.md](../SECURITY.md)).
