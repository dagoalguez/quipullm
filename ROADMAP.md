# Roadmap (v4 and beyond)

This is a list of intentions, **not a promise or a schedule**. Items are ordered by how much they would help real users versus how risky they are. Effort labels (S / M / L) are our estimates, not measurements. Everything here is **not started** unless stated. Want to take one? Open an issue first so we can agree on the test that would prove it works.

The rule for every item is the same as for the existing engine: it is done when the conformance tests say so, not when the output "looks right" (see [How it is verified](README.md#how-it-is-verified)).

## Summary

| # | Item | Why it matters | Effort | Risk | How it would be verified |
|---|---|---|---|---|---|
| 1 | Windowless / service mode and Linux & macOS launch | Unattended servers; reach beyond Windows | M | Medium | Real-GPU runs on Linux, macOS, Windows (needs hardware) |
| 2 | Enforced `response_format` (`json_object` and `json_schema`) | Reliable structured output for apps | M–L | Medium | Synthetic models (output must always validate) |
| 3 | Prompt caching (reuse the KV cache across requests) | Faster multi-turn chat and agents | M | Medium | Token-for-token equality with and without the cache |
| 4 | Tool / function calling | Compatibility with agent frameworks | L | Medium–High | Template rendering vs jinja2; parser tests; real-model checks |
| 5 | f16 KV cache | Half the memory per token of context | M | Medium–High | Compare against the f32 path and an f32 reference |
| 6 | `logprobs`, `n > 1` and other API gaps | Drop-in compatibility | S–M | Low | API tests |
| 7 | Benchmarks on other hardware | We have measured three machines (one against LM Studio) | S (per machine) | Low | `tools/bench.py` output |
| 8 | Vision: follow newer llama.cpp preprocessing | Parity with current llama.cpp for LFM2-VL and Gemma 3 | M | Medium | New references and a full `--engine` run |
| 9 | Later, not planned for v4 | Continuous batching, several models in memory, more architectures | L | High | n/a |

## 1. Windowless / service mode and Linux & macOS launch

**Today:** the engine needs a browser window open on the server PC. The server looks for Edge or Chrome only in Windows locations; on Linux and macOS it falls back to the default browser in a normal tab, and the launch has **not been tested on a real GPU** there.

**What we would do:** (a) find Chrome/Chromium/Edge on Linux and macOS and pass the flags the engine needs; (b) an optional hidden or headless launch mode so the server can run unattended; (c) documentation for running it as a service.
The test suite already runs the real engine in headless Chromium with SwiftShader (software WebGPU), so headless operation works there. Whether headless Chromium reaches a **real** GPU is **unmeasured** and is the first thing to find out.

**Verification:** the same engine suite plus real-GPU runs on each OS. This item depends on benchmarks from other machines (item 7).

## 2. Enforced `response_format` (`json_object` and `json_schema`)

**Today:** `json_object` returns HTTP 400 (as LM Studio did); `json_schema` is accepted but **not enforced**.

**Why it matters:** applications that parse model output need it to be valid every time, not most of the time.

**What we would do:** token sampling already happens in JavaScript on the logits (`web/js/sampling.js`), so a constraint can be applied there by masking tokens that would break the format. A first step is "valid JSON only"; then a supported subset of JSON Schema. The hard part is speed: masks over a vocabulary of many thousands of tokens must be cached or computed incrementally. We do not yet know the cost.

**Verification:** this one is testable with the synthetic random-weight models: with enforcement on, **every** output must parse and validate against the schema, whatever the model says. A negative control (enforcement off) must produce invalid outputs.

## 3. Prompt caching

**Today:** each request resets the model and recomputes the whole prompt, so long multi-turn conversations pay for the full history every time. For scale, measured prefill on the test machine: a 1.2B LFM2 took about 2.3 s for 62 tokens (about 28 tok/s); an 8B Granite 2.3–3.4 s for 58–76 tokens (see [BENCHMARKS.md](docs/BENCHMARKS.md)).

**What we would do:** keep the KV cache between requests and reuse the longest common prefix of token ids; only compute the new tokens. Careful handling is needed for sliding-window layers (Gemma 3), the convolution state of LFM2 and the compressed cache of DeepSeek, which do not all behave like a plain KV cache.

**Verification:** greedy output with the cache must equal greedy output without it, token for token, for every supported architecture.

## 4. Tool / function calling

**Today:** not supported.

**What we would do:** accept `tools` and `tool_choice`; render them into the model's chat template (the template interpreter in `templates.py` already supports `tojson` and `namespace`, which tool templates use); parse the model's tool-call output into the OpenAI `tool_calls` format, including streaming. The output format differs between model families, so it would start with one or two families and grow.

**Verification:** template rendering compared with jinja2 (as in the existing template tests) and parser unit tests. Synthetic random models cannot produce meaningful tool calls, so **end-to-end quality needs real models** and would be reported only as measured. This item works better after item 2 (enforced JSON).

## 5. f16 KV cache

**Today:** the KV cache is f32, which limits context length by memory (`kv_max_mb`).

**What we would do:** store K and V as f16 (WGSL's `pack2x16float` / `unpack2x16float` are already used for f16 weights, so the optional `shader-f16` extension should not be required).

**Why it is risky:** precision. In our own validation work, an f16 KV cache in llama.cpp drifts about 1e-2 from its own exact result on deep MoE models. Any f16 mode must be compared against our f32 path and against an f32 reference, and may need to stay optional.

**Verification:** relative logit difference against the f32 path on every architecture, with the tolerance stated up front; DeepSeek-V2 first, because it is the sensitive case.

## 6. API gaps

`logprobs` and `n > 1` are not implemented. Since the logits are already available in JavaScript for sampling, `logprobs` should be cheap; `n > 1` needs repeated generation in one request. Verified with API tests.

## 7. Benchmarks on other hardware

The only LM Studio comparison comes from one machine (Intel i7-1270P, Intel UHD, Windows, Edge). Two more machines were measured without LM Studio (GTX 1050 Ti on Windows; Intel HD 4000 on Linux, two runs per model). AMD, Apple and recent NVIDIA/Linux GPUs are **unmeasured**, and so are the memory-fit estimates on other hardware. This is the most useful contribution right now: run `tools/bench.py` and open a *Benchmark report* issue (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## 8. Vision: follow newer llama.cpp preprocessing

**Today:** vision follows llama.cpp as of `llama-cpp-python` 0.3.35 and fails the image suites against 0.3.36 because upstream changed image preprocessing (see [LIMITATIONS.md](docs/LIMITATIONS.md)). **Not measured:** how much those differences change the output of the real models.

**Options:** (A) stay pinned and documented (current); (B) follow upstream (Pillow-style bilinear resize, no `<|img_thumbnail|>` for single-tile LFM2 images, area-based tiling rule), regenerate the references and re-run the full `--engine` suite; (C) support both behaviours behind a switch and keep two reference sets. We would decide after measuring real-model output with both.

## 9. Not planned for v4

Continuous batching, several models loaded at once, and models that need new core code (fused QKV, attention softcapping, recurrent/SSM layers, scaled RoPE). They are large changes with a high risk of silent errors. New architectures that fit the existing base classes can already be added by contributors ([docs/ARCH_GUIDE.md](docs/ARCH_GUIDE.md)).
