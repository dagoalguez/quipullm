# How we validated a from-scratch WebGPU LLM engine, number by number, against llama.cpp

*Code: [github.com/dagoalguez/quipullm](https://github.com/dagoalguez/quipullm) (Apache-2.0). Every figure below comes from the repository's test output or from `docs/BENCHMARKS.md` / `docs/LIMITATIONS.md`. Speed numbers come from three machines (listed below); the only one compared against LM Studio is the Intel Core i7-1270P + Intel UHD integrated GPU, Windows, Edge. The other two were run twice per model, without LM Studio.*

## The problem

quipullm is a self-hosted, OpenAI- and LM Studio-compatible LLM server. Its inference engine is written from scratch in WGSL and JavaScript and runs in a browser window; the backend is Python standard library only. It reads standard GGUF files.

A new inference engine has one failure mode that matters more than any crash: **it produces fluent text that is subtly wrong.** A swapped tensor, a wrong attention mask or a missing bias rarely gives gibberish. It gives plausible prose. Reading outputs and saying "looks right" proves nothing. So we compared numbers, not prose, with llama.cpp as the reference.

## Method 1: synthetic models, so the reference is exact and cheap

Real models are large, slow to run in a software-rendered browser, and cannot be shipped in a repo. Instead, generators in `tests/conformance/` write **tiny random-weight models** with the `gguf` Python package for each architecture (Qwen 2/3, Llama-style, Granite, Gemma 3, LFM2, nomic-BERT, DeepSeek-V2 with MLA + MoE + YaRN, and the LFM2-VL / Gemma 3 vision stacks). Odd cases are built in on purpose: explicit head dimension, biases, tied and untied output, sliding windows, different MoE expert counts.

llama.cpp then quantizes each model (Q2_K through Q6_K, Q8_0 in the variants we generate) and records its **token ids, last-token logits and greedy continuation**. The tests run in headless Chromium with SwiftShader (software WebGPU). That proves correctness, not speed or memory; the repository says so everywhere it matters.

## Method 2: compare against the right thing

For each prompt the pass criteria are:

1. **Token ids identical** to llama.cpp's tokenizer output (tokenizer suite: 375 checks, 0 failures in our last full run; chat templates are checked against jinja2: 98 checks, 0 failures).
2. **Last-token logits within a relative difference < 2e-3** of an **F32 copy of the same weights, dequantized**.
3. **Greedy continuation identical** (one documented near-tie is whitelisted: logit gap below 0.002).
4. **End-to-end through the real HTTP API**, the engine page and the chat template; text identical, no JS console errors.

Why compare to a dequantized F32 copy instead of llama.cpp's own quantized output? On CPU, llama.cpp quantizes *activations* to Q8 before the matrix products. Our engine computes activations in f32. Against llama.cpp's quantized path the difference is about 3e-2, which is expected and uninformative; against the dequantized F32 weights it is a tight, meaningful bound. The same split shows up with real models: in our comparison against LM Studio 0.4.25 on the benchmark machine, prompt tokens matched in every test and greedy text matched in most (for example LFM2 44 of 48, Qwen2.5-Coder 8/8, Granite 8/8, Gemma 3 8/8), and every difference was a long text diverging at a near-tie in the logits with a correct meaning. That is the same Q8-activation effect.

## Method 3: negative controls and mutation tests

A test that cannot fail is decoration. We ask each feature to prove it is covered:

- **Mutation procedure (documented in `docs/ARCH_GUIDE.md`)**: temporarily disable the thing you implemented (window, norm, bias, ...). The tests must fail. If they still pass, the generator does not exercise the feature and must be fixed. This is a contributor workflow, not an automated suite.
- **Automated negative control**: in the Gemma 3 vision suite, a run with *causal-only* attention inside the image block must fail, because Gemma 3 uses non-causal attention among image tokens. `run_all.py --engine` reports it as passing only when the mutation is detected. In our last full run it did.

## What the process found

**1. `cos` / `sin` in WebGPU can have about 2e-4 error on some drivers.** That is far larger than a 2e-3 bound can tolerate once errors compound through layers and positions. RoPE therefore uses its own sine/cosine implementation, with error around 5e-6. We cannot tell you which drivers or GPUs show the 2e-4 error beyond "some"; we did not survey them.

**2. An f16 KV cache drifts in deep MoE models, in llama.cpp itself.** With an f16 KV cache, llama.cpp deviates about 1e-2 from its *own* exact result on deep MoE models. A reference that is itself noisy cannot validate anything at 2e-3. The fix is in the method: generate references with an f32 KV cache and compare against the dequantized F32 file. (Our own KV cache is f32; halving it with f16 is on the roadmap.)

**3. The reference moves.** For vision, we validated against `llama-cpp-python` 0.3.35. Against 0.3.36 (ggml 0.25.3), with the same engine code and the same synthetic models, the LFM2-VL encoder suite went from 24/24 to 9/24, the LFM2-VL API suite from 41/41 to 34 passed and 9 failed, and the Gemma 3 encoder suite from 12/12 to 6/12 (cosine 0.9956-0.9985 on the failures). Text, BERT and DeepSeek suites did not move. Three upstream changes explain it: the bilinear resize became a Pillow-style triangle filter that widens when downscaling; LFM2 single-tile images no longer get the `<|img_thumbnail|>` token (one token less per image); and the LFM2 tiling rule changed from per-side to rounded-pixel-area. Even the two reference versions' image embeddings differ from each other (minimum cosine 0.9898) on the same model and image.

The lesson is that image preprocessing is project policy of the reference, not model math. "Validated against llama.cpp" is only meaningful with a version attached. We pin `llama-cpp-python` 0.3.35, `gguf` 0.19.0 and `numpy` 2.4.4, and `run_all.py` warns when the environment differs.

**What we do not know:** how much these differences change the output of the *real* vision models, and which llama.cpp build LM Studio 0.4.25 used when we recorded the real-model references. Neither was measured.

## Platform pitfalls that no reference could catch

These came from running on a real iGPU, not from the conformance suite: Windows' GPU watchdog (TDR) resets the driver after about 2 seconds in one submission, so prefill is submitted per layer and large attention is split; WebGPU limits dispatches to 65535 workgroups per dimension; dynamic indexing of local arrays in WGSL is very slow on integrated GPUs; and WebGPU does not report free VRAM, so our "fits / tight / does not fit" labels are estimates.

## What running on other machines taught us

After the first release we measured two more machines (quipullm 4.0.1, two runs per model, no LM Studio there). Two findings are not about the engine's math:

- **A dedicated card was not automatically faster.** On an i5-8400 + GTX 1050 Ti 4 GB (Windows 11, Edge 143, adapter reported as "nvidia pascal") we got 42.6 / 42.0 tok/s on LFM2 350M Q8_0, 14.7 on LFM2.5 1.2B Q8_0 and 5.0 on Gemma 3 4B Q4_K_M: practically the same as the i7-1270P's integrated GPU (44.3, 15.7, 5.3). We did not investigate why; we have no profile showing whether the limit is the engine, WebGPU or the driver, so we make no claim about the cause.
- **An old Intel GPU on Linux worked, but only after launching Chrome with extra flags.** On an i5-3230M + Intel HD 4000 (Linux Mint 22.3, Chrome 154, adapter "intel gen-7"), Firefox (the default browser) exposes no WebGPU by default and Chrome without flags reported "No WebGPU adapter was found". With `--enable-unsafe-webgpu --enable-features=Vulkan --ignore-gpu-blocklist` it ran: 9.4 tok/s on LFM2 350M Q8_0 and 3.8 tok/s on LFM2.5 1.2B Q8_0. We did not test which of the three flags is the necessary one. Gemma 3 4B was not tested there.

The same exercise exposed three bugs in our own code that conformance tests could not see: a long-poll engine connection that could hang silently (now cut at 35 s with a watchdog that relaunches the engine), a log key mismatch, and Linux browser detection that picked the default browser instead of a Chromium-based one. Our own CI also failed for reasons that were test-harness bugs, not engine bugs (an unread stdout pipe that blocked the server on Windows, and a timing race in a cancel test); we fixed the harness and the CI is now green on Ubuntu and Windows with Python 3.9 and 3.12.

## What this does and does not show

**Shows:** on synthetic models across the listed architectures and quantizations, the engine matches llama.cpp 0.3.35 token for token and within 2e-3 on logits against dequantized F32, end to end through the HTTP API, with negative controls that fail when they should.

**Does not show:** speed in general (three machines, two of them with two runs per model and no reference runtime; AMD and Apple are unmeasured, and the Linux machine needed browser flags), correctness on every real model (real-model checks were done against LM Studio on a small set), or parity with newer llama.cpp vision preprocessing.

For context, that machine gave 44.3 tok/s on a 350M LFM2 Q8_0 and about 15.7 tok/s on a 1.2B one, 5.3 tok/s on Gemma 3 4B Q4_K_M and 2.8-3.1 tok/s on a 7B Q4_K_M. LM Studio on the same machine gave 30.2, 9.6, 5.0 and 3.1 respectively. The two other machines are in `docs/BENCHMARKS.md` (GTX 1050 Ti: 42.6, 14.7, 5.0 tok/s on the same three models; Intel HD 4000 on Linux: 9.4 and 3.8 tok/s on the first two). Two caveats apply: the LM Studio runtime variant (CPU or Vulkan) was not recorded, and we are slower on some 7-8B models (about 1.9 vs 2.4 tok/s on a Q3_K_L Nemo).

## Reproduce it

```
python tests/run_all.py            # API suite with a simulated engine; standard library only
python tests/run_all.py --engine   # full conformance; needs pinned llama-cpp-python, gguf, numpy, Playwright
```

The full `--engine` run is slow under SwiftShader: the DeepSeek suite alone took 1273 s in our last execution. Timings depend heavily on the machine. If you run it on other hardware or find a driver with different `cos`/`sin` behavior, please open an issue with your numbers.

---

*quipullm is designed for PCs where installing software is hard: plain text only, no `pip`, no `npm`, no binaries. Code and docs: [https://github.com/dagoalguez/quipullm](https://github.com/dagoalguez/quipullm).
Built with AI assistance (Claude, Anthropic); the author reviews and is responsible for what is published. Thanks to the llama.cpp / ggml and `gguf` maintainers, whose reference implementation made this validation possible. Models are not distributed and keep their own licenses. Not affiliated with llama.cpp or LM Studio.*
