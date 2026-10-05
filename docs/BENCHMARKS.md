# Benchmarks

Every number here was **measured**. The main comparison against **LM Studio 0.4.25** was done on **one machine** (the first section below), using the llama.cpp runtime installed in it at the time (the exact runtime variant, CPU or Vulkan, was not recorded). Two more machines were measured later **without LM Studio** (section "Other machines"). Nothing is extrapolated. If your hardware differs, run `tools/bench.py` and send a PR with your numbers (see the end).

## Test machine

| | |
|---|---|
| CPU | Intel Core i7-1270P (12 cores / 16 threads) |
| GPU | Intel UHD (integrated), no NVIDIA. WebGPU shows as hardware-accelerated in `edge://gpu` |
| RAM | 15.6 GB (about 88 % in use before loading a model) |
| OS / browser | Windows, Microsoft Edge (`--app` window) |
| GPU `maxBufferSize` | 2 GB |
| Reference | LM Studio 0.4.25 (Windows), default installed llama.cpp runtime; variant not recorded |

Dates: 2026-09-29 and 2026-09-30. Single user, one request at a time.

## Generation speed and load time (this server vs LM Studio)

| Model | Arch / quant | tok/s (ours) | tok/s (LM Studio) | Load s (ours) | Load s (LM Studio) |
|---|---|---|---|---|---|
| lfm2-350m-math | lfm2 Q8_0 | 44.3 | 30.2 | 4.3 | 4.8 |
| lfm2.5-1.2b-thinking | lfm2 Q8_0 | 16.3 | 12.4 | 7.5 | 6.8 |
| lfm2-1.2b-rag | lfm2 Q8_0 | 15.8 | 11.9 | 8.0 | 5.5 |
| lfm2-1.2b-extract | lfm2 Q8_0 | 16.4 | 11.0 | 7.8 | 5.0 |
| liquid/lfm2.5-1.2b | lfm2 Q8_0 | 15.7 | 9.6 | 7.6 | 5.4 |
| lfm2.5-vl-1.6b | lfm2 Q8_0 | 15.1 | 9.2 | 7.9 | 13.2 |
| gemma-3-4b-it | gemma3 Q4_K_M | 5.3 | 5.0 | 12.0 | 18.2 |
| deepseek-r1-distill-qwen-7b | qwen2 Q4_K_M | 3.3 | 3.5 | 18.3 | 12.6 |
| qwen2.5-coder-7b-instruct | qwen2 Q4_K_M | 2.8–3.1 | 3.1 | 20.1 | 18.9 |
| granite-3.2-8b-instruct | granite Q4_K_M | 2.7 | 3.1 | 19.6 | 14.2 |
| mistral-nemo-instruct-2407 | llama Q3_K_L | 1.9 | 2.4 | 28.3 | 38.6 |

(LFM2 rows: engine 1.1.1. 7–8 B rows: engine 2.1.1. Gemma 3: engine 3.0.)

Reading it honestly: on small models we are faster; on 7–8 B models we are roughly on par to ~25 % slower. Decoding on an iGPU is memory-bandwidth bound, so both engines end up close.

## Other machines (this server only, no LM Studio comparison)

Measured on 2026-10-02 with quipullm 4.0.1, through the control panel's "Try it" box (`max_tokens` 300, temperature 0.7 in the Linux runs), **two runs per model**, short prompts (15–23 prompt tokens), one request at a time.
These are first data points, **not averages**. Different prompts per model, so only generation speed is comparable between rows; prompt-processing speed on 15–23 token prompts is not representative and is not quoted.
It was not recorded whether other programs were using the GPU during the runs. There is **no LM Studio measurement** on these machines, and they must not be read against the i7-1270P rows above as "faster" or "slower" hardware: different GPU, driver, browser and OS.

| | Windows PC | Linux laptop |
|---|---|---|
| CPU | Intel Core i5-8400 @ 2.80 GHz | Intel Core i5-3230M @ 2.60 GHz (Ivy Bridge, 2012) |
| GPU | NVIDIA GeForce GTX 1050 Ti, 4 GB, driver 560.94 | Intel HD Graphics 4000 (integrated), Vulkan backend |
| RAM | 16 GB (15.9 GiB) | 15 GiB |
| OS | Windows 11 Pro 10.0.26200 | Linux Mint 22.3, kernel 7.0.0-34 |
| Browser | Microsoft Edge 143.0.3650.139 (`--app` window opened by the server) | Google Chrome 154.0.8037.57, launched with `--enable-unsafe-webgpu --enable-features=Vulkan --ignore-gpu-blocklist` (see below) |
| WebGPU adapter / `maxBufferSize` | "nvidia pascal" / 2147 MB | "intel gen-7" / 1074 MB |

| Model | Windows PC: tok/s (2 runs) | First token | Load s | Linux laptop: tok/s (2 runs) | First token | Load s |
|---|---|---|---|---|---|---|
| lfm2-350m-math (lfm2 Q8_0) | 42.55, 41.98 | 0.31, 0.37 s | 6.0 | 9.44, 9.43 | 0.84, 0.97 s | 4.8 |
| lfm2.5-1.2b-instruct (lfm2 Q8_0) | 14.67, 14.74 | 1.26, 1.25 s | 14.6 | 3.78, 3.78 | 3.5, 3.6 s | 13.5 |
| gemma-3-4b-it (gemma3 Q4_K_M) | 4.99, 5.01 | 0.60, 0.58 s | 39.6 | not tested | | |

Notes:
- Gemma 3 4B was labeled "tight" by the panel on the 4 GB card (context 7680) and still loaded and ran. The label is an estimate.
- The 1.25 GB LFM2.5 model loaded on the Linux laptop with a 1074 MB `maxBufferSize`. Why it worked was not investigated.
- **Linux needed flags.** With default Chrome 154 the engine window said "No WebGPU adapter was found", and `navigator.gpu.requestAdapter()` returned `null` for every option, including `forceFallbackAdapter`, although `chrome://gpu` listed the HD 4000 as "Available". It worked after launching Chrome with the three flags above (`chrome://gpu` then showed "Vulkan: Enabled"). **Which of the three flags is the necessary one was not tested.** Chrome warns that `--enable-unsafe-webgpu` affects stability and security.
- Windows: the first run of each model includes model loading and shader compilation in "Load s"; generation speed was the same on the second run.

## Output agreement with LM Studio

- Prompt tokens: identical in every test (8/8 per model).
- Greedy text identical: LFM2 44 of 48, qwen2.5-coder 8/8, granite 8/8, Gemma 3 8/8, deepseek-r1 7/8, nemo 7/8.
- Every difference is a long text (over 450 characters) diverging at a near-tie in the logits, with a correct meaning. llama.cpp quantizes activations to Q8 on CPU; we compute them in f32, so a few percent of numeric difference is expected.

## Time to first token (prefill)

| Model | Prompt | Time |
|---|---|---|
| lfm2 1.2B | 62 tokens | about 2.3 s (about 28 tok/s prefill) |
| gemma-3-4b | short prompts | 0.6–1.1 s |
| granite 8B | 58–76 tokens | 2.3–3.4 s |
| nemo | — | 1.6–3.2 s |
| deepseek-r1 7B | — | 1.0–1.9 s |
| DeepSeek-Coder-V2-Lite | 538 prompt + 80 generated tokens | 36 s total |

Before tiled prefill (v2.1.1) a 43-token prompt on qwen 7B took about 10 s for the first token; after it, 1.9 s.

## Tool calling (experimental)

Measured on 2026-10-04 with a development version after 4.1.0 (not yet released), on the **Windows PC** of the table above: Intel Core i5-8400, NVIDIA GeForce GTX 1050 Ti 4 GB (driver 32.0.15.6094),
15.9 GiB RAM, Windows 11 Pro 25H2 (build 26200.6584), Microsoft Edge 154.0.4258.53. Script: `tools/probar_tools.py`, temperature 0, **5 runs**.

Each run asks a weather question with one tool (`get_weather(city)`), checks for a valid tool call, sends a fake tool result and checks that the model answers in text.

| Model | Valid tool call | Text answer after the result | Time per run (2 requests) |
|---|---|---|---|
| lfm2.5-1.2b-instruct (Q8_0) | 5/5 | 5/5 | 22–24 s |

What this does **not** tell: whether the model picks the right tool among several, whether it avoids calling a tool when none is needed, harder arguments, streaming with real models, or any other model.
The model wrote its calls in its own native format (`<|tool_call_start|>[get_weather(city="Lima")]<|tool_call_end|>`), which the server understands.
A 7B Qwen2.5-Coder model was refused on this PC by the memory check (about 5.0 GB needed against 3.9 GB estimated), so no 7B result exists.

## Vision

| Test | Result |
|---|---|
| LFM2.5-VL, OCR of a small image ("HOLA 2024") | 16.9 s including model load |
| LFM2.5-VL, screenshot | 26.4 s |
| Gemma 3 4B vision, 972x320 screenshot resized to 896x896 (256 image tokens) | image encoder 29 s, whole request 73 s (about 35 s generating 137 tokens, about 3.9 tok/s) |

## Embeddings

nomic-embed-text-v1.5: vector norm 1, similarities as expected, 0.05 s per short text.

## Not measured (do not quote these)

- Any AMD GPU (discrete or integrated), any Apple or macOS machine, any Linux machine with a recent GPU.
- LM Studio on the two machines in "Other machines"; Gemma 3 4B and the 7–8 B models on the Linux laptop; more than two runs per model on those machines.
- Tool calling with any model other than lfm2.5-1.2b-instruct, with more than one tool, or on a prompt that needs no tool.
- More than one simultaneous user (requests queue; one engine).
- DeepSeek-Coder-V2-Lite generation speed beyond the 36 s total above.
- Memory autoadjust thresholds on other hardware.

## Reproduce

```
python tools/bench.py --url http://localhost:1234 --model <model id> --runs 3
```

It sends a fixed prompt at temperature 0 and prints time to first token, generated tokens and tok/s (from the server's own `usage`/`stats` plus wall-clock). To contribute a result, add a row with your CPU/GPU/RAM, OS, browser, model, quantization and the command output to a PR.
