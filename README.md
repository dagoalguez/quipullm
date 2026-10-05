<p align="center"><img src="docs/img/logo.svg" alt="quipullm logo" width="72"></p>

# quipullm

[English](README.md) · [Español](README.es.md)

A self-hosted LLM server with an **OpenAI- and LM Studio-compatible API**. The inference engine is written from scratch in
**WebGPU (WGSL + JavaScript)** and runs inside a browser window; the backend is **pure Python standard library**.

**Nothing to install: no `pip`, no `npm`, no `.exe` or other compiled binaries, no installers, no Docker.** The whole project is plain text
(`.py .js .html .json .md`), so you can read all of it before running it, and start it with one command. It was designed with PCs in mind where installing software is hard (no admin rights, locked-down machines, no package downloads):

```
python server.py
```

It reads standard **GGUF** files, and its text and embedding paths are checked number by number against llama.cpp
(see [How it is verified](#how-it-is-verified)). Open <http://localhost:1234/chat> for a chat page, or point any OpenAI client at it.

![Control panel](docs/img/panel.png)

![Chat page](docs/img/chat.png)

> **Status: early release (4.1.x).** Measured on three machines: Windows with an Intel iGPU (compared against LM Studio),
> Windows with a GTX 1050 Ti, and an old Linux laptop with an Intel HD 4000 (the last two with only two runs per model). AMD, Apple and
> recent Linux GPUs are **unmeasured**. Everything we know to be missing or weak is in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

**You do not need a dedicated graphics card.** An integrated GPU (Intel, AMD or similar) is enough as long as your browser has hardware-accelerated
WebGPU; the machine we measured against LM Studio is exactly that case. A dedicated card is not automatically faster: a GTX 1050 Ti (Windows, Edge) gave 42 / 14.7 / 5.0 tok/s on the same three models, close to that iGPU (44.3 / 15.7 / 5.3), with only two runs each and no LM Studio comparison on that PC (see [docs/BENCHMARKS.md](docs/BENCHMARKS.md)).

## Is it for you?

**A good fit**

- **Trying small local models (about 1–4B)** to summarize, classify, extract or draft text, with no internet and no per-token cost.
- **A few people sharing one PC:** start it with `python server.py --share` and send them the chat link (see [Share it with your team](#share-it-with-your-team)).
- **Developers** who want an OpenAI-compatible API to test against, offline, and a local embeddings endpoint (`nomic-embed-text`) for prototypes.
- **Anyone with their own model:** drop any GGUF of a supported architecture into the models folder; if your architecture is new, add it yourself
  (often a single JSON manifest) with conformance tests that compare it to llama.cpp; see [Adding models and architectures](#adding-models-and-architectures).
- **Researchers and learners:** a small, readable WebGPU inference engine and a [validation method](#how-it-is-verified) to study or extend.

**Not a good fit (yet)**

- **Fast answers from 7–8B models on an integrated GPU:** about 2–3 tokens/s on the one we measured.
- **Many simultaneous users:** it serves one request at a time; the rest wait.
- **Agents or structured output:** experimental tool calling only (generic text format, not measured on real models yet) and no enforced JSON schema.
- **A server with no screen:** the engine needs a browser window open on the server PC.
- **Anything that needs a certification, an external security audit or encrypted traffic:** it has none of them (no TLS).
  Prompts stay on your machine and we found no outgoing connections, but that is a check, not an audit; review it yourself.
  Read [docs/USE_CASES.md](docs/USE_CASES.md) before proposing it to an organization.

## Quick start (5 steps, nothing to install)

1. **Get the code.** Download the ZIP of this repository (or clone it) and unpack it anywhere.
2. **Check Python 3.** `python --version` should print 3.x (developed and tested with 3.11). You also need a WebGPU browser:
   **Microsoft Edge** on Windows is the tested one (open `edge://gpu` and look for *WebGPU: Hardware accelerated*).
3. **Add a model.** Put a `.gguf` file of a supported architecture in the `models/` folder next to `server.py` (create it).
   A small one is the best first test (see [what to expect](#what-to-expect)). You can pick another folder later in the panel.
4. **Start it.**
   ```
   python server.py
   ```
   A small window called *quipullm engine* opens: **leave it open** (minimizing is fine), it is where the GPU work happens. Open the
   panel at <http://localhost:1234/>; when the badge says **engine: ready**, press *Load* next to your model.
5. **Talk to it.** Open the chat page at <http://localhost:1234/chat>, use the *Try it* box in the panel, or call it from any OpenAI client by changing only the base URL:

   ```
   curl http://localhost:1234/v1/chat/completions -H "Content-Type: application/json" \
     -d '{"model":"<model id shown in the panel>","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
   ```
   ```python
   from openai import OpenAI          # pip install openai (on the *client* machine; the server needs nothing)
   c = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed", timeout=1800)
   print(c.chat.completions.create(model="<model id>", max_tokens=100,
         messages=[{"role": "user", "content": "Hello"}]).choices[0].message.content)
   ```

By default the server listens **only on this PC** (`127.0.0.1`). To serve other computers see [Security](#security).
The panel has an English/Spanish selector (top right; it follows your browser language by default).

### What to expect

Measured on **one** machine, an Intel Core i7-1270P with the Intel UHD integrated GPU, Windows, Edge, one request at a time
(full table and method in [docs/BENCHMARKS.md](docs/BENCHMARKS.md)):

| Model (quantization) | Generation speed |
|---|---|
| LFM2 350M math (Q8_0) | 44.3 tok/s |
| LFM2.5 1.2B (Q8_0) | 15.7 tok/s |
| Gemma 3 4B (Q4_K_M) | 5.3 tok/s |
| Qwen2.5-Coder 7B (Q4_K_M) | 2.8–3.1 tok/s |

Your numbers will differ. Two more machines (a GTX 1050 Ti on Windows and an Intel HD 4000 on Linux) are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md), without an LM Studio comparison. AMD, Apple and recent NVIDIA/Linux GPUs are **unmeasured**; if you run
`tools/bench.py`, please send them (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Features

- OpenAI-compatible endpoints: `/v1/chat/completions` (streaming and not), `/v1/completions`, `/v1/embeddings`, `/v1/models`,
  plus LM Studio's `/api/v0/models`. Existing code only needs a different `base_url` (default port `1234`).
- Reasoning models: `<think>` is returned separately in `reasoning_content`, like LM Studio.
- Vision: images as base64 `image_url` (LFM2-VL and Gemma 3). It runs on the tested machine (timings in
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md)); its conformance suite matches llama.cpp as of `llama-cpp-python` 0.3.35 but not
  yet newer releases, see [How it is verified](#how-it-is-verified).
- Embeddings: `nomic-embed-text` (`float` and `base64`).
- Chat templates are read from the GGUF (`tokenizer.chat_template`) with a small Jinja interpreter written in the standard
  library (compared with real jinja2), with hand-written fallbacks.
- **Chat page** (`/chat`): streaming conversation with history in the page, stop button, system instruction, reasoning shown separately, English/Spanish. Nothing is stored on the server.
- **Example for developers:** [`examples/agent-chat.html`](examples/agent-chat.html) is a one-file web chat agent (system prompt + streaming) built only on the API; open it in a browser and point it at your server. Every setting is explained in [docs/SETTINGS.md](docs/SETTINGS.md).
- Web panel: models, memory estimates, live status and logs, a quick test box, and a settings page (advanced options are folded away).
- Optional API key, request-queue limit (HTTP 429), and a memory estimate per model ("fits / tight / does not fit").
- **Pluggable architectures:** drop a manifest in `web/arch/` and rescan, no restart ([docs/ARCH_GUIDE.md](docs/ARCH_GUIDE.md)).
  Qwen 3 was added with a manifest only.

## Supported architectures

| `general.architecture` | Models tested | Notes |
|---|---|---|
| `lfm2` | LFM2 / LFM2.5 (350M–1.6B), LFM2.5-VL | text, thinking, vision |
| `qwen2` | Qwen2.5-Coder 7B, DeepSeek-R1-Distill-Qwen 7B | K-quants |
| `qwen3` | synthetic models vs llama.cpp | declarative manifest only |
| `llama` | Mistral-Nemo 12B | Llama/Mistral family |
| `granite` | IBM Granite 3.2 8B | |
| `gemma3` | Gemma 3 4B | sliding window, text + vision |
| `deepseek2` | DeepSeek-Coder-V2-Lite | MLA + MoE (Lite variants) |
| `nomic-bert` | nomic-embed-text-v1.5 | embeddings |

Quantizations: F32, F16, BF16, Q8_0 and the K-quants Q2_K–Q6_K; IQ4_NL is implemented for DeepSeek's experts.

The real models we ran, with measured speed, are listed in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Other models of these architectures
should load but are **untested**; please report what works and what does not (an issue with the model file name and quantization is enough).

## How it works

```
client ──HTTP :1234──► server.py (Python stdlib) ──job queue──► engine window (Edge/Chrome tab, WebGPU)
                         API, chat templates, GGUF reader                 WGSL kernels, tokenizer, sampler
```

One process, one engine, one request at a time (others wait in a queue). Details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Adding models and architectures

- **A model of a supported architecture:** copy the `.gguf` to the models folder (a vision model also needs its
  `mmproj*.gguf` next to it) and press *Rescan the folder* in the panel. Models are not distributed with this project and keep their own licenses.
- **A new architecture:** see [docs/ARCH_GUIDE.md](docs/ARCH_GUIDE.md). Many transformer variants need only a JSON manifest;
  truly new ones need a JS module. Either way there is a conformance test kit that compares the engine to llama.cpp, so
  contributions (including AI-written ones) can be accepted on evidence.

## How it is verified

Write-up of the method and what it found: [docs/validation-article.md](docs/validation-article.md).

Each architecture is compared with llama.cpp using small synthetic models (random weights) quantized by llama.cpp itself: token
ids must be identical, last-token logits must match the dequantized F32 reference (relative difference below `2e-3`), and
greedy generation and the full HTTP path must produce the same text. Negative controls (for example, disabling the sliding window)
must make the tests fail.

```
python tests/run_all.py                # API tests with a simulated engine, standard library only
pip install -r tests/requirements-engine.txt && playwright install chromium   # development only
python tests/run_all.py --engine        # engine vs llama.cpp (pins llama-cpp-python 0.3.35, gguf 0.19.0, numpy 2.4.4)
```

Things to know before you rely on this:

- The `--engine` tests run the real engine in headless Chromium with **SwiftShader** (software WebGPU). They prove numerical
  correctness, **not speed**. Speed was measured on real hardware only on the machine listed above.
- Synthetic models prove the implementation matches llama.cpp on small random models. They are not a substitute for running a
  real model on your GPU.
- **Vision depends on the llama.cpp version.** The image tests pass against the llama.cpp bundled in `llama-cpp-python` 0.3.35
  and fail against 0.3.36, because upstream changed how images are preprocessed (the bilinear resize filter, whether a single-tile
  LFM2 image gets a `<|img_thumbnail|>` token, and when LFM2 splits an image into tiles) and the engine still follows the older
  behavior. That is why the versions are pinned. Details and status: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Troubleshooting

- **The panel says the engine is disconnected.** Press *Open engine* in the panel (from the server PC) or restart `server.py`.
  Only one engine window works at a time; the newest takes over.
- **The panel says "Server: connected" but the engine drops, and requests fail with 503 until you press F5 in the engine window.**
  Fixed in 4.0.1 (the engine window now cuts a hung poll after 35 s, and the server reopens the window if requests wait). In Edge,
  also turn off *Settings → System and performance → Save resources with inactive tabs* (Spanish UI: *Ahorrar recursos con pestañas inactivas*)
  or add `http://localhost:1234` to the exclusion list, so the browser does not freeze the engine window in the background.
- **Linux: the engine window says "No WebGPU adapter was found".** Use Chrome or Chromium (Firefox has no WebGPU by default on Linux); the server now looks for `google-chrome`, `chromium` or `brave-browser` on `PATH`. Check `chrome://gpu`. On an old Intel GPU (HD 4000) it only worked starting Chrome with `--enable-unsafe-webgpu --enable-features=Vulkan --ignore-gpu-blocklist` (run `python3 server.py --no-engine` and open `http://localhost:1234/engine` yourself in that Chrome). Which of those flags is needed was not tested. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
- **The engine window shows a WebGPU error.** Open `edge://gpu` (or `chrome://gpu`) and check that WebGPU is hardware accelerated;
  update the graphics driver.
- **The server cannot open port 1234.** Another server (LM Studio, or a second copy of this one) is using it. Close it or change `port` in the panel.
- **Slow, or the PC freezes while loading.** Memory is tight: close other programs. The panel shows a *fits / tight / does not fit*
  estimate per model; it is an estimate, because WebGPU does not report free VRAM.
- **Logs:** the panel (*Registro* section), `logs/server.log`, and the engine window.

## Share it with your team

```
python server.py --share
```

This listens on the whole network and, if you have not set an API key, creates one **for this run** (it is printed in the console and shown in the
panel, and it is not written to `config.json`). The panel shows the address to give your team (for example `http://192.168.1.20:1234`):
anyone who opens it from another PC lands straight on the chat, types the key once, and talks to the model. Developers use the same address plus `/v1`
with `Authorization: Bearer <key>`. Notes:

- **Other PCs do not get the panel.** They are sent to the chat and cannot read the log, the settings or the request history; only the server PC
  manages the server. (Set `"remote_panel": true` in `config.json` if you want them to look at it.)
- **You choose what the chat offers.** In the panel's Models table, tick the models your team will see in the chat and order them with the arrows;
  the first one is the default. This only shapes the chat page: the API still lists every model.
- Use the **IP address**. PC names often do not resolve from other systems (a Linux laptop usually cannot find a Windows PC by name).

- The model runs on **your** GPU, one request at a time: with several people, the others wait in the queue.
- Traffic is **not encrypted** (HTTP). Share only on a network you trust, and never on the internet.
- From another PC the key also allows loading and unloading models; without a key those actions are refused from other PCs.
- For a permanent key, set it in the panel (Settings → API key) or with `LLM_API_KEY`.

## Security

By default the server listens on **127.0.0.1 only**. To open it to your LAN use `--share` (above) or set `"host": "0.0.0.0"` and an API key (panel or
the `LLM_API_KEY` environment variable). Other websites cannot drive the panel or the engine through your browser (origin and
`Host` checks). Read [SECURITY.md](SECURITY.md) before exposing it. There is no TLS; do not expose it to the internet.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the [ROADMAP](ROADMAP.md). Where help matters most, in order:

1. **Benchmarks on other hardware** (AMD, Apple, recent NVIDIA and Linux GPUs, LM Studio comparisons) with `tools/bench.py`: we measured three machines, two of them with only two runs per model and no LM Studio.
2. **Testing the engine launch on Linux (recent GPUs) and macOS** (Linux was tried once, on a 2012 Intel GPU, with special Chrome flags; macOS is untested).
3. **New model architectures** (a JSON manifest plus a test-generator variant; see [docs/ARCH_GUIDE.md](docs/ARCH_GUIDE.md)).
4. **Roadmap features:** measuring tool calling on real models, enforced JSON schema, prompt caching, a windowless mode.
5. **Bug reports** with the engine log, security review, translations and identifier renames.

The interface, messages, configuration keys and HTTP API are in English; comments and docstrings are English, but many internal identifiers
(function and variable names) are still in Spanish (the project started in a Spanish-speaking team); translating them is welcome.

## License and credits

Apache License 2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE). Model files are **not** included and keep their own licenses.
The engine is an independent implementation of the GGUF format and of the model architectures; it is validated against
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) and uses the `gguf` Python package only in the development tests.
This project is not affiliated with LM Studio, llama.cpp, or any model vendor.

**Models used in the tests and benchmarks.** They are not distributed here, and each one keeps its own license and terms (see its model card; check them before using a model in your own project). Thanks to the teams behind them:
LFM2 / LFM2.5 / LFM2.5-VL (Liquid AI), Gemma 3 (Google), Qwen2.5-Coder (Qwen team, Alibaba Cloud), DeepSeek-R1-Distill-Qwen and DeepSeek-Coder-V2-Lite (DeepSeek), Granite 3.2 (IBM),
Mistral-Nemo-Instruct-2407 (Mistral AI and NVIDIA) and nomic-embed-text-v1.5 (Nomic AI). The GGUF files were third-party conversions; model names are trademarks of their owners.
Thanks also to the llama.cpp / ggml and `gguf` maintainers, whose reference implementation made number-by-number validation possible, and to LM Studio, used as the speed reference.

The code was written with the help of Claude (Anthropic) as a technical partner; the author reviews and is responsible for what is published.
