# Changelog

## Unreleased

- **Decision models (Liquid AI d1), text only**: new endpoint `POST /v1/systemone` (request and answer as in llama.cpp), engine job `decidir`
  and `web/js/decision.js`. The registry reads `<arch>.decision.type` (`lfm2-d1`); such a model is excluded from the chat page and the panel's
  chat list and answers 400 on `/v1/chat/completions`; `/api/v0/models` reports type `decision`. New suite `tests/conformance/test_d1.py`
  (95 checks against llama.cpp with a synthetic model; in `run_all.py --engine`) and 34 more API checks. Not run with the real d1 weights yet.
- About dialog (chat and panel): a footer line `quipullm vX · Diego Guevara B. · Apache-2.0` opens version, author, contributions (Claude), license and a note that models keep their own licenses. EN/ES, no external links.
- Tool calling smoke test repeated with gemma-3-4b-it on the GTX 1050 Ti (3 runs per kind): all 12 cases right. Numbers and caveats in docs/BENCHMARKS.md. No code change.

## 4.2.0
- **Chat page**: Markdown is now rendered (headings, bullet and numbered lists, quotes, tables, rules, bold, italics, code), always built from DOM nodes so the model's text cannot inject HTML. From 80 % of the context the counter turns red and a notice suggests starting a new chat; if a prompt no longer fits, the page says so in plain words instead of showing the raw server error; an answer cut by the length limit says so. Nothing is dropped or summarised behind your back (an automatic summary was tried and removed: with a 1.2B model it lost or mixed up facts). On the server side, a prompt that does not fit still returns an error and a reply that reaches the context ends with `finish_reason: "length"`.
- **Chat page: saved chats.** Conversations are now saved in the browser itself (IndexedDB), never on the server, and listed in a side panel: open, rename, delete, delete all, export and import as JSON (imports are validated and only plain text is accepted). Up to 300 chats are kept (the oldest are dropped beyond that); an optional "auto-delete after 30 / 90 days" setting is off by default. Chats belong to the browser, not to a person: two people on the same browser profile share them, and they do not travel between devices (use export/import). If the browser blocks storage (private mode), the page says so. Server-side, per-user history is not implemented. Checked in headless Chromium against a simulated server (tests/conformance/test_chat_ui.py); not tried on other browsers.
- **Chat page**: each answer now shows which model produced it (the `model` the server reports in the response), with tokens, time to first token and speed; this is saved with the chat (in the browser only) and included in export/import, so a reopened chat looks the same. It is not sent back to the server with the history. Chats saved before this show the chat's last model, without the numbers.
- **Icon**: the chat, the panel and the engine window now have the quipullm icon (a small SVG inside each page, so there is no extra request).
- **Panel**: the header and the model-loading banner now stay fixed at the top while you scroll.
- **Chat page**: "Regenerate" button on the last answer; a round arrow to jump to the latest message when you have scrolled up (and the page no longer drags you down while you read above a streaming answer); more space under the token line.
- **Panel**: while a model loads there is now a banner with a progress bar (percentage reported by the engine), the clicked "Load" button shows "Loading…" at once, and the badge says "engine: loading model". `/api/status` `current_job` now includes `action`.
- **Experimental prompt cache for LFM2 models**: when a request starts with the tokens of the previous one (a chat that grows, a regenerated answer), the engine skips re-processing the shared part. LFM2 has convolution state besides attention KV, so the engine keeps snapshots of it: one just before the last prompt token, one after the prompt and one every 64 tokens while processing and generating. When the new prompt differs from the history at some point (for example because the previous answer, converted back to tokens, is not exactly what the model generated), it restarts from the last snapshot before that point instead of recomputing the whole previous answer. Other model families and requests with images are not cached. Setting `prompt_cache` (default on). Checked token for token (logits identical, relative difference 0.0) only on the synthetic test models; on one GTX 1050 Ti with lfm2.5-1.2b-instruct, 7 of 8 follow-up turns of two 5-turn chats reused almost the whole previous answer (first token 1.6 to 2.5 s) and one took 15.0 s after a tokenization mismatch (details in docs/LIMITATIONS.md); other GPUs and models are not measured.
- **Experimental tool calling** (`tools` / `tool_choice` / `tool` messages / `tool_calls`), LM Studio "default mode" style: the tools are described in the
  system message, the model writes `[TOOL_REQUEST]{...}[END_TOOL_REQUEST]`, the server turns it into OpenAI `tool_calls` (`finish_reason: "tool_calls"`, also when streaming)
  and tool results come back as `[TOOL_RESULT]` text. Calls written in the model's own native format are also understood:
  LFM2 (`<|tool_call_start|>[f(a="x")]<|tool_call_end|>`, literals only, nothing is executed) and Qwen/Hermes (`<tool_call>{json}</tool_call>`). The instruction tells the model to keep answering from its own knowledge and treat the tools as optional (with the first wording, lfm2.5-1.2b-instruct refused 3 of 3 questions that needed no tool; with this one, 1 of 3). No constrained decoding: whether a model follows the format depends on the model; see docs/BENCHMARKS.md for what was measured.

## 4.1.0
- New **chat page** at `/chat` (plain text like the rest): streaming conversation with in-page history, stop button, system instruction, reasoning shown apart,
  code blocks and copy button, queue/loading status, English/Spanish, light/dark. Messages are not stored on the server.
- New `--share`: listens on `0.0.0.0` and, if no API key is set, creates one for that run (console and panel; never written to `config.json`).
- Loading, unloading and rescanning models from another PC now requires an API key (they are refused on a keyless server).
- Panel: "Open chat" button, chat link and a sharing status box (only this PC / shared with key / shared WITHOUT key), the API key next to the models folder,
  advanced settings folded away, a hint toward the chat page in the test box.
- Docs: README (EN/ES) reorganised ("Is it for you?" with what it does not fit, sharing section), status line and speed notes updated to the three measured machines.
- Other PCs are sent from `/` to `/chat` and no longer see the panel, log, settings or request history (`remote_panel: true` restores the old behaviour).
- The owner chooses which models the chat offers and in which order (panel Models table, `chat_models` in `config.json`); the first is the default.
- Fix: the panel's Copy button did nothing over plain `http://` (browsers only allow the clipboard API on https or localhost); now it falls back to a method that works.
  The chat page's Copy button got the same fix.
- The panel shows one address to give your team (the IP address when there is one).
- Chat: each code block has its own Copy button.
- Chat: a context counter under the message box ("Context: 150 / 8,192 tokens"): tokens used by the conversation in the last answer against the model's context size.
- Chat: the message box no longer shows scroll arrows while its text is short.
- The chat page no longer shows response-length or temperature boxes: both are set only by the owner (panel, global or per model).
- Response length: new "Default response length" and "Response length limit" settings (general and per model), and a per-model temperature (panel Models card).
  A request that asks for more than the limit is cut to the limit. The chat page and the panel's test box no longer force a temperature and 300–400 tokens:
  with the boxes empty, the administrator's defaults apply.
- The panel's *Open engine* button is disabled while an engine is connected (a second window took control, interrupted the running request and unloaded the model).
- New `examples/agent-chat.html` (a small web chat agent over the API) and `docs/SETTINGS.md` (every setting explained).
- Tests: `/chat` route, `--share` key preparation, remote admin rule, requests from another address (302 to the chat, 403 on log/settings, trimmed status), `chat_models`.

## 4.0.1
- Fix: the engine could be declared disconnected (503 on every request, only F5 in the engine window recovered it). The engine window's long poll
  to `/engine/api/next` had no time limit; if the connection hung (network, suspend, browser freezing the window) it waited forever. Now it is cut after
  35 s (`AbortController`) and polls again.
- New engine watchdog (`vigilar_motor`): if requests are waiting and the engine has been silent for more than 20 s, it reopens the engine window
  (at most once every 90 s). It does nothing when no request is waiting (closing the window on purpose does not reopen it) or while a job is running.
  Only active with automatic engine launch (not with `--no-engine`).
- A request that has not started no longer fails with 503 after 35 s without engine; with automatic launch it gets 90 s more. The up-front
  "engine not connected" check is skipped in that mode so the watchdog can act.
- `ServidorHTTP`: client disconnects (`ConnectionError`/`TimeoutError`, e.g. Windows 10053) log one line instead of a traceback.
- Fix: `/api/logs` entries used the key `t` while the panel reads `time`, so the panel log showed "undefined" instead of the time. Test added.
- Linux: the engine window is opened with `google-chrome`, `chromium`, `chromium-browser`, `brave-browser` or `microsoft-edge` found on `PATH` (with `--app`), instead of the default browser, which may be Firefox (no WebGPU by default). Falls back to the default browser if none is found.
- Docs: BENCHMARKS (two more machines, no LM Studio), LIMITATIONS and README troubleshooting for Linux/WebGPU.
- Tests: `resiliencia()` in `tests/api_tests.py`. Docs: Edge tip (disable "Save resources with inactive tabs" or exclude `http://localhost:1234`).

## 4.0.0 (open edition)
- English public surface (no aliases; 4.0.0 was never released): `server.py`, `templates.py`; config keys `memory_check`, `max_queue`, `gpu_memory_gb`,
  `tiled_prefill`, `open_engine`, `browser`, `timeout_seconds`, `default_sampling`; routes `/api/{architectures,load,unload,rescan,folders}` and
  `/engine/*`; English keys in `/api/status`, `/api/config`, `/api/folders`, `/api/logs`; manifest keys `description`, `options`, `module`,
  `fallback_template`; model-class interface `load/forward/reset/free/embed`, `nLayers`, `batch`; `--engine` (`--motor` still accepted).
- English interface: server messages, template errors, engine page and engine JavaScript messages; control panel with an English/Spanish selector; CLI flags `--port`, `--models-dir`, `--no-engine`.
- Code comments and docstrings translated to English (no code changes; verified by comparing the syntax tree of every Python and JavaScript file before and after).
- Pluggable architectures: `web/arch/*.json` manifests, discovered at runtime (`/api/architectures`); Qwen 3 added with a manifest only.
- Optional API key (`api_key` / `LLM_API_KEY`) and request-queue limit with HTTP 429 (`max_queue`).
- Memory autoadjust: per-model fit estimate, automatic context size, hardware panel.
- Settings page with folder browser; server name shown as an address.
- Conformance test kit and synthetic-model generators moved into `tests/`; Q8 models are compared against their own dequantized F32 weights (`gen_deq.py`).
- Security defaults: listens on `127.0.0.1` unless `host` is set; CORS only on `/v1/*` and `/api/v0/*`; requests from other websites
  (origin / `Sec-Fetch-Site`) rejected on the panel, settings, folder browser and engine routes; `Host` checked against this PC's
  names (`allowed_hosts` to extend); HTML pages cannot be framed.
- Conformance suite pinned to `llama-cpp-python` 0.3.35 / `gguf` 0.19.0 / `numpy` 2.4.4 (`tests/requirements-engine.txt`); `run_all.py --engine`
  prints the versions in use and warns when they differ. Vision conformance passes on 0.3.35 and fails on 0.3.36 (upstream preprocessing changes,
  see docs/LIMITATIONS.md). Added the missing Pillow dependency of the vision generators.
- `models_dir` defaults to `models/` next to `server.py` (relative paths resolve there); `tools/verify.py` works out of the box.
- `docs/BENCHMARKS.md` (measured numbers only), `tools/bench.py`, Spanish README (`README.es.md`).

## Before 4.0 (development history)
- 3.4.x: DeepSeek grouped-expert prefill, Gemma 3 vision, faster ViT attention.
- 3.3: DeepSeek-Coder-V2-Lite. 3.2: LFM2-VL vision. 3.1: embeddings. 3.0: Gemma 3 text.
- 2.x: generic transformer (qwen2/llama/granite) and K-quants. 1.x: LFM2 family.
