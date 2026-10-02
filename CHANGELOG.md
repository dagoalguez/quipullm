# Changelog

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

## 3.x (internal line)
- 3.4.x: DeepSeek grouped-expert prefill, Gemma 3 vision, faster ViT attention.
- 3.3: DeepSeek-Coder-V2-Lite. 3.2: LFM2-VL vision. 3.1: embeddings. 3.0: Gemma 3 text.
- 2.x: generic transformer (qwen2/llama/granite) and K-quants. 1.x: LFM2 family.
