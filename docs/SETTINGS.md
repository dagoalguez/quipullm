# Settings reference

Everything here is in the panel (Settings, on the server PC) and is saved to `config.json`. Other PCs cannot change it.

## Main

| Setting (`config.json` key) | What it does |
|---|---|
| Models folder (`models_dir`) | Where the `.gguf` files are searched, including subfolders. |
| API key (`api_key`) | Password for the API and the chat page. Empty = open to whoever can reach the server. Type `-` to remove it. The `LLM_API_KEY` variable wins over it. |

## Advanced

| Setting (key) | What it does |
|---|---|
| Default context (`default_ctx`) | How many tokens a conversation can hold: your messages plus the answer. A bigger context remembers more but uses more GPU/RAM (the KV cache is f32). It is also capped by what the model supports and by the memory budget. |
| Max. context memory (`kv_max_mb`) | Cap, in MB, for the memory the context uses. 0 = automatic. |
| Memory for models (`gpu_memory_gb`) | The GB budget behind the "fits / tight / does not fit" labels. WebGPU does not report free memory, so 0 means "estimate it". |
| Memory auto-fit (`memory_check`) | Refuses to load a model that almost surely does not fit, so the PC does not hang. |
| Max. wait per request (`timeout_seconds`) | Longest a request may spend in the queue plus generating. |
| Port (`port`) | Needs a restart. |
| Max. queued requests (`max_queue`) | Requests waiting while one is being served. When full, new ones get HTTP 429. 0 = unlimited. |
| Default response length (`default_max_tokens`) | Length of the answer when the request does not say. 0 = until the context is full (a long answer can take minutes on a slow GPU, so a value like 500 is usually kinder). |
| Response length limit (`max_tokens_limit`) | Hard cap: no request can get a longer answer, whatever it asks. 0 = no cap. |
| Open the engine at startup (`open_engine`) | Opens the engine window by itself when the server starts. |

### Response length: what the real limit is

The answer length is the smallest of: what the request asks for (or the default when it asks nothing), the **limit**, and **the context minus the length of the conversation so far**. The boxes accept large numbers, but a value above the context does nothing more: the answer stops when the context is full.

### Per-model defaults

Under the Models table, *Per-model defaults* sets a temperature, a default response length and a limit for each model. An empty box uses the general setting. In `config.json` they live in `model_overrides`, for example `{"model_overrides": {"gemma-3-4b-it": {"temperature": 0.3, "max_tokens": 400, "max_tokens_limit": 1000}}}`. A request that sends its own `temperature` or `max_tokens` is respected, except that the limit always applies. The chat page sends neither unless the user types one, so it follows these defaults.

## Default sampling

Used when a request does not specify them. These control how the next word is picked:

- **Temperature**: 0 is almost always the same answer; around 0.7 to 0.8 is a usual chat value; higher is more varied and more error-prone.
- **Top K**: only the K most likely next tokens are considered. 0 = off.
- **Top P**: only the most likely tokens whose probabilities add up to P are considered.
- **Min P**: drops tokens less likely than this fraction of the best one.
- **Repetition penalty**: above 1 discourages repeating itself.

## Chat page

- *In chat* column in the Models table: which models the chat offers, and in which order (the first is the default). It shapes the chat page only; the API still lists every model.
- The chat page has no sampling or length boxes: temperature, default length and limit are always the administrator's (global or per model). The API still accepts whatever a program sends, except that the length limit always applies.
- `remote_panel` (default `false`): when `false`, other PCs are sent to the chat and cannot see the panel, log or settings.

## Buttons

- **Open engine**: opens the engine window (where the GPU work happens). The server already opens it by itself, so you only need it when the badge says *engine: disconnected*. It is disabled while an engine is connected: a second window would take control, interrupt the current request and unload the model.
- **Free memory**: unloads the loaded model from GPU memory (for example to give the GPU back to another program). The next request loads it again, which takes a few seconds to a minute.
