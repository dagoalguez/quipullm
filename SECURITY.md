# Security

## Threat model (read before exposing the server)

- By default the server listens on **`127.0.0.1` only** (this PC). To serve your network set `"host": "0.0.0.0"` in `config.json`
  (or `python server.py --host 0.0.0.0`) **and** an API key. Without a key, anyone who can reach the port can run models on your GPU,
  load/unload models and read the model list; the server prints a warning at start in that case.
- Set a key in the panel (Settings → API key, done from the server PC) or with the `LLM_API_KEY` environment variable
  (it wins over `config.json`). Clients send `Authorization: Bearer <key>`. The panel and `/api/status` stay readable so people can
  see the status; they never expose the key.
- The key is stored in plain text in `config.json` (unless provided through the environment). Protect that file.
- There is no TLS. Traffic, prompts and the key travel in clear text on the network. Put a reverse proxy with TLS in front if that matters.
- **Do not expose the server to the internet.**
- Engine endpoints (`/engine/*`), settings changes and the folder browser are accepted **only from 127.0.0.1**.
- **Other websites cannot drive the server through your browser.** Outside the OpenAI-compatible API (`/v1/*`, `/api/v0/*`), requests
  that a browser marks as coming from another origin (`Origin` / `Sec-Fetch-Site`) are rejected with 403, and no CORS headers are sent.
  The panel and engine pages cannot be framed (`frame-ancestors 'none'`). This protects `/api/config`, `/api/folders`, `/api/load` and the
  engine bridge against CSRF from a page open on the server PC.
- **DNS rebinding:** the `Host` header must be a name or IP of this PC (`localhost`, its hostname, its LAN IPs). If you put the server behind
  a reverse proxy or use another name, add it to `"allowed_hosts"` (`["*"]` disables the check).
- `/v1/*` keeps `Access-Control-Allow-Origin: *` (as LM Studio does) so browser clients work. Without a key, that also means a web page
  open on a machine that can reach the server can use your models. Set a key if that matters.
- `web/arch/*.js` modules are JavaScript executed in the engine window on the server PC. Install only modules you have reviewed.
- Prompts are not logged; the log records request counts and token numbers. Remote viewers of `/api/status` see only the name of the
  models folder, not its full path.
- The queue limit (`max_queue`, default 64) answers HTTP 429 when full, which limits trivial request floods but is not a rate limiter.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting (Security tab → *Report a vulnerability*) on the repository instead of a public issue.
