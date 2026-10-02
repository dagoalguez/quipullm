# Who is quipullm for, and how to use it in restricted environments

quipullm is a self-hosted LLM server that you can read in full before running it. This page says what that is good for, and, just as important, what it does **not** give you. For the exact limits see [LIMITATIONS.md](LIMITATIONS.md) and [SECURITY.md](../SECURITY.md).

## What you get

- **Plain text only.** No `.exe`, no `.dll`/`.so`, no installers, no compiled binaries, no `pip`, no `npm`, no Docker. The whole project is `.py .js .html .json .md`. It can be copied to a machine behind a TLS-intercepting proxy or a binary-blocking filter, and read before it is run.
- **Your prompts stay on your machine.** Models run on the server PC's own GPU. By default the server listens on `127.0.0.1` only. Prompts are not written to the log (the log keeps request counts and token numbers).
- **No outgoing connections that we could find** (see [How we checked](#how-we-checked-the-no-outgoing-connections-claim)). This is a check, not a security audit.
- **OpenAI/LM Studio-compatible API.** Existing code only changes the base URL.
- **No dedicated graphics card needed.** An integrated GPU works when the browser has hardware-accelerated WebGPU (the only hardware we measured; see [BENCHMARKS.md](BENCHMARKS.md)).
- **Apache-2.0.** No per-use fees. Models are not included and keep their own licenses.

## Who it can help

| Who | Typical use |
|---|---|
| Public sector, health, legal, finance teams | Summarize, classify or draft text with data that must not leave the organization |
| Locked-down institutional networks | Install when only text files can get through |
| Small offices with modest PCs | One shared AI endpoint on a PC with an integrated GPU or an ordinary card |
| Schools and labs | Computer rooms where software cannot be installed |
| Developers | Test against an OpenAI-compatible API without internet and without per-token costs |
| Researchers and learners | A small, readable WebGPU inference engine plus a [validation method](../README.md#how-it-is-verified) to study or extend |

## What it does NOT give you (read before proposing it to an organization)

- **No certification or compliance claim.** quipullm has no ISO 27001, SOC 2, data-protection or other certification, and has had **no external security audit**. Do not describe it as "secure for sensitive data" without your own review.
- **One request at a time.** Others wait in a FIFO queue. It is not built for many simultaneous users.
- **No TLS.** On a shared network, prompts and the API key travel in clear text. Put a TLS reverse proxy in front if that matters.
- **The API key is stored in plain text** in `config.json` (unless set through `LLM_API_KEY`). Protect that file.
- **It needs a browser window open on the server PC** (no headless service mode yet).
- **Each model has its own license.** Review the license of every model you deploy.
- **The LM Studio comparison is from one machine** (Intel i7-1270P + Intel UHD, Windows, Edge). Two more machines were measured without it (see [BENCHMARKS.md](BENCHMARKS.md)); other hardware is unmeasured.

## Checklist for evaluating it in a restricted environment

1. Copy the repository as text; read `server.py`, `templates.py` and `web/` (the whole code base is a few thousand lines).
2. Start with `python server.py` (default: `127.0.0.1` only). Do not set `"host": "0.0.0.0"` until you also set an API key.
3. If other machines must reach it: set an API key, set `"host"`, and put a TLS reverse proxy in front (add its name to `allowed_hosts`).
4. Run `python tests/run_all.py` (standard library only; expect 205 passing checks).
5. Verify the "no outgoing connections" claim yourself in your environment (below).
6. Check the license of each model you plan to load.
7. Do not expose the server to the internet.

## How we checked the "no outgoing connections" claim

On Linux, we ran the API test suite under `strace` and listed every `connect()`:

```
cd tests
strace -f -qq -e trace=connect -o /tmp/connects.txt python3 api_tests.py
grep 'connect(' /tmp/connects.txt | grep -o 'inet_addr("[^"]*")' | sort | uniq -c
```

Result in our run (205 checks passing): almost all connections go to `127.0.0.1`. The only other addresses were `10.255.255.255` and the test machine's own LAN address.
`10.255.255.255` is the server learning its own LAN IP with a UDP `connect()`, which sends no packet (the trace shows no `sendto` to it). The LAN address was the test client reaching the server through the machine's own address.
No connection to any external host appeared.

Limits of this check: it covers the API test suite with a **simulated engine**, not a browser running the real engine, and it is one run on one machine. The engine page in the browser is served by the server and `web/` contains no external URLs, but the browser itself is yours to audit. Repeat the check in your environment before relying on it.
