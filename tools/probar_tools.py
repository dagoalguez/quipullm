#!/usr/bin/env python3
"""Measures whether a loaded model follows the experimental tool-calling format.
Standard library only. Usage:
  python tools/probar_tools.py --model NAME [--url http://127.0.0.1:1235] [--key KEY] [--runs 5]
Each run: (1) asks something that needs a tool and checks that a valid tool_call comes back,
(2) sends a fake tool result and checks that the model answers in text.
Prints a table to paste in an issue/benchmark; it does NOT prove general agent quality."""
import argparse, json, time, urllib.request, urllib.error

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather of a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
PREGUNTAS = ["What is the weather in Lima right now?", "Tell me the current weather in Madrid.",
             "How is the weather in Tokyo today?", "Is it raining in Paris? Check the weather.",
             "Weather in Cusco, please."]

def post(a, body):
    req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **({"Authorization": "Bearer " + a.key} if a.key else {})})
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:   # show the server's own explanation, not just the code
        try:
            msg = json.load(e)["error"]["message"]
        except Exception:
            msg = ""
        raise RuntimeError("HTTP %d: %s" % (e.code, msg or e.reason))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True); p.add_argument("--url", default="http://127.0.0.1:1235")
    p.add_argument("--key", default=""); p.add_argument("--runs", type=int, default=5)
    p.add_argument("--timeout", type=int, default=300)
    a = p.parse_args()
    ok1 = ok2 = 0; filas = []
    for i in range(a.runs):
        q = PREGUNTAS[i % len(PREGUNTAS)]
        msgs = [{"role": "user", "content": q}]
        t0 = time.time()
        try:
            r = post(a, {"model": a.model, "messages": msgs, "tools": TOOLS, "temperature": 0})
        except Exception as e:
            filas.append((i + 1, "ERROR", str(e)[:200], "-")); continue
        ch = r["choices"][0]; m = ch["message"]; tc = m.get("tool_calls") or []
        valida = False; det = (m.get("content") or "")[:60].replace("\n", " ")
        if tc and ch.get("finish_reason") == "tool_calls":
            f = tc[0]["function"]
            try:
                args = json.loads(f["arguments"]); valida = f["name"] == "get_weather" and isinstance(args.get("city"), str)
                det = f["name"] + " " + f["arguments"]
            except Exception:
                det = "bad arguments: " + str(f.get("arguments"))[:50]
        ok1 += valida
        fase2 = "-"
        if valida:
            msgs += [{"role": "assistant", "content": None, "tool_calls": tc},
                     {"role": "tool", "tool_call_id": tc[0]["id"], "content": json.dumps({"temp_c": 21, "sky": "cloudy"})}]
            try:
                r2 = post(a, {"model": a.model, "messages": msgs, "tools": TOOLS, "temperature": 0})
                c2 = r2["choices"][0]
                txt = (c2["message"].get("content") or "").strip()
                good = bool(txt) and not c2["message"].get("tool_calls") and c2.get("finish_reason") == "stop"
                ok2 += good; fase2 = "OK" if good else "FAIL"
            except Exception:
                fase2 = "ERROR"
        filas.append((i + 1, "OK" if valida else "FAIL", det, fase2 + f" ({time.time()-t0:.0f}s)"))
    print(f"\nModel: {a.model}   runs: {a.runs}   temperature 0")
    print(f"{'#':<3} {'tool_call':<9} {'detail':<62} answer after result")
    for f in filas: print(f"{f[0]:<3} {f[1]:<9} {f[2]:<62} {f[3]}")
    print(f"\nValid tool_call: {ok1}/{a.runs}   Answered in text after the result: {ok2}/{ok1 or 0}")

main()
