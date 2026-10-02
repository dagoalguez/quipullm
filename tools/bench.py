"""Measures speed against any OpenAI-compatible server (this one or LM Studio).

    python tools/bench.py --url http://localhost:1234 --model ID --runs 3 [--key KEY]

Standard library only. Fixed prompt, temperature 0. Prints per run:
prompt tokens, generated tokens, total time, tok/s (wall clock) and,
if the server reports them in `stats`, the time to first token and its tok/s.
The first run includes model loading if the model was not in memory.
"""
import argparse, json, statistics, sys, time, urllib.request

PROMPT = ("Explica en un párrafo de unas 120 palabras qué es una base de datos "
          "relacional y para qué sirve una clave primaria.")

def pedir(url, modelo, clave, max_tokens, timeout):
    cuerpo = json.dumps({"model": modelo, "temperature": 0, "max_tokens": max_tokens,
                         "messages": [{"role": "user", "content": PROMPT}]}).encode()
    cab = {"Content-Type": "application/json"}
    if clave:
        cab["Authorization"] = "Bearer " + clave
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=cuerpo, headers=cab)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        datos = json.loads(r.read())
    return datos, time.time() - t0

def resumen(datos, seg):
    u = datos.get("usage") or {}
    s = datos.get("stats") or {}
    gen = u.get("completion_tokens") or 0
    return {"prompt": u.get("prompt_tokens"), "gen": gen, "seg": seg,
            "tps_pared": gen / seg if seg > 0 else 0.0,
            "ttft": s.get("time_to_first_token"), "tps_srv": s.get("tokens_per_second")}

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:1234")
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--key", default="")
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args(argv)
    filas = []
    for i in range(1, a.runs + 1):
        try:
            datos, seg = pedir(a.url, a.model, a.key, a.max_tokens, a.timeout)
        except Exception as e:
            print("corrida %d: error: %s" % (i, e)); return 1
        f = resumen(datos, seg); filas.append(f)
        print("corrida %d: prompt=%s gen=%s total=%.2fs pared=%.2f tok/s ttft=%s srv=%s tok/s" % (
            i, f["prompt"], f["gen"], f["seg"], f["tps_pared"],
            "-" if f["ttft"] is None else "%.2fs" % f["ttft"],
            "-" if f["tps_srv"] is None else "%.2f" % f["tps_srv"]))
    if len(filas) > 1:  # the first one may include model loading: reported separately
        resto = [f["tps_pared"] for f in filas[1:]]
        print("mediana tok/s (sin la 1.ª corrida): %.2f" % statistics.median(resto))
    return 0

if __name__ == "__main__":
    sys.exit(main())
