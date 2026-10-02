"""DeepSeek-Coder-V2-Lite test against the local server (standard library only).

Usage:  python tools/check_deepseek.py            (two tests: a short one and one with a long prompt)
      python tools/check_deepseek.py "write a function that ..."
"""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:1234"
MODELO = "deepseek-coder-v2-lite-instruct"


def buscar_modelo():
    """Picks from /v1/models the id containing 'deepseek-coder' (in case the file name differs)."""
    global MODELO
    try:
        with urllib.request.urlopen(URL + "/v1/models", timeout=30) as r:
            ids = [m["id"] for m in json.loads(r.read())["data"]]
        for i in ids:
            if "deepseek-coder" in i.lower():
                MODELO = i
                break
        else:
            print("No veo ningun modelo 'deepseek-coder' en /v1/models. Modelos: %s" % ", ".join(ids))
    except Exception as e:
        print("No pude consultar /v1/models:", e)


def pedir(mensajes, max_tokens=200):
    cuerpo = {"model": MODELO, "messages": mensajes, "temperature": 0, "max_tokens": max_tokens}
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(cuerpo).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.loads(r.read())
    return res, time.time() - t0


def mostrar(titulo, mensajes, max_tokens):
    print("\n=== " + titulo)
    try:
        res, seg = pedir(mensajes, max_tokens)
    except Exception as e:
        print("Error:", e)
        if hasattr(e, "read"):
            print(e.read().decode("utf-8", "replace"))
        return False
    u = res.get("usage", {})
    print("Tiempo total: %.1f s   prompt: %s tokens   generados: %s tokens" % (seg, u.get("prompt_tokens"), u.get("completion_tokens")))
    print("--- Respuesta ---\n%s\n-----------------" % res["choices"][0]["message"]["content"])
    return True


def main():
    buscar_modelo()
    print("Modelo:", MODELO)
    if len(sys.argv) > 1:
        return 0 if mostrar("Consulta", [{"role": "user", "content": " ".join(sys.argv[1:])}], 400) else 1
    print("La primera petición incluye cargar el modelo (unos 6,4 GB): puede tardar 1-2 minutos.")
    ok = mostrar("1) Prompt corto (mide la velocidad de generación)",
                 [{"role": "user", "content": "Escribe en Python una función que devuelva los n primeros números de Fibonacci."}], 150)
    codigo = "\n".join("def funcion_%d(x):\n    resultado = x * %d + %d\n    return resultado\n" % (i, i + 1, i * 2) for i in range(20))
    ok = mostrar("2) Prompt largo (mide la lectura del prompt)",
                 [{"role": "user", "content": "Explica en dos frases qué hace este código:\n\n" + codigo}], 80) and ok
    return 0 if ok else 1


sys.exit(main())
