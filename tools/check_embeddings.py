"""Quick test of /v1/embeddings against the local server (standard library only)."""
import json, math, sys, time, urllib.request

URL = "http://127.0.0.1:1234"
MODELO = "text-embedding-nomic-embed-text-v1.5"


def pedir(textos, extra=None):
    cuerpo = {"model": MODELO, "input": textos}
    cuerpo.update(extra or {})
    req = urllib.request.Request(URL + "/v1/embeddings", json.dumps(cuerpo).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read()), time.time() - t0


def coseno(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def main():
    textos = ["search_document: Una base de datos guarda información de forma ordenada.",
              "search_document: El servidor responde a las peticiones de la red local.",
              "search_document: La capital de Francia es París.",
              "search_query: ¿qué es una base de datos?",
              "search_query: ciudad capital de Francia"]
    try:
        r, seg = pedir(textos)
    except Exception as e:
        print("Error:", e)
        if hasattr(e, "read"):
            print(e.read().decode("utf-8", "replace"))
        return 1
    v = [d["embedding"] for d in r["data"]]
    print("Primera petición (incluye cargar el modelo): %.1f s, %d vectores de %d números, %d tokens" % (seg, len(v), len(v[0]), r["usage"]["prompt_tokens"]))
    print("Norma del primer vector (debe ser 1): %.4f" % math.sqrt(sum(x * x for x in v[0])))
    r2, seg2 = pedir(textos[:1])
    print("Un texto corto con el modelo ya cargado: %.2f s" % seg2)
    print("\nSimilitud coseno (consulta -> documentos):")
    for i in (3, 4):
        print(" ", textos[i])
        for j in range(3):
            print("     %.3f  %s" % (coseno(v[i], v[j]), textos[j][:60]))
    print("\nEsperado: la consulta sobre bases de datos se parece más al documento 0 y la de Francia al documento 2.")
    return 0


sys.exit(main())
