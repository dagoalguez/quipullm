# -*- coding: utf-8 -*-
"""
verify.py - Compares our server against the recorded LM Studio responses.

Replays exactly the same requests from lmstudio_references.json (temperature 0)
against the new server and compares:
  - prompt tokens  -> must be IDENTICAL (tests chat template + tokenizer)
  - generated text     -> identical or with a long common beginning (small numerical
                          differences between engines can change a token further on)
  - speed               -> our tokens/second vs LM Studio

The included references (tools/lmstudio_references.json) were recorded with LM Studio 0.4.25 on the machine in
docs/BENCHMARKS.md. For other hardware, record your own or compare only the prompt tokens.

Usage (with server.py and the engine running):
  python tools/verify.py
  python tools/verify.py --models lfm2-1.2b-rag,liquid/lfm2.5-1.2b
  python tools/verify.py --referencias otra/ruta/referencias.json

Saves the details to tools/verificacion_resultado.json (attach it if something goes wrong).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

AQUI = os.path.dirname(os.path.abspath(__file__))


def pedir(url, datos=None, timeout=1800):
    req = urllib.request.Request(url, data=None if datos is None else json.dumps(datos).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except ValueError:
            return e.code, {}


def texto_de(resp):
    """Joins reasoning and content for comparison."""
    if not isinstance(resp, dict) or not resp.get("choices"):
        return None
    ch = resp["choices"][0]
    if "message" in ch:
        m = ch["message"] or {}
        r = m.get("reasoning_content") or ""
        return (("<razonamiento>" + r + "</razonamiento>") if r else "") + (m.get("content") or "")
    return ch.get("text")


def prefijo_comun(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:1234")
    ap.add_argument("--referencias", default=os.path.join(AQUI, "lmstudio_references.json"))
    ap.add_argument("--models", default="")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    base = a.url.rstrip("/")
    if not os.path.exists(a.referencias):
        sys.exit("No encuentro %s. Indica la ruta con --referencias." % a.referencias)
    with open(a.referencias, encoding="utf-8") as f:
        ref = json.load(f)
    try:
        cod, st = pedir(base + "/api/status", timeout=10)
    except OSError as e:
        sys.exit("El servidor no responde en %s (%s). Inícialo con: python server.py" % (base, e))
    if not st.get("engine", {}).get("connected"):
        sys.exit("El servidor responde pero el motor no está conectado. Abre http://localhost:%s/engine" % st.get("port"))
    inf = st["engine"].get("info") or {}
    print("Motor: %s (versión del motor %s, del servidor %s)" % ((inf.get("gpu") or {}).get("summary", "?"),
                                                                 inf.get("version"), st.get("version")))
    if inf.get("version") != st.get("version"):
        print("  ATENCIÓN: la ventana del motor es de otra versión que el servidor. Cierre la ventana de Edge del")
        print("  motor y vuelva a abrir el servidor (o pulse 'Abrir motor' en el panel) antes de medir.")
    filtros = [x.strip().lower() for x in a.models.split(",") if x.strip()]
    resultado = {"fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "motor": st["engine"].get("info"), "modelos": {}}
    resumen = []

    for mid, datos in ref["modelos"].items():
        if filtros and not any(f in mid.lower() for f in filtros):
            continue
        if datos.get("tipo") == "embeddings":
            continue
        _, r = pedir(base + "/api/resolve?model=" + urllib.parse.quote(mid), timeout=10)
        m = r.get("model")
        if not m:
            print("\n### %s: NO ENCONTRADO en la carpeta de modelos" % mid)
            resumen.append((mid, "no encontrado", "", "", ""))
            continue
        if not m["supported"]:
            print("\n### %s (%s): pendiente, arquitectura aún no soportada" % (mid, m["arch"]))
            resumen.append((mid, "pendiente (%s)" % m["arch"], "", "", ""))
            continue
        print("\n### %s  ->  %s" % (mid, m["file"]))
        det = {"archivo": m["file"], "pruebas": {}}
        t = time.time()
        cod, _ = pedir(base + "/v1/chat/completions", {"model": mid, "max_tokens": 1, "temperature": 0,
                                                        "messages": [{"role": "user", "content": "Hola"}]})
        carga = time.time() - t
        ref_carga = (datos["pruebas"].get("carga") or {}).get("segundos")
        print("   carga: %.1f s (LM Studio %s s)" % (carga, ref_carga))
        det["carga"] = {"nuestro": round(carga, 2), "lmstudio": ref_carga}
        tok_ok = tok_tot = ident = 0
        velocidades, velocidades_ref = [], []
        for clave, p in datos["pruebas"].items():
            if not (clave.startswith("chat:") or clave.startswith("completion:")) or "peticion" not in p:
                continue
            if clave == "chat:json_object":
                cod, r = pedir(base + "/v1/chat/completions", p["peticion"])
                igual = cod == 400
                print("   %-26s %s" % (clave, "OK (rechazado igual que LM Studio)" if igual else "DIFERENTE: código %s" % cod))
                det["pruebas"][clave] = {"ok": igual, "codigo": cod}
                continue
            ruta = "/v1/chat/completions" if clave.startswith("chat:") else "/v1/completions"
            t = time.time()
            cod, r = pedir(base + ruta, p["peticion"])
            seg = time.time() - t
            if cod != 200:
                print("   %-26s ERROR %s: %s" % (clave, cod, (r.get("error") or {}).get("message")))
                det["pruebas"][clave] = {"error": cod, "respuesta": r}
                continue
            nuestro = texto_de(r) or ""
            suyo = texto_de(p.get("respuesta_cruda")) or ""
            pt, pt_ref = r["usage"]["prompt_tokens"], (p.get("usage") or {}).get("prompt_tokens")
            tok_tot += 1
            tok_ok += int(pt == pt_ref)
            com = prefijo_comun(nuestro, suyo)
            if nuestro == suyo:
                estado = "IDÉNTICO"
                ident += 1
            elif com >= 10:
                estado = "coincide los primeros %d caracteres" % com
            else:
                estado = "DISTINTO desde el inicio (carácter %d)" % com
            stt = r.get("stats") or {}
            tps = stt.get("tokens_per_second")
            if tps:
                velocidades.append(tps)
            if p.get("tokens_por_segundo"):
                velocidades_ref.append(p["tokens_por_segundo"])
            print("   %-26s tokens prompt %s/%s %s | %s | %.1f s (1er token %s s)" % (clave, pt, pt_ref, "OK" if pt == pt_ref else "<- DIFERENTE",
                                                                     estado, seg, stt.get("time_to_first_token")))
            if nuestro != suyo:
                print("        nuestro:   %r" % nuestro[:110])
                print("        LM Studio: %r" % suyo[:110])
            det["pruebas"][clave] = {"prompt_tokens": pt, "prompt_tokens_lmstudio": pt_ref, "estado": estado,
                                     "prefijo_comun": com, "nuestro": nuestro, "lmstudio": suyo, "tps": tps,
                                     "segundos": round(seg, 2), "ttft": stt.get("time_to_first_token"),
                                     "diag": stt.get("diag"), "espera_cola_s": stt.get("espera_cola_s")}
        v = round(max(velocidades), 1) if velocidades else None
        vr = round(max(velocidades_ref), 1) if velocidades_ref else None
        resumen.append((mid, "%d/%d" % (tok_ok, tok_tot), "%d/%d" % (ident, tok_tot), v, vr))
        resultado["modelos"][mid] = det
        with open(os.path.join(AQUI, "verificacion_resultado.json"), "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=1)

    print("\n" + "=" * 96)
    print("%-40s %-18s %-12s %-10s %-10s" % ("Modelo", "Tokens prompt OK", "Idénticos", "tok/s", "LM Studio"))
    print("-" * 96)
    for fila in resumen:
        print("%-40s %-18s %-12s %-10s %-10s" % tuple("" if x is None else x for x in fila))
    print("=" * 96)
    print("Cómo leerlo:")
    print(" - 'Tokens prompt OK' debe estar completo (ej. 8/8). Si no, la plantilla o el tokenizador difieren: pásame")
    print("   verificacion_resultado.json.")
    print(" - 'coincide los primeros N caracteres' es normal: LM Studio redondea los cálculos distinto (en Q8_0")
    print("   incluso las activaciones), y con temperatura 0 una diferencia mínima puede elegir otra palabra más")
    print("   adelante. Lo importante es que el texto tenga sentido. 'DISTINTO desde el inicio' sí es un problema.")
    print("Detalle guardado en verificacion_resultado.json")


if __name__ == "__main__":
    main()
