#!/usr/bin/env python3
"""Runs the repository tests.

  python tests/run_all.py            # fast: API (simulated engine) + bench.py; standard library only
  python tests/run_all.py --engine    # also: templates vs jinja2, tokenizer and the WebGPU engine against llama.cpp
                                     # with synthetic models (slow: generates models). The test browser uses
                                     # SwiftShader (software WebGPU): it tests accuracy, not speed.
  python tests/run_all.py --engine qwen3-q4km   # only one transformer variant

Requirements for the --engine part (development ONLY, not for deployment):
  pip install -r tests/requirements-engine.txt && playwright install chromium
Reference versions: llama-cpp-python 0.3.35, gguf 0.19.0, numpy 2.4.4. The vision tests compare against llama.cpp's image preprocessing,
which changed in llama-cpp-python 0.3.36: with that version (or later) those tests fail (docs/LIMITATIONS.md).
"""
import os, subprocess, sys, time

AQUI = os.path.dirname(os.path.abspath(__file__))
CONF = os.path.join(AQUI, "conformance")
CACHE = os.path.join(CONF, "cache")

REFERENCIA = {"llama-cpp-python": "0.3.35", "gguf": "0.19.0", "numpy": "2.4.4"}

def comprobar_entorno():
    """Shows the versions in use and warns if they are not the reference ones (the llama.cpp references depend on them)."""
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:          # Python < 3.8
        return
    print("Entorno de la parte --engine:")
    distintas = []
    for paquete, esperada in REFERENCIA.items():
        try:
            v = version(paquete)
        except PackageNotFoundError:
            v = None
        print("  %-18s %-10s (referencia %s)" % (paquete, v or "NO INSTALADO", esperada))
        if v != esperada:
            distintas.append(paquete)
    if distintas:
        print("  ADVERTENCIA: versiones distintas a la referencia (%s). Con otra versión de llama.cpp las pruebas de visión pueden fallar\n"
              "  aunque el motor esté bien (cambió el preprocesado de imagen en llama-cpp-python 0.3.36). Ver docs/LIMITATIONS.md." % ", ".join(distintas))
    print()

def correr(nombre, args, cwd=None):
    t0 = time.time()
    print("\n=== %s ===" % nombre, flush=True)
    p = subprocess.run([sys.executable] + args, cwd=cwd or CONF, capture_output=True, text=True)
    out = p.stdout + p.stderr
    resumen = [l for l in out.splitlines() if "fallas" in l or "pruebas OK" in l]
    print("   " + (resumen[-1].strip() if resumen else out.strip().splitlines()[-1] if out.strip() else "(sin salida)"))
    ok = (p.returncode == 0)
    if not ok:
        print(out[-7000:])
    print("   %s en %.0f s" % ("OK" if ok else "FALLÓ", time.time() - t0), flush=True)
    return ok

def generar(script, archivo_ref, *args):
    if os.path.exists(os.path.join(CACHE, archivo_ref)):
        return True
    return correr("generar modelos: " + script, [script] + list(args))

def main():
    motor = "--engine" in sys.argv or "--motor" in sys.argv   # --motor: previous name, still accepted
    solo = [a for a in sys.argv[1:] if not a.startswith("--")]
    res = {}
    res["API (modelos y motor simulados)"] = correr("API", [os.path.join(AQUI, "api_tests.py")], cwd=AQUI)
    res["bench.py (servidor simulado)"] = correr("bench", [os.path.join(AQUI, "test_bench.py")], cwd=AQUI)
    if motor:
        comprobar_entorno()
        os.makedirs(CACHE, exist_ok=True)
        res["plantillas de chat vs jinja2"] = correr("plantillas", ["test_templates.py"])
        res["tokenizador vs llama.cpp"] = correr("tokenizador", ["test_tok.py"])
        if generar("gen_modelos.py", "referencias_llamacpp.json"):
            res["LFM2 vs llama.cpp"] = correr("LFM2", ["test_engine.py"])
        if generar("gen_transformer.py", "referencias_tf.json") and generar("gen_gemma.py", "referencias_gemma.json"):
            res["transformer (qwen2/qwen3/llama/granite/gemma3)"] = correr("transformer", ["test_transformer.py"] + solo)
        if not solo:
            res["decisión d1 (/v1/systemone) vs llama.cpp"] = correr("decisión d1", ["test_d1.py"])
            if generar("gen_bert.py", "referencias_bert.json"):
                res["embeddings (nomic-bert)"] = correr("bert", ["test_bert.py"])
            if generar("gen_ds.py", "referencias_ds.json"):
                res["DeepSeek-V2 (MLA + MoE)"] = correr("deepseek", ["test_ds.py"])
            if generar("gen_vision.py", "referencias_vis.json"):
                for m in ("enc", "lm", "api"):
                    res["visión LFM2-VL (%s)" % m] = correr("visión lfm2 " + m, ["test_vision.py", m])
            if generar("gen_gemma_vis.py", "referencias_visg.json"):
                for m in ("enc", "lm", "api"):
                    res["visión Gemma 3 (%s)" % m] = correr("visión gemma3 " + m, ["test_vision_g.py", m])
                os.environ["VISG_VARIANTE"] = "neg"
                res["control negativo: atención causal en la imagen DEBE fallar"] = correr("control negativo", ["test_vision_g.py", "lm"])
                os.environ.pop("VISG_VARIANTE")
    print("\n================ RESUMEN ================")
    for k, v in res.items():
        print("  %-62s %s" % (k, "OK" if v else "FALLÓ"))
    sys.exit(0 if all(res.values()) else 1)

if __name__ == "__main__":
    main()
