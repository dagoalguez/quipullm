"""Experiment: how fast can the experts of a MoE GGUF be read from disk?  (standard library only)

Streaming experts from disk (the technique of Colibri) only makes sense if reading one expert is fast enough.
This tool reads random experts of a MoE model the way a streaming engine would (the gate, up and down slice of one
expert of one layer) and reports the time per expert, directly from the file and, optionally, through the running server
with HTTP Range requests (the path the engine uses).  It changes nothing and loads no model.

Usage:
  python tools/medir_expertos_disco.py MODEL.gguf
  python tools/medir_expertos_disco.py MODEL.gguf --server http://127.0.0.1:1234 --id MODEL_ID_IN_/v1/models
  options: --reps 300 (experts read per pass)  --seed 1

How to read the result:
  * "first pass" reads experts that were probably not read recently; "second pass" reads the same ones again, so the
    operating system keeps them in RAM (page cache). The truth for a real run is somewhere between them. To get an honest
    first pass, run it right after a reboot, or on a model file you have not loaded in the last hours.
  * "tokens/s ceiling" is plain arithmetic: how many tokens per second you would get if the only cost were reading
    the experts of every MoE layer from disk with NO expert cache. It ignores GPU time, so the real figure is lower.
    A cache that hits makes it better. It is an estimate, not a measurement of any engine.
"""
import argparse, http.client, os, random, statistics, sys, time, urllib.parse

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
import server as sv            # only for its GGUF reader (importing it does not start anything)

SUFIJOS = ("gate", "up", "down")


def expertos_del_modelo(ruta):
    """Returns (info, layers): layers[i] = {"gate": (abs, bytes_per_expert), "up": ..., "down": ...} for each MoE layer."""
    kv, tens, _ = sv.leer_gguf(ruta, completo=False)
    arch = kv.get("general.architecture", "?")
    n_exp = kv.get(arch + ".expert_count")
    usados = kv.get(arch + ".expert_used_count")
    if not n_exp:
        raise SystemExit("This GGUF has no %s.expert_count: it is not a MoE model (architecture %s)." % (arch, arch))
    capas = {}
    for t in tens:
        nombre = t["nombre"]
        if not (nombre.startswith("blk.") and "_exps.weight" in nombre):
            continue
        partes = nombre.split(".")
        i = int(partes[1])
        for s in SUFIJOS:
            if partes[2] == "ffn_%s_exps" % s:
                if not t["bytes"]:
                    raise SystemExit("Tensor %s has a type this tool cannot size (%s)." % (nombre, t["tipo_nombre"]))
                if t["dims"][-1] != n_exp:
                    raise SystemExit("Tensor %s: last dimension %s is not the expert count %s." % (nombre, t["dims"][-1], n_exp))
                capas.setdefault(i, {})[s] = (t["abs"], t["bytes"] // n_exp, t["tipo_nombre"])
    capas = {i: c for i, c in capas.items() if all(s in c for s in SUFIJOS)}
    if not capas:
        raise SystemExit("No blk.N.ffn_{gate,up,down}_exps.weight tensors found (this tool expects stacked experts).")
    info = {"arch": arch, "experts": n_exp, "used": usados, "moe_layers": len(capas),
            "bytes_per_expert": sum(c[1] for c in next(iter(capas.values())).values()),
            "types": sorted({c[2] for l in capas.values() for c in l.values()}),
            "file_bytes": os.path.getsize(ruta)}
    return info, capas


def muestra(capas, n_exp, reps, semilla):
    rnd = random.Random(semilla)
    ids = sorted(capas)
    return [(rnd.choice(ids), rnd.randrange(n_exp)) for _ in range(reps)]


def leer_archivo(ruta):
    f = open(ruta, "rb", buffering=0)
    def leer(abs_, n):
        f.seek(abs_)
        b = f.read(n)
        if len(b) != n:
            raise IOError("short read")
        return n
    return leer, f.close


def leer_http(base, ident):
    u = urllib.parse.urlparse(base)
    cx = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=120)
    ruta = "/engine/file/" + urllib.parse.quote(ident)
    def leer(abs_, n):
        cx.request("GET", ruta, headers={"Range": "bytes=%d-%d" % (abs_, abs_ + n - 1)})
        r = cx.getresponse()
        b = r.read()
        if r.status not in (200, 206) or len(b) != n:
            raise SystemExit("The server answered HTTP %s with %d bytes (expected %d) for the model id '%s'. "
                             "Use the exact id from %s/v1/models, and run this on the server PC (the file route is local only)."
                             % (r.status, len(b), n, ident, base))
        return n
    return leer, cx.close


def medir(leer, capas, plan):
    """Time per expert (gate + up + down slices); returns a list of seconds."""
    tiempos = []
    for capa, e in plan:
        t0 = time.perf_counter()
        for s in SUFIJOS:
            abs_, por_exp, _ = capas[capa][s]
            leer(abs_ + e * por_exp, por_exp)
        tiempos.append(time.perf_counter() - t0)
    return tiempos


def resumen(nombre, tiempos, bytes_exp, info):
    med = statistics.median(tiempos)
    p95 = sorted(tiempos)[int(0.95 * (len(tiempos) - 1))]
    mbs = bytes_exp / med / 1e6
    k = info["used"] or 0
    techo = (1.0 / (k * info["moe_layers"] * med)) if k else None
    print("  %-28s median %7.2f ms/expert   p95 %7.2f ms   %7.1f MB/s%s" % (
        nombre, med * 1e3, p95 * 1e3, mbs,
        ("   tokens/s ceiling (no cache) ~%.2f" % techo) if techo else ""))
    return {"median_ms": round(med * 1e3, 3), "p95_ms": round(p95 * 1e3, 3), "MB_s": round(mbs, 1),
            "tokens_per_s_ceiling_no_cache": round(techo, 3) if techo else None}


def main():
    ap = argparse.ArgumentParser(description="Time reading random experts of a MoE GGUF from disk.")
    ap.add_argument("modelo")
    ap.add_argument("--server", help="e.g. http://127.0.0.1:1234 (to also measure through the server)")
    ap.add_argument("--id", help="model id as shown by /v1/models (needed with --server)")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.server and not a.id:
        raise SystemExit("--server needs --id (the model id from /v1/models).")
    info, capas = expertos_del_modelo(a.modelo)
    print("Model: %s" % os.path.basename(a.modelo))
    print("  architecture %s, %d experts per layer (%s used per token), %d MoE layers, quantization %s" % (
        info["arch"], info["experts"], info["used"], info["moe_layers"], "/".join(info["types"])))
    print("  one expert = %.2f MB (gate+up+down); file = %.2f GB" % (info["bytes_per_expert"] / 1e6, info["file_bytes"] / 1e9))
    if a.server and info["bytes_per_expert"] / 3 < 65536:
        print("  NOTE: slices under 64 KB. On Linux the server's small keep-alive answers wait ~40 ms for a TCP ACK (seen at 4 KB "
              "in a synthetic test, not at 64 KB or more), so the HTTP times below would be dominated by that, not by the disk.")
    plan = muestra(capas, info["experts"], a.reps, a.seed)
    res = {"model": os.path.basename(a.modelo), **info}
    for titulo, fabrica in (("directly from the file", lambda: leer_archivo(a.modelo)),
                            ("through the server (HTTP Range)", (lambda: leer_http(a.server, a.id)) if a.server else None)):
        if fabrica is None:
            continue
        print("\n%s, %d random experts:" % (titulo, a.reps))
        leer, cerrar = fabrica()
        try:
            r1 = medir(leer, capas, plan)
            r2 = medir(leer, capas, plan)
        finally:
            cerrar()
        res[titulo] = {"first_pass": resumen("first pass", r1, info["bytes_per_expert"], info),
                       "second_pass": resumen("second pass (cached by the OS)", r2, info["bytes_per_expert"], info)}
    print("\nPaste this block when asking for help interpreting it:")
    import json
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
