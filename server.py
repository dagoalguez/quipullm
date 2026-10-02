# -*- coding: utf-8 -*-
"""
quipullm
========
OpenAI / LM Studio compatible LLM server built with only the Python standard library.

- OpenAI / LM Studio compatible API at http://localhost:1234/v1/... (on the network: host 0.0.0.0 + api_key)
- The computation is done by our own engine (web/engine.html), which runs in an
  Edge window using the graphics card (WebGPU). This server:
    * reads the .gguf files from the models folder,
    * maps LM Studio-style names to the right file,
    * builds the prompt with each family's chat template,
    * queues requests (one at a time) and hands them to the engine,
    * returns the response (regular or SSE streaming).

Usage:  python server.py                 (opens the engine in Edge automatically; listens on this PC only)
        python server.py --host 0.0.0.0  (accepts network connections; also set an api_key)
        python server.py --no-engine      (does not open Edge; useful for tests)
"""
import argparse
import base64
import json
import logging
import logging.handlers
import os
import queue
import re
import secrets
import socket
import string
import struct
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import templates
except Exception:   # the server keeps working (with fallback templates)
    templates = None

VERSION = "4.0.1"
ESPERA_REAPERTURA = 90   # seconds a waiting request keeps waiting for the engine window to be reopened (auto-relaunch)
DIR = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(DIR, "web")

# Architectures: discovered in web/arch/*.json (see docs/ARCH_GUIDE.md). Re-read when the models folder is scanned.
DIR_ARCH = os.path.join(WEB, "arch")
BASES_NUCLEO = {"transformer", "lfm2", "bert", "deepseek2"}   # classes built into the engine
MANIFIESTOS = {}          # id -> manifest
ARQUITECTURAS_SOPORTADAS = set()
VISION_SOPORTADA = {}     # LM architecture -> supported projector (mmproj) types


def cargar_manifiestos():
    """Reads web/arch/*.json. An invalid manifest is ignored with a warning (it does not bring the server down)."""
    nuevos = {}
    if os.path.isdir(DIR_ARCH):
        for nombre in sorted(os.listdir(DIR_ARCH)):
            if not nombre.endswith(".json"):
                continue
            ruta = os.path.join(DIR_ARCH, nombre)
            try:
                with open(ruta, encoding="utf-8-sig") as f:
                    d = json.load(f)
                if not isinstance(d, dict) or not isinstance(d.get("id"), str) or not d["id"]:
                    raise ValueError("falta 'id'")
                if not (isinstance(d.get("arch"), list) and d["arch"] and all(isinstance(a, str) for a in d["arch"])):
                    raise ValueError("'arch' must be a list of architecture names")
                if bool(d.get("base")) == bool(d.get("module")):
                    raise ValueError("define exactly one of 'base' (declarative) or 'module' (code)")
                if d.get("base") and d["base"] not in BASES_NUCLEO:
                    raise ValueError("unknown 'base': %s (valid: %s)" % (d["base"], ", ".join(sorted(BASES_NUCLEO))))
                if d.get("module"):
                    if not re.fullmatch(r"[A-Za-z0-9_.-]+\.js", d["module"]):
                        raise ValueError("'module' must be a .js file name inside web/arch/")
                    rm = os.path.join(DIR_ARCH, d["module"])
                    if not os.path.isfile(rm):
                        raise ValueError("module web/arch/%s does not exist" % d["module"])
                    d["_v"] = int(os.path.getmtime(rm))
                nuevos[d["id"]] = d
            except Exception as e:
                log.warning("Architecture manifest %s ignored: %s", nombre, e)
    MANIFIESTOS.clear()
    MANIFIESTOS.update(nuevos)
    ARQUITECTURAS_SOPORTADAS.clear()
    VISION_SOPORTADA.clear()
    for d in nuevos.values():
        ARQUITECTURAS_SOPORTADAS.update(d["arch"])
        for a in d["arch"]:
            if d.get("vision"):
                VISION_SOPORTADA[a] = set(d["vision"])


def manifiesto_de(arch):
    for d in MANIFIESTOS.values():
        if arch in d["arch"]:
            return d
    return None
MARCADOR_IMAGEN = "<__media__>"        # the same one llama.cpp (mtmd) uses
MAX_IMAGEN_BYTES = 25 * 1024 * 1024

CONFIG_DEFECTO = {
    "models_dir": os.path.join(DIR, "models"),
    "host": "127.0.0.1",       # this PC only. "0.0.0.0" = the whole network (also set api_key; see SECURITY.md)
    "allowed_hosts": [],       # extra names accepted in the Host header (e.g. a reverse proxy); ["*"] disables the check
    "port": 1234,
    "default_ctx": 8192,
    "kv_max_mb": 0,            # 0 = automatic (based on estimated memory); a number sets the KV cache cap in MB
    "memory_check": True,        # estimate which models fit and warn about / block those that do not
    "api_key": "",             # empty = API open on the network. With a key: Authorization: Bearer <key> (or the LLM_API_KEY variable)
    "max_queue": 64,            # waiting requests; when full it answers 429 (0 = unlimited)
    "gpu_memory_gb": 0,       # 0 = estimate automatically; a number forces the memory budget for models (GB)
    "tiled_prefill": True,   # tiled prefill kernel (K-quant models); False = the v2.0 method
    "open_engine": True,
    "browser": "",            # path to msedge.exe/chrome.exe; empty = look for Edge
    "timeout_seconds": 1800,   # maximum wait for a request (queue + generation)
    "default_sampling": {   # same defaults as LM Studio
        "temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.1
    },
    "aliases": {
        "liquid/lfm2.5-1.2b": "lfm2.5-1.2b-instruct",
        "google/gemma-3-4b": "gemma-3-4b-it",
        "text-embedding-nomic-embed-text-v1.5": "nomic-embed-text-v1.5",
    },
    "model_overrides": {},
}

log = logging.getLogger("server")
LOG_MEMORIA = deque(maxlen=400)


class _MemHandler(logging.Handler):
    def emit(self, record):
        LOG_MEMORIA.append({"time": time.strftime("%H:%M:%S", time.localtime(record.created)),
                            "level": record.levelname, "message": record.getMessage()})


def configurar_logs():
    os.makedirs(os.path.join(DIR, "logs"), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(os.path.join(DIR, "logs", "server.log"),
                                              maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.setLevel(logging.INFO)
    log.handlers = [fh, sh, _MemHandler()]


def cargar_config(ruta):
    cfg = json.loads(json.dumps(CONFIG_DEFECTO))
    if os.path.exists(ruta):
        with open(ruta, encoding="utf-8-sig") as f:
            usuario = json.load(f)
        for k, v in usuario.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    # a relative folder ("models") is resolved against the server folder, not the current directory
    md = os.path.expanduser(str(cfg.get("models_dir") or "models"))
    cfg["models_dir"] = md if os.path.isabs(md) else os.path.join(DIR, md)
    return cfg


# Settings that can be changed from the panel (key -> (type, minimum, maximum))
AJUSTES_PANEL = {
    "default_ctx": (int, 256, 1048576), "kv_max_mb": (int, 0, 65536), "gpu_memory_gb": (float, 0, 4096), "max_queue": (int, 0, 100000),
    "timeout_seconds": (int, 30, 86400), "port": (int, 1, 65535),
}
MUESTREO_PANEL = {"temperature": (float, 0, 5), "top_k": (int, 0, 1000), "top_p": (float, 0, 1),
                  "min_p": (float, 0, 1), "repeat_penalty": (float, 0.5, 3)}


def validar_ajustes(d):
    """Validates what the panel sends. Returns (changes, errors)."""
    cambios, errores = {}, []
    if "models_dir" in d:
        ruta = os.path.abspath(os.path.expanduser(str(d["models_dir"]).strip().strip('"')))
        if not str(d["models_dir"]).strip():
            errores.append("The models folder cannot be empty")
        elif not os.path.isdir(ruta):
            errores.append("That folder does not exist on the server PC: %s" % ruta)
        else:
            cambios["models_dir"] = ruta
    for k, (tipo, lo, hi) in AJUSTES_PANEL.items():
        if k in d:
            try:
                v = tipo(d[k])
            except (TypeError, ValueError):
                errores.append("%s must be a number" % k); continue
            if not lo <= v <= hi:
                errores.append("%s must be between %s and %s" % (k, lo, hi))
            else:
                cambios[k] = v
    if "api_key" in d:
        clave = str(d["api_key"] or "").strip()
        if len(clave) > 200 or any(ch.isspace() for ch in clave):
            errores.append("The API key cannot contain spaces or be longer than 200 characters")
        else:
            cambios["api_key"] = clave
    for k in ("open_engine", "memory_check"):
        if k in d:
            cambios[k] = bool(d[k])
    if isinstance(d.get("default_sampling"), dict):
        m = {}
        for k, (tipo, lo, hi) in MUESTREO_PANEL.items():
            if k in d["default_sampling"]:
                try:
                    v = tipo(d["default_sampling"][k])
                except (TypeError, ValueError):
                    errores.append("%s must be a number" % k); continue
                if not lo <= v <= hi:
                    errores.append("%s must be between %s and %s" % (k, lo, hi))
                else:
                    m[k] = v
        if m:
            cambios["default_sampling"] = m
    return cambios, errores


def guardar_config(ruta, cambios):
    """Writes only the changed keys to config.json (keeps everything else, e.g. aliases)."""
    try:
        with open(ruta, encoding="utf-8-sig") as f:
            actual = json.load(f)
    except (OSError, ValueError):
        actual = {}
    for k, v in cambios.items():
        if isinstance(v, dict) and isinstance(actual.get(k), dict):
            actual[k].update(v)
        else:
            actual[k] = v
    tmp = ruta + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(actual, f, indent=2, ensure_ascii=False)
    os.replace(tmp, ruta)


def listar_carpetas(ruta):
    """Folder browser of the server PC for the panel."""
    if not ruta:
        if os.name == "nt":
            unidades = ["%s:\\" % l for l in string.ascii_uppercase if os.path.isdir("%s:\\" % l)]
            return {"path": "", "parent": None, "folders": unidades, "ggufs": 0, "complete": True}
        ruta = "/"
    ruta = os.path.abspath(ruta)
    if not os.path.isdir(ruta):
        raise ErrorAPI(400, "That folder does not exist: %s" % ruta)
    try:
        entradas = sorted(os.scandir(ruta), key=lambda e: e.name.lower())
    except OSError as e:
        raise ErrorAPI(403, "Cannot open that folder: %s" % e.strerror)

    def es(e, dir_):
        try:
            return e.is_dir() if dir_ else e.is_file()
        except OSError:
            return False
    carpetas = [e.name for e in entradas if not e.name.startswith(("$", ".")) and es(e, True)]
    ggufs = sum(1 for e in entradas if e.name.lower().endswith(".gguf") and es(e, False))
    padre = os.path.dirname(ruta)
    if padre == ruta:
        padre = "" if os.name == "nt" else None
    return {"path": ruta, "parent": padre, "folders": carpetas, "ggufs": ggufs}


# =====================================================================
#  GGUF reader (standard library only)
# =====================================================================
GGML_TIPOS = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
              10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS",
              17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
              23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16"}
# (elements per block, bytes per block)
GGML_BLOQUES = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
                9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
                14: (256, 210), 15: (256, 292), 20: (32, 18), 30: (1, 2)}
FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
              11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S",
              17: "Q5_K_M", 18: "Q6_K", 32: "BF16"}


class LectorGGUF:
    _ESC = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

    def __init__(self, f):
        self.f = f

    def _u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def _u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def _str(self):
        n = self._u64()
        return self.f.read(n).decode("utf-8", "replace")

    def _valor(self, tipo, completo):
        if tipo in self._ESC:
            fmt = self._ESC[tipo]
            return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]
        if tipo == 8:
            return self._str()
        if tipo == 9:
            sub = self._u32()
            n = self._u64()
            if sub in self._ESC:
                fmt = self._ESC[sub]
                tam = struct.calcsize(fmt)
                if not completo and n > 64:
                    self.f.seek(tam * n, 1)
                    return {"_arreglo_omitido": n}
                datos = self.f.read(tam * n)
                return list(struct.unpack("<%d%s" % (n, fmt[1]), datos))
            if sub == 8:
                if not completo and n > 64:
                    for _ in range(n):
                        self.f.seek(self._u64(), 1)
                    return {"_arreglo_omitido": n}
                return [self._str() for _ in range(n)]
            return [self._valor(sub, completo) for _ in range(n)]
        raise ValueError("tipo GGUF desconocido %s" % tipo)


def leer_gguf(ruta, completo=False):
    """Returns (kv, tensors, data_start). With completo=False it skips large arrays."""
    with open(ruta, "rb", buffering=1 << 20) as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        r = LectorGGUF(f)
        version = r._u32()
        if version < 2:
            raise ValueError("GGUF version %d not supported" % version)
        n_tensores = r._u64()
        n_kv = r._u64()
        kv = {}
        for _ in range(n_kv):
            k = r._str()
            t = r._u32()
            kv[k] = r._valor(t, completo)
        tensores = []
        for _ in range(n_tensores):
            nombre = r._str()
            nd = r._u32()
            dims = [r._u64() for _ in range(nd)]
            tipo = r._u32()
            off = r._u64()
            tensores.append({"nombre": nombre, "dims": dims, "tipo": tipo, "offset": off})
        alin = kv.get("general.alignment", 32) or 32
        pos = f.tell()
        inicio = (pos + alin - 1) // alin * alin
    for t in tensores:
        n = 1
        for d in t["dims"]:
            n *= d
        epb, bpb = GGML_BLOQUES.get(t["tipo"], (0, 0))
        t["bytes"] = (n // epb) * bpb if epb else None
        t["tipo_nombre"] = GGML_TIPOS.get(t["tipo"], str(t["tipo"]))
        t["abs"] = inicio + t["offset"]
    return kv, tensores, inicio


# =====================================================================
#  Model registry and name resolution
# =====================================================================
_RE_CUANT = re.compile(r"[-._](q\d(_k)?(_[a-z0-9]+)?|q\d_\d|iq\d[a-z0-9_]*|f16|bf16|f32|fp16)$")


def normalizar(nombre):
    """Lookup keys for a model name (file, folder or LM Studio id)."""
    n = nombre.strip().lower().replace("\\", "/")
    partes = [n, n.rsplit("/", 1)[-1]]
    claves = set()
    for p in partes:
        for _ in range(3):
            p = re.sub(r"\.gguf$", "", p)
            p = re.sub(r"-gguf$", "", p)
            p = _RE_CUANT.sub("", p)
        claves.add(p)
        for suf in ("-instruct", "-it", "-chat"):
            if p.endswith(suf):
                claves.add(p[: -len(suf)])
    return claves


def elegir_mmproj(carpeta, stem=""):
    """Vision projector (mmproj*.gguf) from a model's folder. With several, prefers the one that shares its name with the
    model and then the highest precision (F16 before Q8_0)."""
    cand = [a for a in os.listdir(carpeta) if a.lower().startswith("mmproj") and a.lower().endswith(".gguf")]
    if not cand:
        return None
    def nota(a):
        b = a.lower()
        comun = len(normalizar(stem) & normalizar(a[len("mmproj-"):] if b.startswith("mmproj-") else a))
        prec = 0 if "f16" in b else (1 if "bf16" in b else (2 if "f32" in b else 3))
        return (-comun, prec, b)
    return os.path.join(carpeta, sorted(cand, key=nota)[0])


# =====================================================================
#  Auto-tuning: server PC hardware and per-model memory estimation
# =====================================================================
FACTOR_PESOS = 0.95          # bytes on GPU / bytes of the .gguf (calibrated with the models tested on the real PC)
MARGEN_FIJO = 350e6          # activations, temporary buffers and pipelines
FRACCION_RAM_COMPARTIDA = 0.5  # integrated GPU: the browser is assumed to be able to use ~half of the RAM
UMBRAL_CABE, UMBRAL_BLOQUEO = 0.80, 1.25
_cache_hw = {"t": 0, "v": None}


def ram_total_bytes():
    """Physical RAM of the PC (None if it cannot be read). Standard library only."""
    try:
        if os.name == "nt":
            import ctypes

            class MEMORIA(ctypes.Structure):
                _fields_ = [("largo", ctypes.c_ulong), ("carga", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("libre", ctypes.c_ulonglong), ("tpag", ctypes.c_ulonglong), ("lpag", ctypes.c_ulonglong),
                            ("tvirt", ctypes.c_ulonglong), ("lvirt", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
            ms = MEMORIA()
            ms.largo = ctypes.sizeof(MEMORIA)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return int(ms.total)
            return None
        with open("/proc/meminfo") as f:
            for linea in f:
                if linea.startswith("MemTotal:"):
                    return int(linea.split()[1]) * 1024
    except Exception:
        pass
    return None


def vram_nvidia():
    """(name, total bytes) of the NVIDIA GPU with the most memory, via nvidia-smi if installed; otherwise None."""
    try:
        kw = {"creationflags": 0x08000000} if os.name == "nt" else {}
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, timeout=4, text=True, **kw)
        if r.returncode != 0:
            return None
        mejor = None
        for linea in r.stdout.strip().splitlines():
            nombre, mb = [x.strip() for x in linea.rsplit(",", 1)]
            if mejor is None or float(mb) > mejor[1] / 1048576:
                mejor = (nombre, int(float(mb) * 1048576))
        return mejor
    except Exception:
        return None


def tipo_gpu(info_gpu):
    """'integrated' | 'discrete' | 'unknown' based on what the WebGPU adapter reports."""
    d = ((info_gpu or {}).get("summary") or "").lower() if isinstance(info_gpu, dict) else str(info_gpu or "").lower()
    if not d or d == "unknown":
        return "unknown"
    if "nvidia" in d:
        return "discrete"
    if "apple" in d:
        return "integrated"
    if "intel" in d:
        return "discrete" if ("arc" in d or "xe-hpg" in d or "xe2-hpg" in d) else "integrated"
    return "unknown"


def hardware(cfg, info_gpu):
    """Memory budget for models and machine data. Cached for a few seconds."""
    ahora = time.time()
    clave = (json.dumps(info_gpu, sort_keys=True, default=str), cfg.get("gpu_memory_gb"))
    c = _cache_hw
    if c["v"] is not None and ahora - c["t"] < 5 and c.get("k") == clave:
        return c["v"]
    ram = ram_total_bytes()
    if "_nv" not in c or ahora - c.get("t_nv", 0) > 60:
        c["_nv"], c["t_nv"] = vram_nvidia(), ahora
    nv = c["_nv"]
    tipo = tipo_gpu(info_gpu)
    avisos = []
    if cfg.get("gpu_memory_gb"):
        presup, origen = float(cfg["gpu_memory_gb"]) * 1e9, "set in the configuration"
    elif tipo == "discrete" and nv and "nvidia" in ((info_gpu or {}).get("summary") or "").lower():
        presup, origen = nv[1] * 0.90, "VRAM of %s (nvidia-smi)" % nv[0]
    elif ram:
        presup, origen = ram * FRACCION_RAM_COMPARTIDA, "half of the RAM (memory shared with the GPU)"
        if tipo == "discrete":
            avisos.append("The card is discrete but its VRAM could not be read: the estimate uses system RAM. "
                          "If you know the VRAM, set “Memory for models” in Settings.")
    else:
        presup, origen = None, "could not read the machine's memory"
    if nv and tipo != "discrete" and info_gpu and "nvidia" not in ((info_gpu or {}).get("summary") or "").lower():
        avisos.append("There is an NVIDIA GPU (%s) but Edge is using another GPU. On Windows: Settings → System → Display → "
                      "Graphics → Microsoft Edge → Options → High performance." % nv[0])
    v = {"ram_bytes": ram, "vram_bytes": nv[1] if nv else None, "vram_nombre": nv[0] if nv else None,
         "tipo_gpu": tipo, "presupuesto_bytes": presup, "origen": origen, "avisos": avisos}
    c.update(t=ahora, v=v, k=clave)
    return v


def estimar_memoria(bytes_modelo, kv_glob, kv_local, ctx, bytes_vision, presupuesto):
    """Estimated GPU memory and verdict: fits / tight / no_fit / unknown."""
    necesita = bytes_modelo * FACTOR_PESOS + ctx * kv_glob + kv_local + bytes_vision + MARGEN_FIJO
    if not presupuesto:
        return {"necesita_bytes": int(necesita), "state": "unknown", "ratio": None}
    r = necesita / presupuesto
    return {"necesita_bytes": int(necesita), "ratio": round(r, 2),
            "state": "fits" if r <= UMBRAL_CABE else ("tight" if r <= 1.0 else "no_fit")}


class Registro:
    def __init__(self, cfg):
        self.cfg = cfg
        self.modelos = []   # list of dicts
        self.lock = threading.Lock()
        self._meta_cache = {}

    def escanear(self):
        cargar_manifiestos()
        carpeta = self.cfg["models_dir"]
        encontrados = []
        if not os.path.isdir(carpeta):
            log.warning("The models folder does not exist: %s", carpeta)
        for raiz, _, archivos in os.walk(carpeta):
            for a in sorted(archivos):
                if not a.lower().endswith(".gguf") or a.lower().startswith("mmproj"):
                    continue
                ruta = os.path.join(raiz, a)
                try:
                    kv, tens, _ = leer_gguf(ruta, completo=False)
                except Exception as e:
                    log.warning("Could not read %s: %s", ruta, e)
                    continue
                arch = kv.get("general.architecture", "?")
                stem = a[:-5]
                claves = normalizar(stem)
                rel = os.path.relpath(ruta, carpeta).replace("\\", "/")
                carpeta_modelo = os.path.dirname(rel)
                if carpeta_modelo:
                    claves |= normalizar(carpeta_modelo)
                    claves |= {c for c in normalizar(carpeta_modelo.split("/")[0] + "/" + stem)}
                mid = sorted(normalizar(stem), key=len, reverse=True)[0]
                ctx_modelo = kv.get(arch + ".context_length") or 0
                mmproj = elegir_mmproj(raiz, stem)
                vision = mmproj is not None
                vision_ok = False
                if mmproj:
                    try:
                        kvm, _, _ = leer_gguf(mmproj, completo=False)
                        proy = kvm.get("clip.projector_type") or kvm.get("clip.vision.projector_type")
                        vision_ok = (proy in VISION_SOPORTADA.get(arch, ()) and bool(kvm.get("clip.has_vision_encoder", True)))
                    except Exception as e:
                        log.warning("Could not read projector %s: %s", mmproj, e)
                encontrados.append({
                    "id": mid, "archivo": ruta, "relativo": rel, "claves": claves, "arch": arch,
                    "nombre": kv.get("general.name") or stem, "bytes": os.path.getsize(ruta),
                    "cuantizacion": FILE_TYPES.get(kv.get("general.file_type"), _cuant_de_nombre(stem)),
                    "ctx_modelo": ctx_modelo, "soportado": arch in ARQUITECTURAS_SOPORTADAS,
                    "embedding": "embed" in stem.lower() or bool((manifiesto_de(arch) or {}).get("embedding")),
                    "vision": vision, "mmproj": mmproj, "vision_ok": vision_ok, "publisher": carpeta_modelo.split("/")[0] if carpeta_modelo else "",
                    "n_tensores": len(tens),
                })
        # unique ids
        vistos = {}
        for m in encontrados:
            if m["id"] in vistos:
                vistos[m["id"]] += 1
                m["id"] = "%s:%d" % (m["id"], vistos[m["id"]])
            else:
                vistos[m["id"]] = 1
        with self.lock:
            self.modelos = encontrados
            self._meta_cache.clear()
        log.info("Models found: %d (supported in this version: %d)", len(encontrados),
                 sum(1 for m in encontrados if m["soportado"]))
        return encontrados

    def resolver(self, nombre):
        """Returns the model for a requested name, or None."""
        if not nombre:
            return None
        with self.lock:
            modelos = list(self.modelos)
        for m in modelos:
            if m["id"] == nombre:
                return m
        alias = {k.lower(): v for k, v in self.cfg.get("aliases", {}).items()}
        if nombre.lower() in alias:
            nombre = alias[nombre.lower()]
        pedidas = normalizar(nombre)
        # 1) exact match with the id
        for m in modelos:
            if m["id"] in pedidas:
                return m
        # 2) key match; prefers the longest (most specific) key
        mejor, largo = None, -1
        for m in modelos:
            comunes = pedidas & m["claves"]
            if comunes:
                l = max(len(c) for c in comunes)
                if l > largo:
                    mejor, largo = m, l
        return mejor

    def por_id(self, mid):
        with self.lock:
            for m in self.modelos:
                if m["id"] == mid:
                    return m
        return None

    def kv_coste(self, m):
        """(bytes per token of the global layers, fixed bytes of the local layers) of the KV cache (f32)."""
        with self.lock:
            v = m.get("_kv_coste")
        if v is not None:
            return v
        try:
            kv, _, _ = leer_gguf(m["archivo"], completo=False)
        except Exception:
            return (0, 0)
        a = m["arch"]
        n_capas = int(kv.get(a + ".block_count") or 0)
        cab = kv.get(a + ".attention.head_count") or 0
        nkv = kv.get(a + ".attention.head_count_kv") or cab
        emb = int(kv.get(a + ".embedding_length") or 0)
        ventana = int(kv.get(a + ".attention.sliding_window") or 0) if a == "gemma3" else 0
        patron = int(kv.get(a + ".attention.sliding_window_pattern") or 6)
        glob = local = 0
        if a == "deepseek2":   # compressed cache (MLA): kv_lora_rank + RoPE dimensions, per token and layer
            por_tok = (int(kv.get(a + ".attention.kv_lora_rank") or 512) + int(kv.get(a + ".rope.dimension_count") or 64)) * 4
            with self.lock:
                m["_kv_coste"] = (n_capas * por_tok, 0)
            return (n_capas * por_tok, 0)
        for i in range(n_capas):
            h = nkv[i] if isinstance(nkv, list) else nkv
            hc = cab[i] if isinstance(cab, list) else cab
            if not h:
                continue   # layer without attention (e.g. LFM2 convolution layers)
            hd = int(kv.get(a + ".attention.key_length") or (emb // hc if hc else 0))
            por_tok = int(h) * hd * 2 * 4
            if ventana and (i % patron) < patron - 1:
                local += por_tok * (ventana + 32)   # the engine stores only a ring of window + 32 positions
            else:
                glob += por_tok
        with self.lock:
            m["_kv_coste"] = (glob, local)
        return (glob, local)

    def _hw(self):
        p = ESTADO.get("puente")
        info = ((p.motor.get("info") or {}).get("gpu")) if p else None
        return hardware(self.cfg, info)

    def _bytes_vision(self, m):
        try:
            return os.path.getsize(m["mmproj"]) if (m.get("vision_ok") and m.get("mmproj")) else 0
        except OSError:
            return 0

    def estimar(self, m, ctx=None):
        hw = self._hw()
        glob, local = self.kv_coste(m)
        ctx = ctx if ctx is not None else self.ctx_para(m)
        e = estimar_memoria(m["bytes"], glob, local, ctx, self._bytes_vision(m), hw["presupuesto_bytes"])
        e["presupuesto_bytes"] = hw["presupuesto_bytes"]
        return e

    def _tope_kv_auto(self, m):
        """Memory for the KV cache when kv_max_mb = 0: half of what is left after the weights, between 256 MB and 4 GB."""
        presup = self._hw()["presupuesto_bytes"]
        if not presup:
            return 640e6
        sobra = presup - m["bytes"] * FACTOR_PESOS - self._bytes_vision(m) - MARGEN_FIJO
        return min(4e9, max(256e6, sobra * 0.5))

    def ctx_para(self, m):
        """Effective context: config, model cap and memory budget for the KV cache."""
        ov = self.cfg.get("model_overrides", {}).get(m["id"], {})
        explicito = bool(ov.get("ctx"))
        ctx = int(ov.get("ctx") or self.cfg.get("default_ctx") or 8192)
        if m.get("ctx_modelo"):
            ctx = min(ctx, int(m["ctx_modelo"]))
        if not explicito:
            # The engine's KV cache is f32; on large models it limits the context so RAM/VRAM is not exhausted.
            kvmax = float(self.cfg.get("kv_max_mb") or 0)
            tope = kvmax * 1e6 if kvmax > 0 else self._tope_kv_auto(m)
            glob, local = self.kv_coste(m)
            if glob and ctx * glob + local > tope:
                ctx = max(2048, int((tope - local) / glob) // 256 * 256)
        return ctx

    def plantilla(self, m):
        """Compiled chat template (Jinja) from the GGUF, or None if absent / unsupported."""
        if templates is None:
            return None
        with self.lock:
            if "_plantilla" in m:
                return m["_plantilla"]
        pl = None
        try:
            kv, _, _ = leer_gguf(m["archivo"], completo=True)
            src = kv.get("tokenizer.chat_template")
            toks = kv.get("tokenizer.ggml.tokens") or []

            def tok(k):
                i = kv.get("tokenizer.ggml.%s_token_id" % k)
                return toks[i] if isinstance(i, int) and 0 <= i < len(toks) else ""
            if isinstance(src, str) and src.strip():
                pl = templates.Plantilla(src, tok("bos"), tok("eos"))
                pl.add_bos = bool(kv.get("tokenizer.ggml.add_bos_token", False))
        except Exception as e:
            log.warning("Chat template of %s not usable (%s): using the fallback", m["id"], e)
            pl = None
        with self.lock:
            m["_plantilla"] = pl
        return pl

    def meta_json(self, m):
        """Full metadata (includes vocabulary) for the engine, cached."""
        clave = (m["archivo"], os.path.getmtime(m["archivo"]), self.ctx_para(m))
        with self.lock:
            if clave in self._meta_cache:
                return self._meta_cache[clave]
        kv, tens, inicio = leer_gguf(m["archivo"], completo=True)
        cuerpo = json.dumps({"id": m["id"], "kv": kv, "tensors": tens, "data_start": inicio,
                             "ctx": self.ctx_para(m),
                             "overrides": self.cfg.get("model_overrides", {}).get(m["id"], {}),
                             "arch_options": (manifiesto_de(m["arch"]) or {}).get("options", {}),
                             "options": {"tiled_prefill": bool(self.cfg.get("tiled_prefill", True))}},
                            ensure_ascii=False).encode("utf-8")
        with self.lock:
            if len(self._meta_cache) >= 2:
                self._meta_cache.pop(next(iter(self._meta_cache)))
            self._meta_cache[clave] = cuerpo
        return cuerpo

    def meta_mmproj_json(self, m):
        """Vision projector metadata (without vocabulary) for the engine, cached."""
        if not m.get("mmproj"):
            return None
        clave = (m["mmproj"], os.path.getmtime(m["mmproj"]))
        with self.lock:
            if clave in self._meta_cache:
                return self._meta_cache[clave]
        kv, tens, inicio = leer_gguf(m["mmproj"], completo=True)
        cuerpo = json.dumps({"id": m["id"], "kv": kv, "tensors": tens, "data_start": inicio},
                            ensure_ascii=False).encode("utf-8")
        with self.lock:
            self._meta_cache[clave] = cuerpo
        return cuerpo

    def publico(self, m):
        return {"id": m["id"], "arch": m["arch"], "name": m["nombre"], "file": m["relativo"],
                "gb": round(m["bytes"] / 1e9, 2), "quantization": m["cuantizacion"],
                "supported": m["soportado"], "embedding": m["embedding"], "vision": m["vision"], "vision_ok": m["vision_ok"],
                "match_keys": sorted(m["claves"]), "ctx": self.ctx_para(m), "memory": self._memoria_publica(m)}

    def _memoria_publica(self, m):
        if not m["soportado"]:
            return None
        e = self.estimar(m)
        return {"needs_gb": round(e["necesita_bytes"] / 1e9, 1), "state": e["state"], "ratio": e["ratio"]}


def _cuant_de_nombre(stem):
    mm = _RE_CUANT.search(stem.lower())
    return mm.group(0)[1:].upper() if mm else "?"


# =====================================================================
#  Chat templates
# =====================================================================
def texto_de_contenido(contenido):
    """content can be a string or a list of OpenAI parts. Returns (text, has_image)."""
    if contenido is None:
        return "", False
    if isinstance(contenido, str):
        return contenido, False
    textos, imagen = [], False
    for p in contenido:
        if isinstance(p, dict):
            if p.get("type") == "text":
                textos.append(p.get("text", ""))
            elif p.get("type") in ("image_url", "input_image", "image"):
                imagen = True
        elif isinstance(p, str):
            textos.append(p)
    return "\n".join(textos), imagen


def _url_de_imagen(p):
    v = p.get("image_url", p.get("image"))
    if isinstance(v, dict):
        v = v.get("url")
    return v if isinstance(v, str) else None


def contenido_con_imagenes(contenido, imagenes):
    """Converts 'content' (text or list of OpenAI parts) into a single text with llama.cpp's image marker in
    place of each image; the images (data: URL) are appended to 'imagenes' as {"tipo", "b64"}.
    Joins the parts with a newline, with no newline next to a marker (same as llama.cpp)."""
    if contenido is None:
        return ""
    if isinstance(contenido, str):
        return contenido.replace(MARCADOR_IMAGEN, "")
    texto, ultimo_marcador = "", False
    for p in contenido:
        if isinstance(p, str):
            p = {"type": "text", "text": p}
        if not isinstance(p, dict):
            continue
        tipo = p.get("type")
        if tipo == "text":
            t = str(p.get("text", "")).replace(MARCADOR_IMAGEN, "")
            if texto and not ultimo_marcador:
                texto += "\n"
            ultimo_marcador = False
            texto += t
        elif tipo in ("image_url", "input_image", "image"):
            url = _url_de_imagen(p)
            m = re.match(r"^data:([\w.+/-]*)(;[^,]*)?;base64,(.*)$", url or "", re.S)
            if not m:
                raise ErrorAPI(400, "Only images as base64 data URLs are supported (data:image/png;base64,...).")
            b64 = re.sub(r"\s+", "", m.group(3))
            try:
                n = len(base64.b64decode(b64, validate=True))
            except Exception:
                raise ErrorAPI(400, "The image is not valid base64.")
            if n > MAX_IMAGEN_BYTES:
                raise ErrorAPI(400, "The image is %d MB; the maximum is %d MB." % (n >> 20, MAX_IMAGEN_BYTES >> 20))
            imagenes.append({"tipo": m.group(1) or "image/png", "b64": b64})
            texto += MARCADOR_IMAGEN
            ultimo_marcador = True
    return texto


def _msgs_planos(mensajes):
    out = []
    for m in mensajes:
        texto, _ = texto_de_contenido(m.get("content"))
        out.append({"role": m.get("role", "user"), "content": texto})
    return out


def plantilla_respaldo(arch, plantilla_src, mensajes):
    """Hand-written templates, in case the GGUF's is missing or uses something the interpreter does not understand."""
    msgs = _msgs_planos(mensajes)
    src = plantilla_src or ""
    if arch == "lfm2":
        # Liquid ChatML: <|im_start|>role\ncontent<|im_end|>\n ... <|im_start|>assistant\n
        partes = []
        for m in msgs:
            texto, rol = m["content"], m["role"]
            if rol == "assistant":
                texto = re.sub(r"^\s*<think>.*?</think>\s*", "", texto, flags=re.S)
            partes.append("<|im_start|>%s\n%s<|im_end|>\n" % (rol, texto))
        partes.append("<|im_start|>assistant\n")
        return "".join(partes), False
    if arch == "qwen2" and "\uff5cUser\uff5c" in src:
        # Distilled DeepSeek-R1 (qwen2 architecture)
        u, a = "<\uff5cUser\uff5c>", "<\uff5cAssistant\uff5c>"
        sistema = "".join(m["content"] for m in msgs if m["role"] == "system")
        out = "<\uff5cbegin\u2581of\u2581sentence\uff5c>" + sistema
        for m in msgs:
            if m["role"] == "user":
                out += u + m["content"]
            elif m["role"] == "assistant":
                c = m["content"].split("</think>")[-1]
                out += a + c + "<\uff5cend\u2581of\u2581sentence\uff5c>"
        return out + a + "<think>\n", True
    if arch == "qwen2":
        if not msgs or msgs[0]["role"] != "system":
            msgs = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}] + msgs
        return "".join("<|im_start|>%s\n%s<|im_end|>\n" % (m["role"], m["content"]) for m in msgs) + "<|im_start|>assistant\n", False
    if arch == "granite":
        out = ""
        if not msgs or msgs[0]["role"] != "system":
            msgs = [{"role": "system", "content": "Knowledge Cutoff Date: April 2024.\nToday's Date: %s.\nYou are Granite, developed by IBM. You are a helpful AI assistant." %
                     time.strftime("%B %d, %Y")}] + msgs
        for m in msgs:
            out += "<|start_of_role|>%s<|end_of_role|>%s<|end_of_text|>\n" % (m["role"], m["content"])
        return out + "<|start_of_role|>assistant<|end_of_role|>", False
    if arch == "gemma3":
        # <bos><start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n
        prefijo = ""
        if msgs and msgs[0]["role"] == "system":
            prefijo = msgs[0]["content"].strip() + "\n\n"
            msgs = msgs[1:]
        out = "<bos>"
        for i, m in enumerate(msgs):
            rol = "model" if m["role"] == "assistant" else m["role"]
            out += "<start_of_turn>%s\n%s%s<end_of_turn>\n" % (rol, prefijo if i == 0 else "", m["content"].strip())
        return out + "<start_of_turn>model\n", True
    if arch == "deepseek2":
        # DeepSeek-Coder-V2: <bos>{system}\n\nUser: ...\n\nAssistant: ...<eos>User: ...\n\nAssistant:
        out = "\uff5cbegin\u2581of\u2581sentence\uff5c".join(["<", ">"])
        eos = "\uff5cend\u2581of\u2581sentence\uff5c".join(["<", ">"])
        for m in msgs:
            if m["role"] == "user":
                out += "User: " + m["content"] + "\n\n"
            elif m["role"] == "assistant":
                out += "Assistant: " + m["content"] + eos
            elif m["role"] == "system":
                out += m["content"] + "\n\n"
        return out + "Assistant:", True
    if arch == "llama":
        # Mistral (v3/Tekken): <s>[INST] user[/INST] reply</s>[INST] ...
        sistema = "\n\n".join(m["content"] for m in msgs if m["role"] == "system")
        conv = [m for m in msgs if m["role"] != "system"]
        out = "<s>"
        ult_user = max([i for i, m in enumerate(conv) if m["role"] == "user"], default=-1)
        for i, m in enumerate(conv):
            if m["role"] == "user":
                pre = (sistema + "\n\n") if (i == ult_user and sistema) else ""
                out += "[INST]" + pre + m["content"] + "[/INST]"
            else:
                out += m["content"] + "</s>"
        return out, True
    fam = (manifiesto_de(arch) or {}).get("fallback_template")
    if fam == "chatml":   # generic ChatML (Qwen 3, Yi, OLMo, etc.)
        return "".join("<|im_start|>%s\n%s<|im_end|>\n" % (m["role"], m["content"]) for m in msgs) + "<|im_start|>assistant\n", False
    if fam and fam != arch:
        return plantilla_respaldo(fam, plantilla_src, mensajes)
    raise ValueError("no chat template for architecture " + arch)


def plantilla_chat(arch, mensajes, plantilla=None):
    """Builds the prompt. Returns (text, includes_bos). If includes_bos, the engine does not add another BOS."""
    if arch == "lfm2":   # validated against LM Studio: does not depend on the interpreter
        return plantilla_respaldo(arch, None, mensajes)
    src = plantilla.src if plantilla is not None else None
    if plantilla is not None:
        try:
            texto = plantilla.renderizar(_msgs_planos(mensajes), add_generation_prompt=True)
            incluye = bool(plantilla.bos) and texto.startswith(plantilla.bos)
            return texto, incluye
        except Exception as e:   # NoSoportado, ErrorPlantilla (includes the template's raise_exception)
            if templates is not None and isinstance(e, templates.ErrorPlantilla):
                raise
            log.warning("The GGUF template failed (%s): using the fallback", e)
    return plantilla_respaldo(arch, src, mensajes)


# =====================================================================
#  Text post-processing: stop strings and reasoning separation
# =====================================================================
class FiltroStop:
    """Holds back the end of the text that could be the start of a stop sequence."""

    def __init__(self, stops):
        self.stops = [s for s in (stops or []) if s]
        self.buf = ""
        self.detenido = False

    def alimentar(self, texto):
        if self.detenido:
            return ""
        self.buf += texto
        if not self.stops:
            salida, self.buf = self.buf, ""
            return salida
        idx = min((i for i in (self.buf.find(s) for s in self.stops) if i >= 0), default=-1)
        if idx >= 0:
            salida = self.buf[:idx]
            self.buf = ""
            self.detenido = True
            return salida
        retener = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(self.buf)), 0, -1):
                if self.buf.endswith(s[:k]):
                    retener = max(retener, k)
                    break
        salida = self.buf[:len(self.buf) - retener]
        self.buf = self.buf[len(self.buf) - retener:]
        return salida

    def vaciar(self):
        s, self.buf = ("" if self.detenido else self.buf), ""
        return s


class SeparadorRazonamiento:
    """If the response starts with <think>, sends that to reasoning_content (same as LM Studio)."""
    A, C = "<think>", "</think>"

    def __init__(self, activo=True):
        self.estado = "inicio" if activo else "contenido"
        self.buf = ""

    def alimentar(self, texto):
        out = []
        self.buf += texto
        while self.buf:
            if self.estado == "inicio":
                s = self.buf.lstrip()
                if s.startswith(self.A):
                    self.estado = "razon"
                    self.buf = s[len(self.A):]
                    continue
                if self.A.startswith(s):  # it could still be <think>
                    return out
                self.estado = "contenido"
            if self.estado == "razon":
                i = self.buf.find(self.C)
                if i >= 0:
                    if i:
                        out.append(("razon", self.buf[:i]))
                    self.buf = self.buf[i + len(self.C):].lstrip("\n")
                    self.estado = "post"
                    continue
                retener = 0
                for k in range(min(len(self.C) - 1, len(self.buf)), 0, -1):
                    if self.buf.endswith(self.C[:k]):
                        retener = k
                        break
                if len(self.buf) > retener:
                    out.append(("razon", self.buf[:len(self.buf) - retener]))
                self.buf = self.buf[len(self.buf) - retener:]
                return out
            if self.estado == "post":
                s = self.buf.lstrip("\n")
                if not s:
                    self.buf = ""
                    return out
                self.buf = s
                self.estado = "contenido"
            out.append(("contenido", self.buf))
            self.buf = ""
        return out

    def vaciar(self):
        b, self.buf = self.buf, ""
        if not b:
            return []
        return [("razon" if self.estado == "razon" else "contenido", b)]


# =====================================================================
#  Job queue and bridge to the engine
# =====================================================================
class Trabajo:
    def __init__(self, accion, modelo, prompt="", params=None, origen=""):
        self.id = "t" + secrets.token_hex(6)
        self.accion = accion          # generate | load | unload
        self.modelo = modelo
        self.prompt = prompt
        self.params = params or {}
        self.origen = origen
        self.eventos = queue.Queue()
        self.cancelado = False
        self.creado = time.time()
        self.inicio = None
        self.ultimo_evento = time.time()
        self.fase = "queued"
        self.tokens = 0
        self.terminado = False
        self.motor_id = None

    def para_motor(self, reg):
        d = {"id": self.id, "accion": self.accion, "params": self.params, "prompt": self.prompt}
        if self.modelo:
            d["modelo"] = {"id": self.modelo["id"], "arch": self.modelo["arch"],
                           "meta_url": "/engine/api/meta/" + urllib.parse.quote(self.modelo["id"]),
                           "archivo_url": "/engine/file/" + urllib.parse.quote(self.modelo["id"]),
                           "ctx": reg.ctx_para(self.modelo), "bytes": self.modelo["bytes"]}
            if self.modelo.get("mmproj") and self.modelo.get("vision_ok"):
                d["modelo"]["vision"] = {"meta_url": "/engine/api/meta-mmproj/" + urllib.parse.quote(self.modelo["id"]),
                                         "archivo_url": "/engine/file-mmproj/" + urllib.parse.quote(self.modelo["id"]),
                                         "bytes": os.path.getsize(self.modelo["mmproj"])}
        return d


class ServidorHTTP(ThreadingHTTPServer):
    """ThreadingHTTPServer that logs one line, not a traceback, when a client cuts the connection."""

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionError, TimeoutError)):
            log.info("Connection cut by the client %s (%s)", client_address[0], type(exc).__name__)
            return
        super().handle_error(request, client_address)


class Puente:
    def __init__(self, reg):
        self.reg = reg
        self.cv = threading.Condition()
        self.pendientes = deque()
        self.activos = {}            # id -> Job handed to the engine
        self.motor = {"visto": 0, "info": {}, "modelo_cargado": None, "estado": "desconectado", "id": None}
        self.stats = {"peticiones": 0, "tokens_generados": 0, "errores": 0}
        self.historial = deque(maxlen=30)

    def motor_conectado(self):
        return time.time() - self.motor["visto"] < 35 or bool(self.activos)

    def encolar(self, t):
        with self.cv:
            self.pendientes.append(t)
            self.cv.notify_all()
        return t

    def cancelar(self, t):
        t.cancelado = True
        with self.cv:
            if t in self.pendientes:
                self.pendientes.remove(t)
                t.eventos.put({"tipo": "error", "mensaje": "cancelado"})

    def hola(self, motor_id, info):
        """Registers an engine. If it is another window (or the page was reloaded), it takes control."""
        with self.cv:
            if self.motor.get("id") and self.motor["id"] != motor_id:
                for tid, t in list(self.activos.items()):
                    t.eventos.put({"tipo": "error", "mensaje": "the engine restarted during the request"})
                    del self.activos[tid]
                log.info("A new engine window took over")
            self.motor.update(id=motor_id, info=info, estado="listo", visto=time.time(), modelo_cargado=None)
            self.cv.notify_all()
        # An engine window left open from a previous version keeps the old JS in memory
        # and does not know how to reload: we open a new window (it takes control and the old one becomes inactive).
        if info.get("version") != VERSION and not info.get("recarga") and ESTADO.get("auto_relanzar"):
            log.warning("The engine window is from another version (%s, server %s): opening a new one.",
                        info.get("version"), VERSION)
            threading.Thread(target=abrir_motor, args=(ESTADO["cfg"],), daemon=True).start()

    def es_activo(self, motor_id):
        return not motor_id or not self.motor.get("id") or motor_id == self.motor["id"]

    def siguiente(self, espera=20.0, motor_id=None):
        """Engine long-poll: hands out the next job or None. Returns 'cerrar' if another window is the active one."""
        if not self.es_activo(motor_id):
            return "cerrar"
        self.motor["visto"] = time.time()
        with self.cv:
            # If THIS engine asks for work again while one of its own is active, that job was lost.
            for tid, t in list(self.activos.items()):
                if t.motor_id == motor_id:
                    t.eventos.put({"tipo": "error", "mensaje": "the engine restarted during the request"})
                    del self.activos[tid]
            fin = time.time() + espera
            while True:
                if not self.es_activo(motor_id):
                    return "cerrar"
                if self.pendientes:
                    break
                resto = fin - time.time()
                if resto <= 0:
                    return None
                self.cv.wait(resto)
                self.motor["visto"] = time.time()
            t = self.pendientes.popleft()
            t.inicio = time.time()
            t.fase = "sent to engine"
            t.motor_id = motor_id
            self.activos[t.id] = t
            return t

    def evento(self, d):
        if not self.es_activo(d.get("motor")):
            return {"cancelar": True, "cerrar": True}
        self.motor["visto"] = time.time()
        tipo = d.get("tipo")
        if tipo == "estado":
            for k in ("modelo_cargado", "estado", "info"):
                if k in d:
                    self.motor[k] = d[k]
            return {"ok": True}
        if tipo == "log":
            log.info("[engine] %s", d.get("mensaje", ""))
            return {"ok": True}
        t = self.activos.get(d.get("id"))
        if not t:
            return {"cancelar": True}
        t.ultimo_evento = time.time()
        if tipo == "progreso":
            t.fase = d.get("fase", t.fase)
        elif tipo == "token":
            t.tokens += int(d.get("n", 1))
            t.fase = "generating"
        if tipo in ("fin", "error"):
            t.terminado = True
            with self.cv:
                self.activos.pop(t.id, None)
            if tipo == "fin":
                self.stats["tokens_generados"] += int(d.get("completion_tokens") or 0)
        t.eventos.put(d)
        return {"cancelar": t.cancelado}


# =====================================================================
#  HTTP server
# =====================================================================
ESTADO = {}


def ips_locales():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(i for i in ips if not i.startswith("127.") and not i.startswith("169.254."))


def nombre_pc():
    try:
        return socket.gethostname().lower()
    except OSError:
        return ""


def es_loopback(host):
    return host in ("localhost", "::1") or host.startswith("127.")


_HOSTS = {"t": 0.0, "nombres": set()}


def hosts_locales(forzar=False):
    """Names and IPs by which this PC can be reached (to validate the Host header)."""
    if forzar or time.time() - _HOSTS["t"] > 60:
        nombres = {"localhost", "127.0.0.1", "::1"}
        for n in (nombre_pc(),):
            if n:
                nombres.add(n)
        try:
            fq = socket.getfqdn().lower()
            if fq:
                nombres.add(fq)
        except OSError:
            pass
        nombres.update(ips_locales())
        _HOSTS.update(t=time.time(), nombres=nombres)
    return _HOSTS["nombres"]


def nombre_de_host(valor):
    """'Host: example:1234' / '[::1]:1234' -> 'example' / '::1' (lowercase, no trailing dot)."""
    v = (valor or "").strip().lower()
    if v.startswith("["):
        v = v[1:v.find("]")] if "]" in v else v[1:]
    elif v.count(":") == 1:
        v = v.split(":", 1)[0]
    return v.rstrip(".")


def host_permitido(valor, cfg):
    """Protects against DNS rebinding: the Host header must be a name or IP of this PC (or be in allowed_hosts)."""
    if not valor:
        return True                      # HTTP/1.0 clients without Host
    extra = [str(x).lower() for x in (cfg.get("allowed_hosts") or [])]
    if "*" in extra:
        return True
    h = nombre_de_host(valor)
    if h in extra or es_loopback(h) or h in hosts_locales():
        return True
    return h in hosts_locales(forzar=True)  # the IP may have changed (DHCP)


def _id_respuesta(prefijo):
    return prefijo + secrets.token_hex(12)


def clave_api(cfg):
    """Current API key: the LLM_API_KEY environment variable takes priority over config.json."""
    return (os.environ.get("LLM_API_KEY") or cfg.get("api_key") or "").strip()


class ErrorAPI(Exception):
    def __init__(self, codigo, mensaje, tipo="invalid_request_error", cabeceras=None):
        super().__init__(mensaje)
        self.codigo, self.mensaje, self.tipo, self.cabeceras = codigo, mensaje, tipo, cabeceras or {}


class Manejador(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ServidorLLMLocal/" + VERSION

    def log_message(self, fmt, *args):
        pass

    # ------------------------------------------------------------ utilities
    def _local(self):
        return self.client_address[0] in ("127.0.0.1", "::1", "::ffff:127.0.0.1")

    def _ruta_publica(self):
        """Routes of the OpenAI / LM Studio compatible API: the only ones with open CORS."""
        ruta = urllib.parse.urlparse(self.path).path
        return ruta.startswith("/v1/") or ruta.startswith("/api/v0/")

    def _origen_ajeno(self):
        """True if a browser sends the request from another web page (another origin)."""
        origen = self.headers.get("Origin")
        if origen is not None:
            if origen == "null":
                return True
            red = urllib.parse.urlparse(origen).netloc.lower()
            return red != (self.headers.get("Host") or "").strip().lower()
        return self.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site")

    def _filtro_seguridad(self):
        """Returns (code, message) if the request is rejected, or None.

        - Host must be this PC (DNS rebinding).
        - Outside /v1 and /api/v0, a page from another site cannot call the panel, the configuration or the engine
          (CSRF): only *navigating* to the panel or the engine (opening them in a window) is allowed."""
        if not host_permitido(self.headers.get("Host"), ESTADO["cfg"]):
            return 403, ("Host not allowed: %s. If you use a proxy or a custom name, add it to 'allowed_hosts' in config.json."
                         % self.headers.get("Host"))
        if self._ruta_publica():
            return None
        ruta = urllib.parse.urlparse(self.path).path.rstrip("/") or "/"
        if (self.command in ("GET", "HEAD") and ruta in ("/", "/panel", "/engine")
                and self.headers.get("Sec-Fetch-Mode") == "navigate"):
            return None
        if self._origen_ajeno():
            return 403, "Request rejected: it comes from another web page (origin %s)." % (
                self.headers.get("Origin") or self.headers.get("Sec-Fetch-Site"))
        return None

    def _cabeceras_cors(self):
        if not self._ruta_publica():
            return
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, codigo, datos, cabeceras=None):
        cuerpo = datos if isinstance(datos, bytes) else json.dumps(datos, ensure_ascii=False).encode("utf-8")
        self.send_response(codigo)
        for k, v in (cabeceras or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self._cabeceras_cors()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(cuerpo)

    def _error(self, codigo, mensaje, tipo="invalid_request_error", cabeceras=None):
        self._json(codigo, {"error": {"message": mensaje, "type": tipo, "code": codigo}}, cabeceras)

    def _autorizar(self):
        """Requires the API key (if one is configured) on the routes that use the models."""
        clave = clave_api(ESTADO["cfg"])
        if not clave:
            return
        enviada = self.headers.get("Authorization", "")
        if enviada.lower().startswith("bearer "):
            enviada = enviada[7:]
        if not secrets.compare_digest(enviada.strip().encode("utf-8"), clave.encode("utf-8")):
            raise ErrorAPI(401, "API key missing or incorrect. Send the header 'Authorization: Bearer <key>'.",
                           "invalid_api_key", {"WWW-Authenticate": "Bearer"})

    def _comprobar_cola(self):
        limite = int(ESTADO["cfg"].get("max_queue") or 0)
        if limite and len(ESTADO["puente"].pendientes) >= limite:
            raise ErrorAPI(429, "The queue is full (%d requests waiting). Retry in a few seconds." % limite,
                           "rate_limit_exceeded", {"Retry-After": "10"})

    def _cuerpo(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        datos = self.rfile.read(n)
        try:
            return json.loads(datos.decode("utf-8"))
        except ValueError:
            raise ErrorAPI(400, "The body is not valid JSON")

    def _archivo_estatico(self, ruta):
        tipos = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".json": "application/json", ".wgsl": "text/plain"}
        ruta = os.path.normpath(os.path.abspath(ruta))
        try:
            dentro = os.path.commonpath([ruta, WEB]) == WEB
        except ValueError:          # another drive on Windows
            dentro = False
        if not dentro or not os.path.isfile(ruta):
            return self._error(404, "Not found")
        with open(ruta, "rb") as f:
            cuerpo = f.read()
        self.send_response(200)
        self.send_header("Content-Type", tipos.get(os.path.splitext(ruta)[1], "application/octet-stream"))
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        if ruta.endswith(".html"):
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(cuerpo)

    def _servir_gguf(self, m, ruta=None):
        ruta = ruta or m["archivo"]
        tam = os.path.getsize(ruta)
        ini, fin = 0, tam - 1
        rango = self.headers.get("Range")
        if rango:
            mm = re.match(r"bytes=(\d*)-(\d*)", rango)
            if not mm:
                return self._error(416, "Invalid range")
            if mm.group(1):
                ini = int(mm.group(1))
                if mm.group(2):
                    fin = min(int(mm.group(2)), tam - 1)
            else:
                ini = max(0, tam - int(mm.group(2)))
            if ini > fin:
                return self._error(416, "Invalid range")
        n = fin - ini + 1
        self.send_response(206 if rango else 200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(n))
        if rango:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (ini, fin, tam))
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(ruta, "rb") as f:
            f.seek(ini)
            while n > 0:
                trozo = f.read(min(n, 4 << 20))
                if not trozo:
                    break
                self.wfile.write(trozo)
                n -= len(trozo)

    # ------------------------------------------------------------ routes
    def _rechazar_si_hace_falta(self):
        r = self._filtro_seguridad()
        if r:
            log.warning("Rejected %s %s from %s: %s", self.command, urllib.parse.urlparse(self.path).path,
                        self.client_address[0], r[1])
            self._error(r[0], r[1], "forbidden")
            return True
        return False

    def do_OPTIONS(self):
        if self._rechazar_si_hace_falta():
            return
        self.send_response(204)
        self._cabeceras_cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if self._rechazar_si_hace_falta():
            return
        try:
            self._get()
        except ErrorAPI as e:
            self._error(e.codigo, e.mensaje, e.tipo, e.cabeceras)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            log.exception("Error in GET %s", self.path)
            try:
                self._error(500, "Internal error: %s" % e, "server_error")
            except Exception:
                pass

    def do_POST(self):
        if self._rechazar_si_hace_falta():
            return
        try:
            self._post()
        except ErrorAPI as e:
            self._error(e.codigo, e.mensaje, e.tipo, e.cabeceras)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception as e:
            log.exception("Error in POST %s", self.path)
            ESTADO["puente"].stats["errores"] += 1
            try:
                self._error(500, "Internal error: %s" % e, "server_error")
            except Exception:
                pass

    def _get(self):
        u = urllib.parse.urlparse(self.path)
        ruta = u.path.rstrip("/") or "/"
        reg, puente = ESTADO["registro"], ESTADO["puente"]
        if ruta.startswith("/v1/") or ruta.startswith("/api/v0/"):
            self._autorizar()
        if ruta == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if ruta in ("/", "/panel"):
            return self._archivo_estatico(os.path.join(WEB, "panel.html"))
        if ruta == "/engine":
            return self._archivo_estatico(os.path.join(WEB, "engine.html"))
        if ruta.startswith("/web/"):
            return self._archivo_estatico(os.path.join(WEB, *ruta[5:].split("/")))
        if ruta == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": m["id"], "object": "model", "owned_by": m["publisher"] or "local"}
                for m in reg.modelos]})
        if ruta == "/api/v0/models":
            return self._json(200, {"object": "list", "data": [
                {"id": m["id"], "object": "model",
                 "type": "embeddings" if m["embedding"] else ("vlm" if m["vision"] else "llm"),
                 "publisher": m["publisher"], "arch": m["arch"], "compatibility_type": "gguf",
                 "quantization": m["cuantizacion"],
                 "state": "loaded" if puente.motor.get("modelo_cargado") == m["id"] else "not-loaded",
                 "max_context_length": reg.ctx_para(m)} for m in reg.modelos]})
        if ruta == "/api/status":
            return self._json(200, self._estado())
        if ruta == "/api/resolve":
            q = urllib.parse.parse_qs(u.query).get("model", [""])[0]
            m = reg.resolver(q)
            return self._json(200, {"requested": q, "model": reg.publico(m) if m else None})
        if ruta == "/api/logs":
            return self._json(200, list(LOG_MEMORIA))
        if ruta == "/api/architectures":
            return self._json(200, sorted(MANIFIESTOS.values(), key=lambda d: d["id"]))
        if ruta == "/api/config":
            return self._json(200, self._config_publica())
        if ruta == "/api/folders":
            if not self._local():
                return self._error(403, "Browsing is only possible from the server PC")
            return self._json(200, listar_carpetas(urllib.parse.parse_qs(u.query).get("path", [""])[0]))
        # ---- engine routes (from this PC only)
        if ruta.startswith("/engine/"):
            if not self._local():
                return self._error(403, "Only accessible from the server PC")
            if ruta == "/engine/api/next":
                mid = urllib.parse.parse_qs(u.query).get("motor", [None])[0]
                t = puente.siguiente(motor_id=mid)
                if t == "cerrar":
                    return self._json(200, {"accion": "cerrar"})
                return self._json(200, t.para_motor(reg) if t else {"accion": "nada"})
            if ruta.startswith("/engine/api/meta/"):
                m = reg.por_id(urllib.parse.unquote(ruta[len("/engine/api/meta/"):]))
                if not m:
                    return self._error(404, "Model not found")
                return self._json(200, reg.meta_json(m))
            if ruta.startswith("/engine/api/meta-mmproj/"):
                m = reg.por_id(urllib.parse.unquote(ruta[len("/engine/api/meta-mmproj/"):]))
                cuerpo = reg.meta_mmproj_json(m) if m else None
                if cuerpo is None:
                    return self._error(404, "Model or vision projector not found")
                return self._json(200, cuerpo)
            if ruta.startswith("/engine/file-mmproj/"):
                m = reg.por_id(urllib.parse.unquote(ruta[len("/engine/file-mmproj/"):]))
                if not m or not m.get("mmproj"):
                    return self._error(404, "Vision projector not found")
                return self._servir_gguf(m, m["mmproj"])
            if ruta.startswith("/engine/file/"):
                m = reg.por_id(urllib.parse.unquote(ruta[len("/engine/file/"):]))
                if not m:
                    return self._error(404, "Model not found")
                return self._servir_gguf(m)
        return self._error(404, "Route not found: " + ruta)

    def _post(self):
        ruta = urllib.parse.urlparse(self.path).path.rstrip("/")
        puente, reg = ESTADO["puente"], ESTADO["registro"]
        if ruta.startswith("/v1/") or ruta in ("/api/load", "/api/unload", "/api/rescan"):
            self._autorizar()
        if ruta.startswith("/engine/api/"):
            if not self._local():
                return self._error(403, "Only accessible from the server PC")
            d = self._cuerpo()
            if ruta == "/engine/api/event":
                return self._json(200, puente.evento(d))
            if ruta == "/engine/api/hello":
                puente.hola(d.get("motor"), d)
                log.info("Engine connected: %s", d.get("gpu", {}).get("summary") or d.get("gpu"))
                return self._json(200, {"ok": True, "version": VERSION})
            return self._error(404, "Route not found")
        if ruta in ("/v1/chat/completions", "/v1/completions"):
            return self._generar(ruta.endswith("chat/completions"))
        if ruta == "/v1/embeddings":
            return self._embeddings()
        if ruta == "/api/config":
            return self._guardar_config()
        if ruta == "/api/rescan":
            reg.escanear()
            return self._json(200, {"models": [reg.publico(m) for m in reg.modelos]})
        if ruta in ("/api/load", "/api/unload"):
            d = self._cuerpo()
            m = None
            if ruta == "/api/load":
                m = reg.resolver(d.get("model", ""))
                if not m:
                    raise ErrorAPI(404, "Model not found: %s" % d.get("model"))
                if not m["soportado"]:
                    raise ErrorAPI(400, "Architecture %s is not supported yet" % m["arch"])
                self._comprobar_memoria(m)
            if not puente.motor_conectado() and not ESTADO.get("auto_relanzar"):
                raise ErrorAPI(503, "The engine is not connected", "service_unavailable")
            self._comprobar_cola()
            accion = {"/api/load": "cargar", "/api/unload": "descargar"}[ruta]   # protocol names used with the engine
            puente.encolar(Trabajo(accion, m, origen=self.client_address[0]))
            return self._json(202, {"ok": True})
        return self._error(404, "Route not found: " + ruta)

    def _comprobar_memoria(self, m):
        """Rejects with a clear message a model that almost certainly does not fit (so the PC does not hang)."""
        cfg, reg, puente = ESTADO["cfg"], ESTADO["registro"], ESTADO["puente"]
        if not cfg.get("memory_check", True) or puente.motor.get("modelo_cargado") == m["id"]:
            return
        e = reg.estimar(m)
        if e["state"] == "no_fit" and e["ratio"] > UMBRAL_BLOQUEO:
            raise ErrorAPI(400, "Model '%s' needs ~%.1f GB and the estimated memory for models on this PC is %.1f GB. "
                                "Use a smaller model, or turn off the memory check / set “Memory for models” in the "
                                "panel settings." % (m["id"], e["necesita_bytes"] / 1e9, e["presupuesto_bytes"] / 1e9),
                           "model_too_large")

    # ------------------------------------------------------------ configuration
    def _config_publica(self):
        cfg = ESTADO["cfg"]
        return {"models_dir": cfg["models_dir"], "default_ctx": cfg["default_ctx"], "kv_max_mb": cfg["kv_max_mb"],
                "timeout_seconds": cfg["timeout_seconds"], "port": cfg["port"],
                "open_engine": bool(cfg.get("open_engine", True)),
                "memory_check": bool(cfg.get("memory_check", True)), "gpu_memory_gb": cfg.get("gpu_memory_gb", 0),
                "max_queue": cfg.get("max_queue", 64), "api_key_set": bool(clave_api(cfg)),
                "api_key_from_env": bool(os.environ.get("LLM_API_KEY")),
                "default_sampling": {k: cfg["default_sampling"].get(k) for k in MUESTREO_PANEL},
                "editable": self._local()}

    def _guardar_config(self):
        if not self._local():
            raise ErrorAPI(403, "Settings can only be changed from the server PC (%s)." % socket.gethostname())
        cfg, reg, puente = ESTADO["cfg"], ESTADO["registro"], ESTADO["puente"]
        cambios, errores = validar_ajustes(self._cuerpo())
        if errores:
            raise ErrorAPI(400, "; ".join(errores))
        puerto_nuevo = "port" in cambios and cambios["port"] != cfg["port"]
        carpeta_nueva = "models_dir" in cambios and os.path.normcase(cambios["models_dir"]) != os.path.normcase(os.path.abspath(cfg["models_dir"]))
        try:
            guardar_config(ESTADO["config_ruta"], cambios)
        except OSError as e:
            raise ErrorAPI(500, "Could not write config.json: %s" % e, "server_error")
        for k, v in cambios.items():
            if k == "port":
                continue          # the port only changes on restart
            if isinstance(v, dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
        if carpeta_nueva:
            if puente.motor_conectado() and puente.motor.get("modelo_cargado"):
                puente.encolar(Trabajo("descargar", None, origen="panel"))   # the loaded model no longer belongs to the folder
            reg.escanear()
        log.info("Settings changed from the panel: %s", ", ".join(sorted(cambios)))
        return self._json(200, {"ok": True, "models": len(reg.modelos), "restart_required": puerto_nuevo,
                                "config": self._config_publica()})

    # ------------------------------------------------------------ state
    def _direcciones(self):
        cfg = ESTADO["cfg"]
        if es_loopback(str(cfg.get("host", ""))):
            return ["http://localhost:%d" % cfg["port"]]
        return ((["http://%s:%d" % (nombre_pc(), cfg["port"])] if nombre_pc() else []) +
                ["http://%s:%d" % (ip, cfg["port"]) for ip in ips_locales()])

    def _estado(self):
        reg, puente, cfg = ESTADO["registro"], ESTADO["puente"], ESTADO["cfg"]
        activo = next(iter(puente.activos.values()), None)
        return {
            "version": VERSION, "port": cfg["port"],
            "requires_key": bool(clave_api(cfg)), "pc_name": nombre_pc(), "from_server_pc": self._local(),
            "local_only": es_loopback(str(cfg.get("host", ""))),
            "reachable_at": self._direcciones(),
            "engine": {"connected": puente.motor_conectado(), "state": puente.motor.get("estado"),
                      "info": puente.motor.get("info"), "loaded_model": puente.motor.get("modelo_cargado"),
                      "seconds_since_contact": round(time.time() - puente.motor["visto"], 1)
                      if puente.motor["visto"] else None},
            "current_job": {"id": activo.id, "model": activo.modelo["id"] if activo.modelo else None,
                               "phase": activo.fase, "tokens": activo.tokens,
                               "seconds": round(time.time() - (activo.inicio or time.time()), 1)} if activo else None,
            "queued": len(puente.pendientes),
            "stats": {"requests": puente.stats["peticiones"], "tokens_generated": puente.stats["tokens_generados"],
                      "errors": puente.stats["errores"]}, "history": list(puente.historial),
            "models": [reg.publico(m) for m in reg.modelos],
            "models_dir": cfg["models_dir"] if self._local() else os.path.basename(os.path.normpath(cfg["models_dir"])),
            "hardware": self._hardware_publico(),
        }

    def _hardware_publico(self):
        hw = ESTADO["registro"]._hw()
        gb = lambda x: round(x / 1e9, 1) if x else None
        return {"ram_gb": gb(hw["ram_bytes"]), "vram_gb": gb(hw["vram_bytes"]), "vram_name": hw["vram_nombre"],
                "gpu_type": hw["tipo_gpu"], "budget_gb": gb(hw["presupuesto_bytes"]), "source": hw["origen"],
                "warnings": hw["avisos"], "memory_check": bool(ESTADO["cfg"].get("memory_check", True))}

    # ------------------------------------------------------------ generation
    def _generar(self, es_chat):
        cfg, reg, puente = ESTADO["cfg"], ESTADO["registro"], ESTADO["puente"]
        d = self._cuerpo()
        pedido = d.get("model") or ""
        m = reg.resolver(pedido) if pedido else (reg.por_id(puente.motor.get("modelo_cargado") or "") or
                                                 next((x for x in reg.modelos if x["soportado"]), None))
        if not m:
            raise ErrorAPI(404, "Model not found: '%s'. Check /v1/models." % pedido, "model_not_found")
        if m["embedding"]:
            raise ErrorAPI(400, "'%s' is an embeddings model, not a chat model." % m["id"])
        if not m["soportado"]:
            raise ErrorAPI(400, "Model '%s' (architecture %s) is not supported by this version of the "
                                "engine yet. Supported: %s." % (m["id"], m["arch"], ", ".join(sorted(ARQUITECTURAS_SOPORTADAS))),
                           "model_not_supported")
        self._comprobar_memoria(m)
        rf = d.get("response_format")
        if isinstance(rf, dict) and rf.get("type") not in (None, "text", "json_schema"):
            # Same as LM Studio: this way client code uses its known fallback.
            raise ErrorAPI(400, "'response_format.type' must be 'json_schema' or 'text'")
        if es_chat:
            mensajes = d.get("messages")
            if not isinstance(mensajes, list) or not mensajes:
                raise ErrorAPI(400, "'messages' must be a non-empty list")
            imagenes = []
            if any(texto_de_contenido(x.get("content"))[1] for x in mensajes if isinstance(x, dict)):
                if not m.get("vision_ok"):
                    raise ErrorAPI(400, "Model '%s' has no vision in this version: %s" % (
                        m["id"], "there is no mmproj file in its folder." if not m.get("mmproj")
                        else "its projector (mmproj) is not a supported type (supported: LFM2-VL and Gemma 3)."),
                        "vision_not_supported")
                mensajes = [dict(x, content=contenido_con_imagenes(x.get("content"), imagenes)) if isinstance(x, dict) else x
                            for x in mensajes]
            try:
                prompt, incluye_bos = plantilla_chat(m["arch"], mensajes, reg.plantilla(m))
            except Exception as e:
                if templates is not None and isinstance(e, templates.ErrorPlantilla):
                    raise ErrorAPI(400, "The model's chat template rejected the messages: %s" % e)
                raise
        else:
            imagenes = []
            incluye_bos = False
            prompt = d.get("prompt", "")
            if isinstance(prompt, list):
                prompt = "".join(str(x) for x in prompt)
            if not isinstance(prompt, str):
                raise ErrorAPI(400, "'prompt' must be text")
        if prompt.count(MARCADOR_IMAGEN) != len(imagenes):
            raise ErrorAPI(400, "The model's chat template did not keep all the images in the message "
                                "(%d images, %d markers)." % (len(imagenes), prompt.count(MARCADOR_IMAGEN)))
        stops = d.get("stop") or []
        if isinstance(stops, str):
            stops = [stops]
        defecto = cfg["default_sampling"]
        params = {}
        for k in ("temperature", "top_k", "top_p", "min_p", "repeat_penalty"):
            v = d.get(k, defecto.get(k))
            params[k] = float(v) if v is not None else None
        if d.get("frequency_penalty") or d.get("presence_penalty"):
            params["frequency_penalty"] = float(d.get("frequency_penalty") or 0)
            params["presence_penalty"] = float(d.get("presence_penalty") or 0)
        mt = d.get("max_tokens", d.get("max_completion_tokens"))
        params["max_tokens"] = int(mt) if mt is not None and int(mt) > 0 else -1
        params["seed"] = int(d["seed"]) if d.get("seed") is not None else None
        params["top_k"] = int(params["top_k"]) if params.get("top_k") is not None else 0
        params["add_bos"] = False if incluye_bos else True
        if imagenes:
            params["imagenes"] = imagenes
        # With auto-relaunch the request waits: the engine watchdog reopens the window (see vigilar_motor and _eventos).
        if not puente.motor_conectado() and not ESTADO.get("auto_relanzar"):
            raise ErrorAPI(503, "The engine is not connected. On the server PC open http://localhost:%d/engine "
                                "(or restart server.py)." % cfg["port"], "service_unavailable")
        stream = bool(d.get("stream"))
        t = Trabajo("generar", m, prompt, params, origen=self.client_address[0])
        self._comprobar_cola()
        puente.stats["peticiones"] += 1
        puente.encolar(t)
        log.info("Request %s from %s -> %s (%s, %d prompt characters%s)", t.id, self.client_address[0], m["id"],
                 "chat" if es_chat else "completion", len(prompt), ", stream" if stream else "")
        modelo_resp = pedido or m["id"]
        razon = SeparadorRazonamiento(activo=es_chat)
        filtro = FiltroStop(stops)
        try:
            if stream:
                self._responder_stream(t, es_chat, modelo_resp, razon, filtro)
            else:
                self._responder_normal(t, es_chat, modelo_resp, razon, filtro)
        finally:
            if not t.terminado:
                puente.cancelar(t)

    def _embeddings(self):
        """POST /v1/embeddings (OpenAI / LM Studio format). One normalized vector for each text in 'input'."""
        cfg, reg, puente = ESTADO["cfg"], ESTADO["registro"], ESTADO["puente"]
        d = self._cuerpo()
        pedido = d.get("model") or ""
        m = reg.resolver(pedido) if pedido else next((x for x in reg.modelos if x["embedding"] and x["soportado"]), None)
        if not m:
            raise ErrorAPI(404, "Model not found: '%s'. Check /v1/models." % pedido, "model_not_found")
        if not m["embedding"]:
            raise ErrorAPI(400, "'%s' is not an embeddings model." % m["id"])
        if not m["soportado"]:
            raise ErrorAPI(400, "Model '%s' (architecture %s) is not supported by this version of the engine yet."
                           % (m["id"], m["arch"]), "model_not_supported")
        self._comprobar_memoria(m)
        entrada = d.get("input")
        if isinstance(entrada, str):
            entrada = [entrada]
        if not isinstance(entrada, list) or not entrada or not all(isinstance(x, str) for x in entrada):
            raise ErrorAPI(400, "'input' must be a text or a non-empty list of texts")
        formato = d.get("encoding_format") or "float"
        if formato not in ("float", "base64"):
            raise ErrorAPI(400, "'encoding_format' must be 'float' or 'base64'")
        # With auto-relaunch the request waits: the engine watchdog reopens the window (see vigilar_motor and _eventos).
        if not puente.motor_conectado() and not ESTADO.get("auto_relanzar"):
            raise ErrorAPI(503, "The engine is not connected. On the server PC open http://localhost:%d/engine "
                                "(or restart server.py)." % cfg["port"], "service_unavailable")
        t = Trabajo("embeber", m, "", {"entradas": entrada}, origen=self.client_address[0])
        self._comprobar_cola()
        puente.stats["peticiones"] += 1
        puente.encolar(t)
        log.info("Request %s from %s -> %s (embeddings: %d texts)", t.id, self.client_address[0], m["id"], len(entrada))
        fin = {}
        try:
            for ev in self._eventos(t):
                if ev.get("tipo") == "fin":
                    fin = ev
                elif ev.get("tipo") == "error":
                    raise ErrorAPI(ev.get("codigo", 500), "Engine error: " + ev.get("mensaje", "desconocido"), "engine_error")
        finally:
            if not t.terminado:
                puente.cancelar(t)
        vectores = fin.get("embeddings") or []
        if len(vectores) != len(entrada):
            raise ErrorAPI(500, "The engine returned %d vectors for %d texts" % (len(vectores), len(entrada)), "engine_error")
        datos = []
        for i, v in enumerate(vectores):
            if formato == "base64":
                v = base64.b64encode(struct.pack("<%df" % len(v), *v)).decode("ascii")
            datos.append({"object": "embedding", "embedding": v, "index": i})
        p = int(fin.get("prompt_tokens") or 0)
        self._registrar(t, {"prompt_tokens": p, "completion_tokens": 0, "total_tokens": p}, fin)
        return self._json(200, {"object": "list", "data": datos, "model": pedido or m["id"],
                                "usage": {"prompt_tokens": p, "total_tokens": p}})

    def _eventos(self, t):
        """Event generator for the job, with timeout control."""
        cfg, puente = ESTADO["cfg"], ESTADO["puente"]
        limite = time.time() + float(cfg.get("timeout_seconds", 1800))
        sin_motor_desde = None     # when we first saw the engine disconnected while this request had not started
        while True:
            try:
                ev = t.eventos.get(timeout=2.0)
            except queue.Empty:
                ahora = time.time()
                if ahora > limite:
                    raise ErrorAPI(504, "The maximum wait time was exceeded", "timeout")
                if t.inicio and ahora - t.ultimo_evento > 300:
                    raise ErrorAPI(504, "The engine stopped responding", "timeout")
                if not t.inicio and not puente.motor_conectado():
                    # With auto-relaunch, give the watchdog time to reopen the window before giving up.
                    sin_motor_desde = sin_motor_desde or ahora
                    gracia = ESPERA_REAPERTURA if ESTADO.get("auto_relanzar") else 0
                    if ahora - sin_motor_desde > gracia:
                        raise ErrorAPI(503, "The engine disconnected while the request was waiting", "service_unavailable")
                else:
                    sin_motor_desde = None
                continue
            yield ev
            if ev.get("tipo") in ("fin", "error"):
                return

    def _uso(self, fin, prompt_tokens):
        c = int(fin.get("completion_tokens") or 0)
        p = int(fin.get("prompt_tokens") or prompt_tokens or 0)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}

    def _responder_normal(self, t, es_chat, modelo_resp, razon, filtro):
        contenido, razonamiento, fin, prompt_tokens = [], [], {}, 0
        for ev in self._eventos(t):
            tipo = ev.get("tipo")
            if tipo == "inicio":
                prompt_tokens = ev.get("prompt_tokens", 0)
            elif tipo == "token":
                self._acumular(ev.get("texto", ""), es_chat, razon, filtro, contenido, razonamiento, t)
            elif tipo == "fin":
                fin = ev
            elif tipo == "error":
                raise ErrorAPI(ev.get("codigo", 500), "Engine error: " + ev.get("mensaje", "desconocido"),
                               "engine_error")
        self._acumular(None, es_chat, razon, filtro, contenido, razonamiento, t)
        motivo = "stop" if filtro.detenido else fin.get("razon", "stop")
        uso = self._uso(fin, prompt_tokens)
        self._registrar(t, uso, fin)
        texto = "".join(contenido)
        if es_chat:
            msg = {"role": "assistant", "content": texto}
            if razonamiento:
                msg["reasoning_content"] = "".join(razonamiento)
            elec = {"index": 0, "message": msg, "logprobs": None, "finish_reason": motivo}
            obj, pre = "chat.completion", "chatcmpl-"
        else:
            elec = {"index": 0, "text": texto, "logprobs": None, "finish_reason": motivo}
            obj, pre = "text_completion", "cmpl-"
        self._json(200, {"id": _id_respuesta(pre), "object": obj, "created": int(time.time()),
                         "model": modelo_resp, "choices": [elec], "usage": uso,
                         "stats": {"tokens_per_second": fin.get("tps"), "time_to_first_token": fin.get("ttft"),
                                   "prompt_tokens_per_second": fin.get("tps_prompt"), "diag": fin.get("diag"),
                                   "espera_cola_s": round(t.inicio - t.creado, 2) if t.inicio else None},
                         "system_fingerprint": t.modelo["id"]})

    def _acumular(self, texto, es_chat, razon, filtro, contenido, razonamiento, t):
        """Runs the text through the reasoning separator and the stop filter."""
        piezas = []
        if es_chat:
            partes = razon.alimentar(texto) if texto is not None else razon.vaciar()
        else:
            partes = [("contenido", texto)] if texto is not None else []
        for tipo, s in partes:
            if tipo == "razon":
                razonamiento.append(s)
                piezas.append(("razon", s))
            else:
                ok = filtro.alimentar(s)
                if ok:
                    contenido.append(ok)
                    piezas.append(("contenido", ok))
        if texto is None:
            resto = filtro.vaciar()
            if resto:
                contenido.append(resto)
                piezas.append(("contenido", resto))
        if filtro.detenido and not t.terminado:
            ESTADO["puente"].cancelar(t)
        return piezas

    def _registrar(self, t, uso, fin):
        ESTADO["puente"].historial.appendleft({
            "time": time.strftime("%H:%M:%S"), "model": t.modelo["id"], "source": t.origen,
            "prompt_tokens": uso["prompt_tokens"], "completion_tokens": uso["completion_tokens"],
            "tps": fin.get("tps"), "seconds": round(time.time() - t.creado, 1)})
        log.info("Request %s finished: %d+%d tokens, %s tok/s", t.id, uso["prompt_tokens"],
                 uso["completion_tokens"], fin.get("tps"))

    def _responder_stream(self, t, es_chat, modelo_resp, razon, filtro):
        rid = _id_respuesta("chatcmpl-" if es_chat else "cmpl-")
        creado = int(time.time())
        obj = "chat.completion.chunk" if es_chat else "text_completion"
        cabeceras_enviadas = False
        primero = [True]

        def chunk(delta=None, texto=None, fin=None, uso=None):
            if es_chat:
                elec = {"index": 0, "delta": delta or {}, "logprobs": None, "finish_reason": fin}
            else:
                elec = {"index": 0, "text": texto or "", "logprobs": None, "finish_reason": fin}
            d = {"id": rid, "object": obj, "created": creado, "model": modelo_resp,
                 "system_fingerprint": t.modelo["id"], "choices": [elec]}
            if uso:
                d["usage"] = uso
            enviar("data: " + json.dumps(d, ensure_ascii=False) + "\n\n")

        def enviar(s):
            b = s.encode("utf-8")
            self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
            self.wfile.flush()

        def emitir(piezas):
            for tipo, s in piezas:
                if not s:
                    continue
                if es_chat:
                    delta = {"reasoning_content": s} if tipo == "razon" else {"content": s}
                    if primero[0]:
                        delta = dict(role="assistant", **delta)
                        primero[0] = False
                    chunk(delta=delta)
                else:
                    chunk(texto=s)

        fin, prompt_tokens = {}, 0
        try:
            for ev in self._eventos(t):
                tipo = ev.get("tipo")
                if tipo == "error" and not cabeceras_enviadas:
                    raise ErrorAPI(ev.get("codigo", 500), "Engine error: " + ev.get("mensaje", ""), "engine_error")
                if not cabeceras_enviadas and tipo in ("inicio", "token", "fin"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self._cabeceras_cors()
                    self.end_headers()
                    cabeceras_enviadas = True
                if tipo == "inicio":
                    prompt_tokens = ev.get("prompt_tokens", 0)
                elif tipo == "token":
                    emitir(self._acumular(ev.get("texto", ""), es_chat, razon, filtro, [], [], t))
                    if filtro.detenido:
                        break
                elif tipo == "fin":
                    fin = ev
                elif tipo == "error":
                    enviar("data: " + json.dumps({"error": {"message": ev.get("mensaje", "")}}) + "\n\n")
                    break
            if not cabeceras_enviadas:
                raise ErrorAPI(500, "The engine finished without responding", "engine_error")
            emitir(self._acumular(None, es_chat, razon, filtro, [], [], t))
            uso = self._uso(fin, prompt_tokens)
            motivo = "stop" if filtro.detenido else fin.get("razon", "stop")
            chunk(delta={}, texto="", fin=motivo, uso=uso)
            enviar("data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            self._registrar(t, uso, fin)
        except ErrorAPI as e:
            if cabeceras_enviadas:
                try:
                    enviar("data: " + json.dumps({"error": {"message": e.mensaje}}) + "\n\n")
                    self.wfile.write(b"0\r\n\r\n")
                except OSError:
                    pass
                self.close_connection = True
            else:
                raise
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            log.info("Request %s: the client closed the connection; cancelling.", t.id)
            self.close_connection = True


# =====================================================================
#  Startup
# =====================================================================
NAVEGADORES_LINUX = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "brave-browser", "microsoft-edge")


def buscar_navegador(cfg, plataforma=None, which=None):
    if cfg.get("browser") and os.path.exists(cfg["browser"]):
        return cfg["browser"]
    plataforma = sys.platform if plataforma is None else plataforma
    if plataforma.startswith("linux"):                       # Chromium-based browsers on PATH (not the default one, e.g. Firefox)
        which = shutil.which if which is None else which
        for nombre in NAVEGADORES_LINUX:
            ruta = which(nombre)
            if ruta:
                return ruta
        return None
    candidatos = []
    for var in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA"):
        base = os.environ.get(var)
        if base:
            candidatos.append(os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"))
            candidatos.append(os.path.join(base, "Google", "Chrome", "Application", "chrome.exe"))
    for c in candidatos:
        if os.path.exists(c):
            return c
    return None


def vigilar_motor(cfg, intervalo=5, sin_senal=20, reintento=90, abrir=None):
    """Engine watchdog. If requests are waiting and the engine window has been silent for more than `sin_senal`
    seconds, reopen it (at most once every `reintento` seconds). With nothing waiting it does nothing, so closing
    the window on purpose does not reopen it. Starts a daemon thread and returns its step function `paso()`
    (so it can be tested; `paso(ahora)` accepts an injected clock)."""
    estado = {"ultimo": 0.0}
    if abrir is None:
        def abrir():
            threading.Thread(target=abrir_motor, args=(cfg,), daemon=True).start()

    def paso(ahora=None):
        try:
            ahora = time.time() if ahora is None else ahora
            puente = ESTADO["puente"]
            if not puente.pendientes or puente.activos:       # nothing waiting, or the engine is busy with a job
                return False
            if ahora - puente.motor["visto"] <= sin_senal or ahora - estado["ultimo"] < reintento:
                return False
            estado["ultimo"] = ahora
            log.warning("Requests are waiting and the engine has been silent for %.0f s: reopening the engine window.",
                        ahora - puente.motor["visto"])
            abrir()
            return True
        except Exception:
            log.exception("Engine watchdog error")
            return False

    def bucle():
        while True:
            time.sleep(intervalo)
            paso()

    threading.Thread(target=bucle, daemon=True, name="engine-watchdog").start()
    return paso


def abrir_motor(cfg):
    url = "http://127.0.0.1:%d/engine" % cfg["port"]
    nav = buscar_navegador(cfg)
    if nav:
        perfil = os.path.join(DIR, ".engine_profile")
        args = [nav, "--app=" + url, "--user-data-dir=" + perfil, "--no-first-run",
                "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
                "--disable-backgrounding-occluded-windows", "--window-size=900,700"]
        try:
            subprocess.Popen(args, close_fds=True)
            log.info("Engine opened at %s", os.path.basename(nav))
            return
        except OSError as e:
            log.warning("Could not open %s: %s", nav, e)
    webbrowser.open(url)
    log.info("Engine opened in the default browser: %s", url)


def main():
    ap = argparse.ArgumentParser(description="quipullm: OpenAI-compatible LLM server")
    ap.add_argument("--config", default=os.path.join(DIR, "config.json"))
    ap.add_argument("--no-engine", action="store_true", help="do not open the browser with the engine")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None, help="127.0.0.1 = this PC only (default); 0.0.0.0 = whole network")
    ap.add_argument("--models-dir", default=None, help="models folder (overrides config.json)")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    configurar_logs()
    cfg = cargar_config(a.config)
    if a.port:
        cfg["port"] = a.port
    if a.host:
        cfg["host"] = a.host
    if a.models_dir:
        cfg["models_dir"] = a.models_dir
    reg = Registro(cfg)
    reg.escanear()
    ESTADO.update(cfg=cfg, registro=reg, puente=Puente(reg), config_ruta=a.config)
    try:
        srv = ServidorHTTP((cfg["host"], cfg["port"]), Manejador)
    except OSError as e:
        log.error("Could not open port %d (%s). Is LM Studio or another copy of the server still running?",
                  cfg["port"], e)
        sys.exit(1)
    srv.daemon_threads = True
    log.info("quipullm v%s listening on %s:%d", VERSION, cfg["host"], cfg["port"])
    threading.Thread(target=hosts_locales, daemon=True).start()   # resolves this PC's names in the background
    if es_loopback(str(cfg["host"])):
        log.info("  Only reachable from this PC. For the network: \"host\": \"0.0.0.0\" (or --host 0.0.0.0) and an api_key.")
    else:
        for ip in ips_locales():
            log.info("  Reachable at: http://%s:%d", ip, cfg["port"])
        if nombre_pc():
            log.info("  By name:      http://%s:%d/", nombre_pc(), cfg["port"])
        if not clave_api(cfg):
            log.warning("  The API is open to the whole network WITHOUT a key. Set api_key or LLM_API_KEY (see SECURITY.md).")
    log.info("  Panel:        http://localhost:%d/", cfg["port"])
    ESTADO["auto_relanzar"] = bool(cfg.get("open_engine", True) and not a.no_engine)
    if cfg.get("open_engine", True) and not a.no_engine:
        # Waits a few seconds: if the engine window was already open, it reconnects by itself.
        def quizas_abrir():
            if not ESTADO["puente"].motor_conectado():
                abrir_motor(cfg)
            else:
                log.info("The engine window was already open and reconnected")
        threading.Timer(5.0, quizas_abrir).start()
        vigilar_motor(cfg)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("Server stopped.")


if __name__ == "__main__":
    main()
