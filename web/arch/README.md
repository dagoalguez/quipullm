# Pluggable architectures

Each `.json` file in this folder declares an architecture (the GGUF `general.architecture` value) that the server and the
engine know how to run. To add one, drop its manifest here (and, if needed, a `.js` module) and press *Volver a escanear la
carpeta* (Rescan) in the panel. Nothing needs a restart.

See `docs/ARCH_GUIDE.md` for the full format, the conformance-test workflow and the known pitfalls.
