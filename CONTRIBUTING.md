# Contributing

Thanks for helping. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md). The most valuable contributions are, in order: **measurements on other GPUs/OSes**, **new architectures**
(see [docs/ARCH_GUIDE.md](docs/ARCH_GUIDE.md)), bug reports with the engine log, and documentation/translations.

## Ground rules
- Keep the project **text-only and dependency-free at runtime**: Python standard library on the server, plain JS/WGSL in the engine. The only
  binary files allowed are screenshots under `docs/img/` (`.png`, `.jpg`, `.gif`, at most 1.5 MB each).
  Development and test tools may use extra packages (see `tests/run_all.py`).
- Correctness is decided by tests against llama.cpp, not by how the output reads. A pull request that changes the engine must
  keep `python tests/run_all.py --engine` green (or explain why a case changes).
- New architecture PRs include: manifest/module, generator variant, passing conformance output including a mutation check, and one real-model measurement.
- No model files, no other binaries, no secrets, no personal or institutional names in commits or screenshots (the panel shows the PC name and folder paths: crop them).
- Interface, messages, config keys and HTTP API are English. Comments and docstrings are English; many internal identifiers (function and variable names) are still Spanish. English contributions are welcome and so are renames of existing identifiers (keep the tests green).

## Setup
```
python tests/run_all.py                 # API tests (stdlib only)
pip install -r tests/requirements-engine.txt && playwright install chromium   # pins llama-cpp-python 0.3.35, gguf 0.19.0, numpy 2.4.4
python tests/run_all.py --engine         # engine vs llama.cpp (slow the first time: generates synthetic models)
```

CI runs the API tests on Linux and Windows. The engine conformance suite can be run on demand in GitHub Actions
(*conformance* workflow) or locally as above; it uses SwiftShader, so it checks correctness, not speed.

## Reporting performance
Please include: GPU and driver, OS, browser version, model file and quantization, prompt length, `tools/bench.py` output.
