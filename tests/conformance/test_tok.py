"""JS tokenizer vs llama.cpp for each pre-tokenizer type (vocab_only)."""
import json, os, subprocess, sys, tempfile
import gguf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_modelos as G
import regex
import os
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root

G.CORPUS += """ Hello World! I'm testing CamelCaseWords and snake_case_words, HTTPServer2Go 12345 6789.
Mixed ÁÉÍÓÚ ñandú çedilla 你好世界 日本語のテキスト 한국어 😀🎉 tabs\t\there   many   spaces
line1\r\nline2\n\n\nline3   \n  indented   
x = [1, 2, 3]; y = {"a": 1.5e10}; print(f"{x!r}")  #comentario
""" * 2
tokens, tipos, merges = G.entrenar_bpe(400)

TEXTOS = [
 "Hola mundo", "  Hola   mundo  ", "12345 6789 y 0,5 y 3.14159", "¿Cuánto es 12 por 11? Son 132.",
 "I'm sure they're we've you'd it's THEY'LL", "CamelCaseWords HTTPServer2Go snake_case_words",
 "def f(x):\n    return x*2\n\n\nprint(f(21))\n", "line1\r\nline2\n\n\nline3   \n  indented   \n",
 "你好世界 日本語のテキスト 한국어", "😀🎉 emoji y ñandú ÁÉÍÓÚ", "a\tb\t\tc", "   ", "\n\n", "x" * 50 + "1" * 10,
 "<|im_start|>user\nHola 123<|im_end|>\n<|im_start|>assistant\n", "[INST]Hola[/INST]", "Ünïcödé Ωmega naïve café",
 "!!!???...,,, ---=== ~~~ ###", "Precio: $1,234.56 (IVA 18%) el 2026-09-29 a las 10:30", "ABC DEF ghi JKL mno",
 "end.", " leading space", "trailing space ", "MiXeD cAsE wOrDs 0x1F", "a'b 'c d' 'S 'RE",
]
PRES = ["qwen2", "deepseek-r1-qwen", "qwen35", "refact", "starcoder", "tekken", "gpt-2", "default", "llama3", "lfm2",
        "deepseek-v3", "deepseek-llm", "deepseek-coder", "falcon", "dbrx"]

from llama_cpp import Llama
tmp = tempfile.mkdtemp()
fallas = 0; total = 0
for pre in PRES:
    ruta = os.path.join(tmp, pre + ".gguf")
    w = gguf.GGUFWriter(ruta, "llama")
    w.add_name("t"); w.add_context_length(128); w.add_embedding_length(64); w.add_block_count(1)
    w.add_feed_forward_length(64); w.add_head_count(2); w.add_head_count_kv(2)
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre(pre)
    w.add_token_list(tokens); w.add_token_types(tipos); w.add_token_merges(merges)
    w.add_bos_token_id(1); w.add_eos_token_id(2)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    kv = {"tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": pre, "tokenizer.ggml.tokens": tokens,
          "tokenizer.ggml.token_type": tipos, "tokenizer.ggml.merges": merges,
          "tokenizer.ggml.bos_token_id": 1, "tokenizer.ggml.eos_token_id": 2, "tokenizer.ggml.add_bos_token": False}
    try:
        llm = Llama(model_path=ruta, vocab_only=True, verbose=False)
    except Exception as e:
        print(f"{pre:18s} llama.cpp no lo carga: {e}"); continue
    esp = [list(llm.tokenize(t.encode(), add_bos=False, special=True)) for t in TEXTOS]
    del llm
    json.dump({"kv": kv, "textos": TEXTOS}, open(os.path.join(tmp, "in.json"), "w"))
    js = f"""
import {{ Tokenizador }} from '{RAIZ}/web/js/tokenizer.js';
import fs from 'fs';
const d = JSON.parse(fs.readFileSync('{tmp}/in.json','utf8'));
const t = new Tokenizador(d.kv);
console.log(JSON.stringify(d.textos.map(x => t.codificar(x, {{addBos:false}}))));
"""
    open(os.path.join(tmp, "t.mjs"), "w").write(js)
    r = subprocess.run(["node", os.path.join(tmp, "t.mjs")], capture_output=True, text=True)
    if r.returncode:
        print(pre, "JS falló:", r.stderr[-300:]); fallas += 1; continue
    obt = json.loads(r.stdout)
    malos = [i for i in range(len(TEXTOS)) if obt[i] != esp[i]]
    total += len(TEXTOS)
    fallas += len(malos)
    print(f"{pre:18s} {'OK' if not malos else 'FALLA'}  ({len(TEXTOS)-len(malos)}/{len(TEXTOS)})")
    for i in malos[:3]:
        print("   texto:", repr(TEXTOS[i])); print("   llama:", esp[i][:30]); print("   js   :", obt[i][:30])
print("TOTAL fallas:", fallas, "de", total)
sys.exit(1 if fallas else 0)
