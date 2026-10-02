import json, os, numpy as np, gguf
from gguf.quants import dequantize
from gen_modelos import referencia, OUT
refs = json.load(open(os.path.join(OUT, "..", "referencias_llamacpp.json"), encoding="utf-8"))
for nombre in ["lfm2-test-q8", "lfm2-test-q8-nobos"]:
    r = gguf.GGUFReader(os.path.join(OUT, nombre + ".gguf"))
    ruta = os.path.join(OUT, "..", nombre + "-deq.gguf")
    w = gguf.GGUFWriter(ruta, "lfm2")
    for f in r.fields.values():
        if f.name.startswith("GGUF.") or f.name == "general.architecture": continue
        tipo = f.types[0]
        if tipo == gguf.GGUFValueType.ARRAY:
            sub = f.types[1]
            if sub == gguf.GGUFValueType.STRING:
                val = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
            else:
                val = [f.parts[i].tolist()[0] for i in f.data]
            w.add_key_value(f.name, val, tipo, sub_type=sub)
        elif tipo == gguf.GGUFValueType.STRING:
            w.add_key_value(f.name, bytes(f.parts[f.data[0]]).decode("utf-8"), tipo)
        else:
            w.add_key_value(f.name, f.parts[f.data[0]].tolist()[0], tipo)
    for t in r.tensors:
        shape = [int(x) for x in reversed(t.shape.tolist())]
        a = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(shape)
        w.add_tensor(t.name, a)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    ref = referencia(ruta)
    assert ref["tokens"] == refs[nombre]["tokens"]
    refs[nombre]["prompts"] = ref["prompts"]
    print(nombre, "referencia desde pesos decuantizados en F32")
json.dump(refs, open(os.path.join(OUT, "..", "referencias_llamacpp.json"), "w", encoding="utf-8"), ensure_ascii=False)
