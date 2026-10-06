# Roadmap (v4 y más allá)

Esta es una lista de intenciones, **no una promesa ni un calendario**. Los puntos están ordenados por cuánto ayudarían a usuarios reales frente a cuánto riesgo tienen. Las etiquetas de esfuerzo (S / M / L) son estimaciones nuestras, no mediciones. Todo está **sin empezar** salvo que se indique lo contrario. ¿Quieres tomar uno? Abre primero un issue para acordar la prueba que demostraría que funciona.

La regla para cada punto es la misma que para el motor actual: está terminado cuando las pruebas de conformidad lo dicen, no cuando la salida «se ve bien» (ver [Cómo se verifica](README.es.md#cómo-se-verifica)).

**Lo que funciona hoy (4.1):** API compatible con OpenAI/LM Studio (chat, streaming, completions, embeddings, visión de dos familias), página de chat integrada, panel de control, `--share` para un equipo pequeño y las arquitecturas del README, todo verificado contra llama.cpp. Esta página trata de lo que viene.

**Hecho recientemente:** página de chat y `--share` (4.1.0).

## Resumen

| # | Punto | Por qué importa | Esfuerzo | Riesgo | Cómo se verificaría |
|---|---|---|---|---|---|
| 1 | Modo sin ventana / servicio y arranque en Linux y macOS | Servidores desatendidos; llegar más allá de Windows | M | Medio | Corridas con GPU real en Linux, macOS y Windows (requiere hardware) |
| 2 | `response_format` forzado (`json_object` y `json_schema`) | Salida estructurada confiable para aplicaciones | M–L | Medio | Modelos sintéticos (la salida siempre debe validar) |
| 3 | Caché de prompts (reusar la caché KV entre peticiones) | Chat de varios turnos y agentes más rápidos | M | Medio | Igualdad token a token con y sin caché |
| 4 | Tool calling / llamada a funciones | Compatibilidad con frameworks de agentes | L | Medio–Alto | Plantillas contra jinja2; pruebas del intérprete de salida; pruebas con modelos reales |
| 5 | Caché KV en f16 | La mitad de memoria por token de contexto | M | Medio–Alto | Comparar con el camino f32 y con una referencia f32 |
| 6 | `logprobs`, `n > 1` y otros huecos de la API | Compatibilidad directa | S–M | Bajo | Pruebas de API |
| 7 | Benchmarks en otro hardware | Medimos tres máquinas (una contra LM Studio) | S (por máquina) | Bajo | Salida de `tools/bench.py` |
| 8 | Visión: seguir el preprocesado nuevo de llama.cpp | Paridad con llama.cpp actual en LFM2-VL y Gemma 3 | M | Medio | Referencias nuevas y una corrida `--engine` completa |
| 9 | Más adelante, no planeado para v4 | Batching continuo, varios modelos en memoria, más arquitecturas | L | Alto | n/a |

## 1. Modo sin ventana / servicio y arranque en Linux y macOS

**Hoy:** el motor necesita una ventana del navegador abierta en la PC servidora. Desde 4.0.1 el servidor busca en Linux un navegador basado en Chromium en el PATH (esa búsqueda no se probó en una máquina Linux real); en macOS usa el navegador predeterminado en una pestaña normal, sin probar. En una Intel HD 4000 con Linux Mint, Chrome solo expuso WebGPU con `--enable-unsafe-webgpu --enable-features=Vulkan --ignore-gpu-blocklist` (qué bandera es la necesaria: sin probar) y Firefox no lo expone por defecto; hoy esas banderas las pone el usuario a mano.

**Qué haríamos:** (a) encontrar Chrome/Chromium/Edge en Linux y macOS y pasarle las banderas que el motor necesita; (b) un modo opcional de arranque oculto o sin ventana para que el servidor corra desatendido; (c) documentación para ejecutarlo como servicio.
La suite de pruebas ya corre el motor real en Chromium sin ventana con SwiftShader (WebGPU por software), así que ahí el modo sin ventana funciona. Que Chromium sin ventana llegue a una GPU **real** está **sin medir** y es lo primero que hay que averiguar.

**Verificación:** la misma suite del motor más corridas con GPU real en cada sistema operativo. Este punto depende de benchmarks en otras máquinas (punto 7).

## 2. `response_format` forzado (`json_object` y `json_schema`)

**Hoy:** `json_object` devuelve HTTP 400 (como hacía LM Studio); `json_schema` se acepta pero **no se hace cumplir**.

**Por qué importa:** las aplicaciones que procesan la salida del modelo necesitan que sea válida siempre, no casi siempre.

**Qué haríamos:** el muestreo de tokens ya ocurre en JavaScript sobre los logits (`web/js/sampling.js`), así que ahí se puede aplicar una restricción que enmascare los tokens que romperían el formato. Un primer paso es «solo JSON válido»; después, un subconjunto soportado de JSON Schema. Lo difícil es la velocidad: las máscaras sobre un vocabulario de muchos miles de tokens hay que cachearlas o calcularlas de forma incremental. Todavía no sabemos el costo.

**Verificación:** esto sí se puede probar con los modelos sintéticos de pesos aleatorios: con el forzado activado, **toda** salida debe analizarse y validar contra el esquema, diga lo que diga el modelo. Un control negativo (forzado apagado) debe producir salidas inválidas.

## 3. Caché de prompts

**Hoy:** experimental solo para modelos LFM2 (versión de desarrollo sin publicar): el motor reutiliza el prefijo del prompt que comparte con la petición anterior, con una copia del estado de convolución tomada justo antes del último token del prompt. La igualdad con y sin caché se comprobó con los modelos sintéticos (logits idénticos). Medido una vez en una GTX 1050 Ti con lfm2.5-1.2b-instruct: el segundo turno reutilizó 409 de 425 tokens y el tercero 500 de 514, con primer token en 0,94 s y 1,08 s; en otros turnos de conversaciones más largas el primer token tardó 22,8 a 31,1 s y la causa aún no está identificada. No hay otras familias (ventana deslizante de Gemma 3, caché comprimida de DeepSeek, modelos con KV simple) y las peticiones con imágenes no se cachean. Antes de esto, cada petición reiniciaba el modelo y recalculaba todo el prompt, así que las conversaciones largas pagan el historial completo cada vez. Como referencia de escala, el prefill medido en la máquina de pruebas: un LFM2 de 1,2B tardó unos 2,3 s en 62 tokens (unos 28 tok/s); un Granite de 8B, 2,3–3,4 s en 58–76 tokens (ver [BENCHMARKS.md](docs/BENCHMARKS.md)).

**Qué haríamos:** conservar la caché KV entre peticiones y reutilizar el prefijo común más largo de ids de tokens, calculando solo los tokens nuevos. Hay que cuidar las capas de ventana deslizante (Gemma 3), el estado de convolución de LFM2 y la caché comprimida de DeepSeek, que no se comportan todas como una caché KV simple.

**Verificación:** la salida voraz con caché debe ser igual a la salida sin caché, token a token, en cada arquitectura soportada.

## 4. Tool calling / llamada a funciones

**Hoy:** experimental, en la versión de desarrollo aún sin publicar. Un modo genérico como el «modo por defecto» de LM Studio: las herramientas se describen en el mensaje de sistema, el modelo escribe `[TOOL_REQUEST]{...}[END_TOOL_REQUEST]` (o su formato nativo: se entienden LFM2 y Qwen/Hermes) y el servidor devuelve `tool_calls` de OpenAI, también con streaming. Medido con un modelo en un PC (lfm2.5-1.2b-instruct, 3 corridas por tipo): eligió la herramienta correcta entre dos 6 de 6 veces, pero rechazó 1 de 3 preguntas normales cuando había herramientas ([BENCHMARKS.md](docs/BENCHMARKS.md)). Sin hacer: mostrar las herramientas con la plantilla de chat propia de cada modelo, otras familias, modelos mayores, argumentos JSON forzados.

**Qué haríamos:** aceptar `tools` y `tool_choice`; volcarlos en la plantilla de chat del modelo (el intérprete de plantillas de `templates.py` ya soporta `tojson` y `namespace`, que usan las plantillas con herramientas); interpretar la salida del modelo en el formato `tool_calls` de OpenAI, incluido el streaming. El formato de salida varía entre familias de modelos, así que empezaría con una o dos familias y crecería.

**Verificación:** la plantilla renderizada comparada con jinja2 (como en las pruebas de plantillas actuales) y pruebas del intérprete de salida. Los modelos sintéticos aleatorios no pueden producir llamadas a herramientas con sentido, así que **la calidad de punta a punta necesita modelos reales** y solo se reportaría como medida. Este punto funciona mejor después del 2 (JSON forzado).

## 5. Caché KV en f16

**Hoy:** la caché KV es f32, lo que limita el contexto por memoria (`kv_max_mb`).

**Qué haríamos:** guardar K y V en f16 (`pack2x16float` / `unpack2x16float` de WGSL ya se usan para pesos f16, así que no debería hacer falta la extensión opcional `shader-f16`).

**Por qué es riesgoso:** la precisión. En nuestro propio trabajo de validación, una caché KV f16 en llama.cpp se desvía unos 1e-2 de su propio resultado exacto en modelos MoE profundos. Cualquier modo f16 debe compararse contra nuestro camino f32 y contra una referencia f32, y quizá deba seguir siendo opcional.

**Verificación:** diferencia relativa de logits contra el camino f32 en cada arquitectura, con la tolerancia declarada de antemano; primero DeepSeek-V2, por ser el caso sensible.

## 6. Huecos de la API

`logprobs` y `n > 1` no están implementados. Como los logits ya están disponibles en JavaScript para el muestreo, `logprobs` debería ser barato; `n > 1` necesita generación repetida en una petición. Se verifica con pruebas de API.

La página `/chat` es pequeña a propósito. Candidatos, todos de bajo riesgo: listas y tablas en el renderizador de Markdown, adjuntar una imagen para modelos de visión, exportar una conversación y (tras el punto 3) conversaciones largas más rápidas. Se verificarían en un navegador real contra un motor simulado.

## 7. Benchmarks en otro hardware

La única comparación con LM Studio viene de una máquina (Intel i7-1270P, Intel UHD, Windows, Edge). Otras dos se midieron sin LM Studio (GTX 1050 Ti en Windows; Intel HD 4000 en Linux, dos corridas por modelo). AMD, Apple y NVIDIA/Linux recientes están **sin medir**, igual que las estimaciones de memoria «cabe / justo / no cabe» en otro hardware. Es la contribución más útil ahora: corre `tools/bench.py` y abre un issue *Benchmark report* (ver [CONTRIBUTING.md](CONTRIBUTING.md)).

## 8. Visión: seguir el preprocesado nuevo de llama.cpp

**Hoy:** la visión sigue a llama.cpp según `llama-cpp-python` 0.3.35 y no pasa las suites de imagen contra 0.3.36, porque upstream cambió el preprocesado de imágenes (ver [LIMITATIONS.md](docs/LIMITATIONS.md)). **Sin medir:** cuánto cambian esas diferencias la salida de los modelos reales.

**Opciones:** (A) seguir fijado y documentado (actual); (B) seguir upstream (redimensionado bilineal estilo Pillow, sin `<|img_thumbnail|>` en imágenes LFM2 de una sola tesela, regla de teselado por área), regenerar las referencias y repetir la suite `--engine` completa; (C) soportar ambos comportamientos con un interruptor y dos juegos de referencias. Se decidiría después de medir la salida de modelos reales con ambos.

## 9. No planeado para v4

Batching continuo, varios modelos cargados a la vez y modelos que requieren código nuevo en el núcleo (QKV fusionado, softcapping de atención, capas recurrentes/SSM, RoPE escalado). Son cambios grandes con alto riesgo de errores silenciosos. Las arquitecturas nuevas que encajan en las clases base existentes ya las pueden agregar los colaboradores ([docs/ARCH_GUIDE.md](docs/ARCH_GUIDE.md)).
