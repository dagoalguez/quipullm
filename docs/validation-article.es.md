# Cómo validamos, número por número, un motor LLM WebGPU escrito desde cero contra llama.cpp

*Código: [github.com/dagoalguez/quipullm](https://github.com/dagoalguez/quipullm) (Apache-2.0). Todas las cifras salen de la salida de las pruebas del repositorio o de `docs/BENCHMARKS.md` / `docs/LIMITATIONS.md`. Las velocidades vienen de tres máquinas (listadas abajo); la única comparada con LM Studio es el Intel Core i7-1270P + GPU integrada Intel UHD, Windows, Edge. Las otras dos se corrieron dos veces por modelo, sin LM Studio.*

## El problema

quipullm es un servidor LLM autoalojado, compatible con OpenAI y LM Studio. Su motor de inferencia está escrito desde cero en WGSL y JavaScript y corre en una ventana del navegador; el backend usa solo la librería estándar de Python. Lee archivos GGUF estándar.

Un motor de inferencia nuevo tiene un modo de fallo más importante que cualquier caída: **produce texto fluido pero sutilmente incorrecto.** Un tensor intercambiado, una máscara de atención equivocada o un sesgo (bias) olvidado casi nunca dan galimatías; dan prosa verosímil. Leer salidas y decir «se ve bien» no prueba nada. Por eso comparamos números, no prosa, con llama.cpp como referencia.

## Método 1: modelos sintéticos, para que la referencia sea exacta y barata

Los modelos reales son grandes, lentos de correr en un navegador con renderizado por software, y no se pueden incluir en un repositorio. En su lugar, generadores en `tests/conformance/` escriben **modelos diminutos de pesos aleatorios** con el paquete `gguf` de Python para cada arquitectura (Qwen 2/3, estilo Llama, Granite, Gemma 3, LFM2, nomic-BERT, DeepSeek-V2 con MLA + MoE + YaRN, y las pilas de visión LFM2-VL y Gemma 3). Los casos raros se incluyen a propósito: dimensión de cabeza explícita, sesgos, salida atada y no atada, ventanas deslizantes, distinto número de expertos MoE.

Luego llama.cpp cuantiza cada modelo (de Q2_K a Q6_K, y Q8_0 en las variantes que generamos) y registra sus **ids de tokens, los logits del último token y la continuación voraz (greedy)**. Las pruebas corren en Chromium sin ventana con SwiftShader (WebGPU por software). Eso prueba corrección, no velocidad ni memoria; el repositorio lo dice donde importa.

## Método 2: comparar contra lo correcto

Para cada prompt, los criterios de aprobación son:

1. **Ids de tokens idénticos** a los del tokenizador de llama.cpp (suite del tokenizador: 375 comprobaciones, 0 fallas en nuestra última corrida completa; las plantillas de chat se comparan con jinja2: 98 comprobaciones, 0 fallas).
2. **Logits del último token con diferencia relativa < 2e-3** respecto de una **copia F32 de los mismos pesos, descuantizada**.
3. **Continuación voraz idéntica** (se admite un empate documentado: diferencia de logits menor a 0,002).
4. **De punta a punta por la API HTTP real**, la página del motor y la plantilla de chat; texto idéntico y sin errores en la consola de JS.

¿Por qué comparar con una copia F32 descuantizada y no con la salida cuantizada de llama.cpp? En CPU, llama.cpp cuantiza las *activaciones* a Q8 antes de los productos de matrices. Nuestro motor calcula las activaciones en f32. Frente al camino cuantizado de llama.cpp la diferencia es de unos 3e-2, lo cual es esperable y poco informativo; frente a los pesos F32 descuantizados es una cota ajustada y significativa. La misma división aparece con modelos reales: en la comparación con LM Studio 0.4.25 en la máquina de pruebas, los tokens del prompt coincidieron en todas las pruebas y el texto voraz coincidió en la mayoría (por ejemplo LFM2 44 de 48, Qwen2.5-Coder 8/8, Granite 8/8, Gemma 3 8/8), y cada diferencia fue un texto largo que se separa en un casi-empate de logits, con significado correcto. Es el mismo efecto de las activaciones Q8.

## Método 3: controles negativos y pruebas de mutación

Una prueba que no puede fallar es decoración. A cada funcionalidad le pedimos demostrar que está cubierta:

- **Procedimiento de mutación (documentado en `docs/ARCH_GUIDE.md`)**: desactivar temporalmente lo que implementaste (ventana, norma, sesgo…). Las pruebas deben fallar. Si siguen pasando, el generador no ejercita esa funcionalidad y hay que corregirlo. Es un flujo de trabajo para quien contribuye, no una suite automática.
- **Control negativo automático**: en la suite de visión de Gemma 3, una corrida con atención *solo causal* dentro del bloque de la imagen debe fallar, porque Gemma 3 usa atención no causal entre los tokens de imagen. `run_all.py --engine` lo marca como correcto solo cuando detecta la mutación. En nuestra última corrida completa lo hizo.

## Lo que encontró el proceso

**1. `cos` / `sin` de WebGPU pueden tener unos 2e-4 de error en algunos drivers.** Es mucho más de lo que tolera una cota de 2e-3 cuando los errores se acumulan por capas y posiciones. Por eso RoPE usa su propio seno/coseno, con error cercano a 5e-6. No podemos decir qué drivers o GPU muestran el error de 2e-4 más allá de «algunos»; no hicimos un relevamiento.

**2. Una caché KV en f16 deriva en modelos MoE profundos, en el propio llama.cpp.** Con caché KV en f16, llama.cpp se desvía unos 1e-2 de *su propio* resultado exacto en modelos MoE profundos. Una referencia que ya es ruidosa no puede validar nada a 2e-3. La solución está en el método: generar las referencias con caché KV en f32 y comparar contra el archivo F32 descuantizado. (Nuestra propia caché KV es f32; reducirla a la mitad con f16 está en el roadmap.)

**3. La referencia se mueve.** Para visión validamos contra `llama-cpp-python` 0.3.35. Contra 0.3.36 (ggml 0.25.3), con el mismo código del motor y los mismos modelos sintéticos, la suite del codificador LFM2-VL pasó de 24/24 a 9/24, la suite de API de LFM2-VL de 41/41 a 34 aprobadas y 9 fallas, y la del codificador de Gemma 3 de 12/12 a 6/12 (coseno 0,9956–0,9985 en las fallas). Las suites de texto, BERT y DeepSeek no se movieron. Tres cambios de upstream lo explican: el redimensionado bilineal pasó a ser un filtro triangular al estilo Pillow que se ensancha al reducir; las imágenes LFM2 de una sola tesela ya no reciben el token `<|img_thumbnail|>` (un token menos por imagen); y la regla de teselado de LFM2 pasó de «por lado» a «por área de píxeles redondeada». Incluso los embeddings de imagen de las dos versiones de referencia difieren entre sí (coseno mínimo 0,9898) con el mismo modelo e imagen.

La lección: el preprocesado de imágenes es política del proyecto de referencia, no matemática del modelo. «Validado contra llama.cpp» solo significa algo con una versión adjunta. Fijamos `llama-cpp-python` 0.3.35, `gguf` 0.19.0 y `numpy` 2.4.4, y `run_all.py` avisa cuando el entorno difiere.

**Lo que no sabemos:** cuánto cambian estas diferencias la salida de los modelos de visión *reales*, y qué build de llama.cpp usaba LM Studio 0.4.25 cuando grabamos las referencias reales. Ninguna de las dos cosas se midió.

## Trampas de plataforma que ninguna referencia podía detectar

Salieron de correr en una iGPU real, no de la suite de conformidad: el watchdog de GPU de Windows (TDR) reinicia el driver tras unos 2 segundos en un solo envío, así que el prefill se envía por capa y la atención grande se divide; WebGPU limita los dispatch a 65535 workgroups por dimensión; el indexado dinámico de arreglos locales en WGSL es muy lento en GPU integradas; y WebGPU no informa la VRAM libre, así que nuestras etiquetas «cabe / justo / no cabe» son estimaciones.

## Lo que enseñó correr en otras máquinas

Tras el primer lanzamiento medimos dos máquinas más (quipullm 4.0.1, dos corridas por modelo, sin LM Studio ahí). Dos hallazgos no tienen que ver con la matemática del motor:

- **Una tarjeta dedicada no fue automáticamente más rápida.** En un i5-8400 + GTX 1050 Ti 4 GB (Windows 11, Edge 143, adaptador «nvidia pascal») obtuvimos 42,6 / 42,0 tok/s con LFM2 350M Q8_0, 14,7 con LFM2.5 1.2B Q8_0 y 5,0 con Gemma 3 4B Q4_K_M: prácticamente igual que la GPU integrada del i7-1270P (44,3, 15,7, 5,3). No investigamos por qué; no tenemos un perfil que diga si el límite es el motor, WebGPU o el driver, así que no afirmamos la causa.
- **Una GPU Intel antigua en Linux funcionó, pero solo lanzando Chrome con banderas adicionales.** En un i5-3230M + Intel HD 4000 (Linux Mint 22.3, Chrome 154, adaptador «intel gen-7»), Firefox (el navegador predeterminado) no expone WebGPU por defecto y Chrome sin banderas informó «No WebGPU adapter was found». Con `--enable-unsafe-webgpu --enable-features=Vulkan --ignore-gpu-blocklist` funcionó: 9,4 tok/s con LFM2 350M Q8_0 y 3,8 tok/s con LFM2.5 1.2B Q8_0. No probamos cuál de las tres banderas es la necesaria. Gemma 3 4B no se probó allí.

El mismo ejercicio destapó tres errores de nuestro propio código que las pruebas de conformidad no podían ver: una conexión larga (long-poll) del motor que podía colgarse en silencio (ahora se corta a los 35 s y un watchdog relanza el motor), una clave de log mal nombrada y una detección de navegador en Linux que elegía el predeterminado en lugar de uno basado en Chromium. Nuestro propio CI también falló por errores del arnés de pruebas, no del motor (un pipe de stdout sin leer que bloqueaba al servidor en Windows y una condición de carrera en una prueba de cancelación); corregimos el arnés y el CI está en verde en Ubuntu y Windows con Python 3.9 y 3.12.

## Qué demuestra y qué no

**Demuestra:** en modelos sintéticos de las arquitecturas y cuantizaciones listadas, el motor coincide con llama.cpp 0.3.35 token por token y dentro de 2e-3 en logits frente a F32 descuantizado, de punta a punta por la API HTTP, con controles negativos que fallan cuando deben.

**No demuestra:** velocidad en general (tres máquinas, dos de ellas con dos corridas por modelo y sin runtime de referencia; AMD y Apple no están medidos, y la máquina Linux necesitó banderas del navegador), corrección en todos los modelos reales (las comprobaciones con modelos reales se hicieron contra LM Studio en un conjunto pequeño), ni paridad con el preprocesado de visión más nuevo de llama.cpp.

Como contexto, esa máquina dio 44,3 tok/s con un LFM2 de 350M en Q8_0 y unos 15,7 tok/s con uno de 1,2B, 5,3 tok/s con Gemma 3 4B Q4_K_M y 2,8–3,1 tok/s con un 7B Q4_K_M. LM Studio en la misma máquina dio 30,2, 9,6, 5,0 y 3,1 respectivamente. Las otras dos máquinas están en `docs/BENCHMARKS.md` (GTX 1050 Ti: 42,6, 14,7 y 5,0 tok/s en los mismos tres modelos; Intel HD 4000 en Linux: 9,4 y 3,8 tok/s en los dos primeros). Dos salvedades: no se registró la variante del runtime de LM Studio (CPU o Vulkan), y somos más lentos en algunos modelos de 7–8B (unos 1,9 frente a 2,4 tok/s en un Nemo Q3_K_L).

## Reprodúcelo

```
python tests/run_all.py            # suite de API con motor simulado; solo librería estándar
python tests/run_all.py --engine   # conformidad completa; requiere llama-cpp-python, gguf, numpy fijados y Playwright
```

La corrida completa con `--engine` es lenta bajo SwiftShader: solo la suite de DeepSeek tardó 1273 s en nuestra última ejecución. Los tiempos dependen mucho de la máquina. Si la corres en otro hardware o encuentras un driver con un comportamiento distinto de `cos`/`sin`, abre un issue con tus números.

---

*quipullm está pensado para PCs donde instalar software es difícil: solo texto plano, sin `pip`, sin `npm`, sin binarios. Código y documentación: [https://github.com/dagoalguez/quipullm](https://github.com/dagoalguez/quipullm).
Construido con asistencia de IA (Claude, Anthropic); el autor revisa y es responsable de lo que se publica. Gracias a los mantenedores de llama.cpp / ggml y `gguf`, cuya implementación de referencia hizo posible esta validación. Los modelos no se distribuyen y conservan sus licencias. Sin afiliación con llama.cpp ni LM Studio.*
