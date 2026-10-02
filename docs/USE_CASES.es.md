# ¿Para quién es quipullm y cómo usarlo en entornos restringidos?

quipullm es un servidor LLM autoalojado que puedes leer completo antes de ejecutarlo. Esta página dice para qué sirve eso y, igual de importante, lo que **no** te ofrece. Los límites exactos están en [LIMITATIONS.md](LIMITATIONS.md) y [SECURITY.md](../SECURITY.md) (en inglés).

## Qué obtienes

- **Solo texto.** Sin `.exe`, sin `.dll`/`.so`, sin instaladores, sin binarios compilados, sin `pip`, sin `npm`, sin Docker. Todo el proyecto es `.py .js .html .json .md`. Se puede copiar a una máquina detrás de un proxy que rompe SSL o de un filtro de binarios, y leer antes de ejecutarlo.
- **Tus prompts se quedan en tu máquina.** Los modelos corren en la GPU de la PC servidora. Por defecto el servidor escucha solo en `127.0.0.1`. Los prompts no se escriben en el registro (el registro guarda conteos de peticiones y de tokens).
- **Sin conexiones salientes que hayamos podido encontrar** (ver [Cómo lo comprobamos](#cómo-comprobamos-lo-de-sin-conexiones-salientes)). Es una comprobación, no una auditoría de seguridad.
- **API compatible con OpenAI y LM Studio.** El código existente solo cambia la URL base.
- **No necesitas tarjeta gráfica dedicada.** Una GPU integrada sirve si el navegador tiene WebGPU con aceleración por hardware (es el único hardware que medimos; ver [BENCHMARKS.md](BENCHMARKS.md)).
- **Apache-2.0.** Sin cuotas por uso. Los modelos no se incluyen y conservan su propia licencia.

## A quién puede ayudar

| Quién | Uso típico |
|---|---|
| Sector público, salud, legal, finanzas | Resumir, clasificar o redactar con datos que no deben salir de la organización |
| Redes institucionales restringidas | Instalar cuando solo pasan archivos de texto |
| Oficinas pequeñas con PC modestas | Un punto de IA compartido en una PC con GPU integrada o una tarjeta común |
| Colegios y laboratorios | Aulas donde no se puede instalar software |
| Desarrolladores | Probar contra una API compatible con OpenAI, sin internet y sin costo por token |
| Investigadores y estudiantes | Un motor de inferencia WebGPU pequeño y legible, más un [método de validación](../README.es.md#cómo-se-verifica) para estudiar o ampliar |

## Lo que NO te ofrece (léelo antes de proponerlo en una organización)

- **Ninguna certificación ni cumplimiento normativo.** quipullm no tiene ISO 27001, SOC 2, certificación de protección de datos ni similar, y **no ha tenido auditoría de seguridad externa**. No lo describas como «seguro para datos sensibles» sin tu propia revisión.
- **Una petición a la vez.** Las demás esperan en cola. No está pensado para muchos usuarios simultáneos.
- **Sin TLS.** En una red compartida, los prompts y la clave de API viajan sin cifrar. Pon un proxy inverso con TLS si eso importa.
- **La clave de API queda en texto plano** en `config.json` (salvo que uses `LLM_API_KEY`). Protege ese archivo.
- **Necesita una ventana del navegador abierta en la PC servidora** (aún no hay modo de servicio sin ventana).
- **Cada modelo tiene su propia licencia.** Revisa la de cada modelo que despliegues.
- **La comparación con LM Studio es de una sola máquina** (Intel i7-1270P + Intel UHD, Windows, Edge). Otras dos máquinas se midieron sin ella (ver [BENCHMARKS.md](BENCHMARKS.md)); el resto del hardware no está medido.

## Lista de comprobación para evaluarlo en un entorno restringido

1. Copia el repositorio como texto; lee `server.py`, `templates.py` y `web/` (todo el código son unos pocos miles de líneas).
2. Arranca con `python server.py` (por defecto, solo `127.0.0.1`). No pongas `"host": "0.0.0.0"` hasta definir también una clave de API.
3. Si otras máquinas deben llegar a él: define clave de API, define `"host"` y pon un proxy inverso con TLS (agrega su nombre a `allowed_hosts`).
4. Ejecuta `python tests/run_all.py` (solo librería estándar; se esperan 205 comprobaciones aprobadas).
5. Verifica tú mismo lo de «sin conexiones salientes» en tu entorno (abajo).
6. Revisa la licencia de cada modelo que pienses cargar.
7. No expongas el servidor a internet.

## Cómo comprobamos lo de «sin conexiones salientes»

En Linux corrimos la suite de pruebas de API bajo `strace` y listamos cada `connect()`:

```
cd tests
strace -f -qq -e trace=connect -o /tmp/connects.txt python3 api_tests.py
grep 'connect(' /tmp/connects.txt | grep -o 'inet_addr("[^"]*")' | sort | uniq -c
```

Resultado en nuestra corrida (205 comprobaciones aprobadas): casi todas las conexiones fueron a `127.0.0.1`. Las únicas otras direcciones fueron `10.255.255.255` y la dirección LAN de la propia máquina de pruebas.
`10.255.255.255` es el servidor averiguando su propia IP de red con un `connect()` UDP, que no envía ningún paquete (la traza no muestra ningún `sendto` hacia ella). La dirección LAN fue el cliente de pruebas llegando al servidor por la dirección de la propia máquina.
No apareció ninguna conexión a un host externo.

Límites de esta comprobación: cubre la suite de API con un **motor simulado**, no un navegador ejecutando el motor real, y es una sola corrida en una máquina. La página del motor la sirve el servidor y `web/` no contiene URLs externas, pero el navegador en sí lo debes auditar tú. Repite la comprobación en tu entorno antes de apoyarte en ella.
