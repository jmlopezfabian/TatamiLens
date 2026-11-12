# Rendimiento del análisis (backend `ollama`)

Mediciones del 2026-11-12 en Apple M5 Pro (24 GB), Ollama 0.31.1, video de prueba de 47.8 s a
720p muestreado a 2 fps (96 frames). La estimación para un combate de 5 min (600 frames) es
lineal. Todos los tiempos son de extremo a extremo salvo que se diga lo contrario.

## Resultado

| Configuración | s/frame | Combate de 5 min |
|---|---|---|
| Antes: `qwen3-vl:8b` (Thinking), JSON por frame, 6 imágenes por bloque | 25–80 | varias horas |
| `qwen3-vl:8b-instruct`, mismo prompt | 2.11 | ~21 min |
| + salida compacta (`"mount:A"` por frame) | 1.57 | ~16 min |
| **+ mosaico 2x2, Pydantic AI sobre API nativa (actual)** | **0.49–0.52** | **~5 min** |

## Por qué

1. **El modelo Thinking.** `qwen3-vl:8b` es la variante *Thinking* e ignora `think=False`:
   genera de 2 000 a más de 14 000 tokens de razonamiento por bloque de 6 frames (~37 tok/s),
   es decir, de 1 a más de 8 min por bloque. El backend avisa si el modelo devuelve razonamiento
   y limita los tokens de salida (`max_tokens`) para que no pueda desbocarse.
2. **Imágenes.** Qwen3-VL en Ollama reescala cada imagen a una resolución mínima: mandar 384 px
   cuesta lo mismo que mandar 768 px. Un mosaico 2x2 de 1280x720 cuesta lo que un solo frame y
   lleva 4 (`tile_frames` en `video.py`).
3. **Salida.** Un objeto JSON por frame son ~44 tokens; un código del enum son ~5. Las
   descripciones de los atletas solo se piden en el primer bloque.
4. **API nativa frente a `/v1`.** `pydantic_ai.models.ollama.OllamaModel` usa el endpoint
   OpenAI-compatible. Ahí, el mismo bloque de 8 frames tarda 6.1 s (frente a 3.5 s por la API
   nativa), porque cada imagen se codifica con ~1040 tokens en vez de ~500, y además `num_ctx` se
   ignora (contexto de 4096). Por eso `backends/ollama_native.py` implementa un modelo de Pydantic
   AI (`FunctionModel`) sobre `/api/chat`, que conserva `Agent`, `NativeOutput` y los reintentos.

## Descartado

- **Bajar la resolución:** sin efecto, por la resolución mínima.
- **Bloques de 12 o 24 frames:** igual o más lento por frame.
- **Peticiones en paralelo (`OLLAMA_NUM_PARALLEL=2`):** cero ganancia, porque la GPU ya está
  saturada. Además, partir el video en tramos pierde la identidad A/B entre tramos.
- **Leer frames con seek:** 5 veces más lento que la lectura secuencial. Decodificar un combate
  de 5 min lleva unos 8 s.
- **1 fps:** ~2.5 min por combate, pero con peor continuidad temporal. Se puede elegir con
  `sample_fps`.

## Calidad

Sin conjunto etiquetado, solo se comparó suelo/de pie frente a una referencia hecha a ojo:
84–93 % en todas las variantes, sin pérdida clara por el mosaico. **El VLM de 8B clasifica mal
las posiciones finas en todas las configuraciones** (casi todo sale `mount`) y la identidad A/B
es inestable. Hace falta un conjunto de evaluación etiquetado para medirlo. La vía para mejorar
calidad y velocidad a la vez es el backend `local` (DINOv2-small: 9 ms/frame) entrenado con
`tatami-train`.
