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

Evaluación con `tatami-train bench-vlm` (ver `tatami_train/vlm_eval.py`) contra
`tatami-train/eval/labels/insta2.json`. **Ojo:** son etiquetas preliminares hechas por Claude
(`source: "model"`) sobre el único video disponible (48 s, 96 frames), no verificadas por una
persona. Sirven para comparar variantes a grandes rasgos, no como cifra absoluta: hay bastante
varianza entre ejecuciones (±10 puntos), así que diferencias de ese orden son ruido.

Columnas: *top* es quién está arriba en frames con top etiquetado (entre paréntesis, la referencia
trivial de decir siempre el atleta que más tiempo está arriba); *A/B dicho* es la fracción de esos
frames en que el modelo dijo A o B en vez de una posición neutra; *baile* es, entre esos, la
fracción con la identidad contraria a la mayoritaria. Los s/frame son de pared e incluyen la carga
del modelo; el 0.20 del primer mosaico está falseado porque se ejecutó justo después de otras
pruebas del mismo video con el mismo mosaico y Ollama reutilizó la caché de prompts (sin caché,
~0.5). Las filas de 1 frame/imagen usan bloques de 6; con el prompt actual y contexto de 8192 el
tope es de 5 frames por petición (`max_block_for_context`), así que su s/frame puede variar algo.

| Variante | s/frame | Posición | F1 macro | Grupo | Top (base) | A/B dicho | Baile |
|---|---|---|---|---|---|---|---|
| qwen3-vl, mosaico, prompt antiguo | 0.20* | 11 % | 0.06 | 45 % | 65 % (97 %) | 67 % | 2 % |
| qwen3-vl, 1 frame/imagen, prompt antiguo | 1.70 | 2 % | 0.01 | 47 % | 94 % (97 %) | 97 % | 3 % |
| qwen3-vl, mosaico, prompt con definiciones | 0.52 | 22 % | 0.11 | 33 % | 22 % (97 %) | 41 % | 46 % |
| qwen3-vl, 1 frame/imagen, prompt con definiciones | 1.89 | 18 % | 0.12 | 46 % | 70 % (97 %) | 97 % | 28 % |
| gemma4:e4b, mosaico | 0.46 | 28 % | 0.15 | 31 % | 14 % (97 %) | 19 % | 25 % |
| gemma4:e4b, 1 frame/imagen | 0.61 | 11 % | 0.04 | 34 % | 71 % (97 %) | 79 % | 10 % |

Conclusiones:

- **Ningún VLM pequeño sirve para las posiciones finas ni para quién está arriba.** Cada uno
  colapsa hacia una o dos clases: qwen3-vl hacia `mount` (frame suelto, prompt antiguo) o
  `scramble` (mosaico, prompt nuevo); gemma4 hacia `scramble` u `open_guard`. Las guardias casi
  nunca se reconocen y ninguna variante supera la referencia trivial de *top*.
- **El prompt con definiciones visuales es un cambio mixto.** Corrige el sesgo hacia `mount` y sube
  la posición exacta y el F1 macro, pero en el mosaico desplaza el error hacia `scramble`: baja el
  grupo (45 → 33 %) y el modelo deja de decir quién está arriba en la mayoría de frames. Se mantiene
  porque el F1 macro mejora en las dos configuraciones, pero la evidencia es débil.
- **El mosaico se mantiene por velocidad** (3–4 veces más rápido con qwen3-vl; con gemma4, 1.3
  veces). En posición no hay diferencia clara frente a un frame por imagen; en *top* es peor, sobre
  todo porque más frames salen como posiciones neutras sin A/B.
- La vía razonable es el backend `local` (DINOv2-small: 9 ms/frame) entrenado con `tatami-train`
  sobre etiquetas humanas, usando el VLM como mucho como pseudoetiquetador inicial. Para medir y
  entrenar hacen falta más combates etiquetados a mano en la web.
