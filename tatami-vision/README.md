# tatami-vision

Librería + CLI de Python para analizar videos de jiu-jitsu brasileño (BJJ). Parte de TatamiLens
(contratos compartidos en `../docs/CONTRACTS.md`).

Dado un video produce un `AnalysisResult` (JSON): etiqueta de posición por frame muestreado
(`standing`, `closed_guard`, `mount`, ...), quién está arriba (`top` = `A`/`B`), segmentos
estables, eventos de puntuación IBJJF simplificados y marcador. Opcionalmente renderiza un video
con overlay.

## Cómo funciona

1. Se muestrean frames a N fps (`video.py`), opcionalmente recortados a la "zona de combate"
   detectada con YOLO-pose (`tracking.py`; filtra árbitro y público).
2. Un backend (`backends/`) clasifica la posición por bloques de frames con marcas de tiempo,
   manteniendo la identidad A/B (descripción + últimas etiquetas del bloque anterior).
3. `smoothing.py` quita parpadeos (moda ponderada por confianza).
4. `scoring.py` calcula segmentos y eventos (derribo, raspada, pase, rodilla, montada, espalda).

En el suelo los dos atletas casi siempre salen como una sola caja en detectores de personas, y los
IDs de tracking cambian mucho; por eso no se clasifica con esqueletos individuales sino con un
modelo de visión sobre frames.

## Instalación

Requiere Python 3.12, [uv](https://docs.astral.sh/uv/) y (para el backend por defecto)
[Ollama](https://ollama.com).

```bash
uv sync                      # base (ollama, mock, tracking)
uv sync --extra anthropic    # + backend Claude
uv sync --extra local        # + backend local (torch, transformers, scikit-learn)
ollama pull qwen3-vl:8b-instruct
```

## CLI

```bash
uv run tatami backends
uv run tatami analyze VIDEO.mp4 --backend ollama --model qwen3-vl:8b-instruct --fps 2 --out outputs/x --render
uv run tatami analyze VIDEO.mp4 --backend mock          # sin modelos
uv run tatami analyze VIDEO.mp4 --crop --pose-model yolo11x-pose.pt
uv run tatami render outputs/x/result.json VIDEO.mp4    # re-renderiza el overlay
uv run tatami schema result                             # JSON Schema (o `labels`)
```

`analyze` escribe `result.json` (y `labeled.mp4` con `--render`) en `--out` (default `outputs/<video>`).

## Python

```python
from tatami_vision import analyze_video, rescore, list_backends

result = analyze_video(
    "match.mp4",
    backend="ollama",
    model="qwen3-vl:8b-instruct",
    sample_fps=2.0,
    progress_cb=lambda p: print(p),
)  # dict AnalysisResult
result = rescore(result, intervals)  # intervalos corregidos (formato dataset) -> recalcula
list_backends()  # [{"name","models","available","default_model","detail"}]
```

Opciones extra de `analyze_video`: `crop=False`, `tile=4` (ollama; 1 = un frame por imagen), `smooth=True`, `smooth_window=5`, `block_size=None`,
`pose_model="yolo11m-pose.pt"`, y los kwargs del backend (`host`, `retries`, `temperature`,
`num_ctx`, ...). `rescore` aplica los intervalos (`end` exclusivo) sobre los frames (confianza 1.0
si `source="human"`); los frames fuera de todo intervalo conservan su etiqueta.

## Backends

| Backend | Modelo por defecto | Notas |
|---|---|---|
| `ollama` | `qwen3-vl:8b-instruct` | Local, vía [Pydantic AI](https://ai.pydantic.dev) sobre la API nativa de Ollama. Mosaicos 2x2 (`tile=4`: 4 frames por imagen), bloques de 8 frames, salida compacta (`"mount:A"` por frame) con structured outputs, contexto `TATAMI_OLLAMA_CONTEXT` (8192), temperatura 0.1, reintentos; un bloque que falla queda `not_visible` con confianza 0. Host: `OLLAMA_HOST`. No uses `qwen3-vl:8b` (variante *Thinking*, horas por combate). Ver [`docs/PERFORMANCE.md`](../docs/PERFORMANCE.md). |
| `anthropic` | `claude-opus-5-5` | Extra `[anthropic]`, lee `ANTHROPIC_API_KEY`. Bloques de 48 frames, streaming y `output_config.format`. |
| `local` | artefacto | Clasificador de `tatami-train`. Directorio con `model.joblib` + `meta.json` (ver abajo). `$TATAMI_LOCAL_MODEL` o `./models/local`; `model=` puede ser una ruta. |
| `mock` | `mock` | Determinista, sin modelos; para tests y desarrollo sin GPU. |

### Formato del artefacto `local`

Definido en `../docs/CONTRACTS.md` (sección "Artefacto de modelo local"); lo produce `tatami-train export`.

Para añadir un backend: subclasea `PositionClassifier` (`backends/base.py`) y regístralo en
`backends/__init__.py`.

## Desarrollo

```bash
uv run pytest
uv run ruff check
```

## Limitaciones

- Los VLM pequeños confunden posiciones y pierden identidad A/B; el suavizado ayuda pero no
  corrige errores sistemáticos. Los resultados deben revisarse (la web permite corregir).
- Puntuación simplificada: sin ventajas, penalizaciones ni sumisiones; solo posiciones estables
  de >= 3 s. Un error de `top` puede dar puntos al atleta equivocado.
- Cámara en mano, público y árbitro degradan la clasificación; `--crop` ayuda pero no es perfecto.
- Muestreo a 2 fps: transiciones rápidas pueden perderse.
- Rendimiento (M5 Pro, `ollama`): ~0.5 s por frame muestreado, unos 5 min para un combate de 5 min a 2 fps.
- Probado en macOS Apple Silicon (MPS); sin CUDA.

## Licencia

AGPL-3.0 (ver `LICENSE`). Usa/depende de Ultralytics YOLO, que es AGPL-3.0.
