# TatamiLens — contratos compartidos

Este documento es la fuente de verdad entre paquetes. Si un paquete necesita cambiar algo aquí,
debe hacerse primero en este archivo.

## Paquetes (carpetas del monorepo)

| Carpeta | Qué es | Stack |
|---|---|---|
| `tatami-vision/` | Librería + CLI de visión para BJJ: muestreo de video, pose/tracking, clasificación de posiciones (backends intercambiables), motor de puntuación IBJJF. Produce `AnalysisResult`. | Python 3.12, uv, ultralytics, opencv, ollama, anthropic (opcional) |
| `tatami-train/` | Datasets y entrenamiento: formato de etiquetas por intervalos, extracción de frames, pseudo-etiquetado con VLM, embeddings (DINOv2/CLIP), entrenamiento de clasificador local, evaluación. Exporta modelos que `tatami-vision` carga con el backend `local`. | Python 3.12, uv, torch (MPS), transformers, scikit-learn |
| `tatami-api/` | Backend HTTP: subir videos, lanzar análisis en background, guardar resultados, correcciones de etiquetas, exportar dataset. | Python 3.12, uv, FastAPI, SQLite (SQLModel) |
| `tatami-web/` | Frontend: subir videos, ver análisis con reproductor sincronizado a una línea de tiempo de posiciones, marcador, corregir etiquetas. | React + Vite + TypeScript |

Dependencias: `tatami-api` depende de `tatami-vision` como path dependency (`../tatami-vision`).
`tatami-train` depende de `tatami-vision` (path) para reutilizar schema y muestreo.

Entorno objetivo: macOS, Apple Silicon (M5 Pro, 24 GB), torch con MPS. Sin CUDA.

## Etiquetas de posición

```
standing        ambos de pie
scramble        transición sin posición clara
closed_guard    top = el que está dentro de la guardia (pasador)
half_guard      top = pasador
open_guard      top = pasador (incluye butterfly, DLR, sentado, etc.)
side_control    top = quien controla
north_south     top = quien controla
knee_on_belly   top = quien pone la rodilla
mount           top = quien monta
back_control    top = quien tiene la espalda
turtle          top = quien ataca la tortuga
separated       separados / reinicio / fuera del área
not_visible     tapados o fuera de cuadro
```

`top` ∈ {`A`, `B`, `none`}. Para `standing`, `scramble`, `separated`, `not_visible` siempre `none`.
A y B son los dos competidores; la identidad debe mantenerse en todo el video.

## AnalysisResult (JSON)

Lo produce `tatami-vision`, lo guarda/sirve `tatami-api`, lo consume `tatami-web`.

```json
{
  "schema_version": "1",
  "video": {"path": "insta2.mp4", "fps": 29.97, "duration_s": 47.8, "width": 1280, "height": 720},
  "backend": {"name": "ollama", "model": "qwen3-vl:8b-instruct", "sample_fps": 2.0},
  "athletes": {"A": {"label": "Atleta A", "description": "..."}, "B": {"label": "Atleta B", "description": "..."}},
  "frames": [
    {"t": 0.0, "position": "standing", "top": "none", "confidence": 0.9}
  ],
  "segments": [
    {"start": 0.0, "end": 12.5, "position": "standing", "top": "none", "stable": true}
  ],
  "events": [
    {"t": 15.0, "athlete": "A", "points": 2, "kind": "takedown", "reason": "derribo"}
  ],
  "score": {"A": 2, "B": 0},
  "created_at": "2026-11-11T22:00:00Z"
}
```

- `confidence` es float 0–1.
- `segments`: frames consecutivos con la misma (position, top); `stable` = duración ≥ 3 s.
- `events.kind` ∈ {`takedown`, `sweep`, `guard_pass`, `knee_on_belly`, `mount`, `back_take`}.
- Puntos IBJJF: takedown 2, sweep 2, guard_pass 3, knee_on_belly 2, mount 4, back_take 4.

## Formato de etiquetas (dataset)

Etiquetas humanas o corregidas, por intervalos. Un archivo JSON por video:

```json
{
  "video": "insta2.mp4",
  "athletes": {"A": "descripción", "B": "descripción"},
  "intervals": [
    {"start": 0.0, "end": 12.5, "position": "standing", "top": "none", "source": "human"}
  ]
}
```

`source` ∈ {`human`, `vlm`, `model`}.

Campo opcional `top_side` ∈ {`left`, `right`, `none`} en cada intervalo: lado de la imagen donde está el atleta de arriba. Es visual y relativo (un frame aislado no contiene la identidad A/B); la identidad A/B se resuelve en `tatami-vision` con tracking/contexto temporal.

## Artefacto de modelo local

Lo exporta `tatami-train export` y lo carga el backend `local` de `tatami-vision`
(`model=<directorio>`, o `$TATAMI_LOCAL_MODEL`, o `./models/local`). Directorio con dos archivos:

- `model.joblib`: dict `{"position": Pipeline, "top": Pipeline|None, "top_side": Pipeline|None}`.
  Cada Pipeline es scikit-learn (`StandardScaler` + clasificador) con `predict_proba` y `classes_`;
  `position` clasifica las 13 posiciones, `top` predice A/B/none y `top_side` left/right/none
  (este último aún no lo usa `tatami-vision`). Entrada: el embedding del frame.
- `meta.json`: `{"format_version": 1, "backbone": "facebook/dinov2-small", "pooling": "cls+mean",
  "labels": [...], "top_labels": [...]|null, "sample_fps": 2.0, "model_kind": "logreg",
  "created_at": "..."}`.

Embedding (idéntico en ambos repos): imagen RGB, `AutoImageProcessor` del `backbone`, y `pooling`:
`cls+mean` (DINOv2: CLS concatenado con la media de parches; 768 dims en `dinov2-small`) o `clip`
(`get_image_features`). Si `pooling` falta se deduce del backbone (`clip` en el nombre -> `clip`, si no
`cls+mean`). La predicción de `position` usa argmax de `predict_proba`; `top` sale del modelo `top` y se
fuerza a `none` en standing/scramble/separated/not_visible. `format_version` es obligatorio: si falta o no
está soportado, `tatami-vision` falla con un error claro. Versión actual: 1.

## API REST (`tatami-api`, puerto 8000, prefijo `/api`)

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/health` | `{"status":"ok"}` |
| POST | `/api/videos` | multipart `file` → `Video` (201) |
| GET | `/api/videos` | lista de `Video` |
| GET | `/api/videos/{id}` | `Video` |
| GET | `/api/videos/{id}/stream` | archivo de video (soporta Range) |
| DELETE | `/api/videos/{id}` | borra video y análisis (204) |
| POST | `/api/videos/{id}/analyses` | body `{"backend":"ollama","model":"qwen3-vl:8b-instruct","sample_fps":2.0}` → `Analysis` (201, status `queued`) |
| GET | `/api/analyses/{id}` | `Analysis` (incluye `result: AnalysisResult` cuando `status=done`) |
| GET | `/api/videos/{id}/analyses` | lista de `Analysis` (incluye `result` si `done`) |
| PUT | `/api/videos/{id}/labels` | guarda etiquetas corregidas (formato dataset) → devuelve las etiquetas guardadas |
| GET | `/api/videos/{id}/labels` | etiquetas corregidas o 404 |
| GET | `/api/videos/{id}/labels/export` | descarga del JSON de etiquetas (para `tatami-train`) |
| POST | `/api/analyses/{id}/rescore` | recalcula `events`/`score` a partir de las etiquetas corregidas → `AnalysisResult` (sobrescribe el resultado guardado). 404 si el video no tiene etiquetas; 409 si el análisis no está `done` |
| GET | `/api/backends` | backends disponibles y modelos: `[{"name":"ollama","models":["qwen3-vl:8b-instruct"],"available":true}, ...]` |

```ts
Video    = { id: string, filename: string, duration_s: number|null, fps: number|null, created_at: string }
Analysis = { id: string, video_id: string, status: "queued"|"running"|"done"|"error",
             progress: number /*0-1*/, backend: string, model: string, error: string|null,
             created_at: string, result: AnalysisResult|null }
```

Fechas en ISO 8601 con zona horaria (UTC). Errores de validación → 422.

CORS habilitado para `http://localhost:5173` (Vite).

## Interfaz Python de `tatami-vision` (lo que usa la API)

```python
from tatami_vision import analyze_video, rescore, list_backends
result: dict = analyze_video(path, backend="ollama", model="qwen3-vl:8b-instruct", sample_fps=2.0,
                             progress_cb=lambda p: ...)   # p: float 0–1; devuelve AnalysisResult (dict)
result = rescore(result, intervals)   # recalcula segments/events/score con etiquetas corregidas
backends: list[dict] = list_backends()
```

Backends de clasificación: `ollama` (default, local), `anthropic` (Claude, requiere API key), `local` (clasificador entrenado con `tatami-train`), `mock` (determinista, para tests y desarrollo sin modelos).

## Licencias

No copiar código del repo `ValterH/automatic-positions-detection-and-scoring-in-jiu-jitsu`
(licencia CC BY-NC-ND 4.0). Las reglas de puntuación IBJJF se implementan desde cero.
