# TatamiLens

Análisis automático de combates de jiu-jitsu a partir de video: detección de posiciones,
línea de tiempo del combate y marcador con reglas IBJJF.

## Estructura

| Carpeta | Qué es |
|---|---|
| [`tatami-vision/`](tatami-vision) | Librería y CLI de visión para BJJ: clasificación de posiciones con VLM (Qwen3-VL local por defecto), recorte de la zona de combate con YOLO-pose, suavizado temporal y marcador IBJJF. |
| [`tatami-train/`](tatami-train) | Datasets, etiquetado y entrenamiento de un clasificador local (embeddings DINOv2/CLIP + sklearn). |
| [`tatami-api/`](tatami-api) | Backend FastAPI: videos, cola de análisis, correcciones de etiquetas. |
| [`tatami-web/`](tatami-web) | Frontend React: biblioteca, reproductor con línea de tiempo, marcador y editor de etiquetas. |
| [`docs/CONTRACTS.md`](docs/CONTRACTS.md) | Contratos entre paquetes (etiquetas, formato de resultados, API, modelo local). |

`tatami-api` y `tatami-train` dependen de `tatami-vision` por ruta (`../tatami-vision`).

## Arranque rápido

Requisitos: macOS/Linux, Python 3.12 + [uv](https://docs.astral.sh/uv/), Node 20+,
[Ollama](https://ollama.com) con `ollama pull qwen3-vl:8b-instruct`.

```bash
# API (puerto 8000)
cd tatami-api && uv run tatami-api

# Web (puerto 5173), en otra terminal
cd tatami-web && npm install && npm run dev
```

Abre http://localhost:5173, sube un video y lanza un análisis.
Solo la librería, desde la terminal:

```bash
cd tatami-vision && uv run tatami analyze VIDEO.mp4 --backend ollama --render
```

## Origen

Inspirado en el paper *Video-Based Detection of Combat Positions and Automatic Scoring in
Jiu-jitsu* (Hudovernik & Skočaj, MMSports 2022). No reutiliza su código (licencia CC BY-NC-ND).

## Licencia

AGPL-3.0 (ver `LICENSE`), porque `tatami-vision` usa Ultralytics YOLO (AGPL-3.0).
`tatami-web/` se distribuye bajo MIT (ver `tatami-web/LICENSE`).
