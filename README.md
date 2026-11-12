# tatami-api

Backend HTTP de TatamiLens (FastAPI + SQLModel/SQLite). Contrato: `../docs/CONTRACTS.md`.

## Arranque

```bash
uv sync
uv run tatami-api            # http://localhost:8000  (o: uv run uvicorn tatami_api.main:app --reload --port 8000)
uv run pytest && uv run ruff check .
```

Docs interactivas en `/docs`, esquema OpenAPI en `/openapi.json` (útil para generar tipos en tatami-web).

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `TATAMI_DATA_DIR` | `./data` | Videos (`videos/`), etiquetas (`labels/`) y DB (`tatami.db`) |
| `TATAMI_CORS_ORIGINS` | `http://localhost:5173` | Orígenes permitidos, separados por coma |
| `TATAMI_MAX_CONCURRENT_ANALYSES` | `1` | Análisis simultáneos (el modelo local usa la GPU) |
| `PORT` | `8000` | Puerto de `uv run tatami-api` |

## Endpoints (prefijo `/api`)

`GET /health`, `GET /backends`, `POST /videos` (multipart `file`), `GET /videos`, `GET /videos/{id}`,
`GET /videos/{id}/stream` (HTTP Range), `DELETE /videos/{id}`, `POST /videos/{id}/analyses`,
`GET /videos/{id}/analyses`, `GET /analyses/{id}`, `PUT|GET /videos/{id}/labels`,
`GET /videos/{id}/labels/export` (descarga el JSON para `tatami-train`),
`POST /analyses/{id}/rescore`.

Los análisis corren en un pool de hilos con cola; `status` pasa `queued -> running -> done|error` y
`progress` (0-1) se actualiza en la DB. Hacer polling a `GET /analyses/{id}`. Al reiniciar, los
análisis que estaban `queued/running` quedan en `error`.

Notas: `rescore` guarda el resultado recalculado en el análisis (reemplaza `result`). Códigos:
404 sin etiquetas / no existe, 409 si el análisis no está `done`, 422 backend desconocido o etiquetas
inválidas (`top` debe ser `none` en standing/scramble/separated/not_visible).

## Conexión con otros repos

- **tatami-web**: apuntar a `http://localhost:8000/api`; el `<video src>` usa `/api/videos/{id}/stream`.
- **tatami-vision**: path dependency editable (`../tatami-vision`). `src/tatami_api/vision.py` lo importa
  (`analyze_video`, `rescore`, `list_backends`); `/api/backends` propaga también `default_model` y `detail`.
  Si no está instalado, usa un mock interno determinista (sólo backend `mock`). Los tests corren contra el
  `mock` de tatami-vision (nunca ollama/anthropic).
- **tatami-train**: descargar `labels/export` y usarlo como dataset por intervalos.

## Licencia

AGPL-3.0 (ver `LICENSE`). Usa/depende de Ultralytics YOLO, que es AGPL-3.0.
