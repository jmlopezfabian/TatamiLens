# TatamiLens

Análisis automático de combates de jiu-jitsu a partir de video: detección de posiciones,
línea de tiempo del combate y marcador con reglas IBJJF.

## Repos

- [`tatami-vision/`](tatami-vision) — librería y CLI de visión computacional para BJJ
- [`tatami-train/`](tatami-train) — datasets, etiquetado y entrenamiento de clasificadores
- [`tatami-api/`](tatami-api) — backend FastAPI
- [`tatami-web/`](tatami-web) — frontend React

Contratos entre repos: [`docs/CONTRACTS.md`](docs/CONTRACTS.md).

## Origen

Inspirado en el paper *Video-Based Detection of Combat Positions and Automatic Scoring in
Jiu-jitsu* (Hudovernik & Skočaj, MMSports 2022). No reutiliza su código (licencia CC BY-NC-ND).
El prototipo inicial vive en `../bjj-scoring/`.
