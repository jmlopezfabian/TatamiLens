# TatamiLens

Análisis automático de combates de jiu-jitsu a partir de video: detección de posiciones,
línea de tiempo del combate y marcador con reglas IBJJF.

## Repos

- [`tatami-vision`](https://github.com/jmlopezfabian/tatami-vision) — librería y CLI de visión computacional para BJJ
- [`tatami-train`](https://github.com/jmlopezfabian/tatami-train) — datasets, etiquetado y entrenamiento de clasificadores
- [`tatami-api`](https://github.com/jmlopezfabian/tatami-api) — backend FastAPI
- [`tatami-web`](https://github.com/jmlopezfabian/tatami-web) — frontend React

Contratos entre repos: [`docs/CONTRACTS.md`](docs/CONTRACTS.md).

## Origen

Inspirado en el paper *Video-Based Detection of Combat Positions and Automatic Scoring in
Jiu-jitsu* (Hudovernik & Skočaj, MMSports 2022). No reutiliza su código (licencia CC BY-NC-ND).
Para desarrollo local, clona los cuatro repos dentro de esta carpeta (están en `.gitignore`).

## Licencias

- `tatami-vision`, `tatami-train`, `tatami-api`: AGPL-3.0 (dependen de Ultralytics YOLO, AGPL-3.0).
- `tatami-web` y este repo raíz: MIT.
