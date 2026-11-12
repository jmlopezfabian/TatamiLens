# tatami-train

Datasets, etiquetado y entrenamiento para clasificar **posiciones de BJJ** en video (parte de TatamiLens).
Fuente de verdad de etiquetas y formatos: `../docs/CONTRACTS.md`.

Enfoque: los esqueletos individuales fallan cuando los dos cuerpos se funden, así que se clasifica el
**frame completo** con embeddings de un modelo de visión preentrenado (DINOv2 small por defecto, o CLIP)
y un clasificador ligero (regresión logística / MLP de scikit-learn). Un VLM pre-etiqueta, tú corriges
intervalos, y con eso se entrena un clasificador local, rápido y gratis.

## Instalación

```bash
cd tatami-train
uv sync
uv run tatami-train --help
uv run pytest            # tests sintéticos, sin descargar modelos
uv run pytest -m slow    # además prueba el backbone real (descarga DINOv2 de HF)
```

`labels.py` replica las constantes del contrato. TODO: unificar con `tatami_vision.schema` y declarar
`tatami-vision` como path dependency (`../tatami-vision`) cuando exista.

## Flujo completo

```bash
# 1. Pseudo-etiquetar con un VLM (requiere tatami-vision; o usa --from-result con un AnalysisResult JSON)
uv run tatami-train pseudolabel videos/combate1.mp4 --out labels/ --backend ollama --model qwen3-vl:8b
uv run tatami-train pseudolabel --from-result analysis.json --out labels/

# 2. Corregir a mano los JSON de labels/ (o desde tatami-web / GET /api/videos/{id}/labels/export).
#    Cambia "source" a "human" en lo que corrijas.

# 3. Extraer frames + split por video
uv run tatami-train extract --videos videos/ --labels labels/ --out datasets/v1 --fps 2

# 4. Embeddings (MPS automático; cache en datasets/v1/embeddings/*.npy)
uv run tatami-train embed datasets/v1 --backbone facebook/dinov2-small
#    alternativa: --backbone openai/clip-vit-base-patch32

# 5. Entrenar (logreg + MLP; position y top)
uv run tatami-train train datasets/v1 --out runs/dino-v1

# 6. Evaluar (val o test): accuracy, F1 macro, por clase, accuracy agrupada, PNG de confusión
uv run tatami-train eval runs/dino-v1 --split val

# 7. Exportar para tatami-vision (backend `local`)
uv run tatami-train export runs/dino-v1 --out models/local-v1 --model logreg
```

### Formato de etiquetas (un JSON por video, igual que el contrato)

```json
{"video": "insta2.mp4",
 "intervals": [{"start": 0.0, "end": 12.5, "position": "standing", "top": "none", "source": "human"}]}
```

Intervalos semiabiertos `[start, end)`, sin solapes; `standing/scramble/separated/not_visible` exigen
`top: "none"`. Los frames fuera de todo intervalo se omiten. Se valida al cargar.

### Dataset en disco

`frames/<video>/<t_ms>.jpg`, `index.csv` (`video, t, frame, position, top, top_side, source, split`),
`dataset.json` (`sample_fps`), `embeddings/<backbone>.npy` (+ `.frames.json`, filas alineadas con el índice).

### Split por video

Siempre por video, nunca por frame (los frames consecutivos son casi idénticos y se filtraría información).
1 video: todo en train (aviso); 2: train+val; 3 o más: train/val/test con al menos un video en cada uno.
`tatami-train split datasets/v1 --seed 3` re-asigna. Con pocos videos el val/test es muy ruidoso.

## `top` (A/B) desde un frame: decisión

**A/B depende de la identidad** del atleta, y un frame aislado no la contiene (A es A por todo el video,
no por cómo se ve en este frame). Por eso:

- `position` es el objetivo principal y fiable.
- `top` (A/B/none) se entrena y exporta, pero **espere rendimiento cercano al azar entre A y B**. Se fuerza
  `none` en las posiciones sin top. No lo use como verdad; es sólo pista.
- **Alternativa recomendada: `top_side`** (`left`/`right`/`none`): lado de la imagen donde está el top. Es
  visual y relativo. Requiere un campo opcional `top_side` en los intervalos (propuesta de extensión al
  contrato; es ignorado si no existe). Entrenar: `--targets position top_side`. La identidad A/B se asigna
  después con tracking/apariencia en tatami-vision (p. ej. quién estaba a la izquierda en el frame anterior).

## Cuánta data hace falta (orientativo)

- Mínimo para una prueba de concepto: ~5 videos / ~10 min etiquetados, 2 fps (~1200 frames), con las
  posiciones comunes representadas.
- Razonable: 30-60 min de combates variados (distintos atletas, cámaras, iluminación) y ≥100-200 frames por
  posición frecuente. Posiciones raras (north_south, knee_on_belly, turtle) necesitan al menos ~50.
- Lo que más importa es la diversidad de videos, no los frames: 20 videos cortos > 2 largos.
- Empieza con `logreg`; el MLP rara vez gana con poca data. Mira la matriz de confusión: guardias entre sí
  (`closed/half/open`) se confunden mucho; la accuracy agrupada (`grouped`) lo cuantifica.

## Formato de export (para tatami-vision, backend `local`)

El formato del artefacto (`model.joblib` + `meta.json`) está definido una sola vez en
`../docs/CONTRACTS.md`, sección "Artefacto de modelo local". `tatami-train export` lo produce y
`tatami-vision` (backend `local`) lo carga.

Uso desde tatami-vision (referencia: `tatami_train.export.LocalModel` y `tatami_train.embed.build_embedder`):

```python
from tatami_train.embed import build_embedder
from tatami_train.export import LocalModel
from PIL import Image

model = LocalModel("models/local-v1")
embed = build_embedder(model.backbone)           # una vez
vec = embed([Image.open("frame.jpg").convert("RGB")])
model.predict(vec)   # [{"position": "mount", "top": "none", "confidence": 0.83}]
```

El backend `local` muestrea a `meta["sample_fps"]`, calcular embeddings por lotes y producir los
`frames` del `AnalysisResult`. Mismo loader con sólo `joblib` + `transformers` si se prefiere no depender de
este paquete (las dependencias sklearn deben tener versión compatible con la usada al entrenar).

## Estructura

```
src/tatami_train/  labels.py intervals.py dataset.py split.py embed.py train.py
                   evaluate.py export.py pseudolabel.py cli.py
tests/             datos sintéticos (cv2/numpy) y embeddings falsos
```

Pendiente / ideas: recorte de la zona de combate antes de embeber, modelos con contexto temporal
(ventanas de embeddings), suavizado de predicciones (HMM/mediana) en tatami-vision.

## Licencia

AGPL-3.0 (ver `LICENSE`). Usa/depende de Ultralytics YOLO, que es AGPL-3.0.
