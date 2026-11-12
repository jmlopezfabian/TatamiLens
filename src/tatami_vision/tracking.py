"""Detección de la "zona de combate" con YOLO-pose para recortar frames.

En el suelo los dos atletas suelen salir como UNA sola caja, y hay árbitro y público; por eso
no se clasifican esqueletos: solo se usa la detección de personas para encuadrar el combate
(caja grande, ancha/caída y baja en la imagen) y descartar árbitro/público.
"""

from __future__ import annotations

import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def select_combat_boxes(boxes: list[Box], width: int, height: int) -> list[Box]:
    """Elige la caja principal y las que están pegadas a ella (el otro atleta si salió aparte)."""
    if not boxes:
        return []

    def score(b: Box) -> float:
        w, h = b[2] - b[0], b[3] - b[1]
        ground_bonus = 1.5 if h > 0 and w / h > 0.9 else 1.0  # caídos: cajas anchas
        low = 0.5 + 0.5 * min(1.0, b[3] / height)  # más abajo en la imagen = más cerca
        return _area(b) * ground_bonus * low

    best = max(boxes, key=score)
    bw, bh = best[2] - best[0], best[3] - best[1]
    zone = (best[0] - 0.25 * bw, best[1] - 0.25 * bh, best[2] + 0.25 * bw, best[3] + 0.25 * bh)
    chosen = [best]
    for b in boxes:
        if b is best:
            continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3] and _area(b) >= 0.25 * _area(best):
            chosen.append(b)
    return chosen


def union_box(boxes: list[Box]) -> Box:
    a = np.array(boxes, dtype=float)
    return (a[:, 0].min(), a[:, 1].min(), a[:, 2].max(), a[:, 3].max())


def expand_to_crop(
    box: Box, width: int, height: int, margin: float = 0.2, min_frac: float = 0.4
) -> tuple[int, int, int, int]:
    """Expande con margen, impone tamaño mínimo (fracción del frame) y limita a la imagen."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w = min(float(width), max((box[2] - box[0]) * (1 + 2 * margin), min_frac * width))
    h = min(float(height), max((box[3] - box[1]) * (1 + 2 * margin), min_frac * height))
    x1 = min(max(0.0, cx - w / 2), width - w)  # desplaza la ventana en vez de recortarla
    y1 = min(max(0.0, cy - h / 2), height - h)
    return int(x1), int(y1), int(x1 + w), int(y1 + h)


class CombatZoneTracker:
    """Recorta cada frame alrededor de la zona de combate, con suavizado temporal (EMA)."""

    def __init__(
        self,
        model_path: str = "yolo11m-pose.pt",
        device: str | None = None,
        margin: float = 0.2,
        alpha: float = 0.5,
        conf: float = 0.25,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.device = device or _default_device()
        self.margin, self.alpha, self.conf = margin, alpha, conf
        self._crop: np.ndarray | None = None

    def crop(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        res = self.model.predict(img, device=self.device, conf=self.conf, verbose=False)[0]
        boxes: list[Box] = []
        if res.boxes is not None and len(res.boxes):
            boxes = [tuple(map(float, b)) for b in res.boxes.xyxy.cpu().numpy()]
        chosen = select_combat_boxes(boxes, w, h)
        if chosen:
            new = np.array(expand_to_crop(union_box(chosen), w, h, self.margin), dtype=float)
            self._crop = new if self._crop is None else self.alpha * new + (1 - self.alpha) * self._crop
        if self._crop is None:
            return img
        x1, y1, x2, y2 = (int(v) for v in self._crop)
        return img[y1:y2, x1:x2]


def _default_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
