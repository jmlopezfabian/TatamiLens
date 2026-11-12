"""Lectura de metadatos y muestreo de frames."""

from __future__ import annotations

import base64
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .schema import VideoInfo


@dataclass
class SampledFrame:
    t: float
    image: np.ndarray  # BGR


def read_metadata(path: str | Path) -> VideoInfo:
    p = Path(path)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {p}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        n = float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    duration = n / fps if fps > 0 else 0.0
    return VideoInfo(path=p.name, fps=round(fps, 3), duration_s=round(duration, 3), width=w, height=h)


def resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)


def iter_frames(path: str | Path, sample_fps: float) -> Iterator[SampledFrame]:
    """Genera frames a `sample_fps` (lectura secuencial, robusta con códecs con seek malo)."""
    if sample_fps <= 0:
        raise ValueError("sample_fps debe ser > 0")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = fps / sample_fps
    next_idx = 0.0
    i = 0
    try:
        while True:
            if not cap.grab():
                break
            if i >= next_idx - 1e-9:
                ok, img = cap.retrieve()
                if not ok:
                    break
                yield SampledFrame(t=round(i / fps, 2), image=img)
                next_idx += step
            i += 1
    finally:
        cap.release()


def encode_jpeg(img: np.ndarray, width: int | None = 960, quality: int = 85) -> bytes:
    if width:
        img = resize_to_width(img, width)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("No se pudo codificar el frame a JPEG")
    return buf.tobytes()


def encode_jpeg_b64(img: np.ndarray, width: int | None = 960, quality: int = 85) -> str:
    return base64.standard_b64encode(encode_jpeg(img, width, quality)).decode()


def tile_frames(images: list[np.ndarray], cols: int = 2, cell_width: int = 640) -> np.ndarray:
    """Une varios frames en un mosaico (filas de `cols`, izquierda→derecha, arriba→abajo).

    Cada celda lleva su número (1..n) en la esquina superior izquierda; las celdas sobrantes
    quedan en negro. Un mosaico 2x2 de 640 px cuesta lo mismo que un frame suelto en un VLM
    con resolución mínima (Qwen3-VL en Ollama), así que procesa 4 frames por el precio de 1.
    """
    if not images:
        raise ValueError("tile_frames necesita al menos una imagen")
    h0, w0 = images[0].shape[:2]
    cell_h = max(1, int(round(h0 * cell_width / w0)))
    rows = -(-len(images) // cols)
    canvas = np.zeros((rows * cell_h, cols * cell_width, 3), np.uint8)
    for k, img in enumerate(images):
        r, c = divmod(k, cols)
        cell = cv2.resize(img, (cell_width, cell_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(cell, (0, 0), (60, 44), (0, 0, 0), -1)
        cv2.putText(cell, str(k + 1), (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        canvas[r * cell_h : (r + 1) * cell_h, c * cell_width : (c + 1) * cell_width] = cell
    return canvas
