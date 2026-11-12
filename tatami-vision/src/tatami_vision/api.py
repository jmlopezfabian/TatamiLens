"""Interfaz pública usada por tatami-api (ver docs/CONTRACTS.md)."""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .backends import backend_statuses, get_backend
from .schema import SCHEMA_VERSION, AnalysisResult
from .scoring import compute_segments, estimate_dt, score_timeline, total_score
from .smoothing import smooth_labels
from .video import SampledFrame, iter_frames, read_metadata

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_backends() -> list[dict]:
    """`[{"name","models","available", ...}]`; incluye `default_model` y `detail` extra."""
    return [s.to_dict() for s in backend_statuses()]


def _finalize(result: dict, dt: float, duration: float | None) -> dict:
    frames = result["frames"]
    result["segments"] = compute_segments(frames, dt, duration)
    result["events"] = score_timeline(frames, dt, duration)
    result["score"] = total_score(result["events"])
    return AnalysisResult.model_validate(result).model_dump()


def analyze_video(
    path: str | Path,
    backend: str = "ollama",
    model: str | None = None,
    sample_fps: float = 2.0,
    progress_cb: Callable[[float], None] | None = None,
    *,
    crop: bool = False,
    smooth: bool = True,
    smooth_window: int = 5,
    block_size: int | None = None,
    pose_model: str = "yolo11m-pose.pt",
    **opts: Any,
) -> dict:
    """Analiza un video y devuelve un AnalysisResult (dict). `progress_cb(p)` con p en 0-1.

    `opts` se pasa al constructor del backend (p.ej. host, retries, temperature, num_ctx).
    """
    video = read_metadata(path)
    clf = get_backend(backend, model, block_size=block_size, **opts)
    size = clf.block_size
    tracker = None
    if crop:
        from .tracking import CombatZoneTracker

        tracker = CombatZoneTracker(pose_model)

    total = max(1, int(video.duration_s * sample_fps) + 1)
    labels: list[dict] = []
    athletes: dict[str, str] | None = None
    block: list[SampledFrame] = []

    def flush() -> None:
        nonlocal athletes
        if not block:
            return
        res = clf.classify_block(block, athletes, labels[-4:] or None)
        if athletes is None and res.athletes:
            athletes = res.athletes
        if len(res.labels) != len(block):
            raise RuntimeError("el backend devolvió un número de etiquetas distinto al de frames")
        labels.extend(res.labels)
        block.clear()
        if progress_cb:
            progress_cb(min(1.0, len(labels) / total))

    if progress_cb:
        progress_cb(0.0)
    for fr in iter_frames(path, sample_fps):
        if tracker:
            fr = SampledFrame(fr.t, tracker.crop(fr.image))
        block.append(fr)
        if len(block) >= size:
            flush()
    flush()
    if not labels:
        raise RuntimeError(f"No se pudieron leer frames de {path}")

    if smooth:
        labels = smooth_labels(labels, smooth_window)
    athletes = athletes or {}
    result = {
        "schema_version": SCHEMA_VERSION,
        "video": video.model_dump(),
        "backend": {"name": backend, "model": clf.model, "sample_fps": sample_fps},
        "athletes": {
            "A": {"label": "Atleta A", "description": athletes.get("A", "")},
            "B": {"label": "Atleta B", "description": athletes.get("B", "")},
        },
        "frames": labels,
        "created_at": _now(),
    }
    out = _finalize(result, 1.0 / sample_fps, video.duration_s)
    if progress_cb:
        progress_cb(1.0)
    return out


def rescore(result: dict, intervals: list[dict] | dict) -> dict:
    """Aplica etiquetas corregidas (intervalos, formato dataset) sobre `frames` y recalcula
    segments/events/score. Acepta la lista de intervalos o el archivo de etiquetas completo.
    Los frames fuera de todo intervalo conservan su etiqueta. `end` es exclusivo."""
    from .schema import Interval

    if isinstance(intervals, dict):
        intervals = intervals.get("intervals", [])
    ivs = [Interval.model_validate(i) for i in intervals]
    out = copy.deepcopy(result)
    for f in out["frames"]:
        for iv in ivs:
            if iv.start - 1e-9 <= f["t"] < iv.end - 1e-9:
                f["position"], f["top"] = iv.position, iv.top
                f["confidence"] = 1.0 if iv.source == "human" else max(f["confidence"], 0.5)
    dt = estimate_dt(out["frames"], 1.0 / out["backend"]["sample_fps"])
    return _finalize(out, dt, out["video"].get("duration_s"))
