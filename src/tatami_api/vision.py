"""Capa adaptadora hacia tatami-vision.

Intenta importar `tatami_vision`; si no está instalado usa un mock interno determinista que
cumple el contrato (AnalysisResult válido, progreso simulado).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

try:  # pragma: no cover - depende del entorno
    import tatami_vision as _tv

    if not all(hasattr(_tv, n) for n in ("analyze_video", "rescore", "list_backends")):
        _tv = None  # instalado pero aún sin la interfaz del contrato
except ImportError:  # pragma: no cover
    _tv = None

USING_REAL_VISION = _tv is not None

POINTS = {
    "takedown": 2,
    "sweep": 2,
    "guard_pass": 3,
    "knee_on_belly": 2,
    "mount": 4,
    "back_take": 4,
}
NO_TOP = {"standing", "scramble", "separated", "not_visible"}
GUARDS = {"closed_guard", "half_guard", "open_guard"}
PASSED = {"side_control", "north_south", "mount", "knee_on_belly", "back_control"}

# Guion determinista (position, top, duración relativa)
_SCRIPT = [
    ("standing", "none", 3), ("scramble", "none", 1), ("closed_guard", "A", 4),
    ("half_guard", "A", 3), ("side_control", "A", 4), ("mount", "A", 3),
    ("turtle", "A", 2), ("back_control", "A", 3), ("separated", "none", 2),
]  # fmt: skip


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _probe(path: str) -> dict:
    info = {"path": Path(path).name, "fps": 30.0, "duration_s": 30.0, "width": 0, "height": 0}
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            info.update(
                fps=float(fps),
                duration_s=round(n / fps, 3) if n else 30.0,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
        cap.release()
    except Exception:  # noqa: BLE001
        pass
    return info


def _build_result(video: dict, backend: dict, intervals: list[dict], sample_fps: float) -> dict:
    """Construye frames/segments/events/score a partir de intervalos."""
    intervals = sorted(intervals, key=lambda i: i["start"])
    step = 1.0 / sample_fps
    frames = []
    for iv in intervals:
        t = iv["start"]
        while t < iv["end"] - 1e-9:
            frames.append({"t": round(t, 3), "position": iv["position"], "top": iv["top"],
                           "confidence": float(iv.get("confidence", 0.9))})  # fmt: skip
            t += step
    segments: list[dict] = []
    for iv in intervals:
        last = segments[-1] if segments else None
        if last and last["position"] == iv["position"] and last["top"] == iv["top"] \
                and abs(last["end"] - iv["start"]) < 1e-6:  # fmt: skip
            last["end"] = iv["end"]
        else:
            segments.append({"start": iv["start"], "end": iv["end"], "position": iv["position"],
                             "top": iv["top"]})  # fmt: skip
    for s in segments:
        s["stable"] = (s["end"] - s["start"]) >= 3.0
    events: list[dict] = []
    stable = [s for s in segments if s["stable"]]
    prev = None
    for s in stable:
        if prev is not None and s["top"] != "none":
            ath, pos = s["top"], s["position"]
            kind = reason = None
            if prev["position"] == "standing":
                kind, reason = "takedown", "derribo"
            elif prev["position"] in GUARDS and prev["top"] == ath and pos in PASSED:
                kind, reason = "guard_pass", "pasada de guardia"
            elif prev["position"] in GUARDS and prev["top"] not in ("none", ath) \
                    and pos in GUARDS | {"standing"}:  # fmt: skip
                kind, reason = "sweep", "barrida"
            if pos == "mount" and kind != "guard_pass":
                kind, reason = "mount", "montada"
            elif pos == "mount":
                events.append({"t": s["start"], "athlete": ath, "points": 3, "kind": "guard_pass",
                               "reason": reason})  # fmt: skip
                kind, reason = "mount", "montada"
            elif pos == "knee_on_belly":
                kind, reason = "knee_on_belly", "rodilla en el abdomen"
            elif pos == "back_control":
                kind, reason = "back_take", "toma de espalda"
            if kind and not (
                events and events[-1]["t"] == s["start"] and events[-1]["kind"] == kind
            ):
                events.append({"t": s["start"], "athlete": ath, "points": POINTS[kind],
                               "kind": kind, "reason": reason})  # fmt: skip
        prev = s
    score = {"A": 0, "B": 0}
    for e in events:
        score[e["athlete"]] += e["points"]
    return {
        "schema_version": "1",
        "video": video,
        "backend": backend,
        "athletes": {
            "A": {"label": "Atleta A", "description": ""},
            "B": {"label": "Atleta B", "description": ""},
        },  # fmt: skip
        "frames": frames,
        "segments": segments,
        "events": events,
        "score": score,
        "created_at": _now(),
    }


def _mock_analyze(path, backend="mock", model="mock", sample_fps=2.0, progress_cb=None) -> dict:
    video = _probe(str(path))
    dur = float(video["duration_s"] or 30.0)
    total = sum(w for *_, w in _SCRIPT)
    intervals, t = [], 0.0
    for pos, top, w in _SCRIPT:
        end = min(dur, t + dur * w / total)
        if end > t:
            intervals.append({"start": round(t, 3), "end": round(end, 3), "position": pos,
                              "top": top, "confidence": 0.9})  # fmt: skip
        t = end
    steps = 5
    for i in range(steps):
        time.sleep(0.01)
        if progress_cb:
            progress_cb((i + 1) / steps)
    return _build_result(video, {"name": backend, "model": model, "sample_fps": sample_fps},
                         intervals, sample_fps)  # fmt: skip


def _mock_rescore(result: dict, intervals: list[dict]) -> dict:
    sample_fps = (result.get("backend") or {}).get("sample_fps") or 2.0
    new = _build_result(result["video"], result["backend"], intervals, sample_fps)
    new["athletes"] = result.get("athletes", new["athletes"])
    return new


def analyze_video(path, backend="ollama", model="qwen3-vl:8b", sample_fps=2.0,
                  progress_cb: Callable[[float], None] | None = None) -> dict:  # fmt: skip
    if _tv is not None:
        return _tv.analyze_video(str(path), backend=backend, model=model, sample_fps=sample_fps,
                                 progress_cb=progress_cb)  # fmt: skip
    return _mock_analyze(path, backend, model, sample_fps, progress_cb)


def rescore(result: dict, intervals: list[dict]) -> dict:
    if _tv is not None:
        return _tv.rescore(result, intervals)
    return _mock_rescore(result, intervals)


def list_backends() -> list[dict]:
    if _tv is not None:
        return _tv.list_backends()
    return [
        {"name": "mock", "models": ["mock"], "available": True},
        {"name": "ollama", "models": ["qwen3-vl:8b"], "available": False},
        {"name": "anthropic", "models": [], "available": False},
        {"name": "local", "models": [], "available": False},
    ]
