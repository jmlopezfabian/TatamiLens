"""Etiquetas por intervalos y conversores (AnalysisResult <-> intervalos <-> etiquetas por frame)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .labels import SOURCES, TOP_SIDES, validate_label


@dataclass
class Interval:
    start: float
    end: float
    position: str
    top: str = "none"
    source: str = "human"
    top_side: str | None = None  # extensión opcional: "left" | "right" | "none"

    def validate(self) -> None:
        if not self.end > self.start >= 0:
            raise ValueError(f"intervalo inválido [{self.start}, {self.end})")
        validate_label(self.position, self.top)
        if self.source not in SOURCES:
            raise ValueError(f"source inválido: {self.source!r}")
        if self.top_side is not None and self.top_side not in TOP_SIDES:
            raise ValueError(f"top_side inválido: {self.top_side!r}")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["top_side"] is None:
            del d["top_side"]
        return d


@dataclass
class LabelFile:
    video: str
    intervals: list[Interval]
    athletes: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"video": self.video}
        if self.athletes:
            d["athletes"] = self.athletes
        d["intervals"] = [i.to_dict() for i in self.intervals]
        return d


def validate_intervals(intervals: list[Interval]) -> None:
    """Valida cada intervalo y que no haya solapes (tolerancia 1 ms)."""
    for iv in intervals:
        iv.validate()
    ordered = sorted(intervals, key=lambda i: i.start)
    for a, b in zip(ordered, ordered[1:], strict=False):
        if b.start < a.end - 1e-3:
            raise ValueError(f"intervalos solapados: [{a.start}, {a.end}) y [{b.start}, {b.end})")


def load_label_file(path: str | Path) -> LabelFile:
    data = json.loads(Path(path).read_text())
    intervals = [
        Interval(
            start=float(i["start"]),
            end=float(i["end"]),
            position=i["position"],
            top=i.get("top", "none"),
            source=i.get("source", "human"),
            top_side=i.get("top_side"),
        )
        for i in data["intervals"]
    ]
    validate_intervals(intervals)
    return LabelFile(video=data["video"], intervals=intervals, athletes=data.get("athletes"))


def save_label_file(label_file: LabelFile, path: str | Path) -> None:
    validate_intervals(label_file.intervals)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(label_file.to_dict(), indent=2, ensure_ascii=False) + "\n")


def merge_adjacent(intervals: list[Interval], gap: float = 1e-3) -> list[Interval]:
    """Une intervalos contiguos con la misma etiqueta y fuente."""
    out: list[Interval] = []
    for iv in sorted(intervals, key=lambda i: i.start):
        last = out[-1] if out else None
        if (
            last
            and abs(iv.start - last.end) <= gap
            and (iv.position, iv.top, iv.source, iv.top_side)
            == (last.position, last.top, last.source, last.top_side)
        ):
            last.end = iv.end
        else:
            out.append(Interval(**asdict(iv)))
    return out


def intervals_from_analysis(
    result: dict[str, Any], source: str = "vlm", video_name: str | None = None
) -> LabelFile:
    """Convierte un AnalysisResult (dict, ver CONTRACTS.md) a intervalos.

    Usa `frames` (cada muestra cubre hasta la siguiente, o 1/sample_fps la última) y,
    si no hay frames, cae a `segments`.
    """
    video = result.get("video", {})
    duration = video.get("duration_s")
    frames = sorted(result.get("frames", []), key=lambda f: f["t"])
    intervals: list[Interval] = []
    if frames:
        sample_fps = (result.get("backend") or {}).get("sample_fps") or 1.0
        dt = 1.0 / float(sample_fps)
        for k, f in enumerate(frames):
            end = frames[k + 1]["t"] if k + 1 < len(frames) else f["t"] + dt
            if duration is not None:
                end = min(end, float(duration))
            if end <= f["t"]:
                continue
            intervals.append(
                Interval(f["t"], end, f["position"], f.get("top", "none"), source=source)
            )
        intervals = merge_adjacent(intervals)
    else:
        for s in result.get("segments", []):
            intervals.append(
                Interval(s["start"], s["end"], s["position"], s.get("top", "none"), source=source)
            )
    athletes = {
        k: v.get("description") or v.get("label", "")
        for k, v in (result.get("athletes") or {}).items()
    }
    name = video_name or Path(video.get("path", "unknown")).name
    return LabelFile(video=name, intervals=intervals, athletes=athletes or None)


def label_at(intervals: list[Interval], t: float) -> Interval | None:
    """Intervalo [start, end) que contiene t (None si no hay etiqueta)."""
    for iv in intervals:
        if iv.start <= t < iv.end:
            return iv
    return None


def intervals_to_frame_labels(
    intervals: list[Interval], times: list[float]
) -> list[dict[str, Any] | None]:
    """Etiqueta por frame para cada tiempo en `times` (None = sin etiqueta)."""
    ordered = sorted(intervals, key=lambda i: i.start)
    out: list[dict[str, Any] | None] = []
    for t in times:
        iv = label_at(ordered, t)
        out.append(
            None
            if iv is None
            else {
                "t": t,
                "position": iv.position,
                "top": iv.top,
                "top_side": iv.top_side,
                "source": iv.source,
            }
        )
    return out
