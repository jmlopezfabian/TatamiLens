"""Pseudo-etiquetado con un VLM vía tatami-vision (opcional)."""

from __future__ import annotations

import json
from pathlib import Path

from .intervals import LabelFile, intervals_from_analysis, save_label_file


def run_analysis(video: str | Path, backend: str, model: str | None, sample_fps: float) -> dict:
    try:
        from tatami_vision import analyze_video
    except ImportError as e:  # pragma: no cover - depende del entorno
        raise RuntimeError(
            "tatami_vision no está instalado. Instálalo (path dependency ../tatami-vision) o usa "
            "`--from-result analysis.json` con un AnalysisResult ya generado."
        ) from e
    kwargs = {"backend": backend, "sample_fps": sample_fps}
    if model:
        kwargs["model"] = model
    return analyze_video(str(video), **kwargs)


def pseudolabel_video(
    video: str | Path | None,
    out_dir: str | Path,
    backend: str = "ollama",
    model: str | None = "qwen3-vl:8b",
    sample_fps: float = 2.0,
    from_result: str | Path | None = None,
) -> LabelFile:
    if from_result:
        result = json.loads(Path(from_result).read_text())
    elif video:
        result = run_analysis(video, backend, model, sample_fps)
    else:
        raise ValueError("indica un video o --from-result")
    name = Path(video).name if video else None
    lf = intervals_from_analysis(result, source="vlm", video_name=name)
    save_label_file(lf, Path(out_dir) / f"{Path(lf.video).stem}.json")
    return lf
