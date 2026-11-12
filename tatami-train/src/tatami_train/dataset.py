"""Extracción de frames y manejo del índice del dataset.

Layout de un dataset:
    dataset/
      frames/<video_stem>/<t_ms>.jpg
      index.csv        columnas: video, t, frame, position, top, top_side, source, split
      dataset.json     {"sample_fps": ..., "max_side": ...}
      embeddings/<backbone>.npy   (lo crea `embed`)
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd

from .intervals import intervals_to_frame_labels, load_label_file
from .split import assign_splits

INDEX_COLUMNS = ["video", "t", "frame", "position", "top", "top_side", "source", "split"]
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def sample_video(path: str | Path, fps: float, max_side: int | None = None):
    """Genera (t, frame_bgr) a `fps` muestras/segundo leyendo el video en secuencia."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"no se pudo abrir el video: {path}")
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    next_t, idx = 0.0, 0
    try:
        while True:
            if not cap.grab():
                break
            t = idx / vfps
            idx += 1
            if t + 1e-9 < next_t:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            if max_side:
                h, w = frame.shape[:2]
                s = max_side / max(h, w)
                if s < 1:
                    frame = cv2.resize(
                        frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA
                    )
            yield t, frame
            next_t += 1.0 / fps
    finally:
        cap.release()


def find_video(videos_dir: Path, name: str) -> Path:
    p = videos_dir / name
    if p.exists():
        return p
    for ext in VIDEO_EXTS:
        q = videos_dir / (Path(name).stem + ext)
        if q.exists():
            return q
    raise FileNotFoundError(f"video {name!r} no encontrado en {videos_dir}")


def extract_dataset(
    videos_dir: str | Path,
    labels_dir: str | Path,
    out_dir: str | Path,
    fps: float = 2.0,
    max_side: int = 512,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> pd.DataFrame:
    """Extrae frames etiquetados a `out_dir` y escribe `index.csv`. Frames sin etiqueta se omiten."""
    videos_dir, labels_dir, out = Path(videos_dir), Path(labels_dir), Path(out_dir)
    rows: list[dict] = []
    label_files = sorted(labels_dir.glob("*.json"))
    if not label_files:
        raise FileNotFoundError(f"no hay archivos .json de etiquetas en {labels_dir}")
    for lf_path in label_files:
        lf = load_label_file(lf_path)
        video_path = find_video(videos_dir, lf.video)
        stem = video_path.stem
        (out / "frames" / stem).mkdir(parents=True, exist_ok=True)
        for t, frame in sample_video(video_path, fps, max_side):
            lab = intervals_to_frame_labels(lf.intervals, [t])[0]
            if lab is None:
                continue
            rel = f"frames/{stem}/{round(t * 1000):08d}.jpg"
            cv2.imwrite(str(out / rel), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append(
                {
                    "video": video_path.name,
                    "t": round(t, 3),
                    "frame": rel,
                    "position": lab["position"],
                    "top": lab["top"],
                    "top_side": lab["top_side"] or "",
                    "source": lab["source"],
                    "split": "train",
                }
            )
    df = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    df = assign_splits(df, val_frac=val_frac, test_frac=test_frac, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "index.csv", index=False)
    (out / "dataset.json").write_text(json.dumps({"sample_fps": fps, "max_side": max_side}))
    return df


def load_index(dataset_dir: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(dataset_dir) / "index.csv", keep_default_na=False)
    return df


def load_dataset_meta(dataset_dir: str | Path) -> dict:
    p = Path(dataset_dir) / "dataset.json"
    return json.loads(p.read_text()) if p.exists() else {}
