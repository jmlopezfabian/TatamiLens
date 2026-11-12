import json
from pathlib import Path

import cv2
import numpy as np
import pytest

# posición -> color BGR sintético (el "embedding falso" es el color medio)
COLORS = {"standing": (200, 30, 30), "closed_guard": (30, 200, 30), "mount": (30, 30, 200)}
TOPS = {"standing": "none", "closed_guard": "A", "mount": "B"}


def write_video(path: Path, schedule: list[tuple[str, float]], fps: int = 10) -> None:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (96, 64))
    for pos, dur in schedule:
        frame = np.full((64, 96, 3), COLORS[pos], np.uint8)
        for _ in range(int(dur * fps)):
            w.write(frame)
    w.release()


def fake_embed(images) -> np.ndarray:
    return np.stack([np.asarray(im, dtype=np.float32).mean(axis=(0, 1)) for im in images])


@pytest.fixture
def synthetic(tmp_path):
    """4 videos de 6 s (3 posiciones, 2 s cada una, orden rotado) + etiquetas por intervalos."""
    videos, labels = tmp_path / "videos", tmp_path / "labels"
    videos.mkdir()
    labels.mkdir()
    names = list(COLORS)
    for k in range(4):
        order = names[k % 3 :] + names[: k % 3]
        write_video(videos / f"v{k}.mp4", [(p, 2.0) for p in order])
        intervals = [
            {
                "start": 2.0 * i,
                "end": 2.0 * (i + 1),
                "position": p,
                "top": TOPS[p],
                "source": "human",
            }
            for i, p in enumerate(order)
        ]
        (labels / f"v{k}.json").write_text(
            json.dumps({"video": f"v{k}.mp4", "intervals": intervals})
        )
    return videos, labels, tmp_path / "ds"
