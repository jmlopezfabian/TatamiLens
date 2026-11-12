"""Video con overlay: posición, top, marcador y último evento."""

from __future__ import annotations

import bisect
from collections.abc import Callable
from pathlib import Path

import cv2


def render_video(
    result: dict,
    video_path: str | Path,
    out_path: str | Path,
    progress_cb: Callable[[float], None] | None = None,
) -> Path:
    frames = result["frames"]
    events = result["events"]
    times = [f["t"] for f in frames]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    scale = max(0.5, h / 720)
    i = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            t = i / fps
            if frames:
                lab = frames[max(0, bisect.bisect_right(times, t) - 1)]
                top = f" (arriba: {lab['top']})" if lab["top"] != "none" else ""
                text = f"{lab['position']}{top}  [{lab['confidence']:.2f}]"
            else:
                text = ""
            a = sum(e["points"] for e in events if e["athlete"] == "A" and e["t"] <= t)
            b = sum(e["points"] for e in events if e["athlete"] == "B" and e["t"] <= t)
            bar = int(90 * scale)
            cv2.rectangle(img, (0, 0), (w, bar), (0, 0, 0), -1)
            cv2.putText(img, text, (20, int(38 * scale)), 0, 1.1 * scale, (255, 255, 255), 2)
            cv2.putText(img, f"A {a}  -  {b} B", (20, int(78 * scale)), 0, 1.1 * scale, (0, 220, 255), 2)
            recent = [e for e in events if 0 <= t - e["t"] < 3]
            if recent:
                e = recent[-1]
                cv2.putText(
                    img, f"+{e['points']} {e['athlete']}: {e['reason']}",
                    (w // 2, int(78 * scale)), 0, 1.0 * scale, (0, 255, 0), 2,
                )  # fmt: skip
            writer.write(img)
            i += 1
            if progress_cb and i % 30 == 0:
                progress_cb(min(1.0, i / n))
    finally:
        writer.release()
        cap.release()
    if progress_cb:
        progress_cb(1.0)
    return out_path
