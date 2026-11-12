"""Suavizado temporal de etiquetas (filtro de moda ponderada por confianza)."""

from __future__ import annotations

from collections import defaultdict


def smooth_labels(frames: list[dict], window: int = 5) -> list[dict]:
    """Devuelve copias de `frames` con (position, top) sustituidos por la moda ponderada
    en una ventana centrada de `window` frames (impar). Los frames con confianza 0 (fallos)
    no votan y no se modifican. Los empates conservan la etiqueta original."""
    if window <= 1 or not frames:
        return [dict(f) for f in frames]
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = []
    n = len(frames)
    for i, f in enumerate(frames):
        g = dict(f)
        if f.get("confidence", 1.0) <= 0:
            out.append(g)
            continue
        votes: dict[tuple[str, str], float] = defaultdict(float)
        for j in range(max(0, i - half), min(n, i + half + 1)):
            c = frames[j].get("confidence", 1.0)
            if c <= 0:
                continue
            votes[(frames[j]["position"], frames[j]["top"])] += c
        own = (f["position"], f["top"])
        best = max(votes.values())
        if votes.get(own, 0.0) < best - 1e-9:
            # elige la ganadora; entre empatadas la más cercana al frame actual
            cands = [k for k, v in votes.items() if abs(v - best) < 1e-9]
            pos, top = cands[0]
            g["position"], g["top"] = pos, top
        out.append(g)
    return out
