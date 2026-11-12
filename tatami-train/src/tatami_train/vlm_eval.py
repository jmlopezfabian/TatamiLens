"""Evaluación de análisis de video (AnalysisResult de tatami-vision) contra etiquetas por intervalos.

Métricas por frame muestreado (solo los frames que caen dentro de un intervalo etiquetado):
- `position_acc` / `position_f1`: posición exacta (11 clases).
- `group_acc`: posición agrupada (de pie / guardias / dominante / sin acción; ver `labels.GROUPS`).
- `top_acc`: quién está arriba, en frames con top etiquetado. Los nombres A/B del modelo son
  arbitrarios, así que se usa la asignación (directa o intercambiada) que más acierta.
- `top_flip_rate`: fracción de frames con top etiquetado en que la predicción usa la asignación
  contraria a la mayoritaria; mide cuánto "baila" la identidad dentro del combate.
- `joint_acc`: posición y top correctos a la vez.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .intervals import LabelFile, intervals_to_frame_labels, load_label_file
from .labels import GROUPS, POSITIONS, group_of

SWAP = {"A": "B", "B": "A", "none": "none"}


def _pairs(result: dict, labels: LabelFile) -> list[tuple[dict, dict]]:
    frames = result["frames"]
    gts = intervals_to_frame_labels(labels.intervals, [f["t"] for f in frames])
    return [(f, g) for f, g in zip(frames, gts, strict=True) if g is not None]


def evaluate_analysis(result: dict, labels: LabelFile) -> dict[str, Any]:
    from sklearn.metrics import f1_score

    pairs = _pairs(result, labels)
    if not pairs:
        raise ValueError("ningún frame del resultado cae dentro de un intervalo etiquetado")
    y_true = [g["position"] for _, g in pairs]
    y_pred = [f["position"] for f, _ in pairs]
    present = [p for p in POSITIONS if p in set(y_true) | set(y_pred)]

    topped = [(f, g) for f, g in pairs if g["top"] != "none"]
    direct = sum(f["top"] == g["top"] for f, g in topped)
    swapped = sum(f["top"] == SWAP[g["top"]] for f, g in topped)
    mapping = "direct" if direct >= swapped else "swapped"
    best, other = (direct, swapped) if mapping == "direct" else (swapped, direct)

    def mapped(top: str) -> str:
        return top if mapping == "direct" else SWAP[top]

    joint = sum(f["position"] == g["position"] and f["top"] == mapped(g["top"]) for f, g in pairs)
    confusions = Counter((t, p) for t, p in zip(y_true, y_pred, strict=True) if t != p)
    n = len(pairs)
    return {
        "n": n,
        "position_acc": sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / n,
        "position_f1": float(
            f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
        ),
        "group_acc": sum(group_of(t) == group_of(p) for t, p in zip(y_true, y_pred, strict=True))
        / n,
        "group_recall": {
            name: _recall([group_of(t) for t in y_true], [group_of(p) for p in y_pred], name)
            for name in GROUPS
            if name in {group_of(t) for t in y_true}
        },
        "top_n": len(topped),
        "top_acc": best / len(topped) if topped else None,
        "top_flip_rate": other / len(topped) if topped else None,
        "top_mapping": mapping,
        "joint_acc": joint / n,
        "pred_distribution": dict(Counter(y_pred).most_common()),
        "top_confusions": [
            {"true": t, "pred": p, "count": c} for (t, p), c in confusions.most_common(5)
        ],
    }


def _recall(true: list[str], pred: list[str], name: str) -> float:
    idx = [i for i, t in enumerate(true) if t == name]
    return sum(pred[i] == name for i in idx) / len(idx)


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{v * 100:.0f}%"


def format_table(rows: list[dict[str, Any]]) -> str:
    """Tabla markdown; cada fila: {"name", "metrics", opcional "s_per_frame"}."""
    head = (
        "| variante | s/frame | posición | F1 macro | grupo | top | baile A/B | pos+top |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    lines = [head]
    for r in rows:
        m = r["metrics"]
        spf = r.get("s_per_frame")
        lines.append(
            f"| {r['name']} | {'–' if spf is None else f'{spf:.2f}'} | {_pct(m['position_acc'])} "
            f"| {m['position_f1']:.2f} | {_pct(m['group_acc'])} | {_pct(m['top_acc'])} "
            f"| {_pct(m['top_flip_rate'])} | {_pct(m['joint_acc'])} |"
        )
    return "\n".join(lines)


def parse_variant(spec: str) -> tuple[str, dict[str, Any]]:
    """`nombre:clave=valor,clave=valor` -> (nombre, kwargs de analyze_video)."""
    name, _, rest = spec.partition(":")
    kwargs: dict[str, Any] = {}
    for item in filter(None, rest.split(",")):
        key, _, raw = item.partition("=")
        value: Any = raw
        for cast in (int, float):
            try:
                value = cast(raw)
                break
            except ValueError:
                continue
        if raw in ("true", "false"):
            value = raw == "true"
        kwargs[key.strip()] = value
    return name, kwargs


def bench_vlm(
    video: str | Path,
    labels_path: str | Path,
    variants: list[str],
    out_dir: str | Path,
    backend: str = "ollama",
) -> list[dict[str, Any]]:
    """Ejecuta `analyze_video` por variante, guarda cada resultado y lo evalúa."""
    from tatami_vision import analyze_video

    labels = load_label_file(labels_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in variants:
        name, kwargs = parse_variant(spec)
        t0 = time.perf_counter()
        result = analyze_video(str(video), backend=backend, **kwargs)
        wall = time.perf_counter() - t0
        (out / f"{name}.json").write_text(json.dumps(result))
        rows.append(
            {
                "name": name,
                "kwargs": kwargs,
                "s_per_frame": wall / len(result["frames"]),
                "metrics": evaluate_analysis(result, labels),
            }
        )
    (out / "bench.json").write_text(json.dumps(rows, indent=2))
    return rows
