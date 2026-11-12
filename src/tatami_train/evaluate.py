"""Métricas: accuracy, F1 macro, por clase, accuracy agrupada y matriz de confusión (PNG)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .dataset import load_index  # noqa: E402
from .embed import load_embeddings  # noqa: E402
from .labels import GROUPS, group_of  # noqa: E402
from .train import target_rows  # noqa: E402


def compute_metrics(y_true, y_pred, labels: list[str]) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "per_class": {
            lab: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, lab in enumerate(labels)
        },
    }


def grouped_accuracy(y_true, y_pred) -> dict:
    """Accuracy tras colapsar posiciones en grupos (guardias juntas, etc.). Sólo para `position`."""
    gt = [group_of(v) for v in y_true]
    gp = [group_of(v) for v in y_pred]
    out = {"overall": float(accuracy_score(gt, gp)) if gt else 0.0, "per_group": {}}
    for name in GROUPS:
        idx = [i for i, g in enumerate(gt) if g == name]
        if idx:
            out["per_group"][name] = float(np.mean([gp[i] == name for i in idx]))
    return out


def plot_confusion(y_true, y_pred, labels: list[str], path: str | Path, title: str = "") -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(labels)), max(4, 0.6 * len(labels))))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black",
                )
    ax.set_xlabel("predicho")
    ax.set_ylabel("real")
    ax.set_title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate_run(
    run_dir: str | Path, split: str = "test", dataset_dir: str | Path | None = None
) -> dict:
    """Evalúa todos los modelos de un run sobre `split`; escribe metrics_<split>.json y PNGs."""
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "run.json").read_text())
    dataset_dir = Path(dataset_dir or info["dataset"])
    df = load_index(dataset_dir)
    X = load_embeddings(dataset_dir, info["backbone"])
    results: dict = {"split": split, "targets": {}}
    for target, tinfo in info["targets"].items():
        mask = ((df["split"] == split) & target_rows(df, target)).to_numpy()
        if not mask.any():
            results["targets"][target] = {"error": f"sin filas en split '{split}'"}
            continue
        y = df.loc[mask, target].astype(str).to_numpy()
        results["targets"][target] = {}
        for kind, minfo in tinfo["models"].items():
            model = joblib.load(run_dir / minfo["file"])
            pred = model.predict(X[mask])
            labels = minfo["labels"]
            m = compute_metrics(y, pred, labels)
            if target == "position":
                m["grouped"] = grouped_accuracy(y, pred)
            png = run_dir / f"confusion_{target}_{kind}_{split}.png"
            plot_confusion(y, pred, labels, png, f"{target} / {kind} / {split}")
            m["confusion_png"] = png.name
            results["targets"][target][kind] = m
    (run_dir / f"metrics_{split}.json").write_text(json.dumps(results, indent=2))
    return results


def format_summary(results: dict) -> str:
    lines = [f"Split: {results['split']}"]
    for target, models in results["targets"].items():
        for kind, m in models.items():
            if kind == "error":
                lines.append(f"  {target}: {m}")
                continue
            extra = f"  grouped_acc={m['grouped']['overall']:.3f}" if "grouped" in m else ""
            lines.append(
                f"  {target:9s} {kind:7s} n={m['n']:5d} acc={m['accuracy']:.3f} "
                f"f1_macro={m['f1_macro']:.3f}{extra}"
            )
    return "\n".join(lines)
