"""Entrenamiento de clasificadores ligeros (logreg / MLP) sobre embeddings.

Objetivos:
- `position`: 13 clases del contrato. Es el objetivo principal y el que funciona bien por frame.
- `top`: A/B/none. ADVERTENCIA: A/B depende de la identidad del atleta, que un frame aislado no
  codifica (quién es A sólo se sabe por tracking/vestimenta). Se entrena por completitud, pero
  espera ~azar entre A y B salvo que cada video tenga atletas distintos y el split no los mezcle.
- `top_side` (alternativa recomendada): left/right/none = en qué lado de la imagen está el top.
  Es visual y relativo, no requiere identidad. Se entrena sólo si los intervalos traen el campo
  opcional `top_side`. La identidad A/B se resuelve luego con tracking (en tatami-vision).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset import load_dataset_meta, load_index
from .embed import DEFAULT_BACKBONE, load_embeddings

TARGETS = ("position", "top", "top_side")
MODELS = ("logreg", "mlp")


def make_model(kind: str, seed: int = 0) -> Pipeline:
    if kind == "logreg":
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    elif kind == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, max_iter=300, random_state=seed)
    else:
        raise ValueError(f"modelo desconocido: {kind!r} (opciones: {MODELS})")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def target_rows(df: pd.DataFrame, target: str) -> pd.Series:
    """Máscara de filas usables para el objetivo."""
    if target == "top_side":
        return df["top_side"].astype(str) != ""
    return pd.Series(True, index=df.index)


def train_run(
    dataset_dir: str | Path,
    out_dir: str | Path,
    backbone: str = DEFAULT_BACKBONE,
    targets: tuple[str, ...] = ("position", "top"),
    models: tuple[str, ...] = MODELS,
    seed: int = 0,
    X: np.ndarray | None = None,
) -> dict:
    """Entrena sobre el split `train`. Guarda `<target>_<model>.joblib` y `run.json` en out_dir."""
    dataset_dir, out = Path(dataset_dir), Path(out_dir)
    df = load_index(dataset_dir)
    X = load_embeddings(dataset_dir, backbone) if X is None else X
    if len(X) != len(df):
        raise ValueError("embeddings y index.csv no están alineados; re-ejecuta `embed`")
    out.mkdir(parents=True, exist_ok=True)
    info: dict = {
        "backbone": backbone,
        "sample_fps": load_dataset_meta(dataset_dir).get("sample_fps"),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": str(dataset_dir),
        "targets": {},
    }
    for target in targets:
        if target not in TARGETS:
            raise ValueError(f"objetivo desconocido: {target!r}")
        mask = (df["split"] == "train") & target_rows(df, target)
        y = df.loc[mask, target].astype(str).to_numpy()
        if len(set(y)) < 2:
            raise ValueError(f"'{target}': se necesitan >=2 clases en train (hay {set(y)})")
        info["targets"][target] = {"n_train": int(mask.sum()), "models": {}}
        for kind in models:
            model = make_model(kind, seed).fit(X[mask.to_numpy()], y)
            fname = f"{target}_{kind}.joblib"
            joblib.dump(model, out / fname)
            info["targets"][target]["models"][kind] = {
                "file": fname,
                "labels": [str(c) for c in model.classes_],
            }
    (out / "run.json").write_text(json.dumps(info, indent=2))
    return info
