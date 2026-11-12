"""Export para el backend `local` de tatami-vision y loader de referencia.

Formato del artefacto: ver docs/CONTRACTS.md, sección "Artefacto de modelo local".
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

from .embed import pooling_for
from .labels import NO_TOP_POSITIONS

FORMAT_VERSION = 1


def export_model(run_dir: str | Path, out_dir: str | Path, kind: str = "logreg") -> dict:
    run_dir, out = Path(run_dir), Path(out_dir)
    info = json.loads((run_dir / "run.json").read_text())
    if "position" not in info["targets"] or kind not in info["targets"]["position"]["models"]:
        raise ValueError(f"el run no tiene un modelo 'position' de tipo {kind!r}")
    bundle: dict = {}
    for target in ("position", "top", "top_side"):
        m = info["targets"].get(target, {}).get("models", {}).get(kind)
        bundle[target] = joblib.load(run_dir / m["file"]) if m else None
    meta = {
        "format_version": FORMAT_VERSION,
        "backbone": info["backbone"],
        "pooling": pooling_for(info["backbone"]),
        "labels": info["targets"]["position"]["models"][kind]["labels"],
        "top_labels": (
            info["targets"]["top"]["models"][kind]["labels"] if bundle["top"] is not None else None
        ),
        "sample_fps": info["sample_fps"],
        "model_kind": kind,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    joblib.dump(bundle, out / "model.joblib")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class LocalModel:
    """Loader de referencia: embeddings (N, D) -> predicciones. tatami-vision puede copiarlo."""

    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        self.bundle = joblib.load(d / "model.joblib")

    @property
    def backbone(self) -> str:
        return self.meta["backbone"]

    def predict(self, embeddings: np.ndarray) -> list[dict]:
        """Devuelve [{"position", "top", "confidence"}] por fila.

        `top` sale del modelo `top` si existe (poco fiable, ver README); si la posición es
        standing/scramble/separated/not_visible se fuerza "none".
        """
        proba = self.bundle["position"].predict_proba(embeddings)
        classes = self.bundle["position"].classes_
        tops = self.bundle["top"].predict(embeddings) if self.bundle.get("top") else None
        out = []
        for i, row in enumerate(proba):
            j = int(np.argmax(row))
            pos = str(classes[j])
            top = "none" if pos in NO_TOP_POSITIONS or tops is None else str(tops[i])
            out.append({"position": pos, "top": top, "confidence": float(row[j])})
        return out
