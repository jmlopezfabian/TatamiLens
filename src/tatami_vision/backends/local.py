"""Backend local: clasificador entrenado con tatami-train.

Formato del artefacto (directorio):
  model.joblib   estimador scikit-learn con predict_proba; `classes_` son strings
                 "<position>|<top>", p.ej. "side_control|A", "standing|none".
  meta.json      {"format_version": 1,
                  "embedding": {"backend": "hf", "model": "facebook/dinov2-small",
                                "pooling": "cls" | "mean", "image_size": 224},
                  "labels": "position|top",
                  "name": "opcional"}
Los embeddings (float32) de cada frame, extraídos con el extractor de meta.json, se pasan
a `predict_proba`. Directorio por defecto: $TATAMI_LOCAL_MODEL o ./models/local.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..video import SampledFrame
from .base import BackendStatus, BlockResult, PositionClassifier, normalize_frame

FORMAT_VERSION = 1


def default_artifact_dir() -> Path:
    return Path(os.environ.get("TATAMI_LOCAL_MODEL", "models/local"))


def artifact_ok(d: Path) -> bool:
    return (d / "model.joblib").is_file() and (d / "meta.json").is_file()


class Embedder:
    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if cfg.get("backend", "hf") != "hf":
            raise ValueError(f"extractor no soportado: {cfg.get('backend')}")
        self.torch = torch
        self.pooling = cfg.get("pooling", "cls")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(cfg["model"])
        self.model = AutoModel.from_pretrained(cfg["model"]).to(self.device).eval()

    def __call__(self, images_bgr: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        pil = [Image.fromarray(im[:, :, ::-1]) for im in images_bgr]
        inputs = self.proc(images=pil, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            h = self.model(**inputs).last_hidden_state
        v = h[:, 0] if self.pooling == "cls" else h.mean(dim=1)
        return v.float().cpu().numpy()


class LocalClassifier(PositionClassifier):
    name = "local"
    default_model = None
    block_size = 32
    image_width = 448

    def __init__(self, model: str | None = None, block_size=None, **opts):
        super().__init__(model, **opts)
        import joblib

        self.dir = Path(model) if model else default_artifact_dir()
        if not artifact_ok(self.dir):
            raise FileNotFoundError(f"Artefacto local no encontrado en {self.dir}")
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.clf = joblib.load(self.dir / "model.joblib")
        self.embedder = Embedder(self.meta["embedding"])
        if block_size:
            self.block_size = block_size

    @classmethod
    def status(cls) -> BackendStatus:
        d = default_artifact_dir()
        ok = artifact_ok(d)
        return BackendStatus(
            "local", [str(d)] if ok else [], ok, str(d) if ok else None,
            "" if ok else f"sin artefacto en {d} (entrénalo con tatami-train)",
        )  # fmt: skip

    def classify_block(self, frames: list[SampledFrame], athletes=None, prev_labels=None):
        emb = self.embedder([f.image for f in frames])
        proba = self.clf.predict_proba(emb)
        classes = [str(c) for c in self.clf.classes_]
        labels = []
        for f, p in zip(frames, proba, strict=False):
            k = int(np.argmax(p))
            pos, _, top = classes[k].partition("|")
            labels.append(normalize_frame(f.t, pos, top or "none", float(p[k])))
        return BlockResult(labels)
