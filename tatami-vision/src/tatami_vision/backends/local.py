"""Backend local: clasificador entrenado con tatami-train.

El formato del artefacto está documentado una sola vez en docs/CONTRACTS.md
(sección "Artefacto de modelo local"). Directorio por defecto: $TATAMI_LOCAL_MODEL o ./models/local.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..schema import NEUTRAL_POSITIONS
from ..video import SampledFrame
from .base import BackendStatus, BlockResult, PositionClassifier, normalize_frame

SUPPORTED_FORMAT_VERSIONS = (1,)


def default_artifact_dir() -> Path:
    return Path(os.environ.get("TATAMI_LOCAL_MODEL", "models/local"))


def artifact_ok(d: Path) -> bool:
    return (d / "model.joblib").is_file() and (d / "meta.json").is_file()


def pooling_for(backbone: str, meta_pooling: str | None = None) -> str:
    """Mismo criterio que tatami_train.embed: CLIP -> 'clip'; resto (DINOv2) -> 'cls+mean'."""
    if meta_pooling:
        return meta_pooling
    return "clip" if "clip" in backbone.lower() else "cls+mean"


def load_meta(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    if "format_version" not in meta:
        raise ValueError(f"{d / 'meta.json'}: falta 'format_version'; no es un artefacto de tatami-train")
    if meta["format_version"] not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(
            f"format_version {meta['format_version']!r} no soportado "
            f"(soportados: {SUPPORTED_FORMAT_VERSIONS})"
        )
    if "backbone" not in meta:
        raise ValueError(f"{d / 'meta.json'}: falta 'backbone'")
    return meta


def build_embedder(backbone: str, pooling: str):
    """Devuelve fn(list[BGR ndarray]) -> (N, D) float32, idéntica a tatami_train.embed.build_embedder
    (mismo AutoImageProcessor, RGB, y mismo pooling)."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, CLIPModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(backbone)
    model = (CLIPModel if pooling == "clip" else AutoModel).from_pretrained(backbone).to(device).eval()

    @torch.inference_mode()
    def fn(images_bgr: list[np.ndarray]) -> np.ndarray:
        pil = [Image.fromarray(np.ascontiguousarray(im[:, :, ::-1])) for im in images_bgr]
        inputs = processor(images=pil, return_tensors="pt").to(device)
        if pooling == "clip":
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
        else:
            h = model(**inputs).last_hidden_state
            if pooling == "cls":
                feats = h[:, 0]
            elif pooling == "mean":
                feats = h.mean(dim=1)
            else:  # cls+mean
                feats = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=1)
        return feats.float().cpu().numpy()

    return fn


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
        self.meta = load_meta(self.dir)
        bundle = joblib.load(self.dir / "model.joblib")
        if not isinstance(bundle, dict) or bundle.get("position") is None:
            raise ValueError("model.joblib debe ser un dict con al menos la clave 'position'")
        self.bundle = bundle
        pooling = pooling_for(self.meta["backbone"], self.meta.get("pooling"))
        self.embedder = build_embedder(self.meta["backbone"], pooling)
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
        pos_clf = self.bundle["position"]
        proba = pos_clf.predict_proba(emb)
        classes = [str(c) for c in pos_clf.classes_]
        top_clf = self.bundle.get("top")
        tops = top_clf.predict(emb) if top_clf is not None else None
        labels = []
        for i, (f, p) in enumerate(zip(frames, proba, strict=False)):
            k = int(np.argmax(p))
            pos = classes[k]
            top = "none" if pos in NEUTRAL_POSITIONS or tops is None else str(tops[i])
            labels.append(normalize_frame(f.t, pos, top, float(p[k])))
        return BlockResult(labels)
