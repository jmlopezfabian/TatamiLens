"""Embeddings con un backbone de visión preentrenado (DINOv2 / CLIP) y cache en .npy."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import load_index

DEFAULT_BACKBONE = "facebook/dinov2-small"
# Recibe una lista de imágenes RGB (PIL) y devuelve (N, D) float32.
EmbedFn = Callable[[list[Image.Image]], np.ndarray]


def backbone_slug(backbone: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", backbone)


def pooling_for(backbone: str) -> str:
    """Pooling que aplica `build_embedder` (se escribe en meta.json para tatami-vision)."""
    return "clip" if "clip" in backbone.lower() else "cls+mean"


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_embedder(backbone: str = DEFAULT_BACKBONE, device: str | None = None) -> EmbedFn:
    """Carga el backbone (descarga de HF la primera vez). DINOv2: CLS + media de parches."""
    import torch
    from transformers import AutoImageProcessor, AutoModel, CLIPModel

    device = device or pick_device()
    processor = AutoImageProcessor.from_pretrained(backbone)
    is_clip = pooling_for(backbone) == "clip"
    model = (CLIPModel if is_clip else AutoModel).from_pretrained(backbone).to(device).eval()

    @torch.inference_mode()
    def fn(images: list[Image.Image]) -> np.ndarray:
        inputs = processor(images=images, return_tensors="pt").to(device)
        if is_clip:
            feats = model.get_image_features(pixel_values=inputs["pixel_values"])
        else:
            h = model(**inputs).last_hidden_state
            feats = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=1)
        return feats.float().cpu().numpy()

    return fn


def embed_images(images: list[Image.Image], embed_fn: EmbedFn, batch_size: int = 32) -> np.ndarray:
    chunks = [embed_fn(images[i : i + batch_size]) for i in range(0, len(images), batch_size)]
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros((0, 0), np.float32)


def embeddings_path(dataset_dir: str | Path, backbone: str) -> Path:
    return Path(dataset_dir) / "embeddings" / f"{backbone_slug(backbone)}.npy"


def embed_index(
    dataset_dir: str | Path,
    backbone: str = DEFAULT_BACKBONE,
    embed_fn: EmbedFn | None = None,
    batch_size: int = 32,
    device: str | None = None,
) -> np.ndarray:
    """Calcula embeddings alineados fila a fila con index.csv. Reutiliza el cache por frame."""
    dataset_dir = Path(dataset_dir)
    df = load_index(dataset_dir)
    path = embeddings_path(dataset_dir, backbone)
    keys_path = path.with_suffix(".frames.json")
    cache: dict[str, np.ndarray] = {}
    if path.exists() and keys_path.exists():
        old = np.load(path)
        for k, row in zip(json.loads(keys_path.read_text()), old, strict=True):
            cache[k] = row
    frames = list(df["frame"])
    todo = [f for f in frames if f not in cache]
    if todo:
        fn = embed_fn or build_embedder(backbone, device)
        for i in range(0, len(todo), batch_size):
            chunk = todo[i : i + batch_size]
            imgs = []
            for f in chunk:
                with Image.open(dataset_dir / f) as im:
                    imgs.append(im.convert("RGB"))
            for f, vec in zip(chunk, fn(imgs), strict=True):
                cache[f] = vec
    X = np.stack([cache[f] for f in frames]).astype(np.float32) if frames else np.zeros((0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, X)
    keys_path.write_text(json.dumps(frames))
    return X


def load_embeddings(dataset_dir: str | Path, backbone: str) -> np.ndarray:
    p = embeddings_path(dataset_dir, backbone)
    if not p.exists():
        raise FileNotFoundError(f"faltan embeddings {p}; ejecuta `tatami-train embed` primero")
    return np.load(p)
