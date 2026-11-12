"""Integración: artefacto con el formato de tatami-train + analyze_video(backend='local')."""

import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tatami_vision import analyze_video
from tatami_vision.backends import local

DIM = 8


def fake_embed(images_bgr):
    # el embedding depende del brillo medio del frame
    return np.array([[im.mean() / 255.0 * (k + 1) for k in range(DIM)] for im in images_bgr], np.float32)


def pipe(X, y):
    return Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=200))]).fit(X, y)


def make_artifact(d, format_version=1):
    rng = np.random.default_rng(0)
    X = rng.random((60, DIM)).astype(np.float32)
    pos = np.where(X[:, 0] > 0.5, "mount", "standing")
    top = np.where(pos == "mount", "A", "none")
    bundle = {"position": pipe(X, pos), "top": pipe(X, top), "top_side": None}
    joblib.dump(bundle, d / "model.joblib")
    meta = {
        "backbone": "fake/backbone",
        "labels": ["mount", "standing"],
        "top_labels": ["A", "none"],
        "sample_fps": 2.0,
        "model_kind": "logreg",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    if format_version is not None:
        meta["format_version"] = format_version
    (d / "meta.json").write_text(json.dumps(meta))


def test_analyze_video_local_train_format(tmp_path, synthetic_video, monkeypatch):
    art = tmp_path / "art"
    art.mkdir()
    make_artifact(art)
    seen = {}

    def fake_build(backbone, pooling):
        seen.update(backbone=backbone, pooling=pooling)
        return fake_embed

    monkeypatch.setattr(local, "build_embedder", fake_build)
    res = analyze_video(synthetic_video, backend="local", model=str(art), sample_fps=2.0)
    assert seen == {"backbone": "fake/backbone", "pooling": "cls+mean"}
    assert res["backend"]["name"] == "local"
    assert len(res["frames"]) >= 10
    for f in res["frames"]:
        assert f["position"] in {"mount", "standing"}
        assert (f["top"] == "none") == (f["position"] == "standing")


def test_missing_format_version_is_clear_error(tmp_path, synthetic_video, monkeypatch):
    make_artifact(tmp_path, format_version=None)
    monkeypatch.setattr(local, "build_embedder", lambda b, p: fake_embed)
    with pytest.raises(ValueError, match="format_version"):
        analyze_video(synthetic_video, backend="local", model=str(tmp_path))


def test_pooling_for():
    assert local.pooling_for("openai/clip-vit-base-patch32") == "clip"
    assert local.pooling_for("facebook/dinov2-small") == "cls+mean"
    assert local.pooling_for("x", "cls") == "cls"
