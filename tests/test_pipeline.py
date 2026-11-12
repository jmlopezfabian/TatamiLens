import json

import numpy as np
import pandas as pd
import pytest
from conftest import fake_embed

from tatami_train.cli import main
from tatami_train.dataset import extract_dataset, load_index
from tatami_train.embed import embed_index, embeddings_path
from tatami_train.evaluate import evaluate_run, grouped_accuracy
from tatami_train.export import LocalModel, export_model
from tatami_train.split import assign_splits, check_no_leakage, split_videos
from tatami_train.train import train_run


def test_split_by_video_no_leakage():
    m = split_videos([f"v{i}" for i in range(10)], seed=1)
    assert set(m.values()) == {"train", "val", "test"}
    df = pd.DataFrame({"video": ["a", "a", "b", "c", "c", "d"], "split": "train"})
    out = assign_splits(df)
    check_no_leakage(out)
    assert out.groupby("video")["split"].nunique().max() == 1
    with pytest.warns(UserWarning):
        assert split_videos(["solo"]) == {"solo": "train"}


def test_full_pipeline(synthetic):
    videos, labels, ds = synthetic
    df = extract_dataset(videos, labels, ds, fps=2.0)
    assert len(df) == 4 * 12  # 6 s * 2 fps por video
    assert (ds / df.iloc[0]["frame"]).exists()
    check_no_leakage(df)
    assert set(df["split"]) == {"train", "val", "test"}

    X = embed_index(ds, "fake/backbone", embed_fn=fake_embed, batch_size=5)
    assert X.shape == (len(df), 3)
    assert embeddings_path(ds, "fake/backbone").exists()
    # cache: segunda vez no llama al embedder
    embed_index(ds, "fake/backbone", embed_fn=lambda _: pytest.fail("no debería recalcular"))

    run = ds / "run"
    info = train_run(ds, run, backbone="fake/backbone")
    assert set(info["targets"]) == {"position", "top"}
    res = evaluate_run(run, split="val")
    pos = res["targets"]["position"]["logreg"]
    assert pos["accuracy"] > 0.9 and pos["grouped"]["overall"] > 0.9
    assert (run / pos["confusion_png"]).stat().st_size > 0

    out = ds / "export"
    meta = export_model(run, out, "logreg")
    assert meta["backbone"] == "fake/backbone" and meta["sample_fps"] == 2.0
    assert json.loads((out / "meta.json").read_text())["labels"] == meta["labels"]
    model = LocalModel(out)
    preds = model.predict(X[:3])
    assert len(preds) == 3 and {"position", "top", "confidence"} <= set(preds[0])
    idx = load_index(ds)
    assert preds[0]["position"] == idx.iloc[0]["position"]


def test_grouped_accuracy_merges_guards():
    g = grouped_accuracy(["closed_guard", "half_guard"], ["open_guard", "mount"])
    assert g["per_group"]["guards"] == 0.5 and g["overall"] == 0.5


def test_pseudolabel_cli_from_result(tmp_path):
    res = {
        "video": {"path": "x.mp4", "duration_s": 2.0},
        "backend": {"sample_fps": 1.0},
        "frames": [
            {"t": 0.0, "position": "standing", "top": "none"},
            {"t": 1.0, "position": "standing", "top": "none"},
        ],
    }
    (tmp_path / "r.json").write_text(json.dumps(res))
    assert (
        main(
            ["pseudolabel", "--out", str(tmp_path / "o"), "--from-result", str(tmp_path / "r.json")]
        )
        == 0
    )
    data = json.loads((tmp_path / "o" / "x.json").read_text())
    assert data["intervals"][0]["source"] == "vlm" and data["intervals"][0]["end"] == 2.0


@pytest.mark.slow
def test_real_backbone_dinov2(tmp_path):
    from PIL import Image

    from tatami_train.embed import build_embedder

    fn = build_embedder("facebook/dinov2-small")
    imgs = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), np.uint8)) for _ in range(2)]
    assert fn(imgs).shape == (2, 768)
