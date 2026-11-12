import json

import pytest

from tatami_train.cli import main
from tatami_train.intervals import Interval, LabelFile, save_label_file
from tatami_train.vlm_eval import evaluate_analysis, format_table, parse_variant

LABELS = LabelFile(
    video="v.mp4",
    intervals=[
        Interval(0.0, 2.0, "standing"),
        Interval(2.0, 4.0, "mount", "A"),
        Interval(4.0, 6.0, "closed_guard", "B"),
    ],
)


def result(frames):
    return {
        "frames": [{"t": t, "position": p, "top": top, "confidence": 0.7} for t, p, top in frames]
    }


def test_perfect_with_swapped_identity():
    # El modelo llama "B" a quien las etiquetas llaman "A": la asignación intercambiada acierta todo.
    r = result([(0.0, "standing", "none"), (2.0, "mount", "B"), (4.5, "closed_guard", "A")])
    m = evaluate_analysis(r, LABELS)
    assert m["n"] == 3 and m["position_acc"] == 1.0 and m["group_acc"] == 1.0
    assert m["top_mapping"] == "swapped" and m["top_acc"] == 1.0 and m["top_flip_rate"] == 0.0
    assert m["joint_acc"] == 1.0


def test_errors_flips_and_unlabeled_frames():
    r = result([
        (1.0, "scramble", "none"),     # mismo grupo (neutral), posición distinta
        (3.0, "side_control", "A"),    # grupo dominante, top bien
        (5.0, "closed_guard", "A"),    # identidad al revés
        (9.0, "mount", "A"),           # fuera de las etiquetas: se ignora
    ])  # fmt: skip
    m = evaluate_analysis(r, LABELS)
    assert m["n"] == 3
    assert m["position_acc"] == pytest.approx(1 / 3)
    assert m["group_acc"] == 1.0
    assert m["top_n"] == 2 and m["top_acc"] == 0.5 and m["top_flip_rate"] == 0.5
    assert {"true": "standing", "pred": "scramble", "count": 1} in m["top_confusions"]


def test_no_overlap_raises():
    with pytest.raises(ValueError):
        evaluate_analysis(result([(10.0, "mount", "A")]), LABELS)


def test_parse_variant_casts_values():
    assert parse_variant("x:tile=4,sample_fps=1.5,crop=true,model=qwen3-vl:8b-instruct") == (
        "x",
        {"tile": 4, "sample_fps": 1.5, "crop": True, "model": "qwen3-vl:8b-instruct"},
    )
    assert parse_variant("base") == ("base", {})


def test_eval_analysis_cli(tmp_path, capsys):
    save_label_file(LABELS, tmp_path / "labels.json")
    (tmp_path / "run.json").write_text(json.dumps(result([(2.0, "mount", "A")])))
    assert (
        main(
            ["eval-analysis", str(tmp_path / "run.json"), "--labels", str(tmp_path / "labels.json")]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "| run |" in out and "100%" in out
    assert format_table([]).count("\n") == 1
