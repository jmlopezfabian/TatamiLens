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


def test_parse_variant_validation():
    assert parse_variant("x:crop=False,smooth=TRUE") == ("x", {"crop": False, "smooth": True})
    assert parse_variant("g:model=gemma4:e4b") == ("g", {"model": "gemma4:e4b"})
    for bad in ["model=gemma4:e4b", "tile=4", ":tile=4", "a/b:tile=4", "x:tile", "x:=4", "x:tile="]:
        with pytest.raises(ValueError):
            parse_variant(bad)


def test_top_none_counts_as_uncommitted_not_flip():
    labels = LabelFile(video="v.mp4", intervals=[Interval(0.0, 4.0, "mount", "A")])
    r = result(
        [(0.0, "scramble", "none"), (1.0, "mount", "A"), (2.0, "mount", "A"), (3.0, "mount", "B")]
    )
    m = evaluate_analysis(r, labels)
    assert m["top_acc"] == 0.5 and m["top_committed"] == 0.75
    assert m["top_identity_acc"] == 2 / 3 and m["top_flip_rate"] == 1 / 3
    assert m["top_baseline"] == 1.0


def test_rejects_labels_longer_than_video():
    r = result([(0.0, "standing", "none")]) | {"video": {"duration_s": 2.0}}
    with pytest.raises(ValueError, match="mismo video"):
        evaluate_analysis(r, LABELS)


def test_bench_rejects_duplicate_names_before_running(tmp_path):
    from tatami_train.vlm_eval import bench_vlm

    save_label_file(LABELS, tmp_path / "l.json")
    with pytest.raises(ValueError, match="repetidos"):
        bench_vlm("v.mp4", tmp_path / "l.json", ["a:tile=1", "a:tile=4"], tmp_path / "out")


def test_parse_variant_reserved_name():
    with pytest.raises(ValueError):
        parse_variant("bench:tile=4")


def test_top_metrics_none_without_committed_top():
    r = result([(0.0, "standing", "none"), (2.5, "scramble", "none"), (4.5, "scramble", "none")])
    m = evaluate_analysis(r, LABELS)
    assert m["top_acc"] == 0.0 and m["top_committed"] == 0.0
    assert m["top_identity_acc"] is None and m["top_flip_rate"] is None
    only_standing = LabelFile(video="v.mp4", intervals=[Interval(0.0, 2.0, "standing")])
    m = evaluate_analysis(result([(0.5, "standing", "none")]), only_standing)
    assert m["top_acc"] is None and m["top_committed"] is None and m["top_baseline"] is None


def test_duration_tolerance():
    ok = result([(0.0, "standing", "none")]) | {"video": {"duration_s": 5.0}}
    assert evaluate_analysis(ok, LABELS)["n"] == 1  # etiquetas hasta 6.0 s: dentro de 1 s


def test_bench_writes_partial_results(tmp_path, monkeypatch):
    import tatami_vision
    import tatami_vision.video as tv_video

    from tatami_train.vlm_eval import bench_vlm

    save_label_file(LABELS, tmp_path / "l.json")
    monkeypatch.setattr(tv_video, "read_metadata", lambda p: type("V", (), {"duration_s": 6.0}))
    calls = []

    def fake_analyze(video, backend, **kw):
        calls.append(kw)
        if len(calls) == 2:
            raise RuntimeError("ollama caído")
        return result([(0.0, "standing", "none"), (2.5, "mount", "A")])

    monkeypatch.setattr(tatami_vision, "analyze_video", fake_analyze)
    with pytest.raises(RuntimeError):
        bench_vlm("v.mp4", tmp_path / "l.json", ["a:tile=1", "b:tile=4"], tmp_path / "out")
    rows = json.loads((tmp_path / "out" / "bench.json").read_text())
    assert [r["name"] for r in rows] == ["a"]
