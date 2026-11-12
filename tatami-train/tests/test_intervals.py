import pytest

from tatami_train.intervals import (
    Interval,
    LabelFile,
    intervals_from_analysis,
    intervals_to_frame_labels,
    load_label_file,
    merge_adjacent,
    save_label_file,
    validate_intervals,
)

RESULT = {
    "video": {"path": "/x/insta2.mp4", "duration_s": 3.0},
    "backend": {"sample_fps": 1.0},
    "athletes": {"A": {"label": "Atleta A", "description": "azul"}},
    "frames": [
        {"t": 0.0, "position": "standing", "top": "none", "confidence": 0.9},
        {"t": 1.0, "position": "standing", "top": "none", "confidence": 0.8},
        {"t": 2.0, "position": "mount", "top": "A", "confidence": 0.7},
    ],
}


def test_from_analysis_merges_and_clips():
    lf = intervals_from_analysis(RESULT)
    assert lf.video == "insta2.mp4"
    assert [(i.start, i.end, i.position, i.top, i.source) for i in lf.intervals] == [
        (0.0, 2.0, "standing", "none", "vlm"),
        (2.0, 3.0, "mount", "A", "vlm"),
    ]
    assert lf.athletes == {"A": "azul"}


def test_from_analysis_segments_fallback():
    res = {
        "video": {"path": "a.mp4"},
        "frames": [],
        "segments": [{"start": 0, "end": 4, "position": "turtle", "top": "B"}],
    }
    assert intervals_from_analysis(res).intervals[0].position == "turtle"


def test_frame_labels():
    ivs = [Interval(0, 2, "standing"), Interval(2, 4, "mount", "A")]
    labs = intervals_to_frame_labels(ivs, [0.0, 1.9, 2.0, 5.0])
    assert [None if lab is None else lab["position"] for lab in labs] == [
        "standing",
        "standing",
        "mount",
        None,
    ]


def test_validation_errors():
    with pytest.raises(ValueError):
        validate_intervals([Interval(0, 2, "standing"), Interval(1, 3, "mount", "A")])
    with pytest.raises(ValueError):
        Interval(0, 1, "standing", "A").validate()
    with pytest.raises(ValueError):
        Interval(0, 1, "bad").validate()
    with pytest.raises(ValueError):
        Interval(2, 1, "standing").validate()


def test_roundtrip_and_merge(tmp_path):
    lf = LabelFile(
        "a.mp4", merge_adjacent([Interval(0, 1, "standing"), Interval(1, 2, "standing")])
    )
    assert len(lf.intervals) == 1
    save_label_file(lf, tmp_path / "a.json")
    assert load_label_file(tmp_path / "a.json").intervals[0].end == 2
