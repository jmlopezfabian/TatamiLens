import pytest
from pydantic import ValidationError

from tatami_vision.schema import (
    EVENT_KINDS,
    POINTS,
    POSITIONS,
    FrameLabel,
    Interval,
    LabelFile,
    analysis_result_json_schema,
)  # fmt: skip


def test_constants_match_contract():
    assert len(POSITIONS) == 13 and "not_visible" in POSITIONS
    assert set(POINTS) == set(EVENT_KINDS)
    assert POINTS == {
        "takedown": 2,
        "sweep": 2,
        "guard_pass": 3,
        "knee_on_belly": 2,
        "mount": 4,
        "back_take": 4,
    }


def test_frame_validation():
    FrameLabel(t=0, position="mount", top="A", confidence=0.5)
    with pytest.raises(ValidationError):
        FrameLabel(t=0, position="standing", top="A", confidence=0.5)
    with pytest.raises(ValidationError):
        FrameLabel(t=0, position="nope", top="none", confidence=0.5)
    with pytest.raises(ValidationError):
        FrameLabel(t=0, position="mount", top="A", confidence=1.5)


def test_interval_and_labelfile():
    lf = LabelFile.model_validate(
        {"video": "a.mp4", "athletes": {"A": "x"},
         "intervals": [{"start": 0, "end": 5, "position": "standing", "top": "none", "source": "human"}]}
    )  # fmt: skip
    assert lf.intervals[0].source == "human"
    with pytest.raises(ValidationError):
        Interval(start=0, end=1, position="standing", top="none", source="robot")


def test_json_schema_exports():
    sch = analysis_result_json_schema()
    assert "frames" in sch["properties"]
