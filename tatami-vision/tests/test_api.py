import json

from tatami_vision import analyze_video, list_backends, render_video, rescore
from tatami_vision.schema import AnalysisResult


def test_analyze_video_mock_end_to_end(synthetic_video):
    prog = []
    res = analyze_video(synthetic_video, backend="mock", sample_fps=2.0, progress_cb=prog.append)
    AnalysisResult.model_validate(res)
    assert res["schema_version"] == "1"
    assert res["video"]["width"] == 160 and abs(res["video"]["duration_s"] - 10.0) < 0.2
    assert res["backend"] == {"name": "mock", "model": "mock", "sample_fps": 2.0}
    assert len(res["frames"]) == 20
    assert res["segments"][0]["position"] == "standing"
    assert set(res["score"]) == {"A", "B"}
    assert prog[0] == 0.0 and prog[-1] == 1.0 and prog == sorted(prog)
    json.dumps(res)  # serializable


def test_analyze_block_size(synthetic_video):
    res = analyze_video(synthetic_video, backend="mock", sample_fps=2.0, block_size=3)
    assert len(res["frames"]) == 20


def test_rescore_applies_corrections(synthetic_video):
    res = analyze_video(synthetic_video, backend="mock", sample_fps=2.0)
    ivs = [
        {"start": 0.0, "end": 4.0, "position": "standing", "top": "none", "source": "human"},
        {"start": 4.0, "end": 10.0, "position": "mount", "top": "B", "source": "human"},
    ]
    new = rescore(res, ivs)
    assert new["score"] == {"A": 0, "B": 6}
    assert [e["kind"] for e in new["events"]] == ["takedown", "mount"]
    assert all(f["confidence"] == 1.0 for f in new["frames"])
    assert res["score"] != new["score"]  # original intacto


def test_list_backends_shape():
    for b in list_backends():
        assert {"name", "models", "available"} <= set(b)


def test_render(synthetic_video, tmp_path):
    res = analyze_video(synthetic_video, backend="mock", sample_fps=2.0)
    out = render_video(res, synthetic_video, tmp_path / "o.mp4")
    assert out.exists() and out.stat().st_size > 0
