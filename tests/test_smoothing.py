from tatami_vision.smoothing import smooth_labels


def f(t, pos, top="none", c=0.9):
    return {"t": t, "position": pos, "top": top, "confidence": c}


def test_removes_flicker():
    frames = (
        [f(i, "standing") for i in range(3)] + [f(3, "mount", "A")] + [f(i, "standing") for i in range(4, 8)]
    )
    out = smooth_labels(frames, 5)
    assert all(x["position"] == "standing" for x in out)


def test_keeps_real_transition():
    frames = [f(i, "standing") for i in range(6)] + [f(i, "mount", "A") for i in range(6, 12)]
    out = smooth_labels(frames, 3)
    assert [x["position"] for x in out] == [x["position"] for x in frames]


def test_failed_frames_untouched_and_do_not_vote():
    frames = [f(0, "standing"), f(1, "not_visible", c=0.0), f(2, "standing")]
    out = smooth_labels(frames, 3)
    assert out[1]["position"] == "not_visible"


def test_input_not_mutated():
    frames = [f(0, "standing"), f(1, "mount", "A"), f(2, "standing")]
    smooth_labels(frames, 3)
    assert frames[1]["position"] == "mount"
