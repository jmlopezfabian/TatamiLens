from tatami_vision.scoring import compute_segments, score_timeline, total_score


def tl(spec, dt=1.0):
    """spec: [(segundos, position, top)] -> frames cada dt."""
    out, t = [], 0.0
    for dur, pos, top in spec:
        for _ in range(int(dur / dt)):
            out.append({"t": t, "position": pos, "top": top, "confidence": 1.0})
            t += dt
    return out


def kinds(events):
    return [(e["athlete"], e["kind"], e["points"]) for e in events]


def test_takedown_and_mount():
    ev = score_timeline(tl([(4, "standing", "none"), (4, "side_control", "A"), (4, "mount", "A")]))
    assert kinds(ev) == [("A", "takedown", 2), ("A", "mount", 4)]
    assert total_score(ev) == {"A": 6, "B": 0}


def test_guard_pull_gives_no_takedown():
    ev = score_timeline(tl([(4, "standing", "none"), (4, "open_guard", "A")]))
    assert ev == []


def test_sweep():
    ev = score_timeline(tl([(4, "open_guard", "A"), (4, "side_control", "B")]))
    assert kinds(ev) == [("B", "sweep", 2)]


def test_guard_pass():
    ev = score_timeline(tl([(4, "half_guard", "A"), (4, "side_control", "A")]))
    assert kinds(ev) == [("A", "guard_pass", 3)]


def test_pass_to_mount_gives_pass_and_mount():
    ev = score_timeline(tl([(4, "closed_guard", "A"), (4, "mount", "A")]))
    assert kinds(ev) == [("A", "guard_pass", 3), ("A", "mount", 4)]


def test_back_and_knee_on_belly():
    ev = score_timeline(tl([(4, "turtle", "A"), (4, "back_control", "A"), (4, "knee_on_belly", "B")]))
    assert ("A", "back_take", 4) in kinds(ev) and ("B", "knee_on_belly", 2) in kinds(ev)


def test_unstable_segment_scores_nothing():
    ev = score_timeline(tl([(4, "standing", "none"), (2, "mount", "A"), (4, "standing", "none")]))
    assert ev == []


def test_scramble_absorbed_and_no_double_points():
    ev = score_timeline(
        tl([(4, "standing", "none"), (3, "mount", "A"), (1, "scramble", "none"), (3, "mount", "A")])
    )
    assert kinds(ev) == [("A", "takedown", 2), ("A", "mount", 4)]


def test_event_time_is_start_plus_hold():
    ev = score_timeline(tl([(4, "standing", "none"), (4, "mount", "B")]))
    assert ev[0]["t"] == 7.0 and ev[0]["reason"] == "derribo"


def test_segments_stable_flag():
    segs = compute_segments(tl([(4, "standing", "none"), (2, "scramble", "none"), (3, "mount", "A")]))
    assert [(s["position"], s["start"], s["end"], s["stable"]) for s in segs] == [
        ("standing", 0.0, 4.0, True), ("scramble", 4.0, 6.0, False), ("mount", 6.0, 9.0, True)
    ]  # fmt: skip
