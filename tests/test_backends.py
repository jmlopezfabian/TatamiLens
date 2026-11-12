import numpy as np

from tatami_vision.backends import backend_statuses, get_backend
from tatami_vision.backends.base import align_response
from tatami_vision.backends.ollama import OllamaClassifier
from tatami_vision.schema import NEUTRAL_POSITIONS
from tatami_vision.video import SampledFrame


def frames(n, step=0.5):
    return [SampledFrame(round(i * step, 2), np.zeros((32, 32, 3), np.uint8)) for i in range(n)]


def test_mock_deterministic_and_consistent():
    c = get_backend("mock")
    fr = frames(40)
    a = c.classify_block(fr).labels
    b = c.classify_block(fr).labels
    assert a == b and len(a) == 40
    assert all(x["top"] == "none" for x in a if x["position"] in NEUTRAL_POSITIONS)


def test_statuses_never_raise():
    names = {s.name: s for s in backend_statuses()}
    assert set(names) == {"ollama", "anthropic", "local", "mock"}
    assert names["mock"].available
    assert names["local"].available in (True, False)


def test_align_response_fills_missing():
    fr = frames(3)
    raw = {"frames": [{"t": 0.0, "position": "mount", "top": "A", "confidence": "high"}]}
    out = align_response(fr, raw)
    assert out[0]["position"] == "mount" and out[0]["confidence"] == 0.9
    assert out[1]["position"] == "not_visible" and out[1]["confidence"] == 0.0


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def chat(self, **kw):
        self.calls += 1
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return {"message": {"content": r}}


def test_ollama_retries_then_succeeds():
    good = (
        '{"athlete_A":"a","athlete_B":"b","frames":['
        '{"t":0.0,"position":"standing","top":"A","confidence":"high"},'
        '{"t":0.5,"position":"mount","top":"B","confidence":"low"}]}'
    )
    cl = FakeClient(["not json", good])
    c = OllamaClassifier(client=cl, retries=2)
    res = c.classify_block(frames(2))
    assert cl.calls == 2
    assert res.athletes == {"A": "a", "B": "b"}
    assert res.labels[0]["top"] == "none"  # normalizado
    assert res.labels[1]["position"] == "mount"


def test_ollama_failure_marks_not_visible():
    c = OllamaClassifier(client=FakeClient([RuntimeError("x")] * 3), retries=2)
    res = c.classify_block(frames(3))
    assert all(x["position"] == "not_visible" and x["confidence"] == 0 for x in res.labels)
