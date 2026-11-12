import json

import numpy as np
import pytest

from tatami_vision.backends import backend_statuses, get_backend
from tatami_vision.backends.base import align_response
from tatami_vision.backends.ollama import (
    CODES,
    OllamaClassifier,
    code_to_label,
    label_to_code,
    max_block_for_context,
)
from tatami_vision.schema import NEUTRAL_POSITIONS
from tatami_vision.video import SampledFrame, tile_frames


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
    """Doble de `ollama.Client` (API nativa)."""

    def __init__(self, replies):
        self.replies, self.calls, self.kwargs = list(replies), 0, []

    def chat(self, **kw):
        self.calls += 1
        self.kwargs.append(kw)
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        if isinstance(r, dict):
            return {"message": r, "prompt_eval_count": 10, "eval_count": 5}
        return {"message": {"content": r}, "prompt_eval_count": 10, "eval_count": 5}


def codes_reply(codes, athletes=True):
    body = {"frames": codes}
    if athletes:
        body = {"athlete_A": "a", "athlete_B": "b", **body}
    return json.dumps(body)


def test_ollama_compact_codes_and_mosaic():
    cl = FakeClient([codes_reply(["standing"] * 3 + ["mount:A"] * 4 + ["half_guard:B"])])
    c = OllamaClassifier(client=cl)
    assert c.model == "qwen3-vl:8b-instruct" and c.tile == 4 and c.block_size == 8
    res = c.classify_block(frames(8))
    assert res.athletes == {"A": "a", "B": "b"}
    assert [x["position"] for x in res.labels[2:4]] == ["standing", "mount"]
    assert res.labels[3]["top"] == "A" and res.labels[7] == {
        "t": 3.5, "position": "half_guard", "top": "B", "confidence": 0.7,
    }  # fmt: skip
    kw = cl.kwargs[0]
    user = kw["messages"][-1]
    assert kw["messages"][0]["role"] == "system"
    assert len(user["images"]) == 2  # 8 frames en 2 mosaicos 2x2
    assert kw["format"]["properties"]["frames"]["minItems"] == 8
    assert kw["format"]["properties"]["frames"]["maxItems"] == 8
    assert "athlete_A" in kw["format"]["required"]
    assert kw["options"]["num_ctx"] == 8192 and kw["options"]["num_predict"] > 0


def test_ollama_continuation_omits_athletes_and_sends_context():
    cl = FakeClient([codes_reply(["mount:B", "mount:B"], athletes=False)])
    c = OllamaClassifier(client=cl, tile=1)
    prev = [{"t": 0.0, "position": "mount", "top": "B", "confidence": 0.7}]
    res = c.classify_block(frames(2), {"A": "red", "B": "blue"}, prev)
    assert res.athletes is None and res.labels[0]["top"] == "B"
    kw = cl.kwargs[0]
    assert "athlete_A" not in kw["format"]["properties"]
    assert "red" in kw["messages"][-1]["content"] and "'mount:B'" in kw["messages"][-1]["content"]
    assert len(kw["messages"][-1]["images"]) == 2  # sin mosaico: una imagen por frame


def test_ollama_retries_then_succeeds():
    cl = FakeClient(["not json", codes_reply(["standing", "mount:B"])])
    c = OllamaClassifier(client=cl, retries=2)
    res = c.classify_block(frames(2))
    assert cl.calls == 2  # el reintento lo hace Pydantic AI con el error de validación
    assert res.athletes == {"A": "a", "B": "b"}
    assert res.labels[0]["top"] == "none"
    assert res.labels[1]["position"] == "mount" and res.labels[1]["top"] == "B"


def test_ollama_wrong_count_is_rejected():
    # 3 códigos para 2 frames: no valida ni en el primer intento ni en el reintento interno.
    cl = FakeClient([codes_reply(["standing"] * 3)] * 2)
    res = OllamaClassifier(client=cl, retries=0).classify_block(frames(2))
    assert all(x["position"] == "not_visible" and x["confidence"] == 0 for x in res.labels)


def test_ollama_failure_marks_not_visible():
    c = OllamaClassifier(client=FakeClient([RuntimeError("x")] * 3), retries=2)
    res = c.classify_block(frames(3))
    assert all(x["position"] == "not_visible" and x["confidence"] == 0 for x in res.labels)


def test_ollama_warns_when_model_thinks(caplog):
    reply = {"content": codes_reply(["standing"]), "thinking": "let me think..."}
    c = OllamaClassifier(client=FakeClient([reply]))
    with caplog.at_level("WARNING"):
        c.classify_block(frames(1))
    assert any("razonando" in r.message for r in caplog.records)


def test_ollama_block_size_fits_context():
    c = OllamaClassifier(client=FakeClient([]), tile=1, block_size=50, context_tokens=4096)
    assert c.block_size == max_block_for_context(4096, 1) < 50


def test_codes_roundtrip():
    for code in CODES:
        assert label_to_code(code_to_label(0.0, code)) == code


def test_tile_frames_layout():
    imgs = [np.full((90, 160, 3), k * 40 + 40, np.uint8) for k in range(3)]
    grid = tile_frames(imgs, cols=2, cell_size=(160, 90))
    assert grid.shape == (180, 320, 3)
    assert grid[150, 300].sum() == 0  # celda 4 vacía
    assert grid[150, 100].mean() > 0  # celda 3 con contenido


def test_tile_frames_fixed_size_for_vertical_and_mixed_frames():
    imgs = [np.full((1280, 720, 3), 200, np.uint8), np.full((300, 900, 3), 200, np.uint8)]
    grid = tile_frames(imgs)
    assert grid.shape == (360, 1280, 3)  # 2 celdas de 640x360, sin depender del aspecto
    assert grid[180, 100].sum() == 0  # franja negra a los lados del frame vertical
    assert grid[180, 320].mean() > 0


def test_ollama_validation_failure_is_not_retried_by_outer_loop():
    # Respuestas siempre inválidas: Pydantic AI reintenta `retries` veces con el error de
    # validación y el bucle externo no multiplica los intentos.
    cl = FakeClient([codes_reply(["standing"] * 3)] * 10)
    res = OllamaClassifier(client=cl, retries=1).classify_block(frames(2))
    assert cl.calls == 2
    assert all(x["confidence"] == 0 for x in res.labels)


def test_ollama_network_errors_are_retried():
    cl = FakeClient([ConnectionError("down"), codes_reply(["standing"])])
    res = OllamaClassifier(client=cl, retries=1).classify_block(frames(1))
    assert cl.calls == 2 and res.labels[0]["position"] == "standing"


def test_ollama_think_false_sent_and_fallback_when_unsupported():
    class NoThink(FakeClient):
        def chat(self, **kw):
            if "think" in kw:
                self.kwargs.append(kw)
                raise RuntimeError('"model" does not support thinking')
            return super().chat(**kw)

    cl = FakeClient([codes_reply(["standing"])])
    OllamaClassifier(client=cl).classify_block(frames(1))
    assert cl.kwargs[0]["think"] is False
    cl2 = NoThink([codes_reply(["standing"])])
    res = OllamaClassifier(client=cl2).classify_block(frames(1))
    assert res.labels[0]["position"] == "standing" and "think" not in cl2.kwargs[-1]


def test_ollama_context_options(monkeypatch):
    assert OllamaClassifier(client=FakeClient([]), num_ctx=16384).context_tokens == 16384
    monkeypatch.setenv("TATAMI_OLLAMA_CONTEXT", "12000")
    assert OllamaClassifier(client=FakeClient([])).context_tokens == 12000
    monkeypatch.setenv("TATAMI_OLLAMA_CONTEXT", "mucho")
    with pytest.raises(ValueError, match="TATAMI_OLLAMA_CONTEXT"):
        OllamaClassifier(client=FakeClient([]))


def test_ollama_first_block_limits_descriptions():
    cl = FakeClient([codes_reply(["standing"])])
    OllamaClassifier(client=cl).classify_block(frames(1))
    kw = cl.kwargs[0]
    assert kw["format"]["properties"]["athlete_A"]["maxLength"] == 200
    assert "Describe each athlete" in kw["messages"][-1]["content"]
    assert "confidence" not in kw["messages"][0]["content"]
    assert "Describe each athlete" not in kw["messages"][0]["content"]


def test_cli_rejects_invalid_tile(tmp_path):
    from typer.testing import CliRunner

    from tatami_vision.cli import app

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    res = CliRunner().invoke(app, ["analyze", str(video), "--backend", "mock", "--tile", "3"])
    assert res.exit_code != 0 and "1, 2 o 4" in res.output
