"""Backend determinista sin modelos (tests y desarrollo sin GPU)."""

from __future__ import annotations

from ..video import SampledFrame
from .base import BackendStatus, BlockResult, PositionClassifier, normalize_frame

# (duración en s, posición, top); se repite cíclicamente.
SCRIPT = [
    (5.0, "standing", "none"),
    (5.0, "open_guard", "A"),
    (5.0, "side_control", "A"),
    (5.0, "mount", "A"),
    (2.0, "scramble", "none"),
    (5.0, "back_control", "B"),
    (3.0, "separated", "none"),
]
PERIOD = sum(d for d, _, _ in SCRIPT)


def mock_label(t: float) -> tuple[str, str]:
    x = t % PERIOD
    for dur, pos, top in SCRIPT:
        if x < dur:
            return pos, top
        x -= dur
    return SCRIPT[-1][1], SCRIPT[-1][2]


class MockClassifier(PositionClassifier):
    name = "mock"
    default_model = "mock"
    block_size = 16
    image_width = 64

    @classmethod
    def status(cls) -> BackendStatus:
        return BackendStatus("mock", ["mock"], True, "mock", "determinista, sin modelos")

    def classify_block(self, frames: list[SampledFrame], athletes=None, prev_labels=None):
        labels = []
        for f in frames:
            pos, top = mock_label(f.t)
            labels.append(normalize_frame(f.t, pos, top, 0.9))
        return BlockResult(labels, {"A": "Atleta A (mock)", "B": "Atleta B (mock)"})
