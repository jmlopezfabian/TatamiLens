"""Constantes y modelos pydantic del contrato (docs/CONTRACTS.md)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "1"

POSITIONS: tuple[str, ...] = (
    "standing",
    "scramble",
    "closed_guard",
    "half_guard",
    "open_guard",
    "side_control",
    "north_south",
    "knee_on_belly",
    "mount",
    "back_control",
    "turtle",
    "separated",
    "not_visible",
)
TOP_VALUES: tuple[str, ...] = ("A", "B", "none")
# Posiciones donde `top` es siempre "none".
NEUTRAL_POSITIONS: frozenset[str] = frozenset({"standing", "scramble", "separated", "not_visible"})
GUARDS: frozenset[str] = frozenset({"closed_guard", "half_guard", "open_guard"})

EVENT_KINDS: tuple[str, ...] = (
    "takedown",
    "sweep",
    "guard_pass",
    "knee_on_belly",
    "mount",
    "back_take",
)
POINTS: dict[str, int] = {
    "takedown": 2,
    "sweep": 2,
    "guard_pass": 3,
    "knee_on_belly": 2,
    "mount": 4,
    "back_take": 4,
}
SOURCES: tuple[str, ...] = ("human", "vlm", "model")
STABLE_SECONDS = 3.0


def _check(value: str, allowed: tuple[str, ...], what: str) -> str:
    if value not in allowed:
        raise ValueError(f"{what} inválido: {value!r}; permitidos: {allowed}")
    return value


class _PositionTop(BaseModel):
    position: str
    top: str = "none"

    @field_validator("position")
    @classmethod
    def _v_position(cls, v: str) -> str:
        return _check(v, POSITIONS, "position")

    @field_validator("top")
    @classmethod
    def _v_top(cls, v: str) -> str:
        return _check(v, TOP_VALUES, "top")

    @model_validator(mode="after")
    def _neutral_has_no_top(self):
        if self.position in NEUTRAL_POSITIONS and self.top != "none":
            raise ValueError(f"{self.position} requiere top='none'")
        return self


class VideoInfo(BaseModel):
    path: str
    fps: float
    duration_s: float
    width: int
    height: int


class BackendInfo(BaseModel):
    name: str
    model: str | None = None
    sample_fps: float


class Athlete(BaseModel):
    label: str
    description: str = ""


class FrameLabel(_PositionTop):
    t: float
    confidence: float = Field(ge=0.0, le=1.0)


class Segment(_PositionTop):
    start: float
    end: float
    stable: bool


class Event(BaseModel):
    t: float
    athlete: str
    points: int
    kind: str
    reason: str

    @field_validator("athlete")
    @classmethod
    def _v_athlete(cls, v: str) -> str:
        return _check(v, ("A", "B"), "athlete")

    @field_validator("kind")
    @classmethod
    def _v_kind(cls, v: str) -> str:
        return _check(v, EVENT_KINDS, "kind")


class Score(BaseModel):
    A: int = 0
    B: int = 0


class AnalysisResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    video: VideoInfo
    backend: BackendInfo
    athletes: dict[str, Athlete]
    frames: list[FrameLabel]
    segments: list[Segment]
    events: list[Event]
    score: Score
    created_at: str


class Interval(_PositionTop):
    """Intervalo etiquetado (formato de dataset). `end` es exclusivo."""

    start: float
    end: float
    source: str = "human"

    @field_validator("source")
    @classmethod
    def _v_source(cls, v: str) -> str:
        return _check(v, SOURCES, "source")

    @model_validator(mode="after")
    def _order(self):
        if self.end < self.start:
            raise ValueError("end < start")
        return self


class LabelFile(BaseModel):
    video: str
    athletes: dict[str, str] = Field(default_factory=dict)
    intervals: list[Interval]


def analysis_result_json_schema() -> dict[str, Any]:
    return AnalysisResult.model_json_schema()


def label_file_json_schema() -> dict[str, Any]:
    return LabelFile.model_json_schema()
