"""Esquemas Pydantic (contrato de docs/CONTRACTS.md)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Position = Literal[
    "standing", "scramble", "closed_guard", "half_guard", "open_guard", "side_control",
    "north_south", "knee_on_belly", "mount", "back_control", "turtle", "separated",
    "not_visible",
]  # fmt: skip
Top = Literal["A", "B", "none"]
NO_TOP_POSITIONS = {"standing", "scramble", "separated", "not_visible"}
Status = Literal["queued", "running", "done", "error"]


class Video(BaseModel):
    id: str
    filename: str
    duration_s: float | None = None
    fps: float | None = None
    created_at: datetime


class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


class VideoInfo(_Loose):
    path: str
    fps: float | None = None
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None


class BackendUsed(_Loose):
    name: str
    model: str | None = None
    sample_fps: float | None = None


class AthleteInfo(_Loose):
    label: str
    description: str = ""


class Frame(_Loose):
    t: float
    position: Position
    top: Top
    confidence: float = Field(ge=0, le=1)


class Segment(_Loose):
    start: float
    end: float
    position: Position
    top: Top
    stable: bool


class Event(_Loose):
    t: float
    athlete: Literal["A", "B"]
    points: int
    kind: Literal["takedown", "sweep", "guard_pass", "knee_on_belly", "mount", "back_take"]
    reason: str = ""


class AnalysisResult(_Loose):
    schema_version: str = "1"
    video: VideoInfo
    backend: BackendUsed
    athletes: dict[str, AthleteInfo]
    frames: list[Frame]
    segments: list[Segment]
    events: list[Event]
    score: dict[str, int]
    created_at: str


class AnalysisCreate(BaseModel):
    backend: str = "ollama"
    model: str = "qwen3-vl:8b"
    sample_fps: float = Field(default=2.0, gt=0, le=30)


class Analysis(BaseModel):
    id: str
    video_id: str
    status: Status
    progress: float = Field(ge=0, le=1)
    backend: str
    model: str
    error: str | None = None
    created_at: datetime
    result: AnalysisResult | None = None


class LabelInterval(BaseModel):
    start: float = Field(ge=0)
    end: float
    position: Position
    top: Top
    source: Literal["human", "vlm", "model"] = "human"

    @model_validator(mode="after")
    def _check(self) -> "LabelInterval":
        if self.end <= self.start:
            raise ValueError("end debe ser > start")
        if self.position in NO_TOP_POSITIONS and self.top != "none":
            raise ValueError(f"top debe ser 'none' para {self.position}")
        return self


class Labels(BaseModel):
    video: str
    athletes: dict[str, str] = Field(default_factory=dict)
    intervals: list[LabelInterval]


class BackendInfo(BaseModel):
    name: str
    models: list[str]
    available: bool
    default_model: str | None = None  # los aporta tatami-vision; ausentes en el fallback
    detail: str = ""


class Health(BaseModel):
    status: str = "ok"
