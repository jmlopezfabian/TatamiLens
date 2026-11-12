"""Interfaz común de los clasificadores de posición."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..schema import NEUTRAL_POSITIONS, POSITIONS, TOP_VALUES
from ..video import SampledFrame

CONF_MAP = {"low": 0.4, "medium": 0.7, "high": 0.9}

SYSTEM_PROMPT = """You are an expert Brazilian Jiu-Jitsu referee analysing a match video frame by frame.
The video may be filmed by hand (camera pans/zooms), with spectators, coaches and a referee in view.
Only the two competitors matter; the referee and bystanders must be ignored.

For EVERY frame you receive, label the position between the two competitors:
- position: one of the allowed labels.
- top: which athlete is in the dominant / top role:
    guards (closed, half, open): the athlete INSIDE the guard (the passer) is "top";
    side_control, north_south, knee_on_belly, mount: the athlete on top;
    back_control: the athlete who has the back; turtle: the athlete attacking the turtle;
    standing, scramble, separated, not_visible: "none".
- confidence: low, medium or high.

Label meanings: standing = both on their feet; scramble = transition with no clear position;
open_guard includes butterfly, DLR, seated guard; separated = apart / restart / out of the area;
not_visible = competitors hidden or out of frame.

Identity: athlete A and athlete B must stay consistent across all frames. Use whatever
distinguishes them (shorts, rashguard, tattoos, hair, build, skin tone, shoes/socks) and
use temporal continuity between neighbouring frames. Describe each athlete concretely so
another analyst could tell them apart.
Return exactly one entry per frame, in the same order, with the same t values you were given."""


def response_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "athlete_A": {"type": "string"},
            "athlete_B": {"type": "string"},
            "frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "t": {"type": "number"},
                        "position": {"type": "string", "enum": list(POSITIONS)},
                        "top": {"type": "string", "enum": list(TOP_VALUES)},
                        "confidence": {"type": "string", "enum": list(CONF_MAP)},
                    },
                    "required": ["t", "position", "top", "confidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["athlete_A", "athlete_B", "frames"],
        "additionalProperties": False,
    }


def context_text(athletes: dict[str, str] | None, prev_labels: list[dict] | None) -> str:
    if not athletes:
        return ""
    prev = [{"t": f["t"], "position": f["position"], "top": f["top"]} for f in (prev_labels or [])]
    return (
        "This is a continuation of the same match. Keep the same identities:\n"
        f"Athlete A: {athletes.get('A', '')}\nAthlete B: {athletes.get('B', '')}\n"
        f"Last labels of the previous block: {json.dumps(prev)}\n"
    )


def normalize_frame(t: float, position: str, top: str, confidence: float) -> dict:
    """Fuerza coherencia con el contrato (top='none' en posiciones neutras)."""
    if position not in POSITIONS:
        position, confidence = "not_visible", 0.0
    if top not in TOP_VALUES:
        top = "none"
    if position in NEUTRAL_POSITIONS:
        top = "none"
    return {"t": t, "position": position, "top": top, "confidence": round(float(confidence), 3)}


def failed_frame(t: float) -> dict:
    return normalize_frame(t, "not_visible", "none", 0.0)


def align_response(frames: list[SampledFrame], raw: dict) -> list[dict]:
    """Alinea la respuesta del modelo con los frames pedidos (por orden si coincide la
    longitud, si no por t más cercano); lo que falte queda como not_visible conf 0."""
    items = raw.get("frames") or []
    out = []
    used: set[int] = set()
    for i, fr in enumerate(frames):
        it = None
        if len(items) == len(frames):
            it = items[i]
        elif items:
            j = min(range(len(items)), key=lambda k: abs(float(items[k].get("t", 1e9)) - fr.t))
            if abs(float(items[j].get("t", 1e9)) - fr.t) <= 0.26 and j not in used:
                it = items[j]
                used.add(j)
        if it is None:
            out.append(failed_frame(fr.t))
            continue
        conf = it.get("confidence", "medium")
        conf = CONF_MAP.get(conf, 0.5) if isinstance(conf, str) else float(conf)
        out.append(normalize_frame(fr.t, it.get("position", ""), it.get("top", "none"), conf))
    return out


@dataclass
class BlockResult:
    labels: list[dict]
    athletes: dict[str, str] | None = None  # descripciones, si el backend las produce


@dataclass
class BackendStatus:
    name: str
    models: list[str] = field(default_factory=list)
    available: bool = False
    default_model: str | None = None
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "models": self.models,
            "available": self.available,
            "default_model": self.default_model,
            "detail": self.detail,
        }


class PositionClassifier(ABC):
    name: str = ""
    default_model: str | None = None
    block_size: int = 8  # frames por request por defecto
    image_width: int = 960

    def __init__(self, model: str | None = None, **opts):
        self.model = model or self.default_model
        block_size = opts.pop("block_size", None)
        if block_size:
            self.block_size = block_size
        self.opts = opts

    @classmethod
    @abstractmethod
    def status(cls) -> BackendStatus:
        """Disponibilidad y modelos, sin lanzar excepciones."""

    @abstractmethod
    def classify_block(
        self,
        frames: list[SampledFrame],
        athletes: dict[str, str] | None = None,
        prev_labels: list[dict] | None = None,
    ) -> BlockResult:
        """Etiqueta un bloque de frames consecutivos. Debe devolver una etiqueta por frame."""
