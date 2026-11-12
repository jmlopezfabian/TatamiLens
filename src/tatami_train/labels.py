"""Etiquetas de posición: constantes tomadas de `tatami_vision.schema` (docs/CONTRACTS.md)."""

from __future__ import annotations

from tatami_vision.schema import NEUTRAL_POSITIONS as NO_TOP_POSITIONS
from tatami_vision.schema import POSITIONS, SOURCES
from tatami_vision.schema import TOP_VALUES as TOPS

__all__ = ["NO_TOP_POSITIONS", "POSITIONS", "SOURCES", "TOPS"]

# Agrupaciones para accuracy "gruesa" (no son parte del contrato; sólo para evaluación).
GROUPS: dict[str, tuple[str, ...]] = {
    "neutral": ("standing", "scramble"),
    "guards": ("closed_guard", "half_guard", "open_guard"),
    "dominant": ("side_control", "north_south", "knee_on_belly", "mount", "back_control", "turtle"),
    "no_action": ("separated", "not_visible"),
}

# Extensión propuesta (opcional): lado de la imagen donde está el top.
TOP_SIDES: tuple[str, ...] = ("left", "right", "none")


def group_of(position: str) -> str:
    for name, members in GROUPS.items():
        if position in members:
            return name
    raise ValueError(f"posición desconocida: {position!r}")


def validate_label(position: str, top: str) -> None:
    if position not in POSITIONS:
        raise ValueError(f"posición inválida: {position!r}")
    if top not in TOPS:
        raise ValueError(f"top inválido: {top!r}")
    if position in NO_TOP_POSITIONS and top != "none":
        raise ValueError(f"{position} requiere top='none', se recibió {top!r}")
