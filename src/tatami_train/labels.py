"""Etiquetas de posición, alineadas con docs/CONTRACTS.md.

TODO: cuando `tatami_vision.schema` esté disponible, reutilizar sus constantes y
dejar este módulo como simple re-export (hoy se mantienen aquí para no depender de él).
"""

from __future__ import annotations

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

TOPS: tuple[str, ...] = ("A", "B", "none")
SOURCES: tuple[str, ...] = ("human", "vlm", "model")

# Posiciones donde `top` es siempre "none".
NO_TOP_POSITIONS: frozenset[str] = frozenset({"standing", "scramble", "separated", "not_visible"})

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
