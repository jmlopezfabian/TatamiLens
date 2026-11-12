"""Motor de puntuación IBJJF simplificado (implementación propia).

Entrada: lista de frames {"t", "position", "top", ...} ordenada por t.
Reglas (simplificadas):
- Una posición solo cuenta si se mantiene >= STABLE_SECONDS ("posición estable"); los
  puntos se otorgan al cumplirse ese tiempo.
- Derribo (2): de pie/separados -> suelo con X arriba y sin ser guardia (si es guardia se
  asume que el de abajo jaló guardia: 0 puntos).
- Raspada (2): desde guardia, el que estaba abajo termina arriba.
- Pase de guardia (3): desde guardia, el de arriba llega a side_control, north_south,
  knee_on_belly o mount.
- Rodilla en la barriga (2), montada (4), toma de espalda (4) al estabilizarse.
- `scramble` y `not_visible` no rompen ni crean posiciones.
No modela ventajas, penalizaciones ni sumisiones.
"""

from __future__ import annotations

from .schema import EVENT_KINDS, GUARDS, NEUTRAL_POSITIONS, POINTS, STABLE_SECONDS

HOLD = STABLE_SECONDS
GROUND = {
    "closed_guard", "half_guard", "open_guard", "side_control", "north_south",
    "knee_on_belly", "mount", "back_control", "turtle",
}  # fmt: skip
PASSED = {"side_control", "north_south", "knee_on_belly", "mount"}
IGNORE = {"not_visible", "scramble"}

REASONS = {
    "takedown": "derribo",
    "sweep": "raspada",
    "guard_pass": "pase de guardia",
    "knee_on_belly": "rodilla en la barriga",
    "mount": "montada",
    "back_take": "toma de espalda",
}
assert set(REASONS) == set(EVENT_KINDS)


def other(x: str) -> str:
    return "B" if x == "A" else "A"


def estimate_dt(frames: list[dict], default: float = 1.0) -> float:
    if len(frames) < 2:
        return default
    gaps = sorted(b["t"] - a["t"] for a, b in zip(frames, frames[1:], strict=False) if b["t"] > a["t"])
    return gaps[len(gaps) // 2] if gaps else default


def compute_segments(
    frames: list[dict], dt: float | None = None, duration: float | None = None
) -> list[dict]:
    """Agrupa frames consecutivos con la misma (position, top). `stable` = duración >= 3 s."""
    if not frames:
        return []
    dt = estimate_dt(frames) if dt is None else dt
    segs: list[dict] = []
    for f in frames:
        key = (f["position"], f["top"])
        if segs and (segs[-1]["position"], segs[-1]["top"]) == key:
            continue
        segs.append({"start": f["t"], "end": f["t"], "position": key[0], "top": key[1]})
    last_end = frames[-1]["t"] + dt
    if duration is not None:
        last_end = min(last_end, max(duration, frames[-1]["t"]))
    for s, nxt in zip(segs, segs[1:], strict=False):
        s["end"] = nxt["start"]
    segs[-1]["end"] = max(last_end, segs[-1]["start"])
    for s in segs:
        s["start"] = round(s["start"], 2)
        s["end"] = round(s["end"], 2)
        s["stable"] = (s["end"] - s["start"]) >= HOLD - 1e-9
    return segs


def _scoring_segments(frames: list[dict], dt: float, duration: float | None = None) -> list[dict]:
    """Segmentos estables para puntuar; scramble/not_visible se absorben en el segmento previo."""
    segs: list[dict] = []
    for f in frames:
        if f["position"] in IGNORE:
            if segs:
                segs[-1]["end"] = f["t"] + dt
            continue
        key = (f["position"], f["top"])
        if segs and segs[-1]["key"] == key:
            segs[-1]["end"] = f["t"] + dt
        else:
            if segs:
                segs[-1]["end"] = f["t"]
            segs.append({"key": key, "start": f["t"], "end": f["t"] + dt})
    if segs and duration is not None:
        segs[-1]["end"] = min(segs[-1]["end"], max(duration, segs[-1]["start"]))
    return [s for s in segs if s["end"] - s["start"] >= HOLD - 1e-9]


def _event(t: float, athlete: str, kind: str) -> dict:
    return {
        "t": round(t, 2),
        "athlete": athlete,
        "points": POINTS[kind],
        "kind": kind,
        "reason": REASONS[kind],
    }


def score_timeline(frames: list[dict], dt: float | None = None, duration: float | None = None) -> list[dict]:
    dt = estimate_dt(frames) if dt is None else dt
    events: list[dict] = []
    prev = None
    for seg in _scoring_segments(frames, dt, duration):
        pos, top = seg["key"]
        t = seg["start"] + HOLD
        if prev is not None and prev["key"] == seg["key"]:
            continue
        ppos, ptop = prev["key"] if prev else ("standing", "none")
        if pos in GROUND and top in ("A", "B"):
            if ppos in ("standing", "separated"):
                if pos not in GUARDS:
                    events.append(_event(t, top, "takedown"))
            elif ppos in GUARDS and ptop == other(top):
                events.append(_event(t, top, "sweep"))
            elif ppos in GUARDS and ptop == top and pos in PASSED:
                events.append(_event(t, top, "guard_pass"))
            if pos == "knee_on_belly":
                events.append(_event(t, top, "knee_on_belly"))
            elif pos == "mount":
                events.append(_event(t, top, "mount"))
            elif pos == "back_control":
                events.append(_event(t, top, "back_take"))
        prev = seg
    return events


def total_score(events: list[dict]) -> dict[str, int]:
    return {a: sum(e["points"] for e in events if e["athlete"] == a) for a in ("A", "B")}


__all__ = [
    "HOLD", "NEUTRAL_POSITIONS", "compute_segments", "estimate_dt", "score_timeline", "total_score",
]  # fmt: skip
