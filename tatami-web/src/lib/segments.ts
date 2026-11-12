import type { Frame, Interval, LabelSource, Segment } from '../api/types'

export const STABLE_MIN_S = 3

/**
 * Agrupa frames consecutivos con la misma (position, top) en segmentos.
 * Cada frame cubre desde su t hasta el t del siguiente; el último llega a `endTime`
 * (si no se da, se usa el paso medio entre frames).
 */
export function framesToSegments(frames: Frame[], endTime?: number): Segment[] {
  if (frames.length === 0) return []
  const sorted = [...frames].sort((a, b) => a.t - b.t)
  const step =
    sorted.length > 1 ? (sorted[sorted.length - 1].t - sorted[0].t) / (sorted.length - 1) : 0.5
  const end = endTime !== undefined && endTime > sorted[sorted.length - 1].t ? endTime : sorted[sorted.length - 1].t + step
  const out: Segment[] = []
  for (let i = 0; i < sorted.length; i++) {
    const f = sorted[i]
    const next = i + 1 < sorted.length ? sorted[i + 1].t : end
    const last = out[out.length - 1]
    if (last && last.position === f.position && last.top === f.top) {
      last.end = next
    } else {
      out.push({ start: f.t, end: next, position: f.position, top: f.top, stable: false })
    }
  }
  for (const s of out) s.stable = s.end - s.start >= STABLE_MIN_S
  return out
}

/** Segmento que contiene a t (inicio inclusivo, fin exclusivo; el último incluye su fin). */
export function segmentAt<T extends { start: number; end: number }>(segs: T[], t: number): T | null {
  for (let i = 0; i < segs.length; i++) {
    const s = segs[i]
    if (t >= s.start && (t < s.end || (i === segs.length - 1 && t <= s.end))) return s
  }
  return null
}

export function segmentsToIntervals(segs: Segment[], source: LabelSource = 'vlm'): Interval[] {
  return segs.map((s) => ({ start: s.start, end: s.end, position: s.position, top: s.top, source }))
}

export function intervalsToSegments(intervals: Interval[]): Segment[] {
  return intervals.map((i) => ({
    start: i.start,
    end: i.end,
    position: i.position,
    top: i.top,
    stable: i.end - i.start >= STABLE_MIN_S,
  }))
}
