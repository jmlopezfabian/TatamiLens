import type { Interval, Position, Top } from '../api/types'
import { normalizeTop } from './positions'

/** Duración mínima de un intervalo al partir o mover bordes. */
export const MIN_INTERVAL_S = 0.1

const round = (x: number) => Math.round(x * 1000) / 1000

/** Cambia posición/top de un intervalo y lo marca como humano. */
export function updateInterval(
  intervals: Interval[],
  index: number,
  patch: { position?: Position; top?: Top },
): Interval[] {
  const cur = intervals[index]
  if (!cur) return intervals
  const position = patch.position ?? cur.position
  const top = normalizeTop(position, patch.top ?? cur.top)
  return intervals.map((iv, i) => (i === index ? { ...iv, position, top, source: 'human' } : iv))
}

/** Parte el intervalo que contiene t en dos. No hace nada si t cae fuera o muy cerca de un borde. */
export function splitAt(intervals: Interval[], t: number): Interval[] {
  const idx = intervals.findIndex((iv) => t > iv.start + MIN_INTERVAL_S && t < iv.end - MIN_INTERVAL_S)
  if (idx < 0) return intervals
  const iv = intervals[idx]
  const cut = round(t)
  return [
    ...intervals.slice(0, idx),
    { ...iv, end: cut, source: 'human' },
    { ...iv, start: cut, source: 'human' },
    ...intervals.slice(idx + 1),
  ]
}

/** Une el intervalo `index` con el siguiente; el resultado conserva position/top del primero. */
export function mergeWithNext(intervals: Interval[], index: number): Interval[] {
  const a = intervals[index]
  const b = intervals[index + 1]
  if (!a || !b) return intervals
  return [
    ...intervals.slice(0, index),
    { ...a, end: b.end, source: 'human' },
    ...intervals.slice(index + 2),
  ]
}

/**
 * Mueve el borde entre `index` e `index + 1` a t (acotado para que ninguno quede
 * con menos de MIN_INTERVAL_S).
 */
export function moveBoundary(intervals: Interval[], index: number, t: number): Interval[] {
  const a = intervals[index]
  const b = intervals[index + 1]
  if (!a || !b) return intervals
  const lo = a.start + MIN_INTERVAL_S
  const hi = b.end - MIN_INTERVAL_S
  if (lo > hi) return intervals
  const cut = round(Math.min(hi, Math.max(lo, t)))
  return intervals.map((iv, i) =>
    i === index ? { ...iv, end: cut, source: 'human' } : i === index + 1 ? { ...iv, start: cut, source: 'human' } : iv,
  )
}

/** Devuelve lista de problemas (vacía si los intervalos son válidos y contiguos). */
export function validateIntervals(intervals: Interval[]): string[] {
  const errs: string[] = []
  intervals.forEach((iv, i) => {
    if (!(iv.end > iv.start)) errs.push(`Intervalo ${i + 1}: el fin debe ser mayor que el inicio`)
    if (normalizeTop(iv.position, iv.top) !== iv.top) errs.push(`Intervalo ${i + 1}: ${iv.position} no admite top`)
    const prev = intervals[i - 1]
    if (prev && Math.abs(prev.end - iv.start) > 1e-6) errs.push(`Intervalo ${i + 1}: no es contiguo al anterior`)
  })
  return errs
}

/** Une vecinos idénticos en (position, top). */
export function coalesce(intervals: Interval[]): Interval[] {
  const out: Interval[] = []
  for (const iv of intervals) {
    const last = out[out.length - 1]
    if (last && last.position === iv.position && last.top === iv.top && last.source === iv.source) {
      out[out.length - 1] = { ...last, end: iv.end }
    } else out.push({ ...iv })
  }
  return out
}
