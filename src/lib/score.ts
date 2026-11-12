import type { Athlete, ScoreEvent } from '../api/types'

/** Puntos acumulados por atleta con eventos en t' <= t. */
export function scoreUpTo(events: ScoreEvent[], t: number): Record<Athlete, number> {
  const score: Record<Athlete, number> = { A: 0, B: 0 }
  for (const e of events) if (e.t <= t) score[e.athlete] += e.points
  return score
}

/** Último evento ocurrido en t' <= t (el de mayor t; a igualdad, el último de la lista). */
export function lastEventUpTo(events: ScoreEvent[], t: number): ScoreEvent | null {
  let last: ScoreEvent | null = null
  for (const e of events) if (e.t <= t && (last === null || e.t >= last.t)) last = e
  return last
}

export function eventText(e: ScoreEvent): string {
  return `+${e.points} ${e.athlete} ${e.reason}`
}
