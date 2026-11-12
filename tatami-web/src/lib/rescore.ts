import type { Athlete, EventKind, Interval, ScoreEvent } from '../api/types'
import { KIND_LABELS } from './positions'

export const POINTS: Record<EventKind, number> = {
  takedown: 2,
  sweep: 2,
  guard_pass: 3,
  knee_on_belly: 2,
  mount: 4,
  back_take: 4,
}

const GUARDS = new Set(['closed_guard', 'half_guard', 'open_guard'])
const PASSED_TO = new Set(['side_control', 'north_south'])
const other = (a: Athlete): Athlete => (a === 'A' ? 'B' : 'A')

/**
 * Aproximación local de las reglas IBJJF a partir de transiciones entre intervalos.
 * Sólo la usa el cliente mock; el backend real calcula con `tatami_vision.rescore`.
 */
export function deriveEvents(intervals: Interval[]): ScoreEvent[] {
  const events: ScoreEvent[] = []
  const push = (t: number, athlete: Athlete, kind: EventKind) =>
    events.push({ t, athlete, points: POINTS[kind], kind, reason: KIND_LABELS[kind] })
  for (let i = 1; i < intervals.length; i++) {
    const p = intervals[i - 1]
    const c = intervals[i]
    if (c.top === 'none') continue
    const t = c.start
    if (c.position === 'standing' || c.position === 'scramble') continue
    if (p.position === 'standing' && c.position !== 'back_control') push(t, c.top, 'takedown')
    else if (GUARDS.has(p.position) && PASSED_TO.has(c.position) && p.top === c.top) push(t, c.top, 'guard_pass')
    else if (GUARDS.has(p.position) && GUARDS.has(c.position) && p.top !== 'none' && p.top === other(c.top)) push(t, c.top, 'sweep')
    else if (GUARDS.has(p.position) && p.top !== 'none' && p.top === other(c.top)) push(t, c.top, 'sweep')
    else if (c.position === 'mount' && p.position !== 'mount') push(t, c.top, 'mount')
    else if (c.position === 'knee_on_belly' && p.position !== 'knee_on_belly') push(t, c.top, 'knee_on_belly')
    else if (c.position === 'back_control' && p.position !== 'back_control') push(t, c.top, 'back_take')
  }
  return events
}

export function totalScore(events: ScoreEvent[]): Record<Athlete, number> {
  const s: Record<Athlete, number> = { A: 0, B: 0 }
  for (const e of events) s[e.athlete] += e.points
  return s
}
