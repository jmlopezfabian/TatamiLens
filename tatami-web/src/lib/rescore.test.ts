import { describe, expect, it } from 'vitest'
import type { Interval } from '../api/types'
import { deriveEvents, totalScore } from './rescore'

const iv = (start: number, end: number, position: Interval['position'], top: Interval['top']): Interval => ({ start, end, position, top, source: 'human' })

describe('deriveEvents', () => {
  const ivs = [
    iv(0, 10, 'standing', 'none'),
    iv(10, 20, 'closed_guard', 'A'),
    iv(20, 30, 'side_control', 'A'),
    iv(30, 40, 'mount', 'A'),
  ]
  it('detecta derribo, pase y montada', () => {
    const ev = deriveEvents(ivs)
    expect(ev.map((e) => e.kind)).toEqual(['takedown', 'guard_pass', 'mount'])
    expect(totalScore(ev)).toEqual({ A: 9, B: 0 })
  })
  it('detecta barrida cuando cambia el top en guardia', () => {
    const ev = deriveEvents([iv(0, 5, 'closed_guard', 'A'), iv(5, 10, 'closed_guard', 'B')])
    expect(ev).toHaveLength(1)
    expect(ev[0]).toMatchObject({ kind: 'sweep', athlete: 'B', points: 2 })
  })
  it('sin cambios de posición no hay eventos', () => expect(deriveEvents([iv(0, 5, 'standing', 'none')])).toEqual([]))
})
