import { describe, expect, it } from 'vitest'
import type { ScoreEvent } from '../api/types'
import { eventText, lastEventUpTo, scoreUpTo } from './score'

const ev = (t: number, athlete: 'A' | 'B', points: number, reason: string): ScoreEvent => ({
  t, athlete, points, kind: 'takedown', reason,
})
const events = [ev(5, 'A', 2, 'derribo'), ev(20, 'A', 3, 'pase de guardia'), ev(30, 'B', 2, 'barrida')]

describe('scoreUpTo', () => {
  it('es 0-0 antes del primer evento', () => expect(scoreUpTo(events, 4.9)).toEqual({ A: 0, B: 0 }))
  it('incluye el evento en t exacto', () => expect(scoreUpTo(events, 5)).toEqual({ A: 2, B: 0 }))
  it('acumula por atleta', () => expect(scoreUpTo(events, 31)).toEqual({ A: 5, B: 2 }))
  it('lista vacía', () => expect(scoreUpTo([], 10)).toEqual({ A: 0, B: 0 }))
})

describe('lastEventUpTo', () => {
  it('null antes del primero', () => expect(lastEventUpTo(events, 1)).toBeNull())
  it('devuelve el último ocurrido', () => expect(lastEventUpTo(events, 25)?.reason).toBe('pase de guardia'))
  it('no depende del orden de la lista', () => {
    expect(lastEventUpTo([...events].reverse(), 25)?.t).toBe(20)
  })
  it('formatea el texto', () => expect(eventText(events[1])).toBe('+3 A pase de guardia'))
})
