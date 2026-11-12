import { describe, expect, it } from 'vitest'
import type { Frame } from '../api/types'
import { framesToSegments, intervalsToSegments, segmentAt, segmentsToIntervals } from './segments'

const f = (t: number, position: Frame['position'], top: Frame['top'] = 'none'): Frame => ({ t, position, top, confidence: 0.9 })

describe('framesToSegments', () => {
  it('vacío', () => expect(framesToSegments([])).toEqual([]))
  it('agrupa consecutivos iguales y marca estables (>= 3 s)', () => {
    const frames = [f(0, 'standing'), f(0.5, 'standing'), f(1, 'standing'), f(1.5, 'standing'), f(2, 'standing'), f(2.5, 'standing'),
      f(3, 'closed_guard', 'A'), f(3.5, 'closed_guard', 'A')]
    const segs = framesToSegments(frames, 4)
    expect(segs).toEqual([
      { start: 0, end: 3, position: 'standing', top: 'none', stable: true },
      { start: 3, end: 4, position: 'closed_guard', top: 'A', stable: false },
    ])
  })
  it('distingue por top', () => {
    const segs = framesToSegments([f(0, 'mount', 'A'), f(1, 'mount', 'B')], 2)
    expect(segs).toHaveLength(2)
  })
  it('sin endTime usa el paso medio', () => {
    const segs = framesToSegments([f(0, 'standing'), f(1, 'standing')])
    expect(segs[0].end).toBe(2)
  })
})

describe('segmentAt', () => {
  const segs = [{ start: 0, end: 5 }, { start: 5, end: 10 }]
  it('inicio inclusivo, fin exclusivo', () => expect(segmentAt(segs, 5)).toBe(segs[1]))
  it('el último incluye su fin', () => expect(segmentAt(segs, 10)).toBe(segs[1]))
  it('fuera', () => expect(segmentAt(segs, 11)).toBeNull())
})

describe('conversión intervalos <-> segmentos', () => {
  it('ida y vuelta', () => {
    const segs = [{ start: 0, end: 4, position: 'standing' as const, top: 'none' as const, stable: true }]
    const ivs = segmentsToIntervals(segs, 'human')
    expect(ivs[0].source).toBe('human')
    expect(intervalsToSegments(ivs)).toEqual(segs)
  })
})
