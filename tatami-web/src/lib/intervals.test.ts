import { describe, expect, it } from 'vitest'
import type { Interval } from '../api/types'
import { coalesce, mergeWithNext, moveBoundary, splitAt, updateInterval, validateIntervals } from './intervals'

const base: Interval[] = [
  { start: 0, end: 10, position: 'standing', top: 'none', source: 'vlm' },
  { start: 10, end: 20, position: 'closed_guard', top: 'A', source: 'vlm' },
  { start: 20, end: 30, position: 'side_control', top: 'A', source: 'vlm' },
]

describe('updateInterval', () => {
  it('cambia posición y top y marca human', () => {
    const r = updateInterval(base, 1, { position: 'half_guard', top: 'B' })
    expect(r[1]).toMatchObject({ position: 'half_guard', top: 'B', source: 'human' })
    expect(r[0]).toBe(base[0])
  })
  it('fuerza top none en posiciones sin top', () => {
    expect(updateInterval(base, 1, { position: 'standing' })[1].top).toBe('none')
  })
  it('índice inválido no cambia nada', () => expect(updateInterval(base, 9, { top: 'B' })).toBe(base))
})

describe('splitAt', () => {
  it('parte el intervalo que contiene t', () => {
    const r = splitAt(base, 15)
    expect(r).toHaveLength(4)
    expect(r[1]).toMatchObject({ start: 10, end: 15, source: 'human' })
    expect(r[2]).toMatchObject({ start: 15, end: 20, position: 'closed_guard' })
    expect(validateIntervals(r)).toEqual([])
  })
  it('ignora t en un borde o fuera', () => {
    expect(splitAt(base, 10)).toBe(base)
    expect(splitAt(base, 10.05)).toBe(base)
    expect(splitAt(base, 99)).toBe(base)
  })
})

describe('mergeWithNext', () => {
  it('une con el siguiente conservando el primero', () => {
    const r = mergeWithNext(base, 0)
    expect(r).toHaveLength(2)
    expect(r[0]).toMatchObject({ start: 0, end: 20, position: 'standing', source: 'human' })
    expect(validateIntervals(r)).toEqual([])
  })
  it('el último no tiene siguiente', () => expect(mergeWithNext(base, 2)).toBe(base))
})

describe('moveBoundary', () => {
  it('mueve el borde entre dos intervalos', () => {
    const r = moveBoundary(base, 0, 7.5)
    expect(r[0].end).toBe(7.5)
    expect(r[1].start).toBe(7.5)
    expect(validateIntervals(r)).toEqual([])
  })
  it('acota para no invertir intervalos', () => {
    const r = moveBoundary(base, 0, 25)
    expect(r[0].end).toBeLessThan(r[1].end)
    expect(r[1].end - r[1].start).toBeGreaterThan(0)
    const r2 = moveBoundary(base, 0, -5)
    expect(r2[0].end).toBeGreaterThan(r2[0].start)
  })
})

describe('validateIntervals', () => {
  it('detecta huecos y top inválido', () => {
    const bad: Interval[] = [
      { start: 0, end: 5, position: 'standing', top: 'A', source: 'human' },
      { start: 6, end: 8, position: 'mount', top: 'B', source: 'human' },
    ]
    expect(validateIntervals(bad).length).toBe(2)
  })
})

describe('coalesce', () => {
  it('une vecinos idénticos', () => {
    const r = coalesce([base[0], { ...base[0], start: 10, end: 12 }, base[1]])
    expect(r).toHaveLength(2)
    expect(r[0].end).toBe(12)
  })
})
