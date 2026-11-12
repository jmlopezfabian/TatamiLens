import { describe, expect, it } from 'vitest'
import { formatTime } from './format'

describe('formatTime', () => {
  it('formatea mm:ss', () => expect(formatTime(75.3)).toBe('1:15'))
  it('con decimales', () => expect(formatTime(5.25, 1)).toBe('0:05.3'))
  it('valores inválidos o negativos', () => {
    expect(formatTime(-3)).toBe('0:00')
    expect(formatTime(NaN)).toBe('0:00')
  })
})
