import type { AnalysisResult, Frame, Interval, Position, Top } from './types'
import { deriveEvents, totalScore } from '../lib/rescore'
import { intervalsToSegments } from '../lib/segments'

export const SAMPLE_DURATION = 47.8

const sec = (start: number, end: number, position: Position, top: Top): Interval => ({
  start, end, position, top, source: 'vlm',
})

/** standing -> derribo -> guardia cerrada -> media guardia -> pase -> side control -> montada */
export const SAMPLE_INTERVALS: Interval[] = [
  sec(0, 9.5, 'standing', 'none'),
  sec(9.5, 11, 'scramble', 'none'),
  sec(11, 22, 'closed_guard', 'A'),
  sec(22, 28.5, 'half_guard', 'A'),
  sec(28.5, 38, 'side_control', 'A'),
  sec(38, SAMPLE_DURATION, 'mount', 'A'),
]

export function framesFromIntervals(intervals: Interval[], sampleFps = 2): Frame[] {
  const frames: Frame[] = []
  const last = intervals[intervals.length - 1]
  if (!last) return frames
  const dt = 1 / sampleFps
  for (let t = 0, n = 0; t < last.end; n++, t = n * dt) {
    const iv = intervals.find((i) => t >= i.start && t < i.end) ?? last
    const confidence = 0.72 + 0.25 * Math.abs(Math.sin(n * 1.7))
    frames.push({ t: Math.round(t * 100) / 100, position: iv.position, top: iv.top, confidence: Math.round(confidence * 100) / 100 })
  }
  return frames
}

export function buildResult(
  intervals: Interval[],
  opts: { backend: string; model: string; sampleFps: number; filename: string },
): AnalysisResult {
  const events = deriveEvents(intervals)
  return {
    schema_version: '1',
    video: { path: opts.filename, fps: 29.97, duration_s: SAMPLE_DURATION, width: 1280, height: 720 },
    backend: { name: opts.backend, model: opts.model, sample_fps: opts.sampleFps },
    athletes: {
      A: { label: 'Atleta A', description: 'Kimono blanco, cinta azul' },
      B: { label: 'Atleta B', description: 'Kimono azul, cinta morada' },
    },
    frames: framesFromIntervals(intervals, opts.sampleFps),
    segments: intervalsToSegments(intervals),
    events,
    score: totalScore(events),
    created_at: new Date().toISOString(),
  }
}
