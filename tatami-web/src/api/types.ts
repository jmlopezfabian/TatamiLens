// Tipos del contrato (docs/CONTRACTS.md)

export const POSITION_IDS = [
  'standing',
  'scramble',
  'closed_guard',
  'half_guard',
  'open_guard',
  'side_control',
  'north_south',
  'knee_on_belly',
  'mount',
  'back_control',
  'turtle',
  'separated',
  'not_visible',
] as const

export type Position = (typeof POSITION_IDS)[number]
export type Top = 'A' | 'B' | 'none'
export type Athlete = 'A' | 'B'
export type LabelSource = 'human' | 'vlm' | 'model'
export type EventKind = 'takedown' | 'sweep' | 'guard_pass' | 'knee_on_belly' | 'mount' | 'back_take'

export interface Video {
  id: string
  filename: string
  duration_s: number | null
  fps: number | null
  created_at: string
}

export interface Frame {
  t: number
  position: Position
  top: Top
  confidence: number
}

export interface Segment {
  start: number
  end: number
  position: Position
  top: Top
  stable: boolean
}

export interface ScoreEvent {
  t: number
  athlete: Athlete
  points: number
  kind: EventKind
  reason: string
}

export interface AnalysisResult {
  schema_version: string
  video: { path: string; fps: number; duration_s: number; width: number; height: number }
  backend: { name: string; model: string; sample_fps: number }
  athletes: Record<Athlete, { label: string; description: string }>
  frames: Frame[]
  segments: Segment[]
  events: ScoreEvent[]
  score: Record<Athlete, number>
  created_at: string
}

export type AnalysisStatus = 'queued' | 'running' | 'done' | 'error'

export interface Analysis {
  id: string
  video_id: string
  status: AnalysisStatus
  progress: number
  backend: string
  model: string
  error: string | null
  created_at: string
  result: AnalysisResult | null
}

export interface Interval {
  start: number
  end: number
  position: Position
  top: Top
  source: LabelSource
}

export interface Labels {
  video: string
  athletes: Record<Athlete, string>
  intervals: Interval[]
}

export interface BackendInfo {
  name: string
  models: string[]
  available: boolean
}

export interface CreateAnalysisRequest {
  backend: string
  model: string
  sample_fps: number
}
