import { ApiError, type TatamiApi } from './client'
import { buildResult, SAMPLE_DURATION, SAMPLE_INTERVALS } from './sample'
import type { Analysis, BackendInfo, CreateAnalysisRequest, Labels, Video } from './types'
import { deriveEvents, totalScore } from '../lib/rescore'
import { intervalsToSegments } from '../lib/segments'

interface MockAnalysis {
  base: Analysis
  req: CreateAnalysisRequest
  startedMs: number
  filename: string
}

const QUEUED_MS = 800
const RUN_MS = 3500
const delay = (ms = 120) => new Promise((r) => setTimeout(r, ms))
let counter = 0
const nextId = (p: string) => `${p}-${++counter}-${Math.random().toString(36).slice(2, 6)}`

/** Cliente en memoria: sin backend, sin video real (streamUrl = null). */
export function createMockApi(): TatamiApi {
  const videos = new Map<string, Video>()
  const analyses = new Map<string, MockAnalysis>()
  const labels = new Map<string, Labels>()

  const demo: Video = {
    id: 'demo',
    filename: 'demo_combate.mp4',
    duration_s: SAMPLE_DURATION,
    fps: 29.97,
    created_at: new Date(Date.now() - 3600_000).toISOString(),
  }
  videos.set(demo.id, demo)
  analyses.set('demo-analysis', {
    base: {
      id: 'demo-analysis', video_id: 'demo', status: 'done', progress: 1, backend: 'mock', model: 'mock-1',
      error: null, created_at: demo.created_at, result: null,
    },
    req: { backend: 'mock', model: 'mock-1', sample_fps: 2 },
    startedMs: 0,
    filename: demo.filename,
  })

  function view(a: MockAnalysis): Analysis {
    if (a.base.status === 'done' && a.base.result) return a.base
    const elapsed = Date.now() - a.startedMs
    if (a.startedMs === 0 || elapsed >= QUEUED_MS + RUN_MS) {
      a.base = {
        ...a.base, status: 'done', progress: 1,
        result: buildResult(SAMPLE_INTERVALS, {
          backend: a.req.backend, model: a.req.model, sampleFps: a.req.sample_fps, filename: a.filename,
        }),
      }
    } else if (elapsed < QUEUED_MS) {
      a.base = { ...a.base, status: 'queued', progress: 0 }
    } else {
      a.base = { ...a.base, status: 'running', progress: (elapsed - QUEUED_MS) / RUN_MS }
    }
    return a.base
  }

  const need = <T,>(v: T | undefined, what: string): T => {
    if (v === undefined) throw new ApiError(404, `${what} no encontrado`)
    return v
  }

  return {
    async health() { return { status: 'ok' } },
    async listVideos() {
      await delay()
      return [...videos.values()].sort((a, b) => b.created_at.localeCompare(a.created_at))
    },
    async getVideo(id) { await delay(); return need(videos.get(id), 'Video') },
    async uploadVideo(file, onProgress) {
      const steps = 10
      for (let i = 1; i <= steps; i++) {
        await delay(90)
        onProgress?.(i / steps)
      }
      const v: Video = {
        id: nextId('vid'), filename: file.name, duration_s: SAMPLE_DURATION, fps: 29.97, created_at: new Date().toISOString(),
      }
      videos.set(v.id, v)
      return v
    },
    async deleteVideo(id) {
      await delay()
      need(videos.get(id), 'Video')
      videos.delete(id)
      labels.delete(id)
      for (const [aid, a] of analyses) if (a.base.video_id === id) analyses.delete(aid)
    },
    streamUrl() { return null },
    async createAnalysis(videoId, req) {
      await delay()
      const video = need(videos.get(videoId), 'Video')
      const id = nextId('an')
      const a: MockAnalysis = {
        base: {
          id, video_id: videoId, status: 'queued', progress: 0, backend: req.backend, model: req.model,
          error: null, created_at: new Date().toISOString(), result: null,
        },
        req, startedMs: Date.now(), filename: video.filename,
      }
      analyses.set(id, a)
      return a.base
    },
    async getAnalysis(id) { await delay(60); return view(need(analyses.get(id), 'Análisis')) },
    async listAnalyses(videoId) {
      await delay(60)
      return [...analyses.values()]
        .filter((a) => a.base.video_id === videoId)
        .map(view)
        .sort((a, b) => b.created_at.localeCompare(a.created_at))
    },
    async getLabels(videoId) { await delay(60); return labels.get(videoId) ?? null },
    async putLabels(videoId, l) { await delay(); labels.set(videoId, structuredClone(l)); return l },
    async rescore(analysisId) {
      await delay(300)
      const a = need(analyses.get(analysisId), 'Análisis')
      const l = labels.get(a.base.video_id)
      if (!l) throw new ApiError(409, 'No hay etiquetas corregidas para este video')
      const current = view(a)
      const events = deriveEvents(l.intervals)
      const result = {
        ...need(current.result ?? undefined, 'Resultado'),
        segments: intervalsToSegments(l.intervals),
        events,
        score: totalScore(events),
      }
      a.base = { ...current, result }
      return result
    },
    async listBackends(): Promise<BackendInfo[]> {
      await delay(60)
      return [
        { name: 'mock', models: ['mock-1'], available: true },
        { name: 'ollama', models: ['qwen3-vl:8b', 'llama3.2-vision:11b'], available: true },
        { name: 'local', models: ['dinov2-logreg'], available: true },
        { name: 'anthropic', models: ['claude-sonnet-5-5'], available: false },
      ]
    },
  }
}
