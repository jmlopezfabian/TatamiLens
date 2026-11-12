import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api'
import { useActiveAnalysis, useLabels, useSaveLabels, useVideo } from '../api/hooks'
import type { Analysis, Athlete, Labels } from '../api/types'
import AnalysisLauncher from '../components/AnalysisLauncher'
import AthletesPanel from '../components/AthletesPanel'
import EventList from '../components/EventList'
import Player from '../components/Player'
import Scoreboard from '../components/Scoreboard'
import Timeline from '../components/Timeline'
import { useAthleteNames, usePlayback } from '../hooks'
import { formatDate } from '../lib/format'
import { segmentsToIntervals } from '../lib/segments'

const STATUS: Record<Analysis['status'], string> = {
  queued: 'En cola',
  running: 'Analizando',
  done: 'Listo',
  error: 'Error',
}

export default function VideoPage() {
  const { id = '' } = useParams()
  const video = useVideo(id)
  if (video.isLoading) return <p className="muted">Cargando…</p>
  if (video.isError || !video.data) {
    return (
      <p className="text-sm text-red-600">
        No se pudo cargar el video: {(video.error as Error | null)?.message ?? 'no encontrado'}.{' '}
        <Link className="underline" to="/">Volver</Link>
      </p>
    )
  }
  return <VideoView key={id} id={id} filename={video.data.filename} videoDuration={video.data.duration_s} />
}

function VideoView({ id, filename, videoDuration }: { id: string; filename: string; videoDuration: number | null }) {
  const [picked, setPicked] = useState<string | null>(null)
  const { items, activeId, analysis } = useActiveAnalysis(id, picked)
  const labels = useLabels(id)
  const saveLabels = useSaveLabels(id)
  const [names, setNames] = useAthleteNames(id)

  const a = analysis.data
  const result = a?.result ?? null
  const duration =
    result?.video.duration_s ?? videoDuration ?? (result?.segments.at(-1)?.end ?? 0)
  const src = api.streamUrl(id)
  const playback = usePlayback(src, duration)

  const descriptions: Record<Athlete, string> = {
    A: labels.data?.athletes.A ?? result?.athletes.A.description ?? '',
    B: labels.data?.athletes.B ?? result?.athletes.B.description ?? '',
  }
  const saveDescriptions = (d: Record<Athlete, string>) => {
    if (!result) return
    const next: Labels = {
      video: labels.data?.video ?? result.video.path,
      athletes: d,
      intervals: labels.data?.intervals ?? segmentsToIntervals(result.segments, 'vlm'),
    }
    saveLabels.mutate(next)
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <Link to="/" className="muted hover:underline">← Biblioteca</Link>
          <h1 className="break-all text-2xl font-semibold">{filename}</h1>
        </div>
        {result && (
          <Link className="btn" to={`/videos/${id}/labels${activeId ? `?analysis=${activeId}` : ''}`}>
            Corregir etiquetas
          </Link>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-2">
          <Player src={src} playback={playback} duration={duration} />

          {a && a.status !== 'done' && <StatusCard analysis={a} />}

          {result ? (
            <>
              <Timeline
                duration={duration}
                segments={result.segments}
                events={result.events}
                time={playback.time}
                onSeek={playback.seek}
              />
              <Scoreboard events={result.events} time={playback.time} names={names} />
            </>
          ) : (
            !a && <p className="muted">Este video aún no tiene análisis. Lanza uno desde el panel de la derecha.</p>
          )}
        </div>

        <div className="space-y-4">
          <AnalysisLauncher videoId={id} onCreated={setPicked} />
          {items.length > 0 && (
            <div className="card">
              <h3 className="mb-2 font-semibold">Análisis</h3>
              <ul className="space-y-1">
                {items.map((it) => (
                  <li key={it.id}>
                    <button
                      onClick={() => setPicked(it.id)}
                      className={`w-full cursor-pointer rounded-md px-2 py-1.5 text-left text-sm ${it.id === activeId ? 'bg-indigo-50 dark:bg-indigo-950/50' : 'hover:bg-slate-100 dark:hover:bg-slate-800'}`}
                    >
                      <span className="font-medium">{it.backend}/{it.model}</span>
                      <span className="muted block text-xs">{STATUS[it.status]} · {formatDate(it.created_at)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {result && (
            <>
              <AthletesPanel
                names={names}
                descriptions={descriptions}
                onNames={setNames}
                onDescriptions={saveDescriptions}
                saving={saveLabels.isPending}
              />
              <EventList events={result.events} time={playback.time} onSeek={playback.seek} />
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function StatusCard({ analysis }: { analysis: Analysis }) {
  if (analysis.status === 'error') {
    return (
      <div className="card border-red-300 dark:border-red-900">
        <p className="font-medium text-red-600">El análisis falló</p>
        <p className="muted">{analysis.error ?? 'Error desconocido'}</p>
      </div>
    )
  }
  const pct = Math.round(analysis.progress * 100)
  return (
    <div className="card">
      <div className="mb-2 flex justify-between text-sm">
        <span>{STATUS[analysis.status]} ({analysis.backend}/{analysis.model})</span>
        <span className="tabular-nums">{pct}%</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800" role="progressbar" aria-valuenow={pct}>
        <div className="h-full bg-indigo-600 transition-all" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}
