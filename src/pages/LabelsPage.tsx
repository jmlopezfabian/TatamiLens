import { useState } from 'react'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import { api } from '../api'
import { useActiveAnalysis, useLabels, useRescore, useSaveLabels, useVideo } from '../api/hooks'
import type { Athlete, Interval, Labels, Position, ScoreEvent, Top } from '../api/types'
import Player from '../components/Player'
import Scoreboard from '../components/Scoreboard'
import Timeline from '../components/Timeline'
import { useAthleteNames, usePlayback } from '../hooks'
import { mergeWithNext, moveBoundary, splitAt, updateInterval, validateIntervals } from '../lib/intervals'
import { formatTime } from '../lib/format'
import { positionMeta, POSITIONS } from '../lib/positions'
import { intervalsToSegments, segmentsToIntervals } from '../lib/segments'

export default function LabelsPage() {
  const { id = '' } = useParams()
  const [params] = useSearchParams()
  const video = useVideo(id)
  const labels = useLabels(id)
  const { analysis, activeId } = useActiveAnalysis(id, params.get('analysis'))

  const result = analysis.data?.result ?? null
  if (video.isLoading || labels.isLoading || (activeId && analysis.isLoading)) return <p className="muted">Cargando…</p>
  if (video.isError || !video.data) return <p className="text-sm text-red-600">No se pudo cargar el video.</p>
  if (!result && !labels.data) {
    return (
      <p className="muted">
        No hay análisis terminado ni etiquetas para este video. <Link className="underline" to={`/videos/${id}`}>Volver</Link>
      </p>
    )
  }
  const initial: Labels = labels.data ?? {
    video: result!.video.path,
    athletes: { A: result!.athletes.A.description, B: result!.athletes.B.description },
    intervals: segmentsToIntervals(result!.segments, 'vlm'),
  }
  const duration = result?.video.duration_s ?? video.data.duration_s ?? initial.intervals.at(-1)?.end ?? 0
  return (
    <LabelEditor
      key={id}
      videoId={id}
      filename={video.data.filename}
      initial={initial}
      duration={duration}
      analysisId={result ? (activeId ?? null) : null}
      baseEvents={result?.events ?? []}
    />
  )
}

interface EditorProps {
  videoId: string
  filename: string
  initial: Labels
  duration: number
  analysisId: string | null
  baseEvents: ScoreEvent[]
}

function LabelEditor({ videoId, filename, initial, duration, analysisId, baseEvents }: EditorProps) {
  const [labels, setLabels] = useState<Labels>(initial)
  const [savedJson, setSavedJson] = useState(JSON.stringify(initial))
  const [selected, setSelected] = useState<number | null>(null)
  const [events, setEvents] = useState<ScoreEvent[]>(baseEvents)
  const [names] = useAthleteNames(videoId)
  const playback = usePlayback(api.streamUrl(videoId), duration)
  const save = useSaveLabels(videoId)
  const rescore = useRescore(videoId)

  const intervals = labels.intervals
  const errors = validateIntervals(intervals)
  const dirty = JSON.stringify(labels) !== savedJson
  const setIntervals = (next: Interval[]) => setLabels((l) => ({ ...l, intervals: next }))
  const sel = selected !== null && selected < intervals.length ? selected : null

  const doSave = async () => {
    await save.mutateAsync(labels)
    setSavedJson(JSON.stringify(labels))
  }
  const doRescore = async () => {
    if (!analysisId) return
    if (dirty) await doSave()
    const r = await rescore.mutateAsync(analysisId)
    setEvents(r.events)
  }
  const exportJson = () => {
    const blob = new Blob([JSON.stringify(labels, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${filename.replace(/\.[^.]+$/, '')}.labels.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const t = playback.time
  const busy = save.isPending || rescore.isPending
  const err = (save.error ?? rescore.error) as Error | null

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <Link to={`/videos/${videoId}`} className="muted hover:underline">← Volver al análisis</Link>
          <h1 className="break-all text-2xl font-semibold">Corregir etiquetas · {filename}</h1>
        </div>
        <div className="flex flex-wrap gap-2">
          <button className="btn" onClick={exportJson}>Exportar JSON</button>
          <button className="btn" onClick={doRescore} disabled={busy || !analysisId || errors.length > 0}>
            {rescore.isPending ? 'Recalculando…' : 'Recalcular marcador'}
          </button>
          <button className="btn btn-primary" onClick={doSave} disabled={busy || !dirty || errors.length > 0}>
            {save.isPending ? 'Guardando…' : dirty ? 'Guardar' : 'Guardado'}
          </button>
        </div>
      </div>
      {err && <p className="text-sm text-red-600">{err.message}</p>}
      {errors.length > 0 && (
        <ul className="rounded-lg border border-red-300 p-3 text-sm text-red-600 dark:border-red-900">
          {errors.map((e) => <li key={e}>{e}</li>)}
        </ul>
      )}

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-2">
          <Player src={api.streamUrl(videoId)} playback={playback} duration={duration} />
          <Timeline
            duration={duration}
            segments={intervalsToSegments(intervals)}
            events={events}
            time={t}
            onSeek={playback.seek}
            selected={sel}
            onSelect={setSelected}
          />
          <div className="card flex flex-wrap items-center gap-2">
            <span className="muted mr-2">Tiempo actual {formatTime(t, 1)}</span>
            <button className="btn" onClick={() => setIntervals(splitAt(intervals, t))}>Partir aquí</button>
            <button className="btn" disabled={sel === null || sel >= intervals.length - 1} onClick={() => sel !== null && setIntervals(mergeWithNext(intervals, sel))}>
              Unir con el siguiente
            </button>
            <button className="btn" disabled={sel === null || sel === 0} onClick={() => sel !== null && setIntervals(moveBoundary(intervals, sel - 1, t))}>
              Borde inicial = ahora
            </button>
            <button className="btn" disabled={sel === null || sel >= intervals.length - 1} onClick={() => sel !== null && setIntervals(moveBoundary(intervals, sel, t))}>
              Borde final = ahora
            </button>
          </div>
        </div>
        <div className="space-y-4">
          <Scoreboard events={events} time={t} names={names} />
          <div className="card space-y-2">
            <h3 className="font-semibold">Descripciones</h3>
            {(['A', 'B'] as Athlete[]).map((k) => (
              <input
                key={k}
                className="input w-full"
                aria-label={`Descripción de ${k}`}
                placeholder={`Descripción de ${k}`}
                value={labels.athletes[k]}
                onChange={(e) => setLabels((l) => ({ ...l, athletes: { ...l.athletes, [k]: e.target.value } }))}
              />
            ))}
          </div>
        </div>
      </div>

      <div className="card overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs uppercase text-slate-500">
              <th className="p-1">#</th><th className="p-1">Inicio</th><th className="p-1">Fin</th>
              <th className="p-1">Posición</th><th className="p-1">Arriba</th><th className="p-1">Origen</th><th />
            </tr>
          </thead>
          <tbody>
            {intervals.map((iv, i) => (
              <tr
                key={i}
                onClick={() => { setSelected(i); playback.seek(iv.start) }}
                className={`cursor-pointer border-t border-slate-100 dark:border-slate-800 ${sel === i ? 'bg-indigo-50 dark:bg-indigo-950/40' : ''}`}
              >
                <td className="p-1">
                  <span className="mr-1 inline-block h-2.5 w-2.5 rounded-sm" style={{ background: positionMeta(iv.position).color }} />
                  {i + 1}
                </td>
                <td className="p-1">
                  <TimeInput value={iv.start} disabled={i === 0} onCommit={(v) => setIntervals(moveBoundary(intervals, i - 1, v))} label={`Inicio ${i + 1}`} />
                </td>
                <td className="p-1">
                  <TimeInput value={iv.end} disabled={i === intervals.length - 1} onCommit={(v) => setIntervals(moveBoundary(intervals, i, v))} label={`Fin ${i + 1}`} />
                </td>
                <td className="p-1">
                  <select
                    className="input"
                    aria-label={`Posición ${i + 1}`}
                    value={iv.position}
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => setIntervals(updateInterval(intervals, i, { position: e.target.value as Position }))}
                  >
                    {POSITIONS.map((p) => <option key={p.id} value={p.id}>{p.label}</option>)}
                  </select>
                </td>
                <td className="p-1">
                  <select
                    className="input"
                    aria-label={`Arriba ${i + 1}`}
                    value={iv.top}
                    disabled={positionMeta(iv.position).noTop}
                    onClick={(e) => e.stopPropagation()}
                    onChange={(e) => setIntervals(updateInterval(intervals, i, { top: e.target.value as Top }))}
                  >
                    <option value="none">—</option><option value="A">A</option><option value="B">B</option>
                  </select>
                </td>
                <td className="p-1 text-xs text-slate-500">{iv.source}</td>
                <td className="p-1 text-right">
                  <button
                    className="btn"
                    disabled={i >= intervals.length - 1}
                    onClick={(e) => { e.stopPropagation(); setIntervals(mergeWithNext(intervals, i)) }}
                  >
                    Unir ↓
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function TimeInput({ value, onCommit, disabled, label }: { value: number; onCommit: (v: number) => void; disabled?: boolean; label: string }) {
  return (
    <input
      key={value}
      className="input w-24 font-mono"
      type="number"
      step={0.1}
      min={0}
      defaultValue={value}
      disabled={disabled}
      aria-label={label}
      onClick={(e) => e.stopPropagation()}
      onBlur={(e) => {
        const v = parseFloat(e.target.value)
        if (Number.isFinite(v) && v !== value) onCommit(v)
      }}
      onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
    />
  )
}
