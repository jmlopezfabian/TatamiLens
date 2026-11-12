import type { Playback } from '../hooks'
import { formatTime } from '../lib/format'

interface Props {
  src: string | null
  playback: Playback
  duration: number
}

/** Reproductor <video>; si no hay stream muestra un placeholder con reloj simulado. */
export default function Player({ src, playback, duration }: Props) {
  const { hasVideo, videoRef, videoProps, time, playing, toggle } = playback
  if (hasVideo) {
    return (
      <video
        ref={videoRef}
        src={src ?? undefined}
        controls
        preload="metadata"
        className="aspect-video w-full rounded-xl bg-black"
        {...videoProps}
      />
    )
  }
  return (
    <div className="flex aspect-video w-full flex-col items-center justify-center gap-3 rounded-xl bg-slate-900 text-slate-300">
      <p className="text-sm">Video no disponible (reloj simulado)</p>
      <div className="flex items-center gap-3">
        <button className="btn btn-primary" onClick={toggle}>
          {playing ? 'Pausar' : 'Reproducir'}
        </button>
        <span className="font-mono text-lg tabular-nums">
          {formatTime(time, 1)} / {formatTime(duration, 1)}
        </span>
      </div>
    </div>
  )
}
