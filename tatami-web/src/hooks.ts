import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Reloj de reproducción unificado. Con video real sigue `video.currentTime`;
 * sin video (mock o error de carga) usa un reloj simulado para poder probar la UI.
 */
export function usePlayback(src: string | null, duration: number) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [failedSrc, setFailedSrc] = useState<string | null>(null)
  const hasVideo = !!src && failedSrc !== src
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    if (!playing) return
    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      if (hasVideo) {
        const v = videoRef.current
        if (v) setTime(v.currentTime)
      } else {
        const dt = (now - last) / 1000
        last = now
        setTime((t) => {
          const n = t + dt
          if (n >= duration) {
            setPlaying(false)
            return duration
          }
          return n
        })
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [playing, hasVideo, duration])

  const seek = useCallback(
    (t: number) => {
      const c = Math.min(Math.max(0, t), duration || Infinity)
      const v = videoRef.current
      if (hasVideo && v) v.currentTime = c
      setTime(c)
    },
    [hasVideo, duration],
  )

  const toggle = useCallback(() => {
    const v = videoRef.current
    if (hasVideo && v) {
      if (v.paused) void v.play().catch(() => undefined)
      else v.pause()
    } else {
      setTime((t) => (t >= duration ? 0 : t))
      setPlaying((p) => !p)
    }
  }, [hasVideo, duration])

  const videoProps = {
    onPlay: () => setPlaying(true),
    onPause: () => setPlaying(false),
    onEnded: () => setPlaying(false),
    onSeeked: () => setTime(videoRef.current?.currentTime ?? 0),
    onTimeUpdate: () => setTime(videoRef.current?.currentTime ?? 0),
    onError: () => setFailedSrc(src),
  }

  return { videoRef, hasVideo, time, playing, seek, toggle, videoProps }
}

export type Playback = ReturnType<typeof usePlayback>

const nameKey = (videoId: string) => `tatami:names:${videoId}`

/** Nombres visibles de los atletas (por video, en localStorage). */
export function useAthleteNames(videoId: string) {
  const [names, setNames] = useState<{ A: string; B: string }>(() => {
    try {
      const raw = localStorage.getItem(nameKey(videoId))
      if (raw) return { A: 'Atleta A', B: 'Atleta B', ...JSON.parse(raw) }
    } catch {
      /* ignorar */
    }
    return { A: 'Atleta A', B: 'Atleta B' }
  })
  const update = (next: { A: string; B: string }) => {
    setNames(next)
    try {
      localStorage.setItem(nameKey(videoId), JSON.stringify(next))
    } catch {
      /* ignorar */
    }
  }
  return [names, update] as const
}
