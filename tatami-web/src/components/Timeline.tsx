import { useMemo, type MouseEvent } from 'react'
import type { ScoreEvent, Segment } from '../api/types'
import { formatTime } from '../lib/format'
import { positionLabel, positionMeta, TOP_COLORS, POSITIONS } from '../lib/positions'

interface Props {
  duration: number
  segments: Segment[]
  events?: ScoreEvent[]
  time: number
  onSeek: (t: number) => void
  selected?: number | null
  onSelect?: (index: number) => void
}

const pct = (x: number, d: number) => (d > 0 ? Math.min(100, Math.max(0, (x / d) * 100)) : 0)

export default function Timeline({ duration, segments, events = [], time, onSeek, selected, onSelect }: Props) {
  const used = useMemo(() => {
    const ids = new Set(segments.map((s) => s.position))
    return POSITIONS.filter((p) => ids.has(p.id))
  }, [segments])

  const handleClick = (e: MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const t = ((e.clientX - rect.left) / rect.width) * duration
    onSeek(Math.min(Math.max(0, t), duration))
    const idx = segments.findIndex((s, i) => t >= s.start && (t < s.end || i === segments.length - 1))
    if (idx >= 0) onSelect?.(idx)
  }

  return (
    <div className="card">
      <div className="relative cursor-pointer select-none pt-5" onClick={handleClick} data-testid="timeline">
        {/* marcadores de eventos */}
        {events.map((ev, i) => (
          <div
            key={i}
            className="absolute top-0 -translate-x-1/2 text-[10px] font-bold leading-none"
            style={{ left: `${pct(ev.t, duration)}%`, color: TOP_COLORS[ev.athlete] }}
            title={`${formatTime(ev.t)} · +${ev.points} ${ev.athlete} ${ev.reason}`}
          >
            <span className="block text-center">+{ev.points}</span>
            <span className="mx-auto block h-0 w-0 border-x-4 border-t-4 border-x-transparent" style={{ borderTopColor: TOP_COLORS[ev.athlete] }} />
          </div>
        ))}
        {/* posiciones */}
        <div className="relative h-9 overflow-hidden rounded-md bg-slate-200 dark:bg-slate-800">
          {segments.map((s, i) => (
            <div
              key={i}
              className={`absolute inset-y-0 border-r border-white/40 ${selected === i ? 'ring-2 ring-inset ring-white' : ''}`}
              style={{
                left: `${pct(s.start, duration)}%`,
                width: `${pct(s.end - s.start, duration)}%`,
                background: positionMeta(s.position).color,
              }}
              title={`${positionLabel(s.position)}${s.top !== 'none' ? ` (arriba: ${s.top})` : ''} · ${formatTime(s.start, 1)}–${formatTime(s.end, 1)}`}
            />
          ))}
        </div>
        {/* quién está arriba */}
        <div className="relative mt-1 h-3 overflow-hidden rounded-sm bg-slate-100 dark:bg-slate-800/60">
          {segments.map((s, i) =>
            s.top === 'none' ? null : (
              <div
                key={i}
                className="absolute inset-y-0 flex items-center justify-center text-[9px] font-bold leading-none text-white"
                style={{
                  left: `${pct(s.start, duration)}%`,
                  width: `${pct(s.end - s.start, duration)}%`,
                  background: TOP_COLORS[s.top],
                }}
              >
                {s.top}
              </div>
            ),
          )}
        </div>
        {/* cursor */}
        <div
          className="pointer-events-none absolute bottom-0 top-3 w-0.5 bg-slate-900 dark:bg-white"
          style={{ left: `${pct(time, duration)}%` }}
        />
      </div>
      <div className="mt-2 flex justify-between font-mono text-xs text-slate-500 dark:text-slate-400">
        <span>0:00</span>
        <span>{formatTime(time, 1)}</span>
        <span>{formatTime(duration)}</span>
      </div>
      <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs">
        {used.map((p) => (
          <li key={p.id} className="flex items-center gap-1.5">
            <span className="inline-block h-3 w-3 rounded-sm" style={{ background: p.color }} />
            {p.label}
          </li>
        ))}
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-sm" style={{ background: TOP_COLORS.A }} /> Arriba A
        </li>
        <li className="flex items-center gap-1.5">
          <span className="inline-block h-3 w-3 rounded-sm" style={{ background: TOP_COLORS.B }} /> Arriba B
        </li>
      </ul>
    </div>
  )
}
