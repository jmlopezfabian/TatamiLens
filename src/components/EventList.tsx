import type { ScoreEvent } from '../api/types'
import { formatTime } from '../lib/format'

interface Props {
  events: ScoreEvent[]
  time: number
  onSeek: (t: number) => void
}

export default function EventList({ events, time, onSeek }: Props) {
  const sorted = [...events].sort((a, b) => a.t - b.t)
  return (
    <div className="card">
      <h3 className="mb-2 font-semibold">Eventos</h3>
      {sorted.length === 0 ? (
        <p className="muted">No hay eventos de puntos.</p>
      ) : (
        <ul className="divide-y divide-slate-100 dark:divide-slate-800">
          {sorted.map((e, i) => (
            <li key={i}>
              <button
                onClick={() => onSeek(e.t)}
                className={`flex w-full cursor-pointer items-center gap-3 px-1 py-1.5 text-left text-sm hover:bg-slate-100 dark:hover:bg-slate-800 ${e.t <= time ? '' : 'opacity-60'}`}
              >
                <span className="w-12 font-mono text-xs text-slate-500">{formatTime(e.t)}</span>
                <span className={`w-6 text-center font-bold ${e.athlete === 'A' ? 'text-blue-600' : 'text-orange-600'}`}>{e.athlete}</span>
                <span className="flex-1">{e.reason}</span>
                <span className="font-semibold tabular-nums">+{e.points}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
