import type { ScoreEvent } from '../api/types'
import { eventText, lastEventUpTo, scoreUpTo } from '../lib/score'

interface Props {
  events: ScoreEvent[]
  time: number
  names: { A: string; B: string }
}

export default function Scoreboard({ events, time, names }: Props) {
  const score = scoreUpTo(events, time)
  const last = lastEventUpTo(events, time)
  return (
    <div className="card" aria-live="polite">
      <div className="grid grid-cols-2 gap-3 text-center">
        {(['A', 'B'] as const).map((k) => (
          <div key={k} className={`rounded-lg p-3 ${k === 'A' ? 'bg-blue-600/10' : 'bg-orange-600/10'}`}>
            <div className="truncate text-xs font-medium text-slate-600 dark:text-slate-300">
              {k} · {names[k]}
            </div>
            <div className={`text-5xl font-bold tabular-nums ${k === 'A' ? 'text-blue-600 dark:text-blue-400' : 'text-orange-600 dark:text-orange-400'}`}>
              {score[k]}
            </div>
          </div>
        ))}
      </div>
      <p className="mt-3 text-center text-sm text-slate-600 dark:text-slate-300">
        {last ? <>Último evento: <strong>{eventText(last)}</strong></> : 'Sin eventos todavía'}
      </p>
    </div>
  )
}
