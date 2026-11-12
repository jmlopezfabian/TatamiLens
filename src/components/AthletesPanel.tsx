import type { Athlete } from '../api/types'

interface Props {
  names: { A: string; B: string }
  descriptions: Record<Athlete, string>
  onNames: (n: { A: string; B: string }) => void
  onDescriptions: (d: Record<Athlete, string>) => void
  saving?: boolean
}

export default function AthletesPanel({ names, descriptions, onNames, onDescriptions, saving }: Props) {
  return (
    <div className="card space-y-3">
      <h3 className="font-semibold">Atletas</h3>
      {(['A', 'B'] as const).map((k) => (
        <div key={k} className="space-y-1.5">
          <label className="label" htmlFor={`name-${k}`}>
            Atleta {k}
          </label>
          <input
            id={`name-${k}`}
            className="input w-full"
            value={names[k]}
            onChange={(e) => onNames({ ...names, [k]: e.target.value })}
            placeholder={`Nombre de ${k}`}
          />
          <input
            key={descriptions[k]}
            className="input w-full text-xs"
            defaultValue={descriptions[k]}
            aria-label={`Descripción de ${k}`}
            placeholder="Descripción (ropa, cinta…)"
            disabled={saving}
            onBlur={(e) => {
              if (e.target.value !== descriptions[k]) onDescriptions({ ...descriptions, [k]: e.target.value })
            }}
          />
        </div>
      ))}
      <p className="muted text-xs">El nombre se guarda en este navegador; la descripción se guarda con las etiquetas.</p>
    </div>
  )
}
