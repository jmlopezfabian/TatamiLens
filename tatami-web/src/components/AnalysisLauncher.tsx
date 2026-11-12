import { useState } from 'react'
import { useBackends, useCreateAnalysis } from '../api/hooks'

interface Props {
  videoId: string
  onCreated: (analysisId: string) => void
}

export default function AnalysisLauncher({ videoId, onCreated }: Props) {
  const backends = useBackends()
  const create = useCreateAnalysis(videoId)
  const [backend, setBackend] = useState('')
  const [model, setModel] = useState('')
  const [fps, setFps] = useState(2)

  const list = backends.data ?? []
  const current = list.find((b) => b.name === backend) ?? list.find((b) => b.available) ?? list[0]
  const models = current?.models ?? []
  const selModel = models.includes(model) ? model : (models[0] ?? '')

  const submit = () => {
    if (!current) return
    create.mutate(
      { backend: current.name, model: selModel, sample_fps: fps },
      { onSuccess: (a) => onCreated(a.id) },
    )
  }

  return (
    <div className="card space-y-3">
      <h3 className="font-semibold">Nuevo análisis</h3>
      {backends.isLoading && <p className="muted">Cargando backends…</p>}
      {backends.isError && <p className="text-sm text-red-600">No se pudieron cargar los backends: {(backends.error as Error).message}</p>}
      {current && (
        <>
          <div>
            <label className="label" htmlFor="backend">Backend</label>
            <select id="backend" className="input w-full" value={current.name} onChange={(e) => { setBackend(e.target.value); setModel('') }}>
              {list.map((b) => (
                <option key={b.name} value={b.name} disabled={!b.available}>
                  {b.name}{b.available ? '' : ' (no disponible)'}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="label" htmlFor="model">Modelo</label>
            <select id="model" className="input w-full" value={selModel} onChange={(e) => setModel(e.target.value)}>
              {models.map((m) => <option key={m}>{m}</option>)}
            </select>
          </div>
          <div>
            <label className="label" htmlFor="fps">FPS de muestreo</label>
            <input id="fps" className="input w-full" type="number" min={0.25} max={10} step={0.25} value={fps} onChange={(e) => setFps(Number(e.target.value) || 1)} />
          </div>
          <button className="btn btn-primary w-full" onClick={submit} disabled={create.isPending || !current.available || !selModel}>
            {create.isPending ? 'Enviando…' : 'Analizar'}
          </button>
        </>
      )}
      {create.isError && <p className="text-sm text-red-600">{(create.error as Error).message}</p>}
    </div>
  )
}
