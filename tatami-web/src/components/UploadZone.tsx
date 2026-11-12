import { useRef, useState, type DragEvent } from 'react'
import { useUploadVideo } from '../api/hooks'

export default function UploadZone() {
  const [progress, setProgress] = useState<number | null>(null)
  const [over, setOver] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const upload = useUploadVideo((f) => setProgress(f))

  const start = (file: File | undefined) => {
    if (!file) return
    if (!file.type.startsWith('video/') && !/\.(mp4|mov|mkv|webm|avi|m4v)$/i.test(file.name)) {
      alert('Selecciona un archivo de video.')
      return
    }
    setProgress(0)
    upload.mutate(file, { onSettled: () => setProgress(null) })
  }
  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setOver(false)
    start(e.dataTransfer.files[0])
  }

  const busy = upload.isPending
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault()
        setOver(true)
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
      className={`rounded-xl border-2 border-dashed p-8 text-center transition ${over ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-950/40' : 'border-slate-300 dark:border-slate-700'}`}
    >
      {busy ? (
        <div className="mx-auto max-w-sm">
          <p className="mb-2 text-sm">Subiendo… {Math.round((progress ?? 0) * 100)}%</p>
          <div className="h-2 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800" role="progressbar" aria-valuenow={Math.round((progress ?? 0) * 100)}>
            <div className="h-full bg-indigo-600 transition-all" style={{ width: `${(progress ?? 0) * 100}%` }} />
          </div>
        </div>
      ) : (
        <>
          <p className="mb-3">Arrastra un video aquí o</p>
          <button className="btn btn-primary" onClick={() => input.current?.click()}>
            Elegir archivo
          </button>
          <input ref={input} type="file" accept="video/*" hidden onChange={(e) => { start(e.target.files?.[0]); e.target.value = '' }} />
        </>
      )}
      {upload.isError && <p className="mt-3 text-sm text-red-600">Error al subir: {(upload.error as Error).message}</p>}
    </div>
  )
}
