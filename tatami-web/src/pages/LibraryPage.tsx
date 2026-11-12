import { Link } from 'react-router-dom'
import { useDeleteVideo, useVideos } from '../api/hooks'
import UploadZone from '../components/UploadZone'
import { formatDate, formatTime } from '../lib/format'

export default function LibraryPage() {
  const videos = useVideos()
  const del = useDeleteVideo()

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">Biblioteca</h1>
      <UploadZone />
      {videos.isLoading && <p className="muted">Cargando videos…</p>}
      {videos.isError && (
        <p className="text-sm text-red-600">No se pudo cargar la biblioteca: {(videos.error as Error).message}</p>
      )}
      {videos.data && videos.data.length === 0 && <p className="muted">Todavía no hay videos. Sube el primero.</p>}
      <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {videos.data?.map((v) => (
          <li key={v.id} className="card flex flex-col gap-3">
            <div>
              <Link to={`/videos/${v.id}`} className="break-all font-medium text-indigo-600 hover:underline dark:text-indigo-400">
                {v.filename}
              </Link>
              <p className="muted">
                {v.duration_s != null ? formatTime(v.duration_s) : 'duración desconocida'}
                {v.fps != null && ` · ${v.fps} fps`}
              </p>
              <p className="muted text-xs">{formatDate(v.created_at)}</p>
            </div>
            <div className="mt-auto flex gap-2">
              <Link className="btn btn-primary" to={`/videos/${v.id}`}>Abrir</Link>
              <button
                className="btn btn-danger"
                disabled={del.isPending}
                onClick={() => {
                  if (confirm(`¿Borrar "${v.filename}" y sus análisis?`)) del.mutate(v.id)
                }}
              >
                Borrar
              </button>
            </div>
          </li>
        ))}
      </ul>
      {del.isError && <p className="text-sm text-red-600">Error al borrar: {(del.error as Error).message}</p>}
    </div>
  )
}
