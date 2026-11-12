import { useState } from 'react'
import { Link, Route, Routes } from 'react-router-dom'
import { USE_MOCK } from './api'
import LibraryPage from './pages/LibraryPage'
import VideoPage from './pages/VideoPage'
import LabelsPage from './pages/LabelsPage'

function ThemeToggle() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))
  const toggle = () => {
    const next = !dark
    document.documentElement.classList.toggle('dark', next)
    try {
      localStorage.setItem('tatami:theme', next ? 'dark' : 'light')
    } catch {
      /* sin almacenamiento */
    }
    setDark(next)
  }
  return (
    <button className="btn" onClick={toggle} aria-label="Cambiar tema">
      {dark ? 'Modo claro' : 'Modo oscuro'}
    </button>
  )
}

export default function App() {
  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur dark:border-slate-800 dark:bg-slate-900/80">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-3 px-4 py-3">
          <Link to="/" className="flex items-center gap-2 text-lg font-semibold">
            <span className="inline-block h-6 w-6 rounded-md bg-indigo-600" aria-hidden />
            TatamiLens
          </Link>
          <div className="flex items-center gap-3">
            {USE_MOCK && (
              <span className="rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-medium text-amber-800 dark:bg-amber-900/40 dark:text-amber-300">
                Modo mock
              </span>
            )}
            <ThemeToggle />
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-7xl px-4 py-6">
        <Routes>
          <Route path="/" element={<LibraryPage />} />
          <Route path="/videos/:id" element={<VideoPage />} />
          <Route path="/videos/:id/labels" element={<LabelsPage />} />
          <Route
            path="*"
            element={
              <p className="muted">
                Página no encontrada. <Link className="underline" to="/">Volver a la biblioteca</Link>
              </p>
            }
          />
        </Routes>
      </main>
    </div>
  )
}
