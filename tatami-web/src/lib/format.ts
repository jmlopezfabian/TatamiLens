/** 75.3 -> "1:15.3" (decimales opcionales) */
export function formatTime(t: number, decimals = 0): string {
  const safe = Math.max(0, Number.isFinite(t) ? t : 0)
  const m = Math.floor(safe / 60)
  const s = safe - m * 60
  const sStr = s.toFixed(decimals).padStart(decimals ? 3 + decimals : 2, '0')
  return `${m}:${sStr}`
}

export function formatDate(iso: string): string {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString('es-MX', { dateStyle: 'medium', timeStyle: 'short' })
}
