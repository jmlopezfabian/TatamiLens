import type { Position, Top } from '../api/types'

export interface PositionMeta {
  id: Position
  label: string
  color: string
  /** true si la posición no tiene "top" (siempre none) */
  noTop: boolean
}

export const POSITIONS: PositionMeta[] = [
  { id: 'standing', label: 'De pie', color: '#94a3b8', noTop: true },
  { id: 'scramble', label: 'Scramble', color: '#f59e0b', noTop: true },
  { id: 'closed_guard', label: 'Guardia cerrada', color: '#3b82f6', noTop: false },
  { id: 'half_guard', label: 'Media guardia', color: '#06b6d4', noTop: false },
  { id: 'open_guard', label: 'Guardia abierta', color: '#8b5cf6', noTop: false },
  { id: 'side_control', label: 'Side control', color: '#22c55e', noTop: false },
  { id: 'north_south', label: 'Norte-sur', color: '#14b8a6', noTop: false },
  { id: 'knee_on_belly', label: 'Rodilla en el abdomen', color: '#eab308', noTop: false },
  { id: 'mount', label: 'Montada', color: '#ef4444', noTop: false },
  { id: 'back_control', label: 'Control de espalda', color: '#ec4899', noTop: false },
  { id: 'turtle', label: 'Tortuga', color: '#a16207', noTop: false },
  { id: 'separated', label: 'Separados', color: '#64748b', noTop: true },
  { id: 'not_visible', label: 'No visible', color: '#334155', noTop: true },
]

const BY_ID = new Map(POSITIONS.map((p) => [p.id, p]))

export function positionMeta(id: Position): PositionMeta {
  return BY_ID.get(id) ?? POSITIONS[0]
}

export function positionLabel(id: Position): string {
  return positionMeta(id).label
}

/** Fuerza top = none en posiciones sin top (regla del contrato). */
export function normalizeTop(position: Position, top: Top): Top {
  return positionMeta(position).noTop ? 'none' : top
}

export const TOP_COLORS: Record<Top, string> = { A: '#2563eb', B: '#ea580c', none: 'transparent' }

export const KIND_LABELS: Record<string, string> = {
  takedown: 'derribo',
  sweep: 'barrida',
  guard_pass: 'pase de guardia',
  knee_on_belly: 'rodilla en el abdomen',
  mount: 'montada',
  back_take: 'toma de espalda',
}
