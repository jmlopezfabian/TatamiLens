import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './index'
import type { Analysis, CreateAnalysisRequest, Labels } from './types'

const POLL_MS = 1500
const active = (a?: Analysis) => a?.status === 'queued' || a?.status === 'running'

export const useVideos = () => useQuery({ queryKey: ['videos'], queryFn: () => api.listVideos() })
export const useVideo = (id: string) => useQuery({ queryKey: ['video', id], queryFn: () => api.getVideo(id) })
export const useBackends = () => useQuery({ queryKey: ['backends'], queryFn: () => api.listBackends(), staleTime: 60_000 })

export function useUploadVideo(onProgress: (f: number) => void) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (file: File) => api.uploadVideo(file, onProgress),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['videos'] }),
  })
}

export function useDeleteVideo() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.deleteVideo(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['videos'] }),
  })
}

export const useAnalyses = (videoId: string) =>
  useQuery({
    queryKey: ['analyses', videoId],
    queryFn: () => api.listAnalyses(videoId),
    refetchInterval: (q) => (q.state.data?.some(active) ? POLL_MS : false),
  })

/** Análisis individual (con `result`); hace polling mientras queued/running. */
export const useAnalysis = (id: string | undefined) =>
  useQuery({
    queryKey: ['analysis', id],
    queryFn: () => api.getAnalysis(id as string),
    enabled: !!id,
    refetchInterval: (q) => (active(q.state.data) ? POLL_MS : false),
  })

export function useCreateAnalysis(videoId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (req: CreateAnalysisRequest) => api.createAnalysis(videoId, req),
    onSuccess: (a) => {
      qc.setQueryData(['analysis', a.id], a)
      return qc.invalidateQueries({ queryKey: ['analyses', videoId] })
    },
  })
}

export const useLabels = (videoId: string) =>
  useQuery({ queryKey: ['labels', videoId], queryFn: () => api.getLabels(videoId) })

export function useSaveLabels(videoId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (labels: Labels) => api.putLabels(videoId, labels),
    onSuccess: (saved) => qc.setQueryData(['labels', videoId], saved),
  })
}

export function useRescore(videoId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (analysisId: string) => api.rescore(analysisId),
    onSuccess: (_r, analysisId) => {
      qc.invalidateQueries({ queryKey: ['analysis', analysisId] })
      qc.invalidateQueries({ queryKey: ['analyses', videoId] })
    },
  })
}

/** Elige el análisis a mostrar: el preferido, o el último terminado, o el más reciente. */
export function useActiveAnalysis(videoId: string, preferredId: string | null) {
  const list = useAnalyses(videoId)
  const items = list.data ?? []
  const fallback = items.find((a) => a.status === 'done') ?? items[0]
  const activeId = preferredId ?? fallback?.id
  const analysis = useAnalysis(activeId)
  return { list, items, activeId, analysis }
}
