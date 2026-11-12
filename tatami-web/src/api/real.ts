import { ApiError, type TatamiApi } from './client'
import type { Analysis, AnalysisResult, BackendInfo, Labels, Video } from './types'

async function errorFrom(res: Response): Promise<ApiError> {
  let msg = `${res.status} ${res.statusText}`
  try {
    const body = await res.json()
    if (typeof body?.detail === 'string') msg = body.detail
    else if (body?.detail) msg = JSON.stringify(body.detail)
  } catch {
    /* cuerpo no JSON */
  }
  return new ApiError(res.status, msg)
}

export function createRealApi(baseUrl: string): TatamiApi {
  const base = baseUrl.replace(/\/+$/, '') + '/api'

  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    let res: Response
    try {
      res = await fetch(base + path, init)
    } catch {
      throw new ApiError(0, `No se pudo conectar con el backend (${baseUrl})`)
    }
    if (!res.ok) throw await errorFrom(res)
    if (res.status === 204) return undefined as T
    return (await res.json()) as T
  }

  const json = (method: string, body: unknown): RequestInit => ({
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  return {
    health: () => request('/health'),
    listVideos: () => request<Video[]>('/videos'),
    getVideo: (id) => request<Video>(`/videos/${id}`),
    uploadVideo(file, onProgress) {
      // XMLHttpRequest: fetch no expone progreso de subida
      return new Promise<Video>((resolve, reject) => {
        const xhr = new XMLHttpRequest()
        xhr.open('POST', `${base}/videos`)
        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) onProgress?.(e.loaded / e.total)
        }
        xhr.onerror = () => reject(new ApiError(0, `No se pudo conectar con el backend (${baseUrl})`))
        xhr.onload = () => {
          if (xhr.status >= 200 && xhr.status < 300) {
            onProgress?.(1)
            try {
              resolve(JSON.parse(xhr.responseText) as Video)
            } catch {
              reject(new ApiError(xhr.status, 'Respuesta inválida del servidor'))
            }
          } else {
            let msg = `${xhr.status} ${xhr.statusText}`
            try {
              const d = JSON.parse(xhr.responseText)?.detail
              if (typeof d === 'string') msg = d
            } catch {
              /* ignorar */
            }
            reject(new ApiError(xhr.status, msg))
          }
        }
        const form = new FormData()
        form.append('file', file)
        xhr.send(form)
      })
    },
    deleteVideo: (id) => request<void>(`/videos/${id}`, { method: 'DELETE' }),
    streamUrl: (id) => `${base}/videos/${id}/stream`,
    createAnalysis: (videoId, req) => request<Analysis>(`/videos/${videoId}/analyses`, json('POST', req)),
    getAnalysis: (id) => request<Analysis>(`/analyses/${id}`),
    listAnalyses: (videoId) => request<Analysis[]>(`/videos/${videoId}/analyses`),
    async getLabels(videoId) {
      try {
        return await request<Labels>(`/videos/${videoId}/labels`)
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) return null
        throw e
      }
    },
    putLabels: (videoId, labels) => request<Labels>(`/videos/${videoId}/labels`, json('PUT', labels)),
    rescore: (id) => request<AnalysisResult>(`/analyses/${id}/rescore`, { method: 'POST' }),
    listBackends: () => request<BackendInfo[]>('/backends'),
  }
}
