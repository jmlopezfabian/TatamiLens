import type { Analysis, AnalysisResult, BackendInfo, CreateAnalysisRequest, Labels, Video } from './types'

export interface TatamiApi {
  health(): Promise<{ status: string }>
  listVideos(): Promise<Video[]>
  getVideo(id: string): Promise<Video>
  uploadVideo(file: File, onProgress?: (fraction: number) => void): Promise<Video>
  deleteVideo(id: string): Promise<void>
  /** URL del stream, o null si no hay video reproducible (mock). */
  streamUrl(id: string): string | null
  createAnalysis(videoId: string, req: CreateAnalysisRequest): Promise<Analysis>
  getAnalysis(id: string): Promise<Analysis>
  listAnalyses(videoId: string): Promise<Analysis[]>
  getLabels(videoId: string): Promise<Labels | null>
  putLabels(videoId: string, labels: Labels): Promise<Labels>
  rescore(analysisId: string): Promise<AnalysisResult>
  listBackends(): Promise<BackendInfo[]>
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}
