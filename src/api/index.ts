import type { TatamiApi } from './client'
import { createMockApi } from './mock'
import { createRealApi } from './real'

export const USE_MOCK = import.meta.env.VITE_USE_MOCK === 'true'
export const API_URL: string = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export const api: TatamiApi = USE_MOCK ? createMockApi() : createRealApi(API_URL)
export { ApiError } from './client'
export type { TatamiApi } from './client'
