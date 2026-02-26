import type {
  GenerateResponse,
  QuiltifyResponse,
  GuideResponse,
  QuiltPatternSchema,
} from '../types/pattern'

const BASE = ''  // proxied via Vite to http://localhost:8000

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${err}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  generate(params: {
    prompt: string
    grid_width: number
    grid_height: number
    palette_size: number
    block_size_inches: number
  }): Promise<GenerateResponse> {
    return post('/api/generate', params)
  },

  quiltify(params: {
    image_base64: string
    grid_width: number
    grid_height: number
    palette_size: number
    block_size_inches: number
  }): Promise<QuiltifyResponse> {
    return post('/api/quiltify', params)
  },

  regenerateGuide(pattern: QuiltPatternSchema, title?: string): Promise<GuideResponse> {
    return post('/api/guide', { pattern, title })
  },

  async exportSvg(pattern: QuiltPatternSchema): Promise<Blob> {
    const res = await fetch('/api/export/svg', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pattern }),
    })
    if (!res.ok) throw new Error(`Export failed: ${res.statusText}`)
    return res.blob()
  },

  async exportCsv(pattern: QuiltPatternSchema): Promise<Blob> {
    const res = await fetch('/api/export/csv', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pattern }),
    })
    if (!res.ok) throw new Error(`Export failed: ${res.statusText}`)
    return res.blob()
  },

  async exportPdf(pattern: QuiltPatternSchema): Promise<Blob> {
    const res = await fetch('/api/export/pdf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pattern }),
    })
    if (!res.ok) throw new Error(`Export failed: ${res.statusText}`)
    return res.blob()
  },

  async health(): Promise<{ status: string; ollama: boolean; flux_pipeline: { loaded: boolean; type: string } }> {
    const res = await fetch('/health')
    return res.json()
  },
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}
