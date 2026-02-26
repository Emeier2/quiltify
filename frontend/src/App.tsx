import { useState } from 'react'
import { GeneratePage } from './pages/Generate'
import { QuiltifyPage } from './pages/Quiltify'
import { ExportPage } from './pages/Export'
import type { QuiltPatternSchema } from './types/pattern'

type Page = 'generate' | 'quiltify' | 'export'

export function App() {
  const [page, setPage] = useState<Page>('generate')
  const [exportPattern, setExportPattern] = useState<QuiltPatternSchema | null>(null)

  function handleSendToExport(pattern: QuiltPatternSchema) {
    setExportPattern(pattern)
    setPage('export')
  }

  return (
    <div style={{ minHeight: '100vh', background: '#faf8f5' }}>
      {/* Header / Nav */}
      <header style={{
        background: '#2a1040',
        color: 'white',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        gap: 32,
        height: 56,
        boxShadow: '0 2px 8px rgba(0,0,0,0.2)',
      }}>
        <div style={{
          fontFamily: 'Georgia, serif',
          fontSize: 20,
          fontWeight: 700,
          letterSpacing: 1,
          color: '#e8d8f8',
        }}>
          ✦ Quiltify
        </div>
        <nav style={{ display: 'flex', gap: 0, flex: 1 }}>
          {([
            { id: 'generate', label: 'Generate' },
            { id: 'quiltify', label: 'Quiltify' },
            { id: 'export', label: 'Export' },
          ] as const).map(({ id, label }) => (
            <button
              key={id}
              onClick={() => setPage(id)}
              style={{
                background: 'none',
                border: 'none',
                color: page === id ? '#fff' : 'rgba(255,255,255,0.6)',
                fontSize: 15,
                fontFamily: 'Georgia, serif',
                cursor: 'pointer',
                padding: '0 16px',
                height: 56,
                borderBottom: page === id ? '2px solid #c090e0' : '2px solid transparent',
                transition: 'color 0.15s',
              }}
            >
              {label}
            </button>
          ))}
        </nav>
        <div style={{ fontSize: 12, color: 'rgba(255,255,255,0.4)' }}>
          Pictorial modern quilt pattern generator
        </div>
      </header>

      {/* Page content */}
      <main>
        {page === 'generate' && <GeneratePage onSendToExport={handleSendToExport} />}
        {page === 'quiltify' && <QuiltifyPage onSendToExport={handleSendToExport} />}
        {page === 'export' && <ExportPage pattern={exportPattern} />}
      </main>
    </div>
  )
}
