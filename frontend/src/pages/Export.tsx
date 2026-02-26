import { api, downloadBlob } from '../api/client'
import type { QuiltPatternSchema } from '../types/pattern'

interface Props {
  pattern: QuiltPatternSchema | null
}

export function ExportPage({ pattern }: Props) {
  if (!pattern) {
    return (
      <div style={{ maxWidth: 600, margin: '60px auto', textAlign: 'center', color: '#888' }}>
        <div style={{ fontSize: 40, marginBottom: 16 }}>✦</div>
        <div style={{ fontSize: 18, marginBottom: 8 }}>No pattern loaded</div>
        <div style={{ fontSize: 14 }}>Generate a pattern first using the Generate or Quiltify pages.</div>
      </div>
    )
  }

  async function doExport(fmt: 'svg' | 'csv' | 'pdf') {
    try {
      if (fmt === 'svg') {
        const blob = await api.exportSvg(pattern!)
        downloadBlob(blob, 'quilt-pattern.svg')
      } else if (fmt === 'csv') {
        const blob = await api.exportCsv(pattern!)
        downloadBlob(blob, 'cutting-chart.csv')
      } else {
        const blob = await api.exportPdf(pattern!)
        downloadBlob(blob, 'quilt-guide.pdf')
      }
    } catch (e: any) {
      alert(`Export failed: ${e.message}`)
    }
  }

  const w = pattern.finished_width_in ?? pattern.grid_width * pattern.block_size_in
  const h = pattern.finished_height_in ?? pattern.grid_height * pattern.block_size_in

  return (
    <div style={{ maxWidth: 700, margin: '0 auto', padding: '24px 20px' }}>
      <h1 style={{ fontFamily: 'Georgia, serif', fontSize: 28, marginBottom: 4, color: '#2a1040' }}>
        Export
      </h1>
      <div style={{ color: '#666', fontSize: 14, marginBottom: 32 }}>
        Finished size: {w}" × {h}" &nbsp;·&nbsp; {pattern.fabrics.length} fabrics &nbsp;·&nbsp; {pattern.blocks.length} blocks
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16 }}>
        {[
          { fmt: 'svg' as const, label: 'SVG Pattern', desc: 'Scalable vector quilt grid — open in Inkscape, Illustrator, or a browser', icon: '🖼' },
          { fmt: 'csv' as const, label: 'CSV Cutting Chart', desc: 'Spreadsheet of all cut pieces — import to Excel or Numbers', icon: '📋' },
          { fmt: 'pdf' as const, label: 'PDF Guide', desc: 'Full printable guide with pattern, cutting chart, and instructions', icon: '📄' },
        ].map(({ fmt, label, desc, icon }) => (
          <button
            key={fmt}
            onClick={() => doExport(fmt)}
            style={{
              padding: '20px 16px',
              background: 'white',
              border: '1.5px solid #ddd',
              borderRadius: 8,
              cursor: 'pointer',
              textAlign: 'left',
              transition: 'border-color 0.15s, box-shadow 0.15s',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = '#4a2060'
              e.currentTarget.style.boxShadow = '0 2px 12px rgba(74,32,96,0.1)'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = '#ddd'
              e.currentTarget.style.boxShadow = 'none'
            }}
          >
            <div style={{ fontSize: 28, marginBottom: 8 }}>{icon}</div>
            <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 6, fontFamily: 'Georgia, serif' }}>
              {label}
            </div>
            <div style={{ fontSize: 13, color: '#888', lineHeight: 1.5 }}>{desc}</div>
          </button>
        ))}
      </div>
    </div>
  )
}
