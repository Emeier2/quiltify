import { useState } from 'react'
import { PromptInput } from '../components/PromptInput'
import { QuiltCanvas } from '../components/QuiltCanvas'
import { FabricPalette } from '../components/FabricPalette'
import { CuttingChart } from '../components/CuttingChart'
import { GuideViewer } from '../components/GuideViewer'
import { ConfidenceScore } from '../components/ConfidenceScore'
import { api, downloadBlob } from '../api/client'
import type { GenerateResponse, QuiltPatternSchema } from '../types/pattern'

interface GeneratePageProps {
  onSendToExport?: (pattern: QuiltPatternSchema) => void
}

export function GeneratePage({ onSendToExport }: GeneratePageProps) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<GenerateResponse | null>(null)
  const [pattern, setPattern] = useState<QuiltPatternSchema | null>(null)
  const [selectedFabricId, setSelectedFabricId] = useState<string | null>(null)
  const [recalcLoading, setRecalcLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'chart' | 'guide' | 'cutting-diagram'>('chart')

  async function handleGenerate(params: Parameters<typeof api.generate>[0]) {
    setLoading(true)
    setError(null)
    try {
      const res = await api.generate(params)
      setResult(res)
      setPattern(res.pattern_json)
      setSelectedFabricId(res.pattern_json.fabrics[0]?.id ?? null)
    } catch (e: any) {
      setError(e.message ?? 'Generation failed')
    } finally {
      setLoading(false)
    }
  }

  async function handleRecalculate() {
    if (!pattern) return
    setRecalcLoading(true)
    try {
      const res = await api.regenerateGuide(pattern, 'Updated Pattern')
      setResult(prev => prev ? { ...prev, guide: res.guide, cutting_chart: res.cutting_chart, svg: res.svg } : prev)
      setPattern(res.pattern_json)
    } catch (e: any) {
      setError(e.message ?? 'Recalculation failed')
    } finally {
      setRecalcLoading(false)
    }
  }

  function handleRenameFabric(id: string, newName: string) {
    if (!pattern) return
    setPattern({
      ...pattern,
      fabrics: pattern.fabrics.map(f => f.id === id ? { ...f, name: newName } : f),
    })
  }

  async function handleExport(fmt: 'svg' | 'csv' | 'pdf') {
    if (!pattern) return
    try {
      let blob: Blob
      if (fmt === 'svg') { blob = await api.exportSvg(pattern); downloadBlob(blob, 'quilt-pattern.svg') }
      else if (fmt === 'csv') { blob = await api.exportCsv(pattern); downloadBlob(blob, 'cutting-chart.csv') }
      else { blob = await api.exportPdf(pattern); downloadBlob(blob, 'quilt-guide.pdf') }
    } catch (e: any) {
      setError(e.message)
    }
  }

  return (
    <div style={{ maxWidth: 1400, margin: '0 auto', padding: '24px 20px' }}>
      <h1 style={{ fontFamily: 'Georgia, serif', fontSize: 28, marginBottom: 4, color: '#2a1040' }}>
        Generate from Prompt
      </h1>
      <p style={{ color: '#888', fontSize: 14, marginBottom: 24 }}>
        Describe a subject and Quiltify will generate an Elizabeth Hartman-style quilt pattern.
      </p>

      <PromptInput onSubmit={handleGenerate} loading={loading} />

      {error && (
        <div style={{ background: '#fff0f0', border: '1px solid #f5c0c0', borderRadius: 6,
                      padding: '10px 16px', marginTop: 16, color: '#c00', fontSize: 14 }}>
          {error}
        </div>
      )}

      {result && pattern && (
        <div style={{ marginTop: 28, display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 280px', gap: 24 }}>
          {/* Left: canvas + tabs */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <ConfidenceScore score={result.confidence_score} />
              <div style={{ display: 'flex', gap: 8 }}>
                <button onClick={() => handleExport('svg')} style={exportBtnStyle}>↓ SVG</button>
                <button onClick={() => handleExport('csv')} style={exportBtnStyle}>↓ CSV</button>
                <button onClick={() => handleExport('pdf')} style={exportBtnStyle}>↓ PDF</button>
                <button
                  onClick={handleRecalculate}
                  disabled={recalcLoading}
                  style={{ ...exportBtnStyle, background: '#4a2060', color: '#fff', borderColor: '#4a2060' }}
                >
                  {recalcLoading ? 'Recalculating…' : '↻ Recalculate Guide'}
                </button>
                {onSendToExport && (
                  <button
                    onClick={() => onSendToExport(pattern)}
                    style={{ ...exportBtnStyle, background: '#2a6040', color: '#fff', borderColor: '#2a6040' }}
                  >
                    → Send to Export
                  </button>
                )}
              </div>
            </div>

            <QuiltCanvas
              pattern={pattern}
              selectedFabricId={selectedFabricId}
              onPatternChange={setPattern}
            />

            {/* Tabs */}
            <div style={{ marginTop: 20 }}>
              <div style={{ display: 'flex', borderBottom: '2px solid #e8e0d8', marginBottom: 16 }}>
                {(['chart', 'cutting-diagram', 'guide'] as const).map(tab => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab)}
                    style={{
                      padding: '8px 20px',
                      background: 'none', border: 'none',
                      fontFamily: 'Georgia, serif', fontSize: 15,
                      cursor: 'pointer',
                      borderBottom: activeTab === tab ? '2px solid #4a2060' : '2px solid transparent',
                      marginBottom: -2,
                      color: activeTab === tab ? '#4a2060' : '#888',
                      fontWeight: activeTab === tab ? 600 : 400,
                    }}
                  >
                    {tab === 'chart' ? 'Cutting Chart' : tab === 'cutting-diagram' ? 'Cutting Diagram' : 'Sewing Guide'}
                  </button>
                ))}
              </div>
              {activeTab === 'chart' && <CuttingChart pieces={result.cutting_chart} />}
              {activeTab === 'cutting-diagram' && result.cutting_svg && (
                <div dangerouslySetInnerHTML={{ __html: result.cutting_svg }} />
              )}
              {activeTab === 'guide' && <GuideViewer guide={result.guide} />}
            </div>
          </div>

          {/* Right: fabric palette */}
          <div style={{ borderLeft: '1px solid #e8e0d8', paddingLeft: 20 }}>
            <FabricPalette
              fabrics={pattern.fabrics}
              selectedFabricId={selectedFabricId}
              onSelectFabric={setSelectedFabricId}
              onRenameFabric={handleRenameFabric}
            />
            {result.image_b64 && (
              <div style={{ marginTop: 20 }}>
                <div style={{ fontSize: 12, color: '#888', marginBottom: 6, textTransform: 'uppercase',
                              letterSpacing: 0.5, fontWeight: 700 }}>
                  Source Image
                </div>
                <img
                  src={`data:image/jpeg;base64,${result.image_b64}`}
                  alt="Generated quilt"
                  style={{ width: '100%', borderRadius: 4, border: '1px solid #ddd' }}
                />
              </div>
            )}
          </div>
        </div>
      )}

      {loading && (
        <div style={{ textAlign: 'center', marginTop: 60, color: '#888' }}>
          <div style={{ fontSize: 24, marginBottom: 8 }}>⟳</div>
          <div>Generating your quilt pattern — this may take a few minutes…</div>
          <div style={{ fontSize: 12, marginTop: 8, color: '#aaa' }}>
            FLUX image generation + grid extraction + guide writing
          </div>
        </div>
      )}
    </div>
  )
}

const exportBtnStyle: React.CSSProperties = {
  padding: '5px 12px',
  fontSize: 13,
  background: 'white',
  border: '1px solid #ccc',
  borderRadius: 5,
  cursor: 'pointer',
}
