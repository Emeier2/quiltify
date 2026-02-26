import { useState } from 'react'
import { ImageUpload } from '../components/ImageUpload'
import { QuiltCanvas } from '../components/QuiltCanvas'
import { FabricPalette } from '../components/FabricPalette'
import { CuttingChart } from '../components/CuttingChart'
import { GuideViewer } from '../components/GuideViewer'
import { ConfidenceScore } from '../components/ConfidenceScore'
import { api, downloadBlob } from '../api/client'
import type { QuiltifyResponse, QuiltPatternSchema } from '../types/pattern'

interface QuiltifyPageProps {
  onSendToExport?: (pattern: QuiltPatternSchema) => void
}

export function QuiltifyPage({ onSendToExport }: QuiltifyPageProps) {
  const [imageBase64, setImageBase64] = useState<string | null>(null)
  const [originalPreview, setOriginalPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<QuiltifyResponse | null>(null)
  const [pattern, setPattern] = useState<QuiltPatternSchema | null>(null)
  const [selectedFabricId, setSelectedFabricId] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'chart' | 'guide' | 'cutting-diagram'>('chart')
  const [recalcLoading, setRecalcLoading] = useState(false)

  // Grid settings
  const [gridWidth] = useState(40)
  const [gridHeight] = useState(50)
  const [paletteSize] = useState(6)
  const [blockSize] = useState(2.5)

  function handleImageSelected(base64: string, preview: string) {
    setImageBase64(base64)
    setOriginalPreview(preview)
    setResult(null)
    setPattern(null)
  }

  async function handleQuiltify() {
    if (!imageBase64) return
    setLoading(true)
    setError(null)
    try {
      const res = await api.quiltify({
        image_base64: imageBase64,
        grid_width: gridWidth,
        grid_height: gridHeight,
        palette_size: paletteSize,
        block_size_inches: blockSize,
      })
      setResult(res)
      setPattern(res.pattern_json)
      setSelectedFabricId(res.pattern_json.fabrics[0]?.id ?? null)
    } catch (e: any) {
      setError(e.message ?? 'Quiltification failed')
    } finally {
      setLoading(false)
    }
  }

  async function handleRecalculate() {
    if (!pattern) return
    setRecalcLoading(true)
    try {
      const res = await api.regenerateGuide(pattern)
      setResult(prev => prev ? { ...prev, guide: res.guide, cutting_chart: res.cutting_chart } : prev)
      setPattern(res.pattern_json)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setRecalcLoading(false)
    }
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
        Quiltify an Image
      </h1>
      <p style={{ color: '#888', fontSize: 14, marginBottom: 24 }}>
        Upload any photo and Quiltify will transform it into a buildable geometric quilt pattern.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, maxWidth: 700 }}>
        <div>
          <ImageUpload onImageSelected={handleImageSelected} disabled={loading} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', gap: 12 }}>
          <button
            onClick={handleQuiltify}
            disabled={!imageBase64 || loading}
            style={{
              padding: '12px 24px',
              background: '#4a2060',
              color: 'white',
              border: 'none',
              borderRadius: 6,
              fontSize: 15,
              fontWeight: 600,
              cursor: !imageBase64 || loading ? 'not-allowed' : 'pointer',
              opacity: !imageBase64 || loading ? 0.6 : 1,
            }}
          >
            {loading ? 'Quiltifying…' : '✦ Quiltify'}
          </button>
          {loading && (
            <div style={{ fontSize: 13, color: '#888', textAlign: 'center' }}>
              Segmenting image → quilt-style rendering → extracting grid…
            </div>
          )}
        </div>
      </div>

      {error && (
        <div style={{ background: '#fff0f0', border: '1px solid #f5c0c0', borderRadius: 6,
                      padding: '10px 16px', marginTop: 16, color: '#c00', fontSize: 14 }}>
          {error}
        </div>
      )}

      {result && pattern && (
        <div style={{ marginTop: 32 }}>
          {/* Three-panel: original | quilt render | editable grid */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 24 }}>
            <div>
              <div style={panelLabelStyle}>Original Image</div>
              <img
                src={`data:image/jpeg;base64,${result.original_image_b64}`}
                alt="Original"
                style={{ width: '100%', borderRadius: 4, border: '1px solid #ddd' }}
              />
            </div>
            <div>
              <div style={panelLabelStyle}>AI Quilt Version</div>
              {result.quilt_image_b64 ? (
                <img
                  src={`data:image/jpeg;base64,${result.quilt_image_b64}`}
                  alt="Quilt render"
                  style={{ width: '100%', borderRadius: 4, border: '1px solid #ddd' }}
                />
              ) : (
                <div style={{ background: '#f5f5f0', borderRadius: 4, padding: 20, color: '#aaa',
                              textAlign: 'center', fontSize: 13 }}>
                  ControlNet unavailable — using direct grid extraction
                </div>
              )}
            </div>
            <div>
              <div style={panelLabelStyle}>Editable Grid Pattern</div>
              <div style={{ overflow: 'hidden', borderRadius: 4, border: '1px solid #ddd' }}>
                <div style={{ transform: 'scale(0.4)', transformOrigin: 'top left',
                              width: '250%', height: '250%', pointerEvents: 'none' }}>
                  <QuiltCanvas
                    pattern={pattern}
                    selectedFabricId={null}
                    onPatternChange={() => {}}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Full editor */}
          <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 280px', gap: 24 }}>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <ConfidenceScore score={result.confidence_score} />
                <div style={{ display: 'flex', gap: 8 }}>
                  <button onClick={() => handleExport('svg')} style={exportBtnStyle}>↓ SVG</button>
                  <button onClick={() => handleExport('csv')} style={exportBtnStyle}>↓ CSV</button>
                  <button onClick={() => handleExport('pdf')} style={exportBtnStyle}>↓ PDF</button>
                  <button onClick={handleRecalculate} disabled={recalcLoading}
                          style={{ ...exportBtnStyle, background: '#4a2060', color: '#fff', borderColor: '#4a2060' }}>
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
              <QuiltCanvas pattern={pattern} selectedFabricId={selectedFabricId} onPatternChange={setPattern} />
              <div style={{ marginTop: 20 }}>
                <div style={{ display: 'flex', borderBottom: '2px solid #e8e0d8', marginBottom: 16 }}>
                  {(['chart', 'cutting-diagram', 'guide'] as const).map(tab => (
                    <button key={tab} onClick={() => setActiveTab(tab)} style={{
                      padding: '8px 20px', background: 'none', border: 'none',
                      fontFamily: 'Georgia, serif', fontSize: 15, cursor: 'pointer',
                      borderBottom: activeTab === tab ? '2px solid #4a2060' : '2px solid transparent',
                      marginBottom: -2, color: activeTab === tab ? '#4a2060' : '#888',
                      fontWeight: activeTab === tab ? 600 : 400,
                    }}>
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
            <div style={{ borderLeft: '1px solid #e8e0d8', paddingLeft: 20 }}>
              <FabricPalette
                fabrics={pattern.fabrics}
                selectedFabricId={selectedFabricId}
                onSelectFabric={setSelectedFabricId}
                onRenameFabric={(id, name) => setPattern(prev => prev ? {
                  ...prev, fabrics: prev.fabrics.map(f => f.id === id ? { ...f, name } : f)
                } : prev)}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

const panelLabelStyle: React.CSSProperties = {
  fontSize: 12, color: '#888', marginBottom: 6,
  textTransform: 'uppercase', letterSpacing: 0.5, fontWeight: 700,
}

const exportBtnStyle: React.CSSProperties = {
  padding: '5px 12px', fontSize: 13, background: 'white',
  border: '1px solid #ccc', borderRadius: 5, cursor: 'pointer',
}
