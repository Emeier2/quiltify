import { useState } from 'react'

const EXAMPLE_PROMPTS = [
  'a great horned owl in a pine tree',
  'a red fox in a snowy field',
  'a humpback whale breaching at sunset',
  'a black bear in autumn forest',
  'a monarch butterfly on milkweed',
  'a mallard duck on a pond',
  'a cactus wren on a saguaro cactus',
  'a mountain goat on rocky peaks',
]

interface Params {
  prompt: string
  grid_width: number
  grid_height: number
  palette_size: number
  block_size_inches: number
}

interface Props {
  onSubmit: (params: Params) => void
  loading: boolean
}

export function PromptInput({ onSubmit, loading }: Props) {
  const [prompt, setPrompt] = useState('')
  const [gridWidth, setGridWidth] = useState(40)
  const [gridHeight, setGridHeight] = useState(50)
  const [paletteSize, setPaletteSize] = useState(6)
  const [blockSize, setBlockSize] = useState(2.5)
  const [showAdvanced, setShowAdvanced] = useState(false)

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!prompt.trim() || loading) return
    onSubmit({
      prompt: prompt.trim(),
      grid_width: gridWidth,
      grid_height: gridHeight,
      palette_size: paletteSize,
      block_size_inches: blockSize,
    })
  }

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="Describe your quilt subject (e.g. a great horned owl in a pine tree)"
          disabled={loading}
          style={{
            flex: 1,
            padding: '10px 14px',
            fontSize: 15,
            border: '1.5px solid #ccc',
            borderRadius: 6,
            fontFamily: 'Georgia, serif',
            background: loading ? '#f5f5f5' : 'white',
          }}
        />
        <button
          type="submit"
          disabled={!prompt.trim() || loading}
          style={{
            padding: '10px 22px',
            background: '#4a2060',
            color: 'white',
            border: 'none',
            borderRadius: 6,
            fontSize: 15,
            fontWeight: 600,
            cursor: loading || !prompt.trim() ? 'not-allowed' : 'pointer',
            opacity: loading || !prompt.trim() ? 0.7 : 1,
            transition: 'opacity 0.15s',
          }}
        >
          {loading ? 'Generating…' : 'Generate'}
        </button>
      </div>

      {/* Example prompts */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {EXAMPLE_PROMPTS.map(p => (
          <button
            key={p}
            type="button"
            onClick={() => setPrompt(p)}
            style={{
              padding: '3px 10px',
              fontSize: 12,
              background: 'transparent',
              border: '1px solid #c0a8d0',
              borderRadius: 12,
              cursor: 'pointer',
              color: '#703090',
              transition: 'background 0.12s',
            }}
          >
            {p}
          </button>
        ))}
      </div>

      {/* Advanced options */}
      <div>
        <button
          type="button"
          onClick={() => setShowAdvanced(s => !s)}
          style={{ fontSize: 12, color: '#888', background: 'none', border: 'none',
                   cursor: 'pointer', padding: 0, textDecoration: 'underline' }}
        >
          {showAdvanced ? 'Hide' : 'Show'} advanced options
        </button>
        {showAdvanced && (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginTop: 10 }}>
            <label style={labelStyle}>
              Grid Width
              <input type="number" min={10} max={100} value={gridWidth}
                     onChange={e => setGridWidth(+e.target.value)} style={inputStyle} />
            </label>
            <label style={labelStyle}>
              Grid Height
              <input type="number" min={10} max={100} value={gridHeight}
                     onChange={e => setGridHeight(+e.target.value)} style={inputStyle} />
            </label>
            <label style={labelStyle}>
              Palette Size
              <input type="number" min={2} max={12} value={paletteSize}
                     onChange={e => setPaletteSize(+e.target.value)} style={inputStyle} />
            </label>
            <label style={labelStyle}>
              Block Size (in)
              <input type="number" min={1} max={6} step={0.5} value={blockSize}
                     onChange={e => setBlockSize(+e.target.value)} style={inputStyle} />
            </label>
          </div>
        )}
      </div>
    </form>
  )
}

const labelStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: 4,
  fontSize: 12, color: '#666', fontFamily: 'sans-serif',
}

const inputStyle: React.CSSProperties = {
  padding: '5px 8px', border: '1px solid #ccc', borderRadius: 4, fontSize: 13,
}
