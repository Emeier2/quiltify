import type { CutPiece } from '../types/pattern'

interface Props {
  pieces: CutPiece[]
}

export function CuttingChart({ pieces }: Props) {
  if (pieces.length === 0) {
    return <div style={{ color: '#999', fontSize: 14 }}>No cutting chart available.</div>
  }

  // Group by fabric
  const byFabric = new Map<string, CutPiece[]>()
  for (const p of pieces) {
    const arr = byFabric.get(p.fabric_id) ?? []
    arr.push(p)
    byFabric.set(p.fabric_id, arr)
  }

  return (
    <div style={{ fontSize: 14 }}>
      {Array.from(byFabric.entries()).map(([fabricId, fabricPieces]) => {
        const totalQty = fabricPieces.reduce((s, p) => s + p.quantity, 0)
        const fab = fabricPieces[0]
        return (
          <div key={fabricId} style={{ marginBottom: 20 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
              <div style={{
                width: 18, height: 18, borderRadius: 3,
                background: fab.color_hex,
                border: '1px solid rgba(0,0,0,0.15)',
                flexShrink: 0,
              }} />
              <span style={{ fontWeight: 700, fontSize: 14 }}>{fab.fabric_name}</span>
              <span style={{ color: '#999', fontSize: 12 }}>{totalQty} pieces total</span>
            </div>
            <table style={{ borderCollapse: 'collapse', width: '100%' }}>
              <thead>
                <tr style={{ background: '#f0ede8' }}>
                  <th style={thStyle}>Cut Width</th>
                  <th style={thStyle}>Cut Height</th>
                  <th style={thStyle}>Quantity</th>
                </tr>
              </thead>
              <tbody>
                {fabricPieces
                  .sort((a, b) => b.cut_width_in * b.cut_height_in - a.cut_width_in * a.cut_height_in)
                  .map((piece, i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? 'white' : '#faf8f5' }}>
                      <td style={tdStyle}>{piece.cut_width_in}"</td>
                      <td style={tdStyle}>{piece.cut_height_in}"</td>
                      <td style={{ ...tdStyle, fontWeight: 600 }}>{piece.quantity}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )
      })}
    </div>
  )
}

const thStyle: React.CSSProperties = {
  padding: '6px 12px',
  textAlign: 'left',
  fontWeight: 600,
  fontSize: 12,
  textTransform: 'uppercase',
  letterSpacing: 0.5,
  color: '#666',
  border: '1px solid #ddd',
}

const tdStyle: React.CSSProperties = {
  padding: '5px 12px',
  border: '1px solid #eee',
  fontSize: 13,
}
