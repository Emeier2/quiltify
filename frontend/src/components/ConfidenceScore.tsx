interface Props {
  score: number  // 0.0 to 1.0
}

export function ConfidenceScore({ score }: Props) {
  const pct = Math.round(score * 100)
  const color = pct >= 70 ? '#4a7c3f' : pct >= 40 ? '#d4a42a' : '#c43428'
  const label = pct >= 70 ? 'Good' : pct >= 40 ? 'Fair — review grid' : 'Low — manual edit recommended'

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 14 }}>
      <span style={{ color: '#666' }}>Extraction confidence:</span>
      <span style={{
        background: color,
        color: '#fff',
        padding: '3px 10px',
        borderRadius: 12,
        fontWeight: 600,
      }}>
        {pct}% — {label}
      </span>
    </div>
  )
}
