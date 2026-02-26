import { useState } from 'react'
import type { FabricSchema } from '../types/pattern'

interface Props {
  fabrics: FabricSchema[]
  selectedFabricId: string | null
  onSelectFabric: (id: string) => void
  onRenameFabric: (id: string, newName: string) => void
}

export function FabricPalette({ fabrics, selectedFabricId, onSelectFabric, onRenameFabric }: Props) {
  const [editing, setEditing] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')

  function startEdit(f: FabricSchema) {
    setEditing(f.id)
    setEditValue(f.name)
  }

  function commitEdit(id: string) {
    if (editValue.trim()) onRenameFabric(id, editValue.trim())
    setEditing(null)
  }

  return (
    <div style={{ padding: '12px 0' }}>
      <div style={{ fontWeight: 700, fontSize: 13, textTransform: 'uppercase',
                    letterSpacing: 1, color: '#888', marginBottom: 10 }}>
        Fabrics
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {fabrics.map(f => (
          <div
            key={f.id}
            onClick={() => onSelectFabric(f.id)}
            style={{
              display: 'flex', alignItems: 'center', gap: 10,
              padding: '6px 10px',
              borderRadius: 6,
              cursor: 'pointer',
              background: selectedFabricId === f.id ? '#e8e0d8' : 'transparent',
              border: selectedFabricId === f.id ? '1.5px solid #9a7a5a' : '1.5px solid transparent',
              transition: 'background 0.15s',
            }}
          >
            <div style={{
              width: 24, height: 24, borderRadius: 4,
              background: f.color_hex,
              border: '1px solid rgba(0,0,0,0.15)',
              flexShrink: 0,
            }} />
            {editing === f.id ? (
              <input
                autoFocus
                value={editValue}
                onChange={e => setEditValue(e.target.value)}
                onBlur={() => commitEdit(f.id)}
                onKeyDown={e => { if (e.key === 'Enter') commitEdit(f.id) }}
                style={{ flex: 1, fontSize: 13, border: '1px solid #ccc',
                         borderRadius: 4, padding: '2px 6px' }}
                onClick={e => e.stopPropagation()}
              />
            ) : (
              <span
                style={{ flex: 1, fontSize: 13, lineHeight: 1.3 }}
                onDoubleClick={e => { e.stopPropagation(); startEdit(f) }}
                title="Double-click to rename"
              >
                {f.name}
              </span>
            )}
            <span style={{ fontSize: 11, color: '#999', flexShrink: 0 }}>
              {f.total_sqin ? `${Math.round(f.total_sqin)} sq"` : ''}
            </span>
          </div>
        ))}
      </div>
      <div style={{ fontSize: 11, color: '#aaa', marginTop: 8 }}>
        Click to select • Double-click name to rename
      </div>
    </div>
  )
}
