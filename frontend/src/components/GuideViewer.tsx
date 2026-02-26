import { useState } from 'react'

interface Props {
  guide: string
}

const SECTIONS = [
  'Overview',
  'Materials & Fabric Requirements',
  'Cutting Instructions',
  'Block Assembly',
  'Row Assembly',
  'Quilt Top Assembly',
  'Finishing Notes',
]

function parseGuide(raw: string): Array<{ title: string; content: string }> {
  if (!raw) return []

  const sections: Array<{ title: string; content: string }> = []

  // Split on ## headings
  const parts = raw.split(/^##\s+/m)
  for (const part of parts) {
    const lines = part.split('\n')
    const title = lines[0].trim()
    const content = lines.slice(1).join('\n').trim()
    if (title && content) {
      sections.push({ title, content })
    }
  }

  if (sections.length === 0) {
    return [{ title: 'Guide', content: raw }]
  }
  return sections
}

function renderMarkdown(text: string): string {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/^### (.+)$/gm, '<h4 style="margin:12px 0 4px">$1</h4>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/gs, (m) => `<ul style="margin:4px 0 8px;padding-left:20px">${m}</ul>`)
    .replace(/\n\n/g, '</p><p style="margin:8px 0">')
    .replace(/^(?!<[a-z])(.+)$/gm, '$1')
}

export function GuideViewer({ guide }: Props) {
  const sections = parseGuide(guide)
  const [open, setOpen] = useState<Set<string>>(new Set([sections[0]?.title ?? '']))

  if (!guide) {
    return <div style={{ color: '#999', fontSize: 14 }}>No guide generated yet.</div>
  }

  function toggle(title: string) {
    setOpen(prev => {
      const next = new Set(prev)
      if (next.has(title)) next.delete(title)
      else next.add(title)
      return next
    })
  }

  return (
    <div style={{ fontSize: 14 }}>
      {sections.map(section => (
        <div key={section.title} style={{ borderBottom: '1px solid #e8e0d8' }}>
          <button
            onClick={() => toggle(section.title)}
            style={{
              width: '100%', textAlign: 'left',
              padding: '12px 16px',
              background: 'none', border: 'none',
              cursor: 'pointer',
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              fontFamily: 'Georgia, serif',
              fontSize: 15,
              fontWeight: 600,
              color: '#333',
            }}
          >
            {section.title}
            <span style={{ fontSize: 18, color: '#999', lineHeight: 1 }}>
              {open.has(section.title) ? '−' : '+'}
            </span>
          </button>
          {open.has(section.title) && (
            <div
              style={{ padding: '4px 16px 16px', lineHeight: 1.7, color: '#444' }}
              dangerouslySetInnerHTML={{
                __html: `<p style="margin:8px 0">${renderMarkdown(section.content)}</p>`
              }}
            />
          )}
        </div>
      ))}
    </div>
  )
}
