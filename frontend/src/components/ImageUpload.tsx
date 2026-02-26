import { useRef, useState } from 'react'

interface Props {
  onImageSelected: (base64: string, preview: string) => void
  disabled: boolean
}

export function ImageUpload({ onImageSelected, disabled }: Props) {
  const [isDragging, setIsDragging] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  function handleFile(file: File) {
    if (!file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = e => {
      const dataUrl = e.target?.result as string
      setPreview(dataUrl)
      onImageSelected(dataUrl, dataUrl)
    }
    reader.readAsDataURL(file)
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  return (
    <div>
      <div
        onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        style={{
          border: `2px dashed ${isDragging ? '#4a2060' : '#ccc'}`,
          borderRadius: 8,
          padding: '32px 20px',
          textAlign: 'center',
          cursor: disabled ? 'not-allowed' : 'pointer',
          background: isDragging ? '#f5eef8' : '#faf8f5',
          transition: 'all 0.15s',
          opacity: disabled ? 0.6 : 1,
        }}
      >
        {preview ? (
          <div>
            <img
              src={preview}
              alt="Upload preview"
              style={{ maxHeight: 200, maxWidth: '100%', borderRadius: 4, marginBottom: 8 }}
            />
            <div style={{ fontSize: 12, color: '#888' }}>
              Click or drag to replace
            </div>
          </div>
        ) : (
          <div>
            <div style={{ fontSize: 36, marginBottom: 8 }}>🖼</div>
            <div style={{ fontSize: 15, color: '#666', marginBottom: 4 }}>
              Drag & drop an image here
            </div>
            <div style={{ fontSize: 13, color: '#aaa' }}>
              or click to browse • PNG, JPG, WEBP
            </div>
          </div>
        )}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }}
        disabled={disabled}
      />
    </div>
  )
}
