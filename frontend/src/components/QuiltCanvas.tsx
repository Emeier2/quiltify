import { useCallback, useRef } from 'react'
import type { QuiltPatternSchema, FabricSchema } from '../types/pattern'

interface Props {
  pattern: QuiltPatternSchema
  selectedFabricId: string | null
  onPatternChange: (updated: QuiltPatternSchema) => void
}

const CELL_PX = 12

export function QuiltCanvas({ pattern, selectedFabricId, onPatternChange }: Props) {
  const isDragging = useRef(false)

  const fabricMap = Object.fromEntries(pattern.fabrics.map(f => [f.id, f]))
  const { colOffsets, rowOffsets } = buildOffsets(pattern)

  function paintCell(gx: number, gy: number) {
    if (!selectedFabricId) return
    const blockIdx = pattern.blocks.findIndex(
      b => gx >= b.x && gx < b.x + b.width && gy >= b.y && gy < b.y + b.height
    )
    if (blockIdx === -1) return
    const block = pattern.blocks[blockIdx]
    if (block.fabric_id === selectedFabricId) return

    const newBlocks = splitBlock(pattern.blocks, blockIdx, gx, gy, selectedFabricId)
    onPatternChange({ ...pattern, blocks: mergeBlocks(newBlocks) })
  }

  const handleMouseDown = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    isDragging.current = true
    const { gx, gy } = getCellCoords(e, pattern, colOffsets, rowOffsets)
    paintCell(gx, gy)
  }, [pattern, selectedFabricId, colOffsets, rowOffsets])

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!isDragging.current) return
    const { gx, gy } = getCellCoords(e, pattern, colOffsets, rowOffsets)
    paintCell(gx, gy)
  }, [pattern, selectedFabricId, colOffsets, rowOffsets])

  const handleMouseUp = useCallback(() => {
    isDragging.current = false
  }, [])

  const W = colOffsets[colOffsets.length - 1]
  const H = rowOffsets[rowOffsets.length - 1]

  return (
    <div style={{ overflow: 'auto', border: '1px solid #ddd', borderRadius: 6 }}>
      <svg
        width={W}
        height={H}
        viewBox={`0 0 ${W} ${H}`}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: selectedFabricId ? 'crosshair' : 'default', display: 'block' }}
      >
        <rect width={W} height={H} fill="#f5f5f0" />
        {pattern.blocks.map((block, i) => {
          const fab = fabricMap[block.fabric_id]
          const color = fab?.color_hex ?? '#cccccc'
          const { xPx, yPx, wPx, hPx } = blockRectPx(pattern, block, colOffsets, rowOffsets)
          return (
            <g key={i}>
              <rect
                x={xPx}
                y={yPx}
                width={wPx}
                height={hPx}
                fill={color}
                stroke="#ffffff"
                strokeWidth={0.5}
              />
              {renderCornerTriangles(block, pattern, colOffsets, rowOffsets, fabricMap)}
            </g>
          )
        })}
        {/* Grid lines every 10 cells */}
        {Array.from({ length: Math.floor(pattern.grid_width / 10) + 1 }, (_, i) => (
          <line
            key={`v${i}`}
            x1={colOffsets[Math.min(i * 10, pattern.grid_width)] ?? 0} y1={0}
            x2={colOffsets[Math.min(i * 10, pattern.grid_width)] ?? 0} y2={H}
            stroke="rgba(0,0,0,0.08)" strokeWidth={0.5}
          />
        ))}
        {Array.from({ length: Math.floor(pattern.grid_height / 10) + 1 }, (_, i) => (
          <line
            key={`h${i}`}
            x1={0} y1={rowOffsets[Math.min(i * 10, pattern.grid_height)] ?? 0}
            x2={W} y2={rowOffsets[Math.min(i * 10, pattern.grid_height)] ?? 0}
            stroke="rgba(0,0,0,0.08)" strokeWidth={0.5}
          />
        ))}
      </svg>
    </div>
  )
}

function getCellCoords(
  e: React.MouseEvent<SVGSVGElement>,
  pattern: QuiltPatternSchema,
  colOffsets: number[],
  rowOffsets: number[],
): { gx: number; gy: number } {
  const rect = e.currentTarget.getBoundingClientRect()
  const px = e.clientX - rect.left
  const py = e.clientY - rect.top
  const gx = findIndexFromOffset(colOffsets, px)
  const gy = findIndexFromOffset(rowOffsets, py)
  return { gx, gy }
}

function splitBlock(
  blocks: QuiltPatternSchema['blocks'],
  blockIdx: number,
  cx: number,
  cy: number,
  newFabricId: string,
): QuiltPatternSchema['blocks'] {
  const block = blocks[blockIdx]
  const result = [...blocks.slice(0, blockIdx), ...blocks.slice(blockIdx + 1)]

  if (cy > block.y) {
    const sub = { x: block.x, y: block.y, width: block.width, height: cy - block.y, fabric_id: block.fabric_id }
    result.push({ ...sub, corners: subBlockCorners(block, sub) })
  }
  const bottomY = cy + 1
  if (bottomY < block.y + block.height) {
    const sub = { x: block.x, y: bottomY, width: block.width, height: (block.y + block.height) - bottomY, fabric_id: block.fabric_id }
    result.push({ ...sub, corners: subBlockCorners(block, sub) })
  }
  if (cx > block.x) {
    const sub = { x: block.x, y: cy, width: cx - block.x, height: 1, fabric_id: block.fabric_id }
    result.push({ ...sub, corners: subBlockCorners(block, sub) })
  }
  const rightX = cx + 1
  if (rightX < block.x + block.width) {
    const sub = { x: rightX, y: cy, width: (block.x + block.width) - rightX, height: 1, fabric_id: block.fabric_id }
    result.push({ ...sub, corners: subBlockCorners(block, sub) })
  }
  result.push({ x: cx, y: cy, width: 1, height: 1, fabric_id: newFabricId })

  return result
}

function mergeBlocks(blocks: QuiltPatternSchema['blocks']): QuiltPatternSchema['blocks'] {
  const sorted = [...blocks].sort((a, b) =>
    a.y !== b.y ? a.y - b.y : a.x !== b.x ? a.x - b.x : 0
  )
  const merged: QuiltPatternSchema['blocks'] = []
  const used = new Set<number>()

  for (let i = 0; i < sorted.length; i++) {
    if (used.has(i)) continue
    let current = { ...sorted[i] }
    for (let j = i + 1; j < sorted.length; j++) {
      if (used.has(j)) continue
      const next = sorted[j]
      if (
        next.fabric_id === current.fabric_id &&
        next.y === current.y &&
        next.height === current.height &&
        next.x === current.x + current.width &&
        JSON.stringify(next.corners ?? {}) === JSON.stringify(current.corners ?? {})
      ) {
        current = { ...current, width: current.width + next.width }
        used.add(j)
      }
    }
    merged.push(current)
    used.add(i)
  }
  return merged
}

function buildOffsets(pattern: QuiltPatternSchema): { colOffsets: number[]; rowOffsets: number[] } {
  const colWidths: number[] = []
  const rowHeights: number[] = []
  for (let x = 0; x < pattern.grid_width; x++) {
    const idx = x
    const cell = pattern.cell_sizes[idx] ?? { w: 1, h: 1 }
    colWidths.push(cell.w * CELL_PX)
  }
  for (let y = 0; y < pattern.grid_height; y++) {
    const idx = y * pattern.grid_width
    const cell = pattern.cell_sizes[idx] ?? { w: 1, h: 1 }
    rowHeights.push(cell.h * CELL_PX)
  }
  const colOffsets = [0]
  const rowOffsets = [0]
  for (const w of colWidths) colOffsets.push(colOffsets[colOffsets.length - 1] + w)
  for (const h of rowHeights) rowOffsets.push(rowOffsets[rowOffsets.length - 1] + h)
  return { colOffsets, rowOffsets }
}

function findIndexFromOffset(offsets: number[], value: number): number {
  for (let i = 0; i < offsets.length - 1; i++) {
    if (value >= offsets[i] && value < offsets[i + 1]) return i
  }
  return offsets.length - 2
}

function blockRectPx(
  pattern: QuiltPatternSchema,
  block: QuiltPatternSchema['blocks'][number],
  colOffsets: number[],
  rowOffsets: number[],
): { xPx: number; yPx: number; wPx: number; hPx: number } {
  const xPx = colOffsets[block.x]
  const yPx = rowOffsets[block.y]
  const wPx = colOffsets[block.x + block.width] - colOffsets[block.x]
  const hPx = rowOffsets[block.y + block.height] - rowOffsets[block.y]
  return { xPx, yPx, wPx, hPx }
}

function renderCornerTriangles(
  block: QuiltPatternSchema['blocks'][number],
  pattern: QuiltPatternSchema,
  colOffsets: number[],
  rowOffsets: number[],
  fabricMap: Record<string, FabricSchema>,
) {
  const corners = block.corners ?? {}
  return Object.entries(corners).map(([cornerName, cornerFab]) => {
    let cx = block.x
    let cy = block.y
    if (cornerName === 'ne') {
      cx = block.x + block.width - 1
      cy = block.y
    } else if (cornerName === 'sw') {
      cx = block.x
      cy = block.y + block.height - 1
    } else if (cornerName === 'se') {
      cx = block.x + block.width - 1
      cy = block.y + block.height - 1
    }
    const cell = pattern.cell_sizes[cy * pattern.grid_width + cx] ?? { w: 1, h: 1 }
    const px = colOffsets[cx]
    const py = rowOffsets[cy]
    const pw = cell.w * CELL_PX
    const ph = cell.h * CELL_PX
    const color = fabricMap[cornerFab]?.color_hex ?? '#cccccc'
    let points = ''
    if (cornerName === 'nw') {
      points = `${px},${py} ${px + pw},${py} ${px},${py + ph}`
    } else if (cornerName === 'ne') {
      points = `${px + pw},${py} ${px + pw},${py + ph} ${px},${py}`
    } else if (cornerName === 'sw') {
      points = `${px},${py + ph} ${px + pw},${py + ph} ${px},${py}`
    } else {
      points = `${px + pw},${py + ph} ${px},${py + ph} ${px + pw},${py}`
    }
    return (
      <polygon
        key={`${block.x}-${block.y}-${cornerName}`}
        points={points}
        fill={color}
        stroke="#ffffff"
        strokeWidth={0.5}
      />
    )
  })
}

function subBlockCorners(
  parent: QuiltPatternSchema['blocks'][number],
  sub: { x: number; y: number; width: number; height: number },
): { [key: string]: string } {
  const corners = parent.corners ?? {}
  const result: { [key: string]: string } = {}
  const parentCornerCoords: Record<string, { x: number; y: number }> = {
    nw: { x: parent.x, y: parent.y },
    ne: { x: parent.x + parent.width - 1, y: parent.y },
    sw: { x: parent.x, y: parent.y + parent.height - 1 },
    se: { x: parent.x + parent.width - 1, y: parent.y + parent.height - 1 },
  }
  const subCornerCoords: Record<string, { x: number; y: number }> = {
    nw: { x: sub.x, y: sub.y },
    ne: { x: sub.x + sub.width - 1, y: sub.y },
    sw: { x: sub.x, y: sub.y + sub.height - 1 },
    se: { x: sub.x + sub.width - 1, y: sub.y + sub.height - 1 },
  }
  for (const [name, fab] of Object.entries(corners)) {
    const p = parentCornerCoords[name]
    if (!p) continue
    for (const [subName, coord] of Object.entries(subCornerCoords)) {
      if (coord.x === p.x && coord.y === p.y) {
        result[subName] = fab
      }
    }
  }
  return result
}
