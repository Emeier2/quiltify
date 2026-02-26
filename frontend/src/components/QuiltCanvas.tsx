import { useCallback, useRef, useState } from 'react'
import type { QuiltPatternSchema, FabricSchema } from '../types/pattern'

interface Props {
  pattern: QuiltPatternSchema
  selectedFabricId: string | null
  onPatternChange: (updated: QuiltPatternSchema) => void
}

const CELL_PX = 12

export function QuiltCanvas({ pattern, selectedFabricId, onPatternChange }: Props) {
  const isDragging = useRef(false)

  // Build a flat cell-color grid from blocks
  const cellGrid = buildCellGrid(pattern)
  const fabricMap = Object.fromEntries(pattern.fabrics.map(f => [f.id, f]))

  function paintCell(gx: number, gy: number) {
    if (!selectedFabricId) return
    // Find which block contains this cell
    const blockIdx = pattern.blocks.findIndex(
      b => gx >= b.x && gx < b.x + b.width && gy >= b.y && gy < b.y + b.height
    )
    if (blockIdx === -1) return
    const block = pattern.blocks[blockIdx]
    if (block.fabric_id === selectedFabricId) return

    // Split the block: replace the single cell as a 1×1, keep the rest
    const newBlocks = splitBlock(pattern.blocks, blockIdx, gx, gy, selectedFabricId)
    onPatternChange({ ...pattern, blocks: mergeBlocks(newBlocks) })
  }

  const handleMouseDown = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    isDragging.current = true
    const { gx, gy } = getCellCoords(e)
    paintCell(gx, gy)
  }, [pattern, selectedFabricId])

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!isDragging.current) return
    const { gx, gy } = getCellCoords(e)
    paintCell(gx, gy)
  }, [pattern, selectedFabricId])

  const handleMouseUp = useCallback(() => {
    isDragging.current = false
  }, [])

  const W = pattern.grid_width * CELL_PX
  const H = pattern.grid_height * CELL_PX

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
          return (
            <rect
              key={i}
              x={block.x * CELL_PX}
              y={block.y * CELL_PX}
              width={block.width * CELL_PX}
              height={block.height * CELL_PX}
              fill={color}
              stroke="#ffffff"
              strokeWidth={0.5}
            />
          )
        })}
        {/* Grid lines every 10 cells */}
        {Array.from({ length: Math.floor(pattern.grid_width / 10) + 1 }, (_, i) => (
          <line
            key={`v${i}`}
            x1={i * 10 * CELL_PX} y1={0}
            x2={i * 10 * CELL_PX} y2={H}
            stroke="rgba(0,0,0,0.08)" strokeWidth={0.5}
          />
        ))}
        {Array.from({ length: Math.floor(pattern.grid_height / 10) + 1 }, (_, i) => (
          <line
            key={`h${i}`}
            x1={0} y1={i * 10 * CELL_PX}
            x2={W} y2={i * 10 * CELL_PX}
            stroke="rgba(0,0,0,0.08)" strokeWidth={0.5}
          />
        ))}
      </svg>
    </div>
  )
}

function getCellCoords(e: React.MouseEvent<SVGSVGElement>): { gx: number; gy: number } {
  const rect = e.currentTarget.getBoundingClientRect()
  const px = e.clientX - rect.left
  const py = e.clientY - rect.top
  return {
    gx: Math.floor(px / CELL_PX),
    gy: Math.floor(py / CELL_PX),
  }
}

function buildCellGrid(pattern: QuiltPatternSchema): Map<string, string> {
  const grid = new Map<string, string>()
  for (const block of pattern.blocks) {
    for (let dy = 0; dy < block.height; dy++) {
      for (let dx = 0; dx < block.width; dx++) {
        grid.set(`${block.x + dx},${block.y + dy}`, block.fabric_id)
      }
    }
  }
  return grid
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

  // Surround the target cell with up to 4 sub-blocks from the original block
  // Top slice
  if (cy > block.y) {
    result.push({ x: block.x, y: block.y, width: block.width, height: cy - block.y, fabric_id: block.fabric_id })
  }
  // Bottom slice
  const bottomY = cy + 1
  if (bottomY < block.y + block.height) {
    result.push({ x: block.x, y: bottomY, width: block.width, height: (block.y + block.height) - bottomY, fabric_id: block.fabric_id })
  }
  // Left of cell (in the same row)
  if (cx > block.x) {
    result.push({ x: block.x, y: cy, width: cx - block.x, height: 1, fabric_id: block.fabric_id })
  }
  // Right of cell (in the same row)
  const rightX = cx + 1
  if (rightX < block.x + block.width) {
    result.push({ x: rightX, y: cy, width: (block.x + block.width) - rightX, height: 1, fabric_id: block.fabric_id })
  }
  // The target cell itself
  result.push({ x: cx, y: cy, width: 1, height: 1, fabric_id: newFabricId })

  return result
}

function mergeBlocks(blocks: QuiltPatternSchema['blocks']): QuiltPatternSchema['blocks'] {
  // Simple pass: try to merge horizontally adjacent same-fabric blocks with same y and height
  // This is a lightweight pass, not a full optimal merge
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
      // Horizontal merge: same row/height/fabric, adjacent
      if (
        next.fabric_id === current.fabric_id &&
        next.y === current.y &&
        next.height === current.height &&
        next.x === current.x + current.width
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
