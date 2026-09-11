import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { getCrop, getFertiliser } from '../../engine/catalog'
import {
  canPlace,
  footprintFor,
  type GardenSnapshot,
} from '../../engine/garden'
import { coordKey, type Coord } from '../../engine/types'
import { BuffChips } from './BuffChips'
import type { Tool } from './Palette'

type PaintMode = 'place' | 'erase' | null

interface Props {
  garden: GardenSnapshot
  tool: Tool
  onTilePaint: (coord: Coord) => void
  onTileErase: (coord: Coord) => void
}

function coordFromPoint(clientX: number, clientY: number): Coord | null {
  const el = document.elementFromPoint(clientX, clientY)
  const tile = el?.closest?.('[data-tile]') as HTMLElement | null
  if (!tile) return null
  const x = Number(tile.dataset.x)
  const y = Number(tile.dataset.y)
  if (Number.isNaN(x) || Number.isNaN(y)) return null
  return { x, y }
}

export function GardenGrid({ garden, tool, onTilePaint, onTileErase }: Props) {
  const [hover, setHover] = useState<Coord | null>(null)
  const paintMode = useRef<PaintMode>(null)
  const lastPainted = useRef<string | null>(null)
  const gridRef = useRef<HTMLDivElement>(null)
  const paintRef = useRef(onTilePaint)
  const eraseRef = useRef(onTileErase)
  paintRef.current = onTilePaint
  eraseRef.current = onTileErase

  const applyAt = useCallback((coord: Coord, mode: PaintMode) => {
    if (!mode) return
    const key = coordKey(coord)
    if (lastPainted.current === key) return
    lastPainted.current = key
    if (mode === 'erase') eraseRef.current(coord)
    else paintRef.current(coord)
  }, [])

  const endPaint = useCallback(() => {
    paintMode.current = null
    lastPainted.current = null
  }, [])

  useEffect(() => {
    const onUp = () => endPaint()
    window.addEventListener('pointerup', onUp)
    window.addEventListener('pointercancel', onUp)
    window.addEventListener('blur', onUp)
    return () => {
      window.removeEventListener('pointerup', onUp)
      window.removeEventListener('pointercancel', onUp)
      window.removeEventListener('blur', onUp)
    }
  }, [endPaint])

  const previewKeys = useMemo(() => {
    if (!hover || tool.kind !== 'crop') return new Set<string>()
    const cells = footprintFor(tool.cropId, hover)
    if (!cells) return new Set<string>()
    return new Set(cells.map(coordKey))
  }, [hover, tool])

  const previewValid =
    hover && tool.kind === 'crop' ? canPlace(garden, tool.cropId, hover) : false

  const cellPx = Math.max(26, Math.min(52, Math.floor(480 / garden.size)))

  const onGridPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0 && e.button !== 2) return
    e.preventDefault()
    const mode: PaintMode = e.button === 2 ? 'erase' : 'place'
    paintMode.current = mode
    lastPainted.current = null
    gridRef.current?.setPointerCapture(e.pointerId)
    const coord = coordFromPoint(e.clientX, e.clientY)
    if (coord) {
      setHover(coord)
      applyAt(coord, mode)
    }
  }

  const onGridPointerMove = (e: React.PointerEvent) => {
    const coord = coordFromPoint(e.clientX, e.clientY)
    if (coord) setHover(coord)
    else if (!paintMode.current) setHover(null)

    if (!paintMode.current) return
    if (!(e.buttons & 1 || e.buttons & 2)) {
      endPaint()
      return
    }
    if (coord) applyAt(coord, paintMode.current)
  }

  return (
    <div className="grid-wrap">
      <div
        ref={gridRef}
        className="garden-grid"
        style={{
          gridTemplateColumns: `repeat(${garden.size}, ${cellPx}px)`,
          gridTemplateRows: `repeat(${garden.size}, ${cellPx}px)`,
        }}
        onContextMenu={(e) => e.preventDefault()}
        onPointerDown={onGridPointerDown}
        onPointerMove={onGridPointerMove}
        onPointerLeave={() => {
          if (!paintMode.current) setHover(null)
        }}
        onPointerUp={endPaint}
      >
        {Array.from({ length: garden.size * garden.size }, (_, i) => {
          const x = i % garden.size
          const y = Math.floor(i / garden.size)
          const key = coordKey({ x, y })
          const tile = garden.tiles[key]
          const plant = tile?.plantId ? garden.plants[tile.plantId] : null
          const crop = plant ? getCrop(plant.cropId) : null
          const fert = tile?.fertiliserId
            ? getFertiliser(tile.fertiliserId)
            : null
          const isOrigin =
            !!plant && plant.origin.x === x && plant.origin.y === y
          const isPreview = previewKeys.has(key)

          return (
            <div
              key={key}
              role="gridcell"
              data-tile
              data-x={x}
              data-y={y}
              className={[
                'tile',
                isPreview ? (previewValid ? 'preview-ok' : 'preview-bad') : '',
                plant ? 'has-crop' : '',
                isOrigin ? 'crop-origin' : '',
              ]
                .filter(Boolean)
                .join(' ')}
              title={
                crop
                  ? `${crop.name}${fert ? ` + ${fert.name}` : ''}`
                  : fert
                    ? fert.name
                    : undefined
              }
            >
              {isOrigin && crop && (
                <img
                  className="tile-crop visible"
                  src={`/crops/${crop.id}.webp`}
                  alt=""
                  draggable={false}
                />
              )}
              {!isOrigin && crop && <span className="tile-fill" />}
              {isOrigin && plant && (
                <BuffChips bonuses={plant.bonuses} size={11} />
              )}
              {fert && (
                <img
                  className="fert-img"
                  src={`/fertilisers/${fert.id}.webp`}
                  alt=""
                  draggable={false}
                />
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
