import { useMemo, useState } from 'react'
import { CROP_LIST, FERTILISER_LIST } from '../../engine/catalog'
import type { Bonus, CropDef, CropId, FertiliserId } from '../../engine/types'
import { BuffIcon } from './BuffChips'
import { IconEraser } from '../icons/Icons'

export type Tool =
  | { kind: 'crop'; cropId: CropId }
  | { kind: 'fertiliser'; fertiliserId: FertiliserId }
  | { kind: 'erase' }

export type CropSort =
  | 'name'
  | 'buff'
  | 'size'
  | 'growth'
  | 'value'

const SORT_LABELS: Record<CropSort, string> = {
  name: 'Name',
  buff: 'Buff',
  size: 'Size',
  growth: 'Growth days',
  value: 'Crop value',
}

const BUFF_ORDER: Bonus[] = [
  'Water Retain',
  'Weed Prevention',
  'Harvest Increase',
  'Quality Increase',
  'Speed Increase',
  'None',
]

const SIZE_ORDER = { single: 0, bush: 1, tree: 2 } as const

function sortCrops(crops: CropDef[], sort: CropSort): CropDef[] {
  const list = [...crops]
  list.sort((a, b) => {
    switch (sort) {
      case 'buff': {
        const d =
          BUFF_ORDER.indexOf(a.cropBonus) - BUFF_ORDER.indexOf(b.cropBonus)
        return d || a.name.localeCompare(b.name)
      }
      case 'size': {
        const d = SIZE_ORDER[a.size] - SIZE_ORDER[b.size]
        return d || a.name.localeCompare(b.name)
      }
      case 'growth': {
        const d = a.growthInfo.growthTime - b.growthInfo.growthTime
        return d || a.name.localeCompare(b.name)
      }
      case 'value': {
        const d = b.goldValues.crop - a.goldValues.crop
        return d || a.name.localeCompare(b.name)
      }
      case 'name':
      default:
        return a.name.localeCompare(b.name)
    }
  })
  return list
}

interface Props {
  tool: Tool
  onSelect: (tool: Tool) => void
  /** Dense thumb grid for narrow layouts */
  compact?: boolean
}

export function Palette({ tool, onSelect, compact = false }: Props) {
  const [sort, setSort] = useState<CropSort>('name')
  const crops = useMemo(() => sortCrops(CROP_LIST, sort), [sort])

  return (
    <aside className={`panel palette ${compact ? 'compact' : ''}`}>
      <div className="panel-scroll">
        <div className="panel-heading-row">
          <h2>Crops</h2>
          <label className="sort-field">
            <span className="sr-only">Sort crops</span>
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value as CropSort)}
              title="Sort"
            >
              {(Object.keys(SORT_LABELS) as CropSort[]).map((key) => (
                <option key={key} value={key}>
                  {SORT_LABELS[key]}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className={compact ? 'palette-thumbs' : 'palette-grid'}>
          {crops.map((crop) => {
            const active = tool.kind === 'crop' && tool.cropId === crop.id
            return (
              <button
                key={crop.id}
                type="button"
                className={`${compact ? 'thumb-item' : 'palette-item'} ${active ? 'active' : ''}`}
                onClick={() => onSelect({ kind: 'crop', cropId: crop.id })}
                title={`${crop.name} — ${crop.cropBonus}`}
              >
                <img
                  className={compact ? 'thumb-img' : 'palette-thumb'}
                  src={`/crops/${crop.id}.webp`}
                  alt={crop.name}
                  draggable={false}
                />
                {!compact && (
                  <>
                    <span className="palette-label">{crop.name}</span>
                    <span className="palette-meta">
                      <BuffIcon bonus={crop.cropBonus} size={12} />
                      {crop.cropBonus === 'None' ? 'No buff' : crop.cropBonus}
                      {crop.size !== 'single' ? ` · ${crop.size}` : ''}
                    </span>
                  </>
                )}
              </button>
            )
          })}
        </div>

        <h2>Fertilisers</h2>
        <div className={compact ? 'palette-thumbs' : 'palette-grid'}>
          {FERTILISER_LIST.map((f) => {
            const active =
              tool.kind === 'fertiliser' && tool.fertiliserId === f.id
            return (
              <button
                key={f.id}
                type="button"
                className={`${compact ? 'thumb-item' : 'palette-item'} ${active ? 'active' : ''}`}
                onClick={() =>
                  onSelect({ kind: 'fertiliser', fertiliserId: f.id })
                }
                title={`${f.name} — ${f.effect}`}
              >
                <img
                  className={compact ? 'thumb-img' : 'palette-thumb'}
                  src={`/fertilisers/${f.id}.webp`}
                  alt={f.name}
                  draggable={false}
                />
                {!compact && (
                  <>
                    <span className="palette-label">{f.name}</span>
                    <span className="palette-meta">
                      <BuffIcon bonus={f.effect} size={12} />
                      {f.effect}
                    </span>
                  </>
                )}
              </button>
            )
          })}
        </div>
      </div>

      <div className="panel-footer">
        <button
          type="button"
          className={`erase-btn ${tool.kind === 'erase' ? 'active' : ''}`}
          onClick={() => onSelect({ kind: 'erase' })}
          title="Eraser"
        >
          <IconEraser size={16} />
          Eraser
        </button>
      </div>
    </aside>
  )
}
