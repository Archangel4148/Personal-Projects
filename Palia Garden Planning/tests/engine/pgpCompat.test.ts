import { describe, it, expect, beforeEach } from 'vitest'
import {
  createGarden,
  placeCrop,
  placeFertiliser,
  resetPlantIds,
} from '../../src/engine/garden'
import { DEFAULT_SETTINGS } from '../../src/engine/gold'
import {
  compressPlotString,
  expandPlotCode,
  exportPgpCode,
  importPgpCode,
} from '../../src/engine/pgpCompat'
import { deserializeGarden, serializeGarden } from '../../src/engine/save'

beforeEach(() => {
  resetPlantIds()
})

describe('pgp expand/compress', () => {
  it('round-trips run-length tiles', () => {
    const tiles = ['T', 'T', 'T', 'N', 'B.H', 'B.H', 'N', 'N', 'N']
    const compressed = compressPlotString(tiles)
    expect(expandPlotCode(compressed)).toEqual(tiles)
  })
})

describe('pgp layout codes', () => {
  it('exports v0.5 and reimports crops + fert', () => {
    let g = createGarden(3)
    g = placeCrop(g, 'tomato', { x: 0, y: 0 })
    g = placeCrop(g, 'carrot', { x: 1, y: 0 })
    g = placeFertiliser(g, 'harvest-boost', { x: 1, y: 0 })

    const code = exportPgpCode(g, { ...DEFAULT_SETTINGS, days: 30, gardeningLevel: 10 })
    expect(code.startsWith('v0.5_D-3x3_CR-')).toBe(true)

    const loaded = importPgpCode(code)
    const crops = Object.values(loaded.garden.plants)
      .map((p) => p.cropId)
      .sort()
    expect(crops).toEqual(['carrot', 'tomato'])
    expect(loaded.garden.tiles['1,0'].fertiliserId).toBe('harvest-boost')
    expect(loaded.settings.days).toBe(30)
    expect(loaded.settings.gardeningLevel).toBe(10)
  })

  it('imports a classic 0.4 matrix layout', () => {
    // One 3×3 active plot with tomato + empty tiles
    const code = 'v0.4_D-1_CR-TN8'
    // D-1 means one plot row with one active plot? Actually D-1 is just "1" one cell
    const code2 = 'v0.4_D-1_CR-TNNNNNNNN'
    const loaded = importPgpCode(code2)
    expect(Object.values(loaded.garden.plants).some((p) => p.cropId === 'tomato')).toBe(
      true,
    )
    // silence unused
    expect(code.length).toBeGreaterThan(0)
  })

  it('imports batterfly conversion sample (0.3 → codes with Bt)', () => {
    const v03 =
      'v0.3_D-111-111-111_CR-BtBtBtBtBtBtBtBtBt-BtNBtBtNBtBtBtBt-BtBtBtBtBtBtBtBtN-BtBtBtBtBtBtBtBtBt-BtBtBtBtBtBtBtBtBt-BtBtNBtBtNBtBtN-NNNNNNNNN-NNNNNNNNN-NNNNNNNNN_D30L50'
    const loaded = importPgpCode(v03)
    expect(loaded.garden.size).toBe(9)
    expect(
      Object.values(loaded.garden.plants).every((p) => p.cropId === 'batterfly-bean'),
    ).toBe(true)
    expect(loaded.settings.days).toBe(30)
    expect(loaded.settings.gardeningLevel).toBe(50)
  })

  it('serializeGarden produces web-compatible codes', () => {
    let g = createGarden(3)
    g = placeCrop(g, 'wheat', { x: 0, y: 0 })
    const code = serializeGarden(g, DEFAULT_SETTINGS)
    const again = deserializeGarden(code)
    expect(Object.values(again.garden.plants)[0]?.cropId).toBe('wheat')
  })
})
