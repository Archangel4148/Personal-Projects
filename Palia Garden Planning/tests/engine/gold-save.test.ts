import { describe, it, expect, beforeEach } from 'vitest'
import { createGarden, placeCrop, resetPlantIds } from '../../src/engine/garden'
import { estimateGold, starChance, DEFAULT_SETTINGS } from '../../src/engine/gold'
import {
  deserializeGarden,
  serializeGarden,
  serializeNative,
  toPrettyJson,
  fromPrettyJson,
} from '../../src/engine/save'

beforeEach(() => {
  resetPlantIds()
})

describe('starChance', () => {
  it('matches reference formula', () => {
    expect(starChance(0, false, false)).toBeCloseTo(0.25)
    expect(starChance(0, true, false)).toBeCloseTo(0.5)
    expect(starChance(0, true, true)).toBeCloseTo(1)
    expect(starChance(25, true, false)).toBeCloseTo(1)
  })
})

describe('gold estimate', () => {
  it('returns positive gold for a simple tomato layout', () => {
    let g = createGarden(3)
    g = placeCrop(g, 'tomato', { x: 0, y: 0 })
    const result = estimateGold(g, { ...DEFAULT_SETTINGS, days: 12 })
    expect(result.totalGold).toBeGreaterThan(0)
    expect(result.cropsHarvested).toBeGreaterThan(0)
  })
})

describe('save/load', () => {
  it('round-trips via web-compatible layout code', () => {
    let g = createGarden(6)
    g = placeCrop(g, 'tomato', { x: 0, y: 0 })
    g = placeCrop(g, 'wheat', { x: 1, y: 0 })
    const code = serializeGarden(g, DEFAULT_SETTINGS)
    expect(code.startsWith('v0.5_')).toBe(true)
    const { garden } = deserializeGarden(code)
    expect(garden.size).toBe(6)
    expect(Object.values(garden.plants).map((p) => p.cropId).sort()).toEqual([
      'tomato',
      'wheat',
    ])
  })

  it('round-trips native JSON', () => {
    let g = createGarden(3)
    g = placeCrop(g, 'cotton', { x: 0, y: 0 })
    const json = toPrettyJson(g, DEFAULT_SETTINGS)
    const { garden } = fromPrettyJson(json)
    expect(Object.values(garden.plants)[0]?.cropId).toBe('cotton')
    const native = serializeNative(g, DEFAULT_SETTINGS)
    expect(deserializeGarden(native).garden.size).toBe(3)
  })
})
