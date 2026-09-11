import { describe, it, expect, beforeEach } from 'vitest'
import {
  createGarden,
  placeCrop,
  placeFertiliser,
  eraseAt,
  computeBuffStats,
  resetPlantIds,
  canPlace,
} from '../../src/engine/garden'

beforeEach(() => {
  resetPlantIds()
})

describe('buff rules', () => {
  it('tomato grants Water Retain to orthogonal carrot', () => {
    let g = createGarden(3)
    g = placeCrop(g, 'tomato', { x: 0, y: 0 })
    g = placeCrop(g, 'carrot', { x: 1, y: 0 })
    const carrot = Object.values(g.plants).find((p) => p.cropId === 'carrot')!
    expect(carrot.bonuses).toContain('Water Retain')
    const tomato = Object.values(g.plants).find((p) => p.cropId === 'tomato')!
    expect(tomato.bonuses).toContain('Weed Prevention')
  })

  it('same crop type does not buff itself', () => {
    let g = createGarden(3)
    g = placeCrop(g, 'tomato', { x: 0, y: 0 })
    g = placeCrop(g, 'tomato', { x: 1, y: 0 })
    for (const p of Object.values(g.plants)) {
      expect(p.bonuses).not.toContain('Water Retain')
    }
  })

  it('diagonal does not grant buffs', () => {
    let g = createGarden(3)
    g = placeCrop(g, 'tomato', { x: 0, y: 0 })
    g = placeCrop(g, 'carrot', { x: 1, y: 1 })
    const carrot = Object.values(g.plants).find((p) => p.cropId === 'carrot')!
    expect(carrot.bonuses).not.toContain('Water Retain')
  })

  it('bush needs 2 matching neighbor tiles for a bonus', () => {
    let g = createGarden(6)
    // Blueberry at 1,1 covering 1,1 2,1 1,2 2,2
    g = placeCrop(g, 'blueberry', { x: 1, y: 1 })
    // One tomato adjacent (touches one blueberry tile)
    g = placeCrop(g, 'tomato', { x: 0, y: 1 })
    let berry = Object.values(g.plants).find((p) => p.cropId === 'blueberry')!
    expect(berry.bonuses).not.toContain('Water Retain')

    // Second tomato adjacent to another side
    g = placeCrop(g, 'tomato', { x: 1, y: 0 })
    berry = Object.values(g.plants).find((p) => p.cropId === 'blueberry')!
    expect(berry.bonuses).toContain('Water Retain')
  })

  it('apple tree needs 3 matching neighbor tiles', () => {
    let g = createGarden(9)
    g = placeCrop(g, 'apple', { x: 1, y: 1 })
    g = placeCrop(g, 'tomato', { x: 0, y: 1 })
    g = placeCrop(g, 'tomato', { x: 1, y: 0 })
    let apple = Object.values(g.plants).find((p) => p.cropId === 'apple')!
    expect(apple.bonuses).not.toContain('Water Retain')

    g = placeCrop(g, 'tomato', { x: 4, y: 2 }) // right of tree (tree ends at x=3)
    apple = Object.values(g.plants).find((p) => p.cropId === 'apple')!
    expect(apple.bonuses).toContain('Water Retain')
  })

  it('fertiliser grants bonus without stacking duplicate type from crops beyond threshold', () => {
    let g = createGarden(3)
    g = placeCrop(g, 'carrot', { x: 1, y: 1 })
    g = placeFertiliser(g, 'hydrate-pro', { x: 1, y: 1 })
    const carrot = Object.values(g.plants).find((p) => p.cropId === 'carrot')!
    expect(carrot.bonuses).toContain('Water Retain')
  })

  it('coverage stats count plants with each bonus', () => {
    let g = createGarden(3)
    g = placeCrop(g, 'tomato', { x: 0, y: 0 })
    g = placeCrop(g, 'carrot', { x: 1, y: 0 })
    const stats = computeBuffStats(g)
    expect(stats.plantCount).toBe(2)
    expect(stats.withBonus['Water Retain']).toBe(1)
    expect(stats.withBonus['Weed Prevention']).toBe(1)
  })
})

describe('placement', () => {
  it('blocks overlapping multi-tile crops', () => {
    let g = createGarden(6)
    g = placeCrop(g, 'blueberry', { x: 0, y: 0 })
    expect(canPlace(g, 'blueberry', { x: 1, y: 0 })).toBe(false)
    expect(canPlace(g, 'tomato', { x: 2, y: 0 })).toBe(true)
  })

  it('erase removes whole multi-tile plant', () => {
    let g = createGarden(6)
    g = placeCrop(g, 'blueberry', { x: 0, y: 0 })
    g = eraseAt(g, { x: 1, y: 1 })
    expect(Object.keys(g.plants)).toHaveLength(0)
    expect(g.tiles['0,0'].plantId).toBeNull()
  })
})
