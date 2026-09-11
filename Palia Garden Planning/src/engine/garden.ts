import { getCrop, getFertiliser } from './catalog'
import {
  BUFF_THRESHOLD,
  ORTHOGONAL,
  SIZE_DIMS,
  TRACKED_BONUSES,
  coordKey,
  type Bonus,
  type Coord,
  type CropId,
  type FertiliserId,
} from './types'

export interface PlantInstance {
  id: string
  cropId: CropId
  origin: Coord
  /** Active bonuses after threshold check */
  bonuses: Bonus[]
}

export interface TileState {
  plantId: string | null
  fertiliserId: FertiliserId | null
  /** Bonuses contributed onto this tile (neighbor crop tiles + fertiliser) */
  received: Bonus[]
}

export interface GardenSnapshot {
  size: number
  tiles: Record<string, TileState>
  plants: Record<string, PlantInstance>
}

let nextPlantId = 1

function newPlantId(): string {
  return `p${nextPlantId++}`
}

export function createGarden(size: number): GardenSnapshot {
  const tiles: Record<string, TileState> = {}
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      tiles[coordKey({ x, y })] = {
        plantId: null,
        fertiliserId: null,
        received: [],
      }
    }
  }
  return { size, tiles, plants: {} }
}

export function footprintFor(cropId: CropId, origin: Coord): Coord[] | null {
  const crop = getCrop(cropId)
  if (!crop) return null
  const { w, h } = SIZE_DIMS[crop.size]
  const cells: Coord[] = []
  for (let dy = 0; dy < h; dy++) {
    for (let dx = 0; dx < w; dx++) {
      cells.push({ x: origin.x + dx, y: origin.y + dy })
    }
  }
  return cells
}

export function canPlace(
  garden: GardenSnapshot,
  cropId: CropId,
  origin: Coord,
): boolean {
  const cells = footprintFor(cropId, origin)
  if (!cells) return false
  for (const c of cells) {
    if (c.x < 0 || c.y < 0 || c.x >= garden.size || c.y >= garden.size) {
      return false
    }
    const tile = garden.tiles[coordKey(c)]
    if (!tile || tile.plantId !== null) return false
  }
  return true
}

export function placeCrop(
  garden: GardenSnapshot,
  cropId: CropId,
  origin: Coord,
): GardenSnapshot {
  if (!canPlace(garden, cropId, origin)) return garden
  const cells = footprintFor(cropId, origin)!
  const plantId = newPlantId()
  const plants = { ...garden.plants }
  plants[plantId] = { id: plantId, cropId, origin, bonuses: [] }

  const tiles = { ...garden.tiles }
  for (const c of cells) {
    const key = coordKey(c)
    tiles[key] = { ...tiles[key], plantId }
  }

  return recalculateBonuses({ ...garden, tiles, plants })
}

export function eraseAt(garden: GardenSnapshot, at: Coord): GardenSnapshot {
  const key = coordKey(at)
  const tile = garden.tiles[key]
  if (!tile) return garden

  let tiles = { ...garden.tiles }
  let plants = { ...garden.plants }

  // Prefer clearing fertiliser so planted tiles can drop fert without uprooting
  if (tile.fertiliserId) {
    tiles[key] = { ...tile, fertiliserId: null }
    return recalculateBonuses({ ...garden, tiles })
  }

  if (tile.plantId) {
    const plantId = tile.plantId
    const plant = plants[plantId]
    if (plant) {
      const cells = footprintFor(plant.cropId, plant.origin) ?? [at]
      for (const c of cells) {
        const k = coordKey(c)
        tiles[k] = { ...tiles[k], plantId: null, received: [] }
      }
      delete plants[plantId]
    }
    return recalculateBonuses({ ...garden, tiles, plants })
  }

  return garden
}

export function placeFertiliser(
  garden: GardenSnapshot,
  fertiliserId: FertiliserId,
  at: Coord,
): GardenSnapshot {
  const key = coordKey(at)
  const tile = garden.tiles[key]
  if (!tile) return garden
  if (!getFertiliser(fertiliserId)) return garden

  const tiles = {
    ...garden.tiles,
    [key]: { ...tile, fertiliserId },
  }
  return recalculateBonuses({ ...garden, tiles })
}

export function clearFertiliser(garden: GardenSnapshot, at: Coord): GardenSnapshot {
  const key = coordKey(at)
  const tile = garden.tiles[key]
  if (!tile?.fertiliserId) return garden
  const tiles = {
    ...garden.tiles,
    [key]: { ...tile, fertiliserId: null },
  }
  return recalculateBonuses({ ...garden, tiles })
}

export function resizeGarden(garden: GardenSnapshot, size: number): GardenSnapshot {
  let next = createGarden(size)
  for (const plant of Object.values(garden.plants)) {
    const cells = footprintFor(plant.cropId, plant.origin)
    if (!cells) continue
    if (cells.every((c) => c.x < size && c.y < size)) {
      next = placeCrop(next, plant.cropId, plant.origin)
    }
  }
  for (let y = 0; y < Math.min(garden.size, size); y++) {
    for (let x = 0; x < Math.min(garden.size, size); x++) {
      const key = coordKey({ x, y })
      const old = garden.tiles[key]
      if (old?.fertiliserId) {
        next = placeFertiliser(next, old.fertiliserId, { x, y })
      }
    }
  }
  return next
}

export function clearGarden(garden: GardenSnapshot): GardenSnapshot {
  return createGarden(garden.size)
}

/**
 * Orthogonal neighbor contribution + fertiliser on tile.
 * Multi-tile crops aggregate received counts; threshold by size (1/2/3).
 */
export function recalculateBonuses(garden: GardenSnapshot): GardenSnapshot {
  const tiles: Record<string, TileState> = {}
  for (const [key, tile] of Object.entries(garden.tiles)) {
    tiles[key] = { ...tile, received: [] }
  }

  // Pass 1: collect received bonuses per tile
  for (const [key, tile] of Object.entries(tiles)) {
    const [x, y] = key.split(',').map(Number)
    const received: Bonus[] = []

    for (const d of ORTHOGONAL) {
      const nKey = coordKey({ x: x + d.x, y: y + d.y })
      const neighbor = tiles[nKey]
      if (!neighbor?.plantId) continue

      const nPlant = garden.plants[neighbor.plantId]
      if (!nPlant) continue

      // Same plant instance does not buff itself
      if (tile.plantId && neighbor.plantId === tile.plantId) continue

      const selfPlant = tile.plantId ? garden.plants[tile.plantId] : null
      // Same crop type cannot buff itself
      if (selfPlant && selfPlant.cropId === nPlant.cropId) continue

      const nCrop = getCrop(nPlant.cropId)
      if (!nCrop || nCrop.cropBonus === 'None') continue
      received.push(nCrop.cropBonus)
    }

    if (tile.fertiliserId) {
      const fert = getFertiliser(tile.fertiliserId)
      if (fert && fert.effect !== 'None') {
        received.push(fert.effect)
      }
    }

    tiles[key] = { ...tile, received }
  }

  // Pass 2: aggregate per plant and apply thresholds
  const plants: Record<string, PlantInstance> = {}
  for (const plant of Object.values(garden.plants)) {
    const crop = getCrop(plant.cropId)
    if (!crop) continue
    const cells = footprintFor(plant.cropId, plant.origin) ?? []
    const counts = new Map<Bonus, number>()

    for (const c of cells) {
      const t = tiles[coordKey(c)]
      if (!t) continue
      for (const bonus of t.received) {
        counts.set(bonus, (counts.get(bonus) ?? 0) + 1)
      }
    }

    const threshold = BUFF_THRESHOLD[crop.size]
    const bonuses: Bonus[] = []
    for (const [bonus, count] of counts) {
      if (count >= threshold) bonuses.push(bonus)
    }

    plants[plant.id] = { ...plant, bonuses }
  }

  return { ...garden, tiles, plants }
}

export interface BuffStats {
  plantCount: number
  withBonus: Record<Bonus, number>
  coveragePct: Record<Bonus, number>
}

export function computeBuffStats(garden: GardenSnapshot): BuffStats {
  const withBonus = Object.fromEntries(TRACKED_BONUSES.map((b) => [b, 0])) as Record<
    Bonus,
    number
  >
  const plants = Object.values(garden.plants)
  for (const plant of plants) {
    for (const b of plant.bonuses) {
      if (b in withBonus) withBonus[b]++
    }
  }
  const plantCount = plants.length
  const coveragePct = Object.fromEntries(
    TRACKED_BONUSES.map((b) => [
      b,
      plantCount === 0 ? 0 : Math.round((withBonus[b] / plantCount) * 100),
    ]),
  ) as Record<Bonus, number>

  return { plantCount, withBonus, coveragePct }
}

/** Reset plant id counter (tests). */
export function resetPlantIds(): void {
  nextPlantId = 1
}
