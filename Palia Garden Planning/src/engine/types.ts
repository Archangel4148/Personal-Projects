/** Core types for the garden planner engine. Values mirror VincentAmante/palia-tools. */

export type Bonus =
  | 'None'
  | 'Water Retain'
  | 'Harvest Increase'
  | 'Quality Increase'
  | 'Speed Increase'
  | 'Weed Prevention'

export type CropSize = 'single' | 'bush' | 'tree'

export type CropId = string
export type FertiliserId = string

export interface GrowthInfo {
  base: number
  growthTime: number
  withBonus: number
  isReharvestable: boolean
  reharvestCooldown: number
  reharvestLimit: number
}

export interface GoldValues {
  crop: number
  cropStar: number
  seed: number
  seedStar: number
  hasPreserve: boolean
  preserve: number
  preserveStar: number
}

export interface ConversionInfo {
  cropsPerSeed: number
  seedsPerConversion: number
  cropsPerPreserve: number
  seedProcessMinutes: number
  preserveProcessMinutes: number
}

export interface CropDef {
  id: CropId
  name: string
  cropBonus: Bonus
  size: CropSize
  color: string
  code: string
  growthInfo: GrowthInfo
  goldValues: GoldValues
  costs: { zekiPrice: number; guildPrice: number; potionPrice: number }
  conversion: ConversionInfo
}

export interface FertiliserDef {
  id: FertiliserId
  name: string
  effect: Bonus
  color: string
  code: string
  costs: {
    zekiBatchPrice: number
    zekiBatchCount: number
    guildBatchPrice: number
    guildBatchCount: number
    goldSellValue: number
  }
}

export type SellMode = 'crop' | 'seed' | 'preserve'

export interface SimSettings {
  days: number
  gardeningLevel: number
  useStarSeeds: boolean
  sellMode: SellMode
  includeReplantCost: boolean
  useGrowthBoost: boolean
}

export interface Coord {
  x: number
  y: number
}

export function coordKey(c: Coord): string {
  return `${c.x},${c.y}`
}

export function parseCoordKey(key: string): Coord {
  const [x, y] = key.split(',').map(Number)
  return { x, y }
}

export const SIZE_DIMS: Record<CropSize, { w: number; h: number }> = {
  single: { w: 1, h: 1 },
  bush: { w: 2, h: 2 },
  tree: { w: 3, h: 3 },
}

export const BUFF_THRESHOLD: Record<CropSize, number> = {
  single: 1,
  bush: 2,
  tree: 3,
}

export const TRACKED_BONUSES: Bonus[] = [
  'Water Retain',
  'Weed Prevention',
  'Harvest Increase',
  'Quality Increase',
  'Speed Increase',
]

export const ORTHOGONAL: Coord[] = [
  { x: 0, y: -1 },
  { x: 1, y: 0 },
  { x: 0, y: 1 },
  { x: -1, y: 0 },
]
