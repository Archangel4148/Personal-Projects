import cropsJson from '../data/crops.json'
import fertilisersJson from '../data/fertilisers.json'
import type { CropDef, CropId, FertiliserDef, FertiliserId } from './types'

export const CROPS: Record<CropId, CropDef> = cropsJson as Record<CropId, CropDef>
export const FERTILISERS: Record<FertiliserId, FertiliserDef> =
  fertilisersJson as Record<FertiliserId, FertiliserDef>

export const CROP_LIST: CropDef[] = Object.values(CROPS)
export const FERTILISER_LIST: FertiliserDef[] = Object.values(FERTILISERS)

export function getCrop(id: CropId): CropDef | undefined {
  return CROPS[id]
}

export function getFertiliser(id: FertiliserId): FertiliserDef | undefined {
  return FERTILISERS[id]
}
