import { getCrop } from './catalog'
import type { GardenSnapshot } from './garden'
import type { Bonus, SellMode, SimSettings } from './types'

/**
 * Community / reference planner star-chance estimate:
 * 0.25 + 0.25*starSeeds + 0.02*level + 0.5*qualityBoost
 */
export function starChance(
  gardeningLevel: number,
  useStarSeeds: boolean,
  hasQualityBoost: boolean,
): number {
  const chance =
    0.25 +
    (useStarSeeds ? 0.25 : 0) +
    0.02 * gardeningLevel +
    (hasQualityBoost ? 0.5 : 0)
  return Math.min(1, Math.max(0, chance))
}

function unitValue(
  cropId: string,
  sellMode: SellMode,
  isStar: boolean,
): number {
  const crop = getCrop(cropId)
  if (!crop) return 0
  const g = crop.goldValues
  switch (sellMode) {
    case 'seed':
      return isStar ? g.seedStar : g.seed
    case 'preserve':
      if (!g.hasPreserve) return isStar ? g.cropStar : g.crop
      return isStar ? g.preserveStar : g.preserve
    case 'crop':
    default:
      return isStar ? g.cropStar : g.crop
  }
}

function seedCost(cropId: string, useStarSeeds: boolean): number {
  const crop = getCrop(cropId)
  if (!crop) return 0
  // Prefer Zeki price; fall back to guild / potion
  const buy = crop.costs.zekiPrice || crop.costs.guildPrice || crop.costs.potionPrice
  if (buy > 0) return buy
  // Approximate star seed premium via seed sell values when buy price missing
  return useStarSeeds ? crop.goldValues.seedStar : crop.goldValues.seed
}

interface HarvestEvent {
  day: number
  plantId: string
  cropId: string
  amount: number
  hasHarvestBoost: boolean
  hasQualityBoost: boolean
  hasSpeedBoost: boolean
}

function plantHas(bonuses: Bonus[], bonus: Bonus): boolean {
  return bonuses.includes(bonus)
}

/**
 * Simulate harvests over N days with instant processing (v1).
 * Growth uses Speedy Gro / Speed Increase when enabled via settings.useGrowthBoost
 * only for growth-time reduction from Speed Increase bonus (fertiliser Speedy Gro).
 */
export function estimateGold(
  garden: GardenSnapshot,
  settings: SimSettings,
): {
  totalGold: number
  goldPerDay: number
  cropsHarvested: number
  events: HarvestEvent[]
} {
  const events: HarvestEvent[] = []
  let totalGold = 0
  let cropsHarvested = 0
  let replantCost = 0

  for (const plant of Object.values(garden.plants)) {
    const crop = getCrop(plant.cropId)
    if (!crop) continue

    const hasHarvest = plantHas(plant.bonuses, 'Harvest Increase')
    const hasQuality = plantHas(plant.bonuses, 'Quality Increase')
    const hasSpeed = plantHas(plant.bonuses, 'Speed Increase')

    const growthTime =
      settings.useGrowthBoost && hasSpeed
        ? Math.max(1, Math.floor(crop.growthInfo.growthTime * 0.5)) // Speedy Gro halves; approximate
        : crop.growthInfo.growthTime

    const yieldPerHarvest = hasHarvest
      ? crop.growthInfo.withBonus
      : crop.growthInfo.base

    const chance = starChance(
      settings.gardeningLevel,
      settings.useStarSeeds,
      hasQuality,
    )

    // Initial plant cost
    if (settings.includeReplantCost) {
      replantCost += seedCost(plant.cropId, settings.useStarSeeds)
    }

    let day = growthTime
    let reharvestsDone = 0

    while (day <= settings.days) {
      events.push({
        day,
        plantId: plant.id,
        cropId: plant.cropId,
        amount: yieldPerHarvest,
        hasHarvestBoost: hasHarvest,
        hasQualityBoost: hasQuality,
        hasSpeedBoost: hasSpeed,
      })

      const starAmount = yieldPerHarvest * chance
      const normalAmount = yieldPerHarvest * (1 - chance)
      const gold =
        starAmount * unitValue(plant.cropId, settings.sellMode, true) +
        normalAmount * unitValue(plant.cropId, settings.sellMode, false)

      totalGold += gold
      cropsHarvested += yieldPerHarvest

      if (!crop.growthInfo.isReharvestable) {
        // Replant single-harvest crops
        if (settings.includeReplantCost) {
          replantCost += seedCost(plant.cropId, settings.useStarSeeds)
        }
        day += growthTime
        continue
      }

      reharvestsDone++
      if (
        crop.growthInfo.reharvestLimit > 0 &&
        reharvestsDone > crop.growthInfo.reharvestLimit
      ) {
        // After final reharvest, replant
        if (settings.includeReplantCost) {
          replantCost += seedCost(plant.cropId, settings.useStarSeeds)
        }
        reharvestsDone = 0
        day += growthTime
      } else {
        const cd = crop.growthInfo.reharvestCooldown || growthTime
        day += cd
      }
    }
  }

  const net = totalGold - replantCost
  return {
    totalGold: Math.round(net),
    goldPerDay: settings.days > 0 ? Math.round(net / settings.days) : 0,
    cropsHarvested: Math.round(cropsHarvested),
    events,
  }
}

export const DEFAULT_SETTINGS: SimSettings = {
  days: 30,
  gardeningLevel: 0,
  useStarSeeds: false,
  sellMode: 'crop',
  includeReplantCost: false,
  useGrowthBoost: true,
}
