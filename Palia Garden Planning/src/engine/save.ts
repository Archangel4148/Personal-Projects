import { getCrop, getFertiliser } from './catalog'
import {
  createGarden,
  placeCrop,
  placeFertiliser,
  type GardenSnapshot,
} from './garden'
import type { SimSettings } from './types'
import { DEFAULT_SETTINGS } from './gold'
import {
  exportPgpCode,
  importPgpCode,
  isPgpLayoutCode,
} from './pgpCompat'

export interface SavePayload {
  version: 1
  size: number
  plants: { cropId: string; x: number; y: number }[]
  fertilisers: { fertiliserId: string; x: number; y: number }[]
  settings?: Partial<SimSettings>
}

function encodeBase64(text: string): string {
  const bytes = new TextEncoder().encode(text)
  let binary = ''
  for (const b of bytes) binary += String.fromCharCode(b)
  return btoa(binary)
}

function decodeBase64(code: string): string {
  const binary = atob(code.trim())
  const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0))
  return new TextDecoder().decode(bytes)
}

function buildPayload(
  garden: GardenSnapshot,
  settings: SimSettings,
): SavePayload {
  const plants = Object.values(garden.plants).map((p) => ({
    cropId: p.cropId,
    x: p.origin.x,
    y: p.origin.y,
  }))
  const fertilisers: SavePayload['fertilisers'] = []
  for (const [key, tile] of Object.entries(garden.tiles)) {
    if (!tile.fertiliserId) continue
    const [x, y] = key.split(',').map(Number)
    fertilisers.push({ fertiliserId: tile.fertiliserId, x, y })
  }
  return {
    version: 1,
    size: garden.size,
    plants,
    fertilisers,
    settings,
  }
}

function hydrate(payload: SavePayload): {
  garden: GardenSnapshot
  settings: SimSettings
} {
  if (payload.version !== 1) {
    throw new Error('Unsupported save version')
  }

  let garden = createGarden(payload.size)
  for (const p of payload.plants) {
    if (!getCrop(p.cropId)) continue
    garden = placeCrop(garden, p.cropId, { x: p.x, y: p.y })
  }
  for (const f of payload.fertilisers) {
    if (!getFertiliser(f.fertiliserId)) continue
    garden = placeFertiliser(garden, f.fertiliserId, { x: f.x, y: f.y })
  }

  return {
    garden,
    settings: { ...DEFAULT_SETTINGS, ...payload.settings },
  }
}

/** Copyable code — web planner v0.5 format for cross-tool sharing. */
export function serializeGarden(
  garden: GardenSnapshot,
  settings: SimSettings = DEFAULT_SETTINGS,
): string {
  return exportPgpCode(garden, settings)
}

/**
 * Load a layout code from this app (JSON/base64) or the web planner (v0.x).
 */
export function deserializeGarden(code: string): {
  garden: GardenSnapshot
  settings: SimSettings
} {
  const trimmed = code.trim()

  if (isPgpLayoutCode(trimmed)) {
    return importPgpCode(trimmed)
  }

  // Native JSON (pretty or base64)
  try {
    if (trimmed.startsWith('{')) {
      return hydrate(JSON.parse(trimmed) as SavePayload)
    }
    const json = decodeBase64(trimmed)
    if (json.startsWith('{')) {
      return hydrate(JSON.parse(json) as SavePayload)
    }
  } catch {
    /* fall through */
  }

  // Last resort: try as PGP anyway (bare CR payloads, etc.)
  try {
    return importPgpCode(trimmed)
  } catch {
    throw new Error('Unrecognized layout code')
  }
}

export function toPrettyJson(
  garden: GardenSnapshot,
  settings: SimSettings,
): string {
  return JSON.stringify(buildPayload(garden, settings), null, 2)
}

export function fromPrettyJson(text: string): {
  garden: GardenSnapshot
  settings: SimSettings
} {
  const trimmed = text.trim()
  if (isPgpLayoutCode(trimmed)) return importPgpCode(trimmed)
  return hydrate(JSON.parse(trimmed) as SavePayload)
}

/** Keep base64 helper available for tests / backups */
export function serializeNative(
  garden: GardenSnapshot,
  settings: SimSettings = DEFAULT_SETTINGS,
): string {
  return encodeBase64(JSON.stringify(buildPayload(garden, settings)))
}

export { exportPgpCode, importPgpCode, isPgpLayoutCode }
