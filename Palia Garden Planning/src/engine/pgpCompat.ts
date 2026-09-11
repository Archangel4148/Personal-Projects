/**
 * Import/export compatible with VincentAmante/palia-tools layout codes (v0.1–v0.5).
 * Latest export format: v0.5
 *
 * v0.5 shape: v0.5_D-{W}x{H}_CR-{plotX}x{plotY}{tiles}-...[_settings]
 * Each plot is a 3×3 block; tiles are crop[+.fert] run-length compressed.
 */

import {
  createGarden,
  placeCrop,
  placeFertiliser,
  type GardenSnapshot,
} from './garden'
import { DEFAULT_SETTINGS } from './gold'
import { coordKey, type CropId, type FertiliserId, type SimSettings } from './types'

/** Current web planner crop codes (v0.4 / v0.5) */
const CROP_TO_CODE: Record<string, string> = {
  tomato: 'T',
  potato: 'P',
  rice: 'R',
  wheat: 'W',
  carrot: 'C',
  onion: 'O',
  cotton: 'Co',
  blueberry: 'B',
  apple: 'A',
  corn: 'Cr',
  'spicy-pepper': 'S',
  'napa-cabbage': 'Cb',
  'bok-choy': 'Bk',
  'rockhopper-pumpkin': 'Pm',
  'batterfly-bean': 'Bt',
}

const CODE_TO_CROP: Record<string, CropId> = Object.fromEntries(
  Object.entries(CROP_TO_CODE).map(([id, code]) => [code, id]),
)

const FERT_TO_CODE: Record<string, string> = {
  'quality-up': 'Q',
  'harvest-boost': 'H',
  'weed-block': 'W',
  'speedy-gro': 'S',
  'hydrate-pro': 'Y',
}

const CODE_TO_FERT: Record<string, FertiliserId> = {
  Q: 'quality-up',
  H: 'harvest-boost',
  W: 'weed-block',
  S: 'speedy-gro',
  Y: 'hydrate-pro',
  Hp: 'hydrate-pro',
}

/** Older crop aliases → current codes */
const LEGACY_CROP: Record<string, string> = {
  Na: 'N',
  To: 'T',
  Po: 'P',
  Ri: 'R',
  Wh: 'W',
  Ca: 'C',
  On: 'O',
  Bl: 'B',
  Ap: 'A',
  Sp: 'S',
  Bb: 'Bt',
}

const TILE_TOKEN = /([A-Z][a-z]*)(?:\.([A-Z][a-z]*))?(\d*)/g

export function isPgpLayoutCode(raw: string): boolean {
  const s = raw.trim()
  if (/^v0\.\d/i.test(s)) return true
  if (s.includes('layout=')) return true
  if (/_CR(?:OPS)?-/i.test(s)) return true
  if (/^D(?:IM)?-[01]/.test(s) || /^D(?:IM)?-\d+x\d+/.test(s)) return true
  return false
}

export function expandPlotCode(code: string): string[] {
  const tokens = code.match(/[A-Z][a-z]*(?:\.[A-Z][a-z]*)?\d*/g) || []
  return tokens.flatMap((token) => {
    const match = token.match(/^([A-Z][a-z]*(?:\.[A-Z][a-z]*)?)(\d*)$/)
    if (!match) return []
    const base = match[1]
    const count = match[2] ? Math.min(parseInt(match[2], 10), 1000) : 1
    return Array.from({ length: count }, () => base)
  })
}

export function compressPlotString(tiles: string[]): string {
  if (tiles.length === 0) return ''
  const compressed: string[] = []
  let current = tiles[0]
  let count = 0
  for (const tile of tiles) {
    if (tile === current) count++
    else {
      compressed.push(`${current}${count > 1 ? count : ''}`)
      current = tile
      count = 1
    }
  }
  compressed.push(`${current}${count > 1 ? count : ''}`)
  return compressed.join('')
}

function normalizeCropCode(code: string): string {
  if (code === 'N' || CODE_TO_CROP[code]) return code
  if (LEGACY_CROP[code]) return LEGACY_CROP[code]
  return code
}

function normalizeFertCode(code: string | undefined): string | null {
  if (!code || code === 'N') return null
  if (code === 'Hp') return 'Y'
  if (CODE_TO_FERT[code]) return code
  return code
}

function parseTileToken(token: string): { cropCode: string; fertCode: string | null } {
  const m = /^([A-Z][a-z]*)(?:\.([A-Z][a-z]*))?$/.exec(token)
  if (!m) return { cropCode: 'N', fertCode: null }
  return {
    cropCode: normalizeCropCode(m[1]),
    fertCode: normalizeFertCode(m[2]),
  }
}

function tileToken(cropCode: string, fertCode: string | null): string {
  return fertCode ? `${cropCode}.${fertCode}` : cropCode
}

function upgradeCropInfoCodes(cropInfo: string): string {
  return cropInfo.replace(/CROPS-/g, 'CR-').replace(TILE_TOKEN, (_full, crop, fert, count) => {
    const c = normalizeCropCode(crop)
    const f = normalizeFertCode(fert)
    const body = f ? `${c}.${f}` : c
    return `${body}${count || ''}`
  })
}

/** Convert pre-0.5 plot matrix (D-111-111-111) into tile dimensions + plots */
function matrixToPlots(
  dimensionInfo: string,
  cropInfo: string,
): { width: number; height: number; plots: { x: number; y: number; tiles: string[] }[] } {
  const dim = dimensionInfo.replace(/^D(?:IM)?-/, '')
  const rows = dim.split('-').filter(Boolean)
  const cropSections = cropInfo.replace(/^CR(?:OPS)?-/, '').split('-')

  const plots: { x: number; y: number; tiles: string[] }[] = []
  let plotIndex = 0
  let maxW = 0
  let maxH = 0

  for (let r = 0; r < rows.length; r++) {
    const row = rows[r]
    for (let c = 0; c < row.length; c++) {
      if (row[c] !== '1') continue
      const x = c * 3
      const y = r * 3
      maxW = Math.max(maxW, x + 3)
      maxH = Math.max(maxH, y + 3)
      const section = cropSections[plotIndex++] ?? 'N9'
      const tiles = expandPlotCode(section)
      if (tiles.length !== 9) {
        throw new Error(`Plot at ${x}x${y} does not have 9 tiles`)
      }
      plots.push({ x, y, tiles })
    }
  }

  return { width: maxW || 3, height: maxH || 3, plots }
}

function parseV05Dimensions(dimensionInfo: string): { width: number; height: number } {
  const m = dimensionInfo.replace(/^D-/, '').match(/^(\d+)x(\d+)/)
  if (!m) throw new Error('Invalid v0.5 dimension info')
  return { width: parseInt(m[1], 10), height: parseInt(m[2], 10) }
}

function parseV05Plots(cropInfo: string): { x: number; y: number; tiles: string[] }[] {
  const parts = cropInfo.replace(/^CR-/, '').split('-').filter(Boolean)
  return parts.map((part) => {
    const m = part.match(/^(\d+)x(\d+)(.*)$/)
    if (!m) throw new Error(`Invalid plot code: ${part}`)
    const tiles = expandPlotCode(m[3])
    if (tiles.length !== 9) {
      throw new Error(`Plot ${m[1]}x${m[2]} does not have 9 tiles (${tiles.length})`)
    }
    return { x: parseInt(m[1], 10), y: parseInt(m[2], 10), tiles }
  })
}

function parseSettingsInfo(settingsInfo: string): Partial<SimSettings> {
  const next: Partial<SimSettings> = {}
  if (!settingsInfo) return next

  const general = settingsInfo.split('Cr0.')[0] || settingsInfo
  const tokens = general.match(/[A-Z][a-z0-9]*/g) || []
  for (const setting of tokens) {
    if (setting === 'Nss') next.useStarSeeds = false
    else if (setting === 'Gb') next.useGrowthBoost = true
    else if (setting === 'Nrc') next.includeReplantCost = false
    else {
      const m = setting.match(/^([A-Z])(\d+)$/)
      if (!m) continue
      const n = parseInt(m[2], 10)
      if (m[1] === 'D') next.days = n
      if (m[1] === 'L') next.gardeningLevel = n
    }
  }

  const cropIdx = settingsInfo.indexOf('Cr0.')
  if (cropIdx !== -1) {
    const chunk = settingsInfo.slice(cropIdx + 4).split('Fr0.')[0]
    const first = chunk.split('-').find((s) => s.length > 0)
    if (first) {
      const proc = first.match(/~?([PS])\d*$/)
      if (proc?.[1] === 'P') next.sellMode = 'preserve'
      if (proc?.[1] === 'S') next.sellMode = 'seed'
    }
  }

  return next
}

function encodeSettings(settings: SimSettings): string {
  let out = ''
  if (settings.days > 0) out += `D${settings.days}`
  if (settings.gardeningLevel > 0) out += `L${settings.gardeningLevel}`
  if (!settings.useStarSeeds) out += 'Nss'
  if (settings.useGrowthBoost) out += 'Gb'
  if (!settings.includeReplantCost) out += 'Nrc'
  out += 'Nfc'
  return out
}

function preferredGardenSize(width: number, height: number): number {
  const max = Math.max(width, height)
  if (max <= 3) return 3
  if (max <= 6) return 6
  if (max <= 9) return 9
  return Math.ceil(max / 3) * 3
}

function loadPlotsIntoGarden(
  width: number,
  height: number,
  plots: { x: number; y: number; tiles: string[] }[],
): GardenSnapshot {
  let garden = createGarden(preferredGardenSize(width, height))

  for (const plot of plots) {
    let i = 0
    for (let dy = 0; dy < 3; dy++) {
      for (let dx = 0; dx < 3; dx++) {
        const token = plot.tiles[i++]
        const { cropCode, fertCode } = parseTileToken(token)
        const x = plot.x + dx
        const y = plot.y + dy
        if (x >= garden.size || y >= garden.size) continue

        if (cropCode !== 'N') {
          const cropId = CODE_TO_CROP[cropCode]
          if (cropId) garden = placeCrop(garden, cropId, { x, y })
        }

        if (fertCode) {
          const fertId = CODE_TO_FERT[fertCode]
          if (fertId) garden = placeFertiliser(garden, fertId, { x, y })
        }
      }
    }
  }

  return garden
}

function isMatrixDimension(dimensionInfo: string): boolean {
  const body = dimensionInfo.replace(/^D(?:IM)?-/, '')
  return /^[01]+(-[01]+)*$/.test(body)
}

/**
 * Parse a web planner save/share code into a garden + settings.
 */
export function importPgpCode(raw: string): {
  garden: GardenSnapshot
  settings: SimSettings
} {
  let save = raw.trim()
  try {
    if (save.includes('layout=')) {
      const u = new URL(save)
      save = u.searchParams.get('layout') || save
    }
  } catch {
    /* not a URL */
  }
  save = decodeURIComponent(save.trim())

  if (!save.startsWith('v') && (save.startsWith('D-') || save.startsWith('DIM-'))) {
    // Bare dimension+crop payloads — assume modern-enough codes
    save = save.includes('x') && /D-\d+x\d+/.test(save) ? `v0.5_${save}` : `v0.4_${save}`
  }

  const [versionRaw, ...rest] = save.split('_')
  const version = (versionRaw || '').replace(/^v/, '')
  let dimensionInfo = (rest[0] || '').replace(/^DIM-/, 'D-')
  let cropInfo = upgradeCropInfoCodes(rest[1] || '')
  const settingsInfo = rest[2] || ''

  if (!version && !dimensionInfo) throw new Error('Missing layout code')

  let width = 0
  let height = 0
  let plots: { x: number; y: number; tiles: string[] }[] = []

  if (/^D-\d+x\d+/.test(dimensionInfo) || version === '0.5') {
    ;({ width, height } = parseV05Dimensions(dimensionInfo))
    plots = parseV05Plots(cropInfo)
  } else if (isMatrixDimension(dimensionInfo)) {
    const converted = matrixToPlots(dimensionInfo, cropInfo)
    width = converted.width
    height = converted.height
    plots = converted.plots
  } else {
    throw new Error(`Unsupported layout dimension format: ${dimensionInfo}`)
  }

  const garden = loadPlotsIntoGarden(width, height, plots)
  return {
    garden,
    settings: { ...DEFAULT_SETTINGS, ...parseSettingsInfo(settingsInfo) },
  }
}

function originCropCode(garden: GardenSnapshot, x: number, y: number): string {
  const tile = garden.tiles[coordKey({ x, y })]
  if (!tile?.plantId) return 'N'
  const plant = garden.plants[tile.plantId]
  if (!plant) return 'N'
  if (plant.origin.x !== x || plant.origin.y !== y) return 'N'
  return CROP_TO_CODE[plant.cropId] ?? 'N'
}

function fertCodeAt(garden: GardenSnapshot, x: number, y: number): string | null {
  const tile = garden.tiles[coordKey({ x, y })]
  if (!tile?.fertiliserId) return null
  return FERT_TO_CODE[tile.fertiliserId] ?? null
}

/**
 * Encode garden as a v0.5 web-planner layout code.
 */
export function exportPgpCode(
  garden: GardenSnapshot,
  settings: SimSettings = DEFAULT_SETTINGS,
): string {
  const width = garden.size
  const height = garden.size
  const plots: string[] = []

  for (let py = 0; py < height; py += 3) {
    for (let px = 0; px < width; px += 3) {
      const tiles: string[] = []
      for (let dy = 0; dy < 3; dy++) {
        for (let dx = 0; dx < 3; dx++) {
          const x = px + dx
          const y = py + dy
          if (x >= width || y >= height) {
            tiles.push('N')
            continue
          }
          tiles.push(tileToken(originCropCode(garden, x, y), fertCodeAt(garden, x, y)))
        }
      }
      plots.push(`${px}x${py}${compressPlotString(tiles)}`)
    }
  }

  const dimensionInfo = `D-${width}x${height}`
  const cropInfo = `CR-${plots.join('-')}`
  const settingsInfo = encodeSettings(settings)
  return `v0.5_${dimensionInfo}_${cropInfo}_${settingsInfo}`
}
