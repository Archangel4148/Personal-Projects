import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  clearGarden,
  computeBuffStats,
  createGarden,
  eraseAt,
  placeCrop,
  placeFertiliser,
  resizeGarden,
  type GardenSnapshot,
} from './engine/garden'
import { DEFAULT_SETTINGS, estimateGold } from './engine/gold'
import {
  deserializeGarden,
  fromPrettyJson,
  serializeGarden,
  toPrettyJson,
} from './engine/save'
import type { Coord, SimSettings } from './engine/types'
import { GardenGrid } from './ui/components/GardenGrid'
import { Palette, type Tool } from './ui/components/Palette'
import { SettingsModal } from './ui/components/SettingsModal'
import { StatsPanel } from './ui/components/StatsPanel'
import { Toolbar } from './ui/components/Toolbar'
import {
  applyTheme,
  loadThemeId,
  saveThemeId,
  type ThemeId,
} from './ui/themes/themes'
import './ui/themes/themes.css'
import './App.css'

type MobileTab = 'plant' | 'stats'

function useNarrow(query = '(max-width: 700px)') {
  const [narrow, setNarrow] = useState(() =>
    typeof window !== 'undefined' ? window.matchMedia(query).matches : false,
  )
  useEffect(() => {
    const mq = window.matchMedia(query)
    const onChange = () => setNarrow(mq.matches)
    onChange()
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [query])
  return narrow
}

export default function App() {
  const narrow = useNarrow()
  const [garden, setGarden] = useState<GardenSnapshot>(() => createGarden(9))
  const [tool, setTool] = useState<Tool>({ kind: 'crop', cropId: 'tomato' })
  const [settings, setSettings] = useState<SimSettings>(DEFAULT_SETTINGS)
  const [status, setStatus] = useState('')
  const [themeId, setThemeId] = useState<ThemeId>(() => loadThemeId())
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [mobileTab, setMobileTab] = useState<MobileTab>('plant')

  const stats = useMemo(() => computeBuffStats(garden), [garden])
  const gold = useMemo(
    () => estimateGold(garden, settings),
    [garden, settings],
  )

  useEffect(() => {
    applyTheme(themeId)
  }, [themeId])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (settingsOpen) setSettingsOpen(false)
        else setTool({ kind: 'erase' })
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [settingsOpen])

  const flash = useCallback((msg: string) => {
    setStatus(msg)
    window.setTimeout(() => setStatus(''), 2500)
  }, [])

  const onTheme = (id: ThemeId) => {
    setThemeId(id)
    saveThemeId(id)
    applyTheme(id)
  }

  const onTilePaint = useCallback(
    (coord: Coord) => {
      setGarden((g) => {
        if (tool.kind === 'erase') return eraseAt(g, coord)
        if (tool.kind === 'crop') return placeCrop(g, tool.cropId, coord)
        return placeFertiliser(g, tool.fertiliserId, coord)
      })
    },
    [tool],
  )

  const onTileErase = useCallback((coord: Coord) => {
    setGarden((g) => eraseAt(g, coord))
  }, [])

  const onCopyCode = async () => {
    const code = serializeGarden(garden, settings)
    await navigator.clipboard.writeText(code)
    flash('Copied')
  }

  const onLoadCode = () => {
    const code = window.prompt('Paste layout code')
    if (!code) return
    try {
      const loaded = deserializeGarden(code)
      setGarden(loaded.garden)
      setSettings(loaded.settings)
      flash('Loaded')
    } catch {
      flash('Invalid code')
    }
  }

  const onDownload = () => {
    const blob = new Blob([toPrettyJson(garden, settings)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'palia-garden.json'
    a.click()
    URL.revokeObjectURL(url)
    flash('Saved')
  }

  const onUpload = async (file: File) => {
    try {
      const text = await file.text()
      const loaded = fromPrettyJson(text)
      setGarden(loaded.garden)
      setSettings(loaded.settings)
      flash('Loaded')
    } catch {
      flash('Invalid file')
    }
  }

  const grid = (
    <GardenGrid
      garden={garden}
      tool={tool}
      onTilePaint={onTilePaint}
      onTileErase={onTileErase}
    />
  )

  return (
    <div className={`app-shell ${narrow ? 'is-narrow' : ''}`}>
      <Toolbar
        size={garden.size}
        onSize={(size) => setGarden((g) => resizeGarden(g, size))}
        onClear={() => setGarden((g) => clearGarden(g))}
        onCopyCode={onCopyCode}
        onLoadCode={onLoadCode}
        onDownload={onDownload}
        onUpload={onUpload}
        onOpenSettings={() => setSettingsOpen(true)}
        status={status}
        compact={narrow}
      />

      {narrow ? (
        <main className="workspace mobile-workspace">
          <section className="canvas">{grid}</section>
          <nav className="mobile-tabs" aria-label="Panels">
            <button
              type="button"
              className={mobileTab === 'plant' ? 'active' : ''}
              onClick={() => setMobileTab('plant')}
            >
              Plant
            </button>
            <button
              type="button"
              className={mobileTab === 'stats' ? 'active' : ''}
              onClick={() => setMobileTab('stats')}
            >
              Stats
            </button>
          </nav>
          <div className="mobile-drawer">
            {mobileTab === 'plant' ? (
              <Palette tool={tool} onSelect={setTool} compact />
            ) : (
              <StatsPanel
                stats={stats}
                settings={settings}
                onSettings={setSettings}
                gold={gold}
              />
            )}
          </div>
        </main>
      ) : (
        <main className="workspace">
          <Palette tool={tool} onSelect={setTool} />
          <section className="canvas">{grid}</section>
          <StatsPanel
            stats={stats}
            settings={settings}
            onSettings={setSettings}
            gold={gold}
          />
        </main>
      )}

      <SettingsModal
        open={settingsOpen}
        themeId={themeId}
        onClose={() => setSettingsOpen(false)}
        onTheme={onTheme}
      />
    </div>
  )
}
