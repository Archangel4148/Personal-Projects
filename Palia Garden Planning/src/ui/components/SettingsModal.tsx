import { BuffIcon, BUFF_COLORS } from './BuffChips'
import type { Bonus } from '../../engine/types'
import {
  THEMES,
  type ThemeId,
  type ThemeDef,
} from '../themes/themes'
import { IconClose } from '../icons/Icons'

interface Props {
  open: boolean
  themeId: ThemeId
  onClose: () => void
  onTheme: (id: ThemeId) => void
}

function ThemeCard({
  theme,
  active,
  onSelect,
}: {
  theme: ThemeDef
  active: boolean
  onSelect: () => void
}) {
  return (
    <button
      type="button"
      className={`theme-card ${active ? 'active' : ''}`}
      onClick={onSelect}
    >
      <span className="theme-swatches" aria-hidden>
        {theme.preview.map((c) => (
          <span key={c} style={{ background: c }} />
        ))}
      </span>
      <span className="theme-card-meta">
        <strong>{theme.name}</strong>
        <em>
          {theme.mode} · {theme.tone}
        </em>
      </span>
    </button>
  )
}

export function SettingsModal({ open, themeId, onClose, onTheme }: Props) {
  if (!open) return null

  const light = THEMES.filter((t) => t.mode === 'light')
  const dark = THEMES.filter((t) => t.mode === 'dark')

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <div
        className="modal-sheet"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-title"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="modal-header">
          <h2 id="settings-title">Settings</h2>
          <button
            type="button"
            className="icon-btn"
            onClick={onClose}
            aria-label="Close settings"
            title="Close"
          >
            <IconClose />
          </button>
        </header>

        <section>
          <h3>Appearance</h3>
          <h4>Light</h4>
          <div className="theme-grid">
            {light.map((t) => (
              <ThemeCard
                key={t.id}
                theme={t}
                active={themeId === t.id}
                onSelect={() => onTheme(t.id)}
              />
            ))}
          </div>
          <h4>Dark</h4>
          <div className="theme-grid">
            {dark.map((t) => (
              <ThemeCard
                key={t.id}
                theme={t}
                active={themeId === t.id}
                onSelect={() => onTheme(t.id)}
              />
            ))}
          </div>
        </section>

        <section className="legend-section">
          <h3>Buff icons</h3>
          <ul className="buff-legend">
            {(
              [
                'Water Retain',
                'Weed Prevention',
                'Harvest Increase',
                'Quality Increase',
                'Speed Increase',
              ] as Bonus[]
            ).map((b) => (
              <li key={b}>
                <BuffIcon bonus={b} size={16} />
                <span style={{ color: BUFF_COLORS[b] }}>{b}</span>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  )
}
