import { TRACKED_BONUSES, type SimSettings, type Bonus } from '../../engine/types'
import { BuffIcon } from './BuffChips'
import type { BuffStats } from '../../engine/garden'

interface Props {
  stats: BuffStats
  settings: SimSettings
  onSettings: (next: SimSettings) => void
  gold: { totalGold: number; goldPerDay: number; cropsHarvested: number }
}

export function StatsPanel({ stats, settings, onSettings, gold }: Props) {
  return (
    <aside className="panel stats">
      <div className="panel-scroll">
        <h2>Buff coverage</h2>
        <ul className="stat-list">
          {TRACKED_BONUSES.filter((b) => b !== 'Speed Increase').map(
            (b: Bonus) => (
              <li key={b}>
                <BuffIcon bonus={b} size={14} />
                <span className="stat-name">{b}</span>
                <span className="stat-val">
                  {stats.withBonus[b]}/{stats.plantCount || 0}
                  <em>{stats.coveragePct[b]}%</em>
                </span>
              </li>
            ),
          )}
        </ul>

        <h2>Gold</h2>
        <div className="gold-block">
          <div>
            <strong>{gold.totalGold.toLocaleString()}g</strong>
            <span>over {settings.days} days</span>
          </div>
          <div>
            <strong>{gold.goldPerDay.toLocaleString()}g</strong>
            <span>per day</span>
          </div>
          <div>
            <strong>{gold.cropsHarvested.toLocaleString()}</strong>
            <span>crops harvested</span>
          </div>
        </div>

        <h2>Simulation</h2>
        <label className="field">
          Days
          <input
            type="number"
            min={1}
            max={180}
            value={settings.days}
            onChange={(e) =>
              onSettings({ ...settings, days: Number(e.target.value) || 1 })
            }
          />
        </label>
        <label className="field">
          Gardening level
          <input
            type="number"
            min={0}
            max={50}
            value={settings.gardeningLevel}
            onChange={(e) =>
              onSettings({
                ...settings,
                gardeningLevel: Number(e.target.value) || 0,
              })
            }
          />
        </label>
        <label className="field">
          Sell as
          <select
            value={settings.sellMode}
            onChange={(e) =>
              onSettings({
                ...settings,
                sellMode: e.target.value as SimSettings['sellMode'],
              })
            }
          >
            <option value="crop">Crops</option>
            <option value="seed">Seeds</option>
            <option value="preserve">Preserves</option>
          </select>
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={settings.useStarSeeds}
            onChange={(e) =>
              onSettings({ ...settings, useStarSeeds: e.target.checked })
            }
          />
          Star seeds
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={settings.includeReplantCost}
            onChange={(e) =>
              onSettings({ ...settings, includeReplantCost: e.target.checked })
            }
          />
          Include replant cost
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={settings.useGrowthBoost}
            onChange={(e) =>
              onSettings({ ...settings, useGrowthBoost: e.target.checked })
            }
          />
          Use Speedy Gro timing
        </label>
      </div>
    </aside>
  )
}
