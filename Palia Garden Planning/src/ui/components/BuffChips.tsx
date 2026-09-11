import type { Bonus } from '../../engine/types'

export const BUFF_COLORS: Record<string, string> = {
  'Water Retain': '#3b9eff',
  'Weed Prevention': '#3ecf6e',
  'Harvest Increase': '#f0b429',
  'Quality Increase': '#ff6bb5',
  'Speed Increase': '#6ec8ff',
}

/** Colored SVG icons matching the reference planner's Font Awesome semantics. */
export function BuffIcon({
  bonus,
  size = 14,
  className = '',
}: {
  bonus: Bonus
  size?: number
  className?: string
}) {
  if (bonus === 'None') return null
  const color = BUFF_COLORS[bonus] ?? '#888'
  const common = {
    width: size,
    height: size,
    viewBox: '0 0 24 24',
    fill: color,
    className: `buff-icon ${className}`.trim(),
    'aria-hidden': true as const,
  }

  switch (bonus) {
    case 'Water Retain':
      return (
        <svg {...common} title="Water Retain">
          <path d="M12 2.2C12 2.2 5.5 10.1 5.5 14.6a6.5 6.5 0 0 0 13 0C18.5 10.1 12 2.2 12 2.2zm0 16.4a3.5 3.5 0 0 1-3.5-3.5c0-2.2 2.4-5.7 3.5-7.3 1.1 1.6 3.5 5.1 3.5 7.3a3.5 3.5 0 0 1-3.5 3.5z" />
        </svg>
      )
    case 'Weed Prevention':
      return (
        <svg {...common}>
          <path d="M12 2 4 5.2v6.3c0 5 3.4 9.6 8 10.8 4.6-1.2 8-5.8 8-10.8V5.2L12 2zm0 17.6c-3.3-1.1-5.8-4.6-5.8-8.3V6.7L12 4.4l5.8 2.3v4.8c0 3.7-2.5 7.2-5.8 8.3z" />
        </svg>
      )
    case 'Harvest Increase':
      return (
        <svg {...common}>
          <path d="M18.6 3.2c-3.4.2-6.3 1.8-8 4.2-1.2 1.7-1.8 3.7-1.9 5.7l-4.5 4.5 1.4 1.4 4.5-4.5c2 .1 4-.4 5.7-1.7 2.5-1.8 4.1-4.8 4.2-8.3l-1.4-.3zm-6.7 9.1c-.1-1.4.3-2.8 1.1-4 1.1-1.6 2.9-2.7 5-3.1-.3 2.2-1.4 4.1-3.1 5.2-1.1.8-2.4 1.2-3 1.9zM7.2 17.8l-2.8 2.8 1.4 1.4 2.8-2.8-1.4-1.4z" />
        </svg>
      )
    case 'Quality Increase':
      return (
        <svg {...common}>
          <path d="M12 2.4 14.8 9l7.2.6-5.5 4.6 1.7 7-6.2-3.8-6.2 3.8 1.7-7L2 9.6 9.2 9 12 2.4z" />
        </svg>
      )
    case 'Speed Increase':
      return (
        <svg {...common}>
          <path d="M11 4 5 12h4.5L8 20l9-10h-4.5L15 4H11z" />
          <path d="M16 5.5 13.2 9h2.3L14.2 14 19.5 9.2h-2.2L18.4 5.5H16z" opacity="0.75" />
        </svg>
      )
    default:
      return null
  }
}

export function BuffChips({
  bonuses,
  size = 12,
}: {
  bonuses: Bonus[]
  size?: number
}) {
  const shown = bonuses.filter((b) => b !== 'None')
  if (shown.length === 0) return null
  return (
    <div className="buff-chips" role="list">
      {shown.map((b) => (
        <span key={b} className="buff-chip" title={b} role="listitem">
          <BuffIcon bonus={b} size={size} />
        </span>
      ))}
    </div>
  )
}
