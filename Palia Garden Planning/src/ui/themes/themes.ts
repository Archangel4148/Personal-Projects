export const THEME_STORAGE_KEY = 'pgp-theme'

export type ThemeId =
  | 'grove-light'
  | 'grove-dark'
  | 'meadow-light'
  | 'meadow-dark'
  | 'clay-light'
  | 'clay-dark'
  | 'mist-light'
  | 'mist-dark'
  | 'orchard-light'
  | 'orchard-dark'

export interface ThemeDef {
  id: ThemeId
  name: string
  mode: 'light' | 'dark'
  tone: 'muted' | 'balanced' | 'colorful'
  preview: [string, string, string]
}

export const THEMES: ThemeDef[] = [
  {
    id: 'grove-light',
    name: 'Grove',
    mode: 'light',
    tone: 'balanced',
    preview: ['#eef4e8', '#2f6b3a', '#c4a35a'],
  },
  {
    id: 'grove-dark',
    name: 'Grove Night',
    mode: 'dark',
    tone: 'balanced',
    preview: ['#152018', '#7cb87a', '#c4a35a'],
  },
  {
    id: 'meadow-light',
    name: 'Meadow',
    mode: 'light',
    tone: 'colorful',
    preview: ['#f3faf0', '#1f8a4c', '#f0b429'],
  },
  {
    id: 'meadow-dark',
    name: 'Meadow Night',
    mode: 'dark',
    tone: 'colorful',
    preview: ['#0f1a14', '#3ecf6e', '#f0b429'],
  },
  {
    id: 'clay-light',
    name: 'Clay',
    mode: 'light',
    tone: 'muted',
    preview: ['#f2ebe3', '#6b5344', '#a67c52'],
  },
  {
    id: 'clay-dark',
    name: 'Clay Night',
    mode: 'dark',
    tone: 'muted',
    preview: ['#1c1714', '#c4a484', '#8b6b4a'],
  },
  {
    id: 'mist-light',
    name: 'Mist',
    mode: 'light',
    tone: 'muted',
    preview: ['#eef2f4', '#4a6670', '#7a9eaa'],
  },
  {
    id: 'mist-dark',
    name: 'Mist Night',
    mode: 'dark',
    tone: 'muted',
    preview: ['#12181c', '#9bb4be', '#5a7a86'],
  },
  {
    id: 'orchard-light',
    name: 'Orchard',
    mode: 'light',
    tone: 'colorful',
    preview: ['#fff6ef', '#c0392b', '#2f6b3a'],
  },
  {
    id: 'orchard-dark',
    name: 'Orchard Night',
    mode: 'dark',
    tone: 'colorful',
    preview: ['#1a1210', '#e85d4c', '#7cb87a'],
  },
]

export const DEFAULT_THEME: ThemeId = 'grove-light'

export function loadThemeId(): ThemeId {
  try {
    const raw = localStorage.getItem(THEME_STORAGE_KEY)
    if (raw && THEMES.some((t) => t.id === raw)) return raw as ThemeId
  } catch {
    /* ignore */
  }
  return DEFAULT_THEME
}

export function saveThemeId(id: ThemeId): void {
  try {
    localStorage.setItem(THEME_STORAGE_KEY, id)
  } catch {
    /* ignore */
  }
}

export function applyTheme(id: ThemeId): void {
  document.documentElement.dataset.theme = id
}
