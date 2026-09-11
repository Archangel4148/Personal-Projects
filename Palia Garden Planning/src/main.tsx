import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { applyTheme, loadThemeId } from './ui/themes/themes'
import './ui/themes/themes.css'
import './index.css'
import App from './App.tsx'

applyTheme(loadThemeId())

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
