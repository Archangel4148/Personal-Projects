import {
  IconCopy,
  IconFolder,
  IconGear,
  IconPaste,
  IconSave,
  IconTrash,
} from '../icons/Icons'

interface Props {
  size: number
  onSize: (size: number) => void
  onClear: () => void
  onCopyCode: () => void
  onLoadCode: () => void
  onDownload: () => void
  onUpload: (file: File) => void
  onOpenSettings: () => void
  status: string
  compact?: boolean
}

export function Toolbar({
  size,
  onSize,
  onClear,
  onCopyCode,
  onLoadCode,
  onDownload,
  onUpload,
  onOpenSettings,
  status,
  compact = false,
}: Props) {
  return (
    <header className={`toolbar ${compact ? 'compact' : ''}`}>
      <div className="brand">
        <img className="brand-mark" src="/app-icon.svg" alt="" />
        {!compact && (
          <div>
            <h1>Palia Garden Planner</h1>
          </div>
        )}
      </div>
      <div className="toolbar-actions">
        <div className="size-toggle" role="group" aria-label="Plot size">
          {[3, 6, 9].map((n) => (
            <button
              key={n}
              type="button"
              className={size === n ? 'active' : ''}
              onClick={() => onSize(n)}
              title={`${n}×${n}`}
            >
              {n}×{n}
            </button>
          ))}
        </div>
        <button
          type="button"
          className="icon-btn"
          onClick={onCopyCode}
          title="Copy"
          aria-label="Copy layout"
        >
          <IconCopy />
        </button>
        <button
          type="button"
          className="icon-btn"
          onClick={onLoadCode}
          title="Paste"
          aria-label="Paste layout"
        >
          <IconPaste />
        </button>
        <button
          type="button"
          className="icon-btn"
          onClick={onDownload}
          title="Save"
          aria-label="Save"
        >
          <IconSave />
        </button>
        <label className="icon-btn file-btn" title="Open">
          <IconFolder />
          <span className="sr-only">Open</span>
          <input
            type="file"
            accept="application/json,.json"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) onUpload(file)
              e.target.value = ''
            }}
          />
        </label>
        <button
          type="button"
          className="icon-btn danger"
          onClick={onClear}
          title="Clear"
          aria-label="Clear"
        >
          <IconTrash />
        </button>
        <button
          type="button"
          className="icon-btn"
          onClick={onOpenSettings}
          title="Settings"
          aria-label="Settings"
        >
          <IconGear />
        </button>
      </div>
      {status && <p className="status">{status}</p>}
    </header>
  )
}
