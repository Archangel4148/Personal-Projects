# Palia Garden Planner

A clean desktop garden planner for [Palia](https://palia.com). Place crops and fertilisers on a soil grid, see live companion buffs, and estimate gold over a harvest window.

Crop values, growth data, and buff rules are aligned with the community reference planner ([VincentAmante/palia-tools](https://github.com/VincentAmante/palia-tools), MIT). This app is a clean rewrite with its own UI — not a fork.

Fan-made and unofficial. Not affiliated with Singularity 6.

## Features (v1)

- 3×3 / 6×6 / 9×9 plots
- All reference crops (1×1, 2×2 bushes, 3×3 apple)
- All five fertilisers
- Orthogonal companion buffs with bush/tree thresholds (1 / 2 / 3)
- Buff coverage stats
- Multi-day gold estimate (crop / seed / preserve), community star-chance formula
- Copyable layout codes + JSON save/load
- Windows NSIS / MSI installers via Tauri 2

## Develop

Requirements: Node 20+, Rust (for Tauri), WebView2 on Windows.

```bash
npm install
npm run dev          # browser UI
npm run tauri:dev    # desktop shell
npm test             # engine accuracy tests
```

## Build installer

```bash
npm run tauri:build
```

Installers land under:

`src-tauri/target/release/bundle/nsis/` and `.../msi/`

A successful local build also produces:

- `Palia Garden Planner_0.1.0_x64-setup.exe` (NSIS, double-click install)
- `Palia Garden Planner_0.1.0_x64_en-US.msi` (MSI)

Copied builds may be placed in `release/` for convenience.

## Project layout

```
src/engine/   # Pure TS grid, buffs, gold, save (no React / Tauri)
src/data/     # Ported crop & fertiliser stats
src/ui/       # React workspace
src-tauri/    # Desktop shell & bundling
tests/engine/ # Vitest buff / gold / save tests
```

## Layout codes

**Copy code** exports a `v0.5_...` string compatible with the community web planner
([palia-garden-planner.vercel.app](https://palia-garden-planner.vercel.app/) /
[paliagardenplanner.com](https://paliagardenplanner.com/)).

**Load code** accepts:

- Web planner codes (`v0.1`–`v0.5`, including share URLs with `?layout=`)
- This app’s JSON files (Save JSON / Load JSON)

## Fertilisers

In Palia, each tile holds **one fertiliser type** at a time (up to 99 stacked units of that type for multi-day coverage). Applying a different type replaces the old one. This planner mirrors type-per-tile; day-stack counts are not simulated in v1.


- Buffs apply orthogonally only; same crop type never buffs itself.
- Multi-tile crops need enough neighboring *tiles* of a bonus type: single 1, bush 2, tree 3.
- Fertiliser effects count toward that tile’s received bonuses and do not stack past the threshold with identical crop buffs.
- Star chance: `0.25 + 0.25*starSeeds + 0.02*level + 0.5*qualityBoost` (community estimate, capped at 100%).
- v1 assumes instant processing (no jar/collector queue). Speedy Gro timing is a simple growth-time reduction when Speed Increase is active.

## Future mobile APK

Keep `src/engine` and `src/ui` free of `@tauri-apps` imports. When ready:

1. Build the Vite web app (`npm run build`)
2. Wrap `dist/` with [Capacitor](https://capacitorjs.com/) for Android
3. Produce a release APK / AAB from Android Studio or `cap build android`

## License

Application code: MIT.

Game data concepts come from community research and the MIT-licensed reference planner. Do not redistribute official Singularity 6 art assets.
