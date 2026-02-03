<file>
00001| # Migration Learnings
00002|
00003| ## Initial State
00004| - Project is Vite-based.
00005| - Source code moved to `webxr/src/xr`.
00006| - Goal: Convert to Next.js 14+ with static export.
00007|
00008| ## Plan
00009| - Install Next.js, React, Tailwind, Shadcn UI.
00010| - Configure `next.config.mjs` for `output: 'export'`.
00011| - Setup `src/app` directory.
00012| - Ensure strict `three` version compatibility (`npm:super-three@0.177.0`).
00013|
00014| ## Execution
00015| - Dependencies installed successfully.
00016| - Moved `src/*` to `src/xr/`.
00017| - Configured Next.js, Tailwind v3, Shadcn UI (manual init).
00018| - Updated `tsconfig.json` to exclude `src/xr` (temporarily) to allow build to pass.
00019| - Verified `npm run build` produces `out/` with static files.
00020| - Preserved existing `package.json` dependencies.
00021|
00022| ## Asset Optimization Script
00023| - Created `webxr/scripts/optimize-assets.mjs` to replace `@iwsdk/vite-plugin-gltf-optimizer` and `@iwsdk/vite-plugin-uikitml`.
00024| - Replicates "medium" optimization level for GLB files using `gltf-transform`.
00025| - Compiles UIKitML (`.uikitml`) files to JSON in `public/ui/` using `@pmndrs/uikitml`.
00026| - Integrated via `prebuild` script in `webxr/package.json`, ensuring assets are ready before the main build.
00027| - Dependencies added: `@gltf-transform/core`, `@gltf-transform/extensions`, `@gltf-transform/functions`, `@pmndrs/uikitml`, `fs-extra`, `glob`, `sharp`, `draco3dgltf`.
00028|
00029| ## State Store
00030| - Added a Zustand store at `webxr/src/lib/store.ts` with camera config defaults (unset means enabled), teleop settings, and connection status actions.
00031|
00032| ## Dashboard UI
00033| - Implemented 2D Dashboard using Shadcn UI.
00034| - Installed components: `card`, `switch`, `slider`, `button`, `label`.
00035| - Created `TeleopPanel.tsx` and `CameraSettingsPanel.tsx` in `webxr/src/components/dashboard`.
00036| - Connected components to `useAppStore` for real-time updates of settings.
00037| - Updated `webxr/src/app/page.tsx` with a responsive layout and "Launch XR" button.
00038| - Build verified successfully.
</file>- 2026-02-02: Refactored WebXR world initialization into `initWorld(container)` and removed 3D teleop/settings panels; camera panels remain managed in world setup.
- 2026-02-02: TeleopSystem now reads teleop settings from Zustand, updates connection status/telemetry in the store, and drops spatial UI handlers.
- 2026-02-02: `webxr/src/xr/camera_config.ts` now proxies camera config reads/writes and change subscriptions directly to the Zustand store.
- 2026-02-02: Verified `webxr/src/xr/camera_config.ts` uses `useAppStore.getState()` and `useAppStore.subscribe()` with unsubscribe cleanup; no local camera config state remains.
