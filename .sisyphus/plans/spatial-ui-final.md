# Final Cleanup: Spatial UI Features

## TL;DR

> **Quick Summary**: Remove debug artifacts from the codebase and commit the final Spatial UI features. Document the known headless rendering limitations regarding glass materials.
>
> **Deliverables**:
> - Clean `webxr/index.html` (no debug box)
> - Clean `webxr/src/components/spatial-panel.ts` (no debug logs)
> - New `docs/SPATIAL_UI_NOTES.md` (limitations & verification steps)
> - Git Commit of the finalized state
>
> **Estimated Effort**: Quick
> **Parallel Execution**: Sequential (Cleanup → Doc → Commit)

---

## Context

### Original Request
User wants to ensure the "A-Frame Spatial UI" glass look is implemented, despite screenshot artifacts. Needs cleanup of debug code (`<a-box>`, logs) and a final commit.

### Technical Constraints
- **Headless Browser Limitation**: The Playwright environment (Chromium/SwiftShader/EGL) on Linux cannot render WebGL 2 transmission/refraction correctly. It falls back to opaque rendering.
- **Trust the Code**: The `MeshPhysicalMaterial` implementation is correct for Quest headsets.

---

## Work Objectives

### Core Objective
Clean up the codebase and document the state of the Spatial UI for handover.

### Concrete Deliverables
- `webxr/index.html` (modified)
- `webxr/src/components/spatial-panel.ts` (modified)
- `docs/SPATIAL_UI_NOTES.md` (created)

### Definition of Done
- [ ] No red/blue debug box in `index.html`
- [ ] No `[spatial-panel]` logs in browser console
- [ ] `docs/SPATIAL_UI_NOTES.md` exists with headset verification instructions
- [ ] Changes committed to git

---

## Execution Strategy

### Dependency Matrix
| Task | Depends On | Blocks |
|------|------------|--------|
| 1. Cleanup | None | 3 |
| 2. Documentation | None | 3 |
| 3. Commit | 1, 2 | None |

---

## TODOs

- [ ] 1. Remove debug artifacts
  **What to do**:
  - Edit `webxr/index.html`: Remove the `<a-box spatial-panel ...>` element (Line 35).
  - Edit `webxr/src/components/spatial-panel.ts`: Remove console logs at lines 25, 41, and 95 (approximate).

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `["simplify"]`

  **Verification**:
  - `grep "a-box spatial-panel" webxr/index.html` → Returns nothing (or only the intended panel, not the debug one)
  - `grep "console.log" webxr/src/components/spatial-panel.ts` → Returns clean output (no debug logs)

- [ ] 2. Create Spatial UI Documentation
  **What to do**:
  - Create `docs/SPATIAL_UI_NOTES.md`.
  - Explain: "Why screenshots look blue/opaque" (Headless WebGL limitation).
  - Explain: "How to verify on Headset" (Sideload/Serve and look for glass effect).

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: `["crafting-effective-readmes"]`

  **Content Draft**:
  ```markdown
  # Spatial UI Implementation Notes

  ## Visual Artifacts in CI/Debug
  The "solid blue square" seen in automated screenshots is a known artifact of the headless browser environment (Linux/SwiftShader). It lacks full WebGL 2 transmission support required for the glass/refraction effect.

  ## Headset Verification
  To verify the correct "frosted glass" look:
  1. Load the app on a Meta Quest headset.
  2. Observe the panel materials.
  3. Expected: Transparent, refractive, blurry background (transmission > 0).
  ```

- [ ] 3. Commit Final State
  **What to do**:
  - Stage changes: `webxr/index.html`, `webxr/src/components/spatial-panel.ts`, `docs/SPATIAL_UI_NOTES.md`.
  - Commit message: `chore(webxr): cleanup debug artifacts and finalize spatial UI`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `["git-master", "commit-work"]`

  **Verification**:
  - `git log -1` shows correct message.
  - `git status` is clean.
