# Draft: Fix Spatial UI Glass Effect

## Requirements (confirmed)
- Target: "A-Frame Spatial UI" glass look (transparent, refractive).
- Hypotheses: Headless limitation, missing envMap, renderer config.
- Constraints: Use existing `MeshPhysicalMaterial`, optimize logic.

## Technical Decisions
- **Environment Map**: Use `THREE.PMREMGenerator` + `RoomEnvironment` from `three-stdlib` (avoids downloading assets).
- **Renderer**: Enable `colorManagement` and `physicallights`.
- **Component Logic**: Remove `tick` loop; use event-driven `applyMaterial`.

## Scope Boundaries
- INCLUDE: `spatial-panel.ts` refactor, `index.html` config.
- EXCLUDE: New feature development, other UI changes.

## Verification
- Run `verify_ui.js` (headless).
- Inspect `spatial_ui_verification.png`.
- Fallback: Rely on code correctness if headless renderer fails to handle transmission.
