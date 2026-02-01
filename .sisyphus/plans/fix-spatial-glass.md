# Fix Spatial UI Glass Effect

## Context
The user wants to achieve a "glass-like" appearance (transparent, refractive) for the spatial UI panels in the WebXR application. Current implementation uses `MeshPhysicalMaterial` with `transmission` and `opacity` set, but the panels appear opaque in verification screenshots. This is likely due to the lack of an environment map (essential for PBR transmission/reflection) and missing A-Frame renderer configuration for physical lighting and color management.

**Findings:**
- `spatial-panel.ts` has inefficient logic (`tick` loop re-assigning material) and lacks `envMap`.
- `index.html` lacks `renderer` configuration (`colorManagement`, `physicallights`).
- Headless verification is limited but we can use `PMREMGenerator` to generate a fallback environment map.
- `three-stdlib` is available, allowing use of `RoomEnvironment` for PMREM generation.

## Task Dependency Graph

| Task | Depends On | Reason |
|------|------------|--------|
| Task 1 | None | Core component refactor, independent of global config |
| Task 2 | None | Global configuration, can be done in parallel with component |
| Task 3 | Task 1, Task 2 | Verification requires both component fixes and renderer config |

## Parallel Execution Graph

Wave 1 (Start immediately):
├── Task 1: Refactor Spatial Panel Component (Logic + EnvMap)
└── Task 2: Configure A-Frame Renderer (Physicallights + ColorMgmt)

Wave 2 (After Wave 1 completes):
└── Task 3: Verification (Build & Screenshot)

## Tasks

### Task 1: Refactor Spatial Panel Component
**Description**:
Refactor `webxr/src/components/spatial-panel.ts` to:
1.  **Implement Environment Map**: Use `THREE.PMREMGenerator` with `RoomEnvironment` (from `three-stdlib`) to generate a default environment map in `init`. Assign this `envMap` to the `MeshPhysicalMaterial`.
2.  **Optimize Logic**: Remove the `tick` function entirely. Move material application logic to `applyMaterial` which should be idempotent.
3.  **Fix Material Lifecycle**: Ensure `this.material` is created once (or updated) and reused. Do not recreate `MeshPhysicalMaterial` in loops or `traverse`.
4.  **Handle Async Loading**: Ensure environment map generation happens after scene load or is handled async.
5.  **Clean Up**: properly dispose of PMREM generator and textures if the component is removed (optional but good practice).

**Delegation Recommendation**:
- Category: `visual-engineering` - Requires Three.js/A-Frame expertise and understanding of PBR materials.
- Skills: [`typescript-programmer`, `frontend-ui-ux`] - For robust TS code and visual implementation.

**Skills Evaluation**:
- INCLUDED `typescript-programmer`: Essential for writing correct TS code.
- INCLUDED `frontend-ui-ux`: Focus on visual outcome (glass effect).
- OMITTED `python-programmer`: Not needed for frontend work.

**Depends On**: None
**Acceptance Criteria**:
- `tick` function is removed.
- `PMREMGenerator` is used to create an envMap.
- `MeshPhysicalMaterial` is assigned with `envMap` and `transmission: 0.95`.
- Material is not recreated every frame.

### Task 2: Configure A-Frame Renderer
**Description**:
Update `webxr/index.html` to enable physical lighting and color management.
Add `renderer="colorManagement: true; physicallights: true; exposure: 1; toneMapping: ACESFilmic;"` to the `<a-scene>` tag.
This ensures that PBR materials (like glass) render correctly with proper light falloff and color space.

**Delegation Recommendation**:
- Category: `visual-engineering` - Simple config change but affects visuals.
- Skills: [`frontend-ui-ux`] - Understanding of renderer settings.

**Skills Evaluation**:
- INCLUDED `frontend-ui-ux`: Knowledge of A-Frame renderer settings.
- OMITTED `typescript-programmer`: HTML change only.

**Depends On**: None
**Acceptance Criteria**:
- `index.html` `<a-scene>` tag includes `renderer` attribute with specified values.

### Task 3: Verification
**Description**:
1.  Run `npm run build` in `webxr` directory.
2.  Start the teleop server (`python -m teleop_xr.demo`).
3.  Run `node verify_ui.js` to take a screenshot.
4.  Analyze `spatial_ui_verification.png` (using `look_at` or `ls -l` to verify generation).
5.  If headless limitation prevents visual verification, rely on code correctness (logic + config).

**Delegation Recommendation**:
- Category: `quick` - Running verification scripts.
- Skills: [`agent-browser`, `bash`] - To run commands and check files.

**Skills Evaluation**:
- INCLUDED `agent-browser`: For screenshot verification if needed (though script does it).
- INCLUDED `bash`: For running build/server commands.

**Depends On**: Task 1, Task 2
**Acceptance Criteria**:
- Build succeeds.
- Verification script runs without error.
- Screenshot is generated.

## Commit Strategy
- `fix(webxr): refactor spatial-panel for glass effect and performance` (Task 1)
- `config(webxr): enable physical lights and color management` (Task 2)

## Success Criteria
- Codebase has `PMREMGenerator` implemented.
- `index.html` has correct renderer settings.
- Build passes.
- UI logic is performant (no `tick` loop).
