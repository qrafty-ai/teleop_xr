# Fix Spatial UI Glass Material

## Context

**User Request**: Fix the "A-Frame Spatial UI" glass look (currently solid blue) by forcing material properties and debugging mesh structure. The user suspects material fighting, incorrect property application, or geometry mismatches.

**Current State**:
- `spatial-panel.ts` re-creates `MeshPhysicalMaterial` every frame in `tick()`, causing performance issues and potential memory leaks.
- The material might be getting overwritten by `PanelUI` or other systems.
- Visual result is a "solid blue square" instead of glass.

**Strategy**:
1.  **Refactor `spatial-panel.ts`**: Optimize to cache material (create once, apply only when needed), fix geometry detection, and force specific glass properties (white color, low opacity).
2.  **Add Debugging**: Instrument the component to log what it's touching.
3.  **Automated Verification**: Add Playwright tests to verify material properties and stability (no per-frame changes).
4.  **Visual Verification**: Add an isolation test primitive (`<a-box>`) to `index.html`.

## Task Dependency Graph

| Task | Depends On | Reason |
|------|------------|--------|
| Task 1 | None | Implementation of the core fix and optimization |
| Task 2 | None | HTML change, independent of component logic |
| Task 3 | Task 1 | Requires the refactored component to verify properties |
| Task 4 | Task 2, Task 3 | Final visual check requires both code and scene setup |

## Parallel Execution Graph

Wave 1 (Start Immediately):
├── Task 1: Refactor `spatial-panel.ts` (Fix & Optimize)
└── Task 2: Add Isolation Test Primitive to `index.html`

Wave 2 (After Wave 1):
└── Task 3: Create Playwright Verification Test

Wave 3 (After Wave 2):
└── Task 4: Visual Verification (Screenshot)

Critical Path: Task 1 → Task 3 → Task 4

## Tasks

### Task 1: Refactor `spatial-panel.ts` (Fix & Optimize)
**Description**:
Refactor the component to:
1.  Add `debug: boolean` to schema.
2.  Store a persistent `MeshPhysicalMaterial` instance on the component (do NOT create in `tick`).
3.  In `update()`: update the stored material properties.
4.  In `tick()`: Check `node.material.uuid`. Only re-apply if it differs from our stored material.
5.  Improve `applyMaterial`:
    - Log mesh names/types if `debug: true`.
    - Broaden the check to apply to `isMesh` (with optional filtering if needed), ensuring we hit the plane.
6.  Force defaults: `color: #ffffff`, `opacity: 0.1`, `transparent: true`.

**Delegation Recommendation**:
- Category: `visual-engineering` - Three.js material handling and performance optimization.
- Skills: [`typescript-programmer`, `frontend-ui-ux`] - Complex logic and visual outcome.

**Skills Evaluation**:
- INCLUDED `typescript-programmer`: Required for robust Three.js/A-Frame component typing.
- INCLUDED `frontend-ui-ux`: Understanding of PBR materials (transmission, roughness).
- OMITTED `git-master`: Not a complex git operation.

**Depends On**: None
**Acceptance Criteria**:
- `applyMaterial` does NOT call `new MeshPhysicalMaterial` in `tick()` loop.
- `tick()` logic only re-assigns material if `node.material` has changed (is dirty).
- Component logs mesh hierarchy when `debug: true`.

### Task 2: Add Isolation Test Primitive
**Description**:
Modify `webxr/index.html` to add a simple test entity:
```html
<a-box
  position="1 1.6 -1"
  scale="0.2 0.2 0.2"
  spatial-panel="color: #ffffff; opacity: 0.2; transmission: 0.95; debug: true"
></a-box>
```
This isolates the component from the complex `PanelUI` system to verify if the shader works at all.

**Delegation Recommendation**:
- Category: `quick` - Simple HTML edit.
- Skills: [`frontend-ui-ux`] - Scene composition.

**Skills Evaluation**:
- INCLUDED `frontend-ui-ux`: Knowledge of A-Frame primitives.
- OMITTED `typescript-programmer`: HTML only.

**Depends On**: None
**Acceptance Criteria**:
- `webxr/index.html` contains the test `<a-box>`.
- `npm --prefix webxr run build` passes.

### Task 3: Automated Verification (Playwright)
**Description**:
Create `webxr/tests/spatial-panel.spec.ts` to verify:
1.  The `a-box` with `spatial-panel` exists.
2.  The underlying mesh material is `MeshPhysicalMaterial`.
3.  `material.transparent` is `true`.
4.  `material.uuid` remains constant over 100ms (verifying no per-frame reallocation).

**Delegation Recommendation**:
- Category: `unspecified-low` - Writing a standard test.
- Skills: [`playwright`, `typescript-programmer`] - Browser automation and TS test code.

**Skills Evaluation**:
- INCLUDED `playwright`: Essential for checking runtime WebGL/DOM state.
- INCLUDED `typescript-programmer`: Writing the test spec.

**Depends On**: Task 1 (component logic), Task 2 (target entity)
**Acceptance Criteria**:
- `npx playwright test tests/spatial-panel.spec.ts` passes.
- Confirms material type and stability.

### Task 4: Visual Verification
**Description**:
Manually capture a screenshot or check logs to confirm the fix.
1.  Check console logs for "SpatialPanel applied to..." (from Task 1 debug).
2.  Check visual appearance of the box (should be glass).

**Delegation Recommendation**:
- Category: `visual-engineering` - Visual confirmation.
- Skills: [`agent-browser`] - To capture evidence.

**Skills Evaluation**:
- INCLUDED `agent-browser`: Capturing screenshot.

**Depends On**: Task 3
**Acceptance Criteria**:
- Screenshot saved to `.sisyphus/evidence/spatial_ui_glass.png`.

## Commit Strategy
- Commit 1: "fix(ui): refactor spatial-panel to optimize material application and force glass look" (Task 1)
- Commit 2: "test(ui): add isolation primitive and playwright verification" (Task 2, 3)

## Success Criteria
1.  `npx playwright test` passes (proving material is correct and stable).
2.  Screenshot shows a transparent/glass-like cube (proving shader works).
3.  Console logs confirm correct mesh targeting.
