# Fix Spatial UI Glass Material

## TL;DR

> **Quick Summary**: Fix the "solid blue square" issue by correctly applying `MeshPhysicalMaterial` (glass) to the UI panel background while preserving text readability. Add necessary environment lighting for PBR materials to work.
>
> **Deliverables**:
> - Updated `spatial-panel.ts` (robust material application, text exclusion)
> - Updated `index.html` (lights, component attachment)
> - New Playwright test `tests/visual-regression.spec.ts`
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Fix Component → Update HTML → Verify

---

## Context

### Original Request
User sees a "solid blue square" instead of a glass panel. Suspects material override, missing component, or missing lighting.

### Interview Summary
**Key Discussions**:
- The "solid blue" is likely the fallback behavior of the panel or the robot placeholder box.
- `spatial-panel.ts` currently blindly overwrites *all* meshes, which would make text invisible/glassy.
- Missing PBR environment (lighting/envMap) is a major cause of "flat" looking glass.
- AR mode (passthrough) requires specific handling (frosted tint vs transmission).

**Research Findings**:
- `spatial-panel` is defined but not attached to `#robot-settings-panel` in `index.html`.
- `index.html` lacks `<a-light>` or `environment` components.
- `package.json` only has `@playwright/test` (dev). No extra A-Frame components installed.

### Metis Review
**Identified Gaps** (addressed):
- **Text Visibility**: Component updated to exclude text meshes.
- **Material Leaks**: Component updated to reuse material instances.
- **Verification**: Added automated Playwright test to verify material properties without VR headset.

---

## Work Objectives

### Core Objective
Ensure the UI panel renders with a "glass" aesthetic (translucent, refractive/frosted) while keeping text content legible.

### Concrete Deliverables
- `webxr/src/components/spatial-panel.ts`: Logic to target background plane only.
- `webxr/index.html`: Added lighting and `spatial-panel` attribute.
- `tests/visual-regression.spec.ts`: Playwright test for material verification.

### Definition of Done
- [ ] Panel background is `MeshPhysicalMaterial` with `transmission > 0` or `opacity < 1`.
- [ ] Text meshes are NOT `MeshPhysicalMaterial`.
- [ ] No console errors from material application.

### Must Have
- Robust material application (doesn't get overwritten by next tick).
- Text/Icons remain opaque.
- Basic lighting in scene.

### Must NOT Have (Guardrails)
- Do NOT implement full `robot_settings` UI logic (only styling).
- Do NOT introduce heavy assets (HDRIs) that bloat the repo.
- Do NOT break existing robot model rendering.

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (Playwright in package.json)
- **User wants tests**: YES (Implicit in "Verify" task)
- **Framework**: Playwright

### Automated Verification Only (NO User Intervention)

**For Frontend/UI changes** (using playwright skill):
```typescript
// tests/visual-regression.spec.ts
import { test, expect } from '@playwright/test';

test('panel has glass material', async ({ page }) => {
  await page.goto('http://localhost:8080'); // Adjust port as needed
  await page.waitForSelector('a-scene', { state: 'attached' });

  // Wait for A-Frame to load
  await page.waitForFunction(() => (window as any).AFRAME && (window as any).AFRAME.scenes[0].hasLoaded);

  // Evaluate Three.js state
  const isGlass = await page.evaluate(() => {
    const el = document.querySelector('#robot-settings-panel');
    const mesh = el.object3D.getObjectByProperty('type', 'Mesh'); // Finds first mesh (usually background plane)
    const mat = mesh.material;
    return mat.type === 'MeshPhysicalMaterial' && mat.transmission > 0;
  });

  expect(isGlass).toBe(true);
});
```

---

## Task Dependency Graph

| Task | Depends On | Reason |
|------|------------|--------|
| Task 1 | None | Core logic fix, independent of HTML |
| Task 2 | None | HTML structure update, independent of TS logic |
| Task 3 | Task 1, Task 2 | Verification requires both code and HTML to be ready |

---

## Parallel Execution Graph

Wave 1 (Start Immediately):
├── Task 1: Fix `spatial-panel.ts` logic
└── Task 2: Update `index.html` (lights + component)

Wave 2 (After Wave 1):
└── Task 3: Verify with Playwright

Critical Path: Task 1 → Task 3
Estimated Parallel Speedup: ~40% faster than sequential

---

## Tasks

### Task 1: Fix spatial-panel.ts Logic
**Description**:
- Modify `spatial-panel.ts` to:
  1.  Filter meshes: Apply glass ONLY to `PlaneGeometry` or specific names, exclude `TextGeometry` or meshes with `map` (text often uses map).
  2.  Implement `tick` handler (throttled) to check if material was reset by PanelUI, and re-apply if needed.
  3.  Reuse material instance (don't create `new` every time).
  4.  Add logs for debugging.

**Delegation Recommendation**:
- Category: `visual-engineering` - Requires Three.js/A-Frame knowledge and aesthetic judgment.
- Skills: [`typescript-programmer`, `frontend-ui-ux`] - For robust TS code and design intent.
**Skills Evaluation**:
- INCLUDED `typescript-programmer`: Complex logic for material management.
- INCLUDED `frontend-ui-ux`: Ensuring the "glass" look is aesthetically correct.
**Depends On**: None
**Acceptance Criteria**:
- [ ] Material is `MeshPhysicalMaterial`.
- [ ] Text remains opaque/visible.
- [ ] Material instance is reused across updates.

### Task 2: Update index.html
**Description**:
- Add `spatial-panel` attribute to `#robot-settings-panel`.
- Add basic lighting to support PBR:
  - Ambient light (intensity 0.5)
  - Directional light (castShadow true)
- Ensure scene has `renderer="colorManagement: true"`.

**Delegation Recommendation**:
- Category: `visual-engineering` - HTML/Scene composition.
- Skills: [`frontend-ui-ux`] - Understanding of lighting and scene setup.
**Skills Evaluation**:
- INCLUDED `frontend-ui-ux`: Scene composition.
- OMITTED `typescript-programmer`: Minimal code, mostly markup.
**Depends On**: None
**Acceptance Criteria**:
- [ ] `#robot-settings-panel` has `spatial-panel` attribute.
- [ ] Scene contains `<a-light>` or equivalent.
- [ ] Render loop runs without errors.

### Task 3: Verify with Playwright
**Description**:
- Create `tests/visual-regression.spec.ts`.
- Implement checks for material type on the panel background.
- Run tests to verify the fix.

**Delegation Recommendation**:
- Category: `quick` - Test creation and execution.
- Skills: [`agent-browser`, `typescript-programmer`] - Playwright execution.
**Skills Evaluation**:
- INCLUDED `agent-browser`: Essential for Playwright.
- INCLUDED `typescript-programmer`: Writing the test spec.
**Depends On**: Task 1, Task 2
**Acceptance Criteria**:
- [ ] Test passes: `npx playwright test`.
- [ ] Material is confirmed as `MeshPhysicalMaterial`.

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1, 2 | `fix(ui): enhance spatial-panel and add lighting` | `webxr/src/components/spatial-panel.ts`, `webxr/index.html` | Build/Run |
| 3 | `test(ui): add material verification` | `tests/visual-regression.spec.ts` | `npx playwright test` |

## Success Criteria

### Verification Commands
```bash
# Verify build
npm --prefix webxr run build

# Verify visual state (automated)
npx playwright test
```

### Final Checklist
- [ ] Glass material visible on panel background
- [ ] Text remains readable (not glass)
- [ ] No flashing/flickering from material fighting
- [ ] Tests pass
