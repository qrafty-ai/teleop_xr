# UI Migration to Horizon Kit

## TL;DR

> **Quick Summary**: Update all `.uikitml` files to use `pmndrs/uikit` Horizon Kit components, replacing raw HTML tags and custom styles.
>
> **Deliverables**:
> - Updated `package.json` with Horizon Kit dependencies
> - Updated `src/index.ts` enabling `spatialUI` feature
> - Refactored `.uikitml` files (teleop, camera, settings, welcome)
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Install -> Configure -> Refactor UI

---

## Context

### Original Request
Update all UI components in `@webxr/` to use `uikit` (Horizon Kit), preferring existing components.

### Interview Summary
**Key Discussions**:
- **Architecture**: Stick to `@iwsdk/core` + `uikitml` workflow (no React rewrite).
- **Theme**: Use Strict Horizon Theme (Horizon defaults > Custom CSS).
- **Scope**: All `.uikitml` files.

**Research Findings**:
- Project uses `vite-plugin-uikitml`.
- Dependencies missing: `@pmndrs/uikit-horizon`, `@pmndrs/uikit-lucide`.
- Critical runtime bindings rely on specific Element IDs and CSS classes.

### Metis Review
**Identified Gaps** (addressed):
- **State Styling**: Must preserve functional classes (`.hidden`, `.active`, `.connected`) even in "Strict" theme.
- **ID Preservation**: List of critical IDs identified and added to verification.
- **Config Pattern**: Validated `features.spatialUI.kits` pattern from guide.

---

## Work Objectives

### Core Objective
Replace custom UI implementation with standardized Horizon Kit components while maintaining all existing functionality.

### Concrete Deliverables
- `package.json` (dependencies added)
- `src/index.ts` (kits configured)
- `ui/teleop.uikitml` (refactored)
- `ui/camera.uikitml` (refactored)
- `ui/camera_settings.uikitml` (refactored)
- `ui/welcome.uikitml` (refactored)

### Definition of Done
- [x] Build succeeds (`npm run build`)
- [x] No forbidden `<style>` blocks (except allowed state classes)
- [x] All critical IDs present in new markup

### Must Have
- `<Panel>`, `<Button>`, `<ButtonIcon>`, `<Input>`, `<Card>` where applicable.
- Minimal state-handling CSS (`.hidden`, `.active`, `.connected`).

### Must NOT Have (Guardrails)
- Custom visual CSS (colors, borders, shadows) - rely on Horizon defaults.
- Removal of IDs used by TypeScript systems.

---

## Verification Strategy

### Automated Verification Only
> All verification is executable by the agent.

**Type 1: Build Verification**
```bash
npm --prefix webxr install && npm --prefix webxr run build
# Assert: Exit code 0
# Assert: public/ui/*.json files exist
```

**Type 2: ID Preservation Check**
```bash
# Script to verify all critical IDs exist in source files
node -e "
const fs=require('fs');
const checks=[
  ['webxr/ui/camera.uikitml',['id=\"camera-label\"']],
  ['webxr/ui/teleop.uikitml',['id=\"status-text\"','id=\"fps-text\"','id=\"latency-text\"','id=\"camera-button\"','id=\"camera-settings-btn\"','id=\"xr-button\"']],
  ['webxr/ui/camera_settings.uikitml',['id=\"close-btn\"','id=\"row-0\"','id=\"label-0\"','id=\"btn-0\"','id=\"row-5\"','id=\"label-5\"','id=\"btn-5\"']],
  ['webxr/ui/welcome.uikitml',['id=\"xr-button\"']]
];
let err=0;
for(const [f,need] of checks){
  if(!fs.existsSync(f)) { console.error('Missing file: '+f); err=1; continue; }
  const s=fs.readFileSync(f,'utf8');
  for(const n of need){
    if(!s.includes(n)){ console.error('Missing '+n+' in '+f); err=1;}
  }
}
process.exit(err);
"
```

**Type 3: Style Guardrail Check**
```bash
# Verify no custom styles (allowing only specific state classes)
node -e "
const fs=require('fs');
const files=['webxr/ui/teleop.uikitml','webxr/ui/camera.uikitml','webxr/ui/camera_settings.uikitml','webxr/ui/welcome.uikitml'];
const allowedClasses = ['.hidden', '.active', '.connected'];
let err=0;
for(const f of files){
  if(!fs.existsSync(f)) continue;
  const s=fs.readFileSync(f,'utf8');
  // Simple check for <style> blocks - strict mode
  // If we need state classes, we might allow <style> but check content.
  // For this plan: WARN if <style> exists, but allow if it only contains state logic.
  if(s.includes('<style>')){
     console.log('WARNING: <style> block found in '+f+'. Verify it only contains state classes: ' + allowedClasses.join(', '));
  }
}
"
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1:
├── Task 1: Install & Config (Global setup)
└── Task 2: Refactor Camera UI (Independent)

Wave 2 (After Wave 1):
├── Task 3: Refactor Teleop UI (Depends on Config)
├── Task 4: Refactor Settings UI (Depends on Config)
└── Task 5: Refactor Welcome UI (Depends on Config)
```

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1 | `build` (Setup) |
| 1 | 2 | `visual-engineering` (UI work) |
| 2 | 3, 4, 5 | `visual-engineering` (UI work) |

---

## TODOs

- [x] 1. Install & Configure Horizon Kit

  **What to do**:
  - Install `@pmndrs/uikit-horizon`, `@pmndrs/uikit-lucide`.
  - Modify `webxr/src/index.ts`:
    - Import `* as horizonKit` from `@pmndrs/uikit-horizon`.
    - Import `{ LogInIcon, SettingsIcon, XIcon, VideoIcon, SignalIcon, SignalZeroIcon }` from `@pmndrs/uikit-lucide` (optimize imports based on usage).
    - Update `World.create` options to include `features: { spatialUI: { kits: [horizonKit, { ...icons }] } }`.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`javascript`, `node`]

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 2)
  - **Parallel Group**: Wave 1
  - **Blocks**: 3, 4, 5

  **Acceptance Criteria**:
  - [ ] `webxr/package.json` contains `@pmndrs/uikit-horizon`
  - [ ] `webxr/src/index.ts` imports horizonKit and passes it to `World.create`
  - [ ] `npm --prefix webxr run build` passes

- [x] 2. Refactor Camera UI (`ui/camera.uikitml`)

  **What to do**:
  - Rewrite `webxr/ui/camera.uikitml`.
  - Replace `<div class="panel-container">` with `<Panel>`.
  - Replace header `<div>` with `Horizon` structural equivalents if available, or just use `<Panel>` structure.
  - Maintain ID `camera-label`.
  - Ensure video overlay logic (which appends video mesh) works. (Note: Video mesh is added to `panelEntity.object3D`, so as long as `PanelUI` creates a root, it should be fine).

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: [`frontend-ui-ux`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1

  **Acceptance Criteria**:
  - [ ] File uses `<Panel>` tag
  - [ ] ID `camera-label` exists
  - [ ] No custom CSS for panel colors (use default)

- [x] 3. Refactor Teleop UI (`ui/teleop.uikitml`)

  **What to do**:
  - Rewrite `webxr/ui/teleop.uikitml`.
  - Structure: `<Panel>` root.
  - Status section: Use `<Card>` or `<Text>`? Prefer Horizon components.
  - Buttons: Replace `<div class="button">` with `<Button>`.
  - Icons: Use `<ButtonIcon><IconName/></ButtonIcon>`.
  - IDs: Preserve `status-text`, `fps-text`, `latency-text`, `camera-button`, `camera-settings-btn`, `xr-button`.
  - State: Keep `.connected` class style if needed for color toggle, OR rely on text update. (System toggles class, so keep class style).

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: [`frontend-ui-ux`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: 1

  **Acceptance Criteria**:
  - [ ] Uses `<Panel>`, `<Button>`
  - [ ] All IDs present
  - [ ] Build verification passes

- [x] 4. Refactor Settings UI (`ui/camera_settings.uikitml`)

  **What to do**:
  - Rewrite `webxr/ui/camera_settings.uikitml`.
  - Root: `<Panel>`.
  - List items: Use `<Card>` or flex container.
  - Buttons: `<Button size="sm">` (or similar).
  - IDs: Preserve `close-btn`, `row-N`, `label-N`, `btn-N`.
  - State: Keep `.hidden` and `.active` styles.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: [`frontend-ui-ux`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: 1

  **Acceptance Criteria**:
  - [ ] Uses `<Panel>`, `<Button>`
  - [ ] All IDs present
  - [ ] Build verification passes

- [x] 5. Refactor Welcome UI (`ui/welcome.uikitml`)

  **What to do**:
  - Rewrite `webxr/ui/welcome.uikitml`.
  - Simple `<Panel>` with `<Text>` and `<Button id="xr-button">`.
  - Use `<LoginIcon>` (from lucide) inside `<ButtonIcon>`.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: [`frontend-ui-ux`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocked By**: 1

  **Acceptance Criteria**:
  - [ ] Uses `<Panel>`, `<Button>`
  - [ ] ID `xr-button` present
  - [ ] Build verification passes
