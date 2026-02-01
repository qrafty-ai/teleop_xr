# Plan: Debug Visibility

## TL;DR
> **Quick Summary**: Fix blank screenshot by adjusting camera rig to look at the dashboard and adding a placeholder for the robot model.
>
> **Deliverables**:
> - Updated `webxr/index.html` (Camera pose fix)
> - Updated `webxr/src/components/robot-model.ts` (Placeholder geometry)
> - `final_verification_fixed.png` (Evidence)
>
> **Estimated Effort**: Small
> **Parallel Execution**: Sequential (Verification depends on Fixes)
> **Critical Path**: Fix Camera -> Fix Robot -> Verify

---

## Context

### Original Request
User sees a blank grey screen in `final_verification.png`. Wants to see `teleop-dashboard` and `robot-model`.

### Investigation Summary
**Root Causes**:
1.  **Camera Framing**: `index.html` camera is at `0 0 1` looking down (`-20 0 0`). Dashboard is at `0 1.5 -1.5` (above and forward). The camera is likely missing the dashboard entirely.
2.  **Invisible Robot**: `robot-model.ts` only renders after a WebSocket `robot_config` event. In the screenshot test (no WS server or early capture), the robot is effectively empty.

**Key Decisions**:
- **Camera**: Move rig to `0 1.6 0` (standard VR standing height) and reset rotation to `0 0 0` to face forward/dashboard.
- **Robot**: Add a temporary "placeholder" box in `init()` that gets removed when the real robot loads. This ensures *something* is visible for debug/screenshots.
- **Dashboard**: "Connecting..." text is created in `init()`, so framing fix is sufficient.

### Metis Review
**Identified Gaps** (addressed):
- **Screenshot Pipeline**: Confirmed `final_verification.spec.ts` captures the canvas via Playwright on `localhost:4443`.
- **Guardrails**: Placeholder must be cleared in `onRobotConfig` to avoid double-rendering.

---

## Work Objectives

### Core Objective
Ensure the dashboard and a robot representation are visible in the automated screenshot.

### Concrete Deliverables
- `webxr/index.html`: Camera rig transform updated.
- `webxr/src/components/robot-model.ts`: Placeholder logic added.
- `tests/final_verification.spec.ts`: Test runs successfully.

### Definition of Done
- [ ] `final_verification_fixed.png` contains non-grey pixels (dashboard and box visible).
- [ ] Test passes with `expect(canvasExists).toBeGreaterThan(0)`.

---

## Verification Strategy

### Automated Verification Only (NO User Intervention)

> **CRITICAL PRINCIPLE: ZERO USER INTERVENTION**
> ALL verification MUST be automated.

**For UI/Scene Changes** (using Playwright + Python Server):
```bash
# Agent executes:
# 1. Start backend server in background
python -m teleop_xr.demo &
PID=$!
sleep 5 # Wait for startup

# 2. Run Playwright test
npx playwright test tests/final_verification.spec.ts

# 3. Cleanup
kill $PID
```

**Evidence to Capture:**
- `final_verification.png` (artifact from test).
- Terminal output showing test pass.

---

## Execution Strategy

### Parallel Execution Waves
Sequential execution required as verification depends on code changes.

```
Wave 1 (Fixes):
├── Task 1: Fix Camera Rig (index.html)
└── Task 2: Add Robot Placeholder (robot-model.ts)

Wave 2 (Verification):
└── Task 3: Run Verification Test
```

---

## TODOs

- [ ] 1. Fix Camera Rig Framing

  **What to do**:
  - Modify `webxr/index.html`.
  - Change `#rig` or `#head` position to `0 1.6 0`.
  - Change rotation to `0 0 0` (looking forward).

  **Reference**:
  - `webxr/index.html:20-21` - Current rig definition.

  **Acceptance Criteria**:
  - [ ] `grep 'position="0 1.6 0"' webxr/index.html` returns match.
  - [ ] `grep 'rotation="0 0 0"' webxr/index.html` returns match.

  **Recommended Agent Profile**:
  - **Category**: `quick` (Simple HTML attribute edit)
  - **Skills**: `["code-reviewer"]`

- [ ] 2. Add Robot Placeholder

  **What to do**:
  - Modify `webxr/src/components/robot-model.ts`.
  - In `init()`: Create a `THREE.Mesh` (BoxGeometry 0.5m) with `THREE.MeshBasicMaterial` (wireframe or color).
  - Add it to `this.el.object3D`.
  - Store it in `self.placeholder`.
  - In `onRobotConfig()`: Check if `self.placeholder` exists, remove it from `this.el.object3D`, and set to null.

  **Reference**:
  - `webxr/src/components/robot-model.ts:16` - `init` function.
  - `webxr/src/components/robot-model.ts:42` - `onRobotConfig` function.

  **Acceptance Criteria**:
  - [ ] Placeholder code exists in `init`.
  - [ ] Placeholder removal code exists in `onRobotConfig`.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `["frontend-ui-ux"]` (Three.js knowledge)

- [ ] 3. Run Final Verification

  **What to do**:
  - Start `python -m teleop_xr.demo`.
  - Run `npx playwright test tests/final_verification.spec.ts`.
  - Capture output and screenshot.

  **Acceptance Criteria**:
  - [ ] Playwright test passes.
  - [ ] Screenshot `final_verification.png` is generated.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
  - **Skills**: `["playwright"]`

---

## Success Criteria

### Final Checklist
- [ ] Camera is at standing height (1.6m).
- [ ] Robot component shows a box by default.
- [ ] Screenshot shows dashboard + box.
