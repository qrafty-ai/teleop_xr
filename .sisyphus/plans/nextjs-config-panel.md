# Plan: Next.js Advanced Config Panel

## TL;DR

> **Quick Summary**: Add an "Advanced Settings" panel to the WebXR dashboard for Network and Visualization tuning.
>
> **Deliverables**:
> - New `AdvancedSettingsPanel.tsx` in `webxr/src/components/dashboard/`
> - Updated `webxr/src/lib/store.ts` with persisted `advancedSettings`
> - Updated `webxr/src/xr/teleop_system.ts` (update rate)
> - Updated `webxr/src/xr/console_stream.ts` (log levels)
> - Updated `webxr/src/xr/robot_system.ts` (visibility, spawn settings, reset trigger)
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Store Update → Panel Creation & System Wiring

---

## Context

### Original Request
The user wants to "see what options we can further provide in the nextjs config panel."

### Revised Requirements
**Added**:
- Robot Visibility Toggle
- "Reset Robot Position" Button
- Network Update Rate
- Log Level

**Removed** (per user request):
- Reconnect Delay
- Deadzones
- Input Curves

### Metis Review
**Identified Gaps (addressed)**:
- **Persistence**: Confirmed `zustand/middleware` is safe if lazily initialized for SSR.
- **Validation**: Need to clamp/sanitize persisted values to avoid crashes.
- **Build Safety**: Ensure `use client` and lazy storage to prevent Next.js build errors.

---

## Work Objectives

### Core Objective
Empower users to tune WebXR teleoperation performance and visualization via a persistent configuration UI.

### Concrete Deliverables
- `webxr/src/lib/store.ts`: Updated with `advancedSettings` and `persist` middleware.
- `webxr/src/components/dashboard/AdvancedSettingsPanel.tsx`: New UI component.
- `webxr/src/xr/*.ts`: Systems updated to consume new settings.
- `webxr/src/test/advanced_settings.test.ts`: Unit tests for logic.

### Definition of Done
- [x] `npm run build` in `webxr` passes.
- [x] `npm run test` in `webxr` passes (including new tests).
- [x] Settings persist across page reloads.
- [x] Default values match previous hardcoded constants exactly.

### Must Have
- **Exact Defaults**: 100Hz, 1.0m spawn, -0.3m height.
- **Persistence**: Settings survive reload.
- **Reset Action**: Button immediately respawns robot.

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
> ALL verification is executed by the agent using tools.

### Test Decision
- **Infrastructure exists**: YES (vitest in `webxr`)
- **Automated tests**: YES (TDD for logic)
- **Framework**: vitest

### TDD Workflow
1. **RED**: Write failing unit tests for store defaults and persistence simulation.
2. **GREEN**: Implement store updates and system logic.
3. **REFACTOR**: Clean up integration.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Update Store & Logic Utilities (TDD)
└── Task 2: Create AdvancedSettingsPanel UI

Wave 2 (After Wave 1):
└── Task 3: Integrate Systems (Teleop, Console, Robot) & Verify
```

---

## TODOs

- [x] 1. Update Store & Implement Logic Utilities (TDD)

  **What to do**:
  - Create `webxr/src/test/settings.test.ts` to test store defaults and persistence.
  - Update `webxr/src/lib/store.ts`:
    - Define `AdvancedSettings` type: `{ updateRate, logLevel, robotVisible, spawnDistance, spawnHeight }`.
    - Add `advancedSettings` to `AppState`.
    - Add `robotResetTrigger` (number) to `AppState` (ephemeral, NOT persisted).
    - Implement `persist` middleware (lazy init for SSR safety).

  **Must NOT do**:
  - Break existing `teleopSettings`.
  - Import `store` in server-side contexts.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`code-reviewer`] (for TDD/logic correctness)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 3
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] `webxr/src/test/settings.test.ts` created.
  - [ ] `npm test` passes.
  - [ ] `store.ts` exports typed `advancedSettings`.
  - [ ] `robotResetTrigger` updates when action called.

- [x] 2. Create AdvancedSettingsPanel UI

  **What to do**:
  - Create `webxr/src/components/dashboard/AdvancedSettingsPanel.tsx`.
  - Use `Card`, `Switch`, `Slider`, `Label`, `Button` from `webxr/src/components/ui/`.
  - Implement collapsible "Advanced Settings" section (simple React state `isOpen`).
  - Wire up controls to `useAppStore`.
  - "Reset Robot Position" button calls `setRobotResetTrigger(Date.now())`.
  - Add panel to `webxr/src/app/page.tsx` (below existing panels).

  **Must NOT do**:
  - Add new UI dependencies.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: [`frontend-ui-ux`]

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 3
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] Panel component file exists.
  - [ ] Compiles without TS errors.
  - [ ] `npm run lint` passes.

- [x] 3. Integrate Systems & Final Verification

  **What to do**:
  - **TeleopSystem** (`webxr/src/xr/teleop_system.ts`):
    - Subscribe to `advancedSettings.updateRate`.
    - Use rate for throttle loop.
  - **ConsoleStream** (`webxr/src/xr/console_stream.ts`):
    - Subscribe to `advancedSettings.logLevel`.
    - Filter messages before pushing to queue.
  - **RobotSystem** (`webxr/src/xr/robot_system.ts`):
    - Subscribe to `advancedSettings.robotVisible` -> toggle `model.visible`.
    - Subscribe to `advancedSettings.spawnDistance/Height`.
    - Subscribe to `robotResetTrigger` -> call `spawnRobot()` immediately.

  **Must NOT do**:
  - Change `0.01s` default behavior.
  - Introduce cyclic dependencies.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`code-reviewer`, `simplify`]

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: None
  - **Blocked By**: Task 1, Task 2

  **Acceptance Criteria**:
  - [ ] All hardcoded constants replaced with store values.
  - [ ] `npm run build` passes.
  - [ ] `npm test` passes.

---

## Success Criteria

### Final Checklist
- [ ] `npm run build` passes.
- [ ] Store has `persist` middleware.
- [ ] Defaults match original constants.
- [ ] "Reset Robot Position" button triggers respawn.
