# Migration: WebXR to A-Frame

## TL;DR

> **Quick Summary**: Complete rewrite of the frontend view layer from `@iwsdk/core` (Three.js ECS) to A-Frame 1.6.0, preserving the core Teleop/Video logic.
>
> **Deliverables**:
> - `webxr/package.json` updated (A-Frame installed, IWSDK removed)
> - `webxr/src/index.html` (A-Frame scene entry point)
> - `webxr/src/systems/teleop.ts` (Ported Teleop logic)
> - `webxr/src/components/*.ts` (UI, Video, Robot components)
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Setup → Systems/Logic → Components → Integration

---

## Context

### Original Request
Migrate `webxr` frontend to A-Frame as per [issue #18](https://github.com/qrafty-ai/teleop_xr/issues/18).

### Interview Summary
**Key Discussions**:
- **UI**: Rebuild using native A-Frame primitives (`a-plane`, `a-text`) instead of external GUI libraries.
- **Hand Tracking**: Maintain current behavior (ignore joint data) to strictly control scope.
- **Input Mode**: Support dynamic switching via backend config.

**Research Findings**:
- **Architecture**: Current app uses proprietary `@iwsdk/core`. `video.ts` and `console_stream.ts` are reusable.
- **Pitfalls**: Billboard panels must be siblings of controllers (not children) to allow independent rotation facing the head.
- **Layout**: A-Frame lacks Flexbox; UI elements require hardcoded relative positions.

---

## Work Objectives

### Core Objective
Replace the proprietary ECS framework with A-Frame while maintaining exact feature parity (Video, Teleop, Robot Model, Console Stream).

### Concrete Deliverables
- Functional A-Frame scene with WebRTC video panels.
- Teleop system transmitting Head/Controller poses to backend.
- UI for Status, FPS, and Camera settings.
- Robot model visualization via URDF.

### Definition of Done
- [ ] `npm run build` succeeds with A-Frame.
- [ ] Teleop demo (`python -m teleop_xr.demo`) receives controller inputs from A-Frame app.
- [ ] Video streams appear on wrist/floating panels.

### Must Have
- Exact JSON structure for `DevicePose` (backend compatibility).
- `console_stream.ts` integration for debugging.

### Must NOT Have (Guardrails)
- **NO** dependency on `@iwsdk/*`.
- **NO** Flexbox layout engines (keep UI simple).
- **NO** new hand-tracking features (unless backend protocol changes).

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (Vitest)
- **User wants tests**: YES (Maintain existing logic coverage)
- **Framework**: Vitest

### Automated Verification Only
Each task includes verification steps executable by the agent:
- **Unit Tests**: `npm run test` (Vitest) for logic.
- **Build**: `npm run build` (Vite) for integration.
- **Runtime**: `curl` to check file existence.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Setup & Clean):
└── Task 1: Environment Setup (Clean deps, Install A-Frame, config)

Wave 2 (Core Logic - Parallel):
├── Task 2: Port TeleopSystem (Logic)
├── Task 3: Port VideoClient (Component wrapper)
└── Task 4: Create RobotModel Component

Wave 3 (UI & Integration - Parallel):
├── Task 5: Create UI Components (Billboard, Wrist, Main)
└── Task 6: Scene Assembly (index.html + Rig)
```

---

## TODOs

- [x] 1. Environment Setup & Dependency Migration

  **What to do**:
  - Remove `@iwsdk/*` dependencies from `webxr/package.json`.
  - Install `aframe` and `@types/aframe`.
  - Update `webxr/vite.config.ts` (remove iwsdk plugins, keep mkcert).
  - Create `webxr/src/aframe-types.d.ts` if needed for TS.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`simplify`]

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: All subsequent tasks

  **References**:
  - `webxr/package.json`
  - `webxr/vite.config.ts`

  **Acceptance Criteria**:
  - [ ] `npm install` completes successfully.
  - [ ] `webxr/node_modules/aframe` exists.
  - [ ] `webxr/vite.config.ts` is clean of `@iwsdk`.
  - [ ] `grep "@iwsdk" webxr/package.json` returns empty.

- [x] 2. Port TeleopSystem Logic

  **What to do**:
  - Create `webxr/src/systems/teleop.ts`.
  - Port `TeleopSystem` from `webxr/src/teleop_system.ts`.
  - Implement `AFRAME.registerSystem('teleop', { ... })`.
  - **Logic**:
    - `tick()`: Gather input from `tracked-controls` components on rig.
    - Format data as `DevicePose` (match existing structure EXACTLY).
    - Send/Receive WebSocket messages.
    - Handle `input_mode` config (emit events to toggle controllers vs hands).

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: [`code-reviewer`] (Logic integrity)

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 2)
  - **Blocked By**: Task 1

  **References**:
  - `webxr/src/teleop_system.ts` (Source logic)
  - `webxr/src/index.ts` (WS connection logic)

  **Acceptance Criteria**:
  - [ ] Unit test `webxr/src/systems/teleop.test.ts` passes (mock WS and A-Frame scene).
  - [ ] `DevicePose` structure matches source file exactly.

- [x] 3. Port VideoClient & Console Stream

  **What to do**:
  - Keep `webxr/src/video.ts` (Logic is good).
  - Create `webxr/src/components/video-stream.ts`.
  - Component `video-stream`:
    - Schema: `{ trackId: string }`.
    - Logic: Use `VideoClient` to get stream, create `<video>` element, apply as texture to entity material.
  - Ensure `console_stream.ts` is imported in main entry.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: [`frontend-ui-ux`]

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 2)
  - **Blocked By**: Task 1

  **References**:
  - `webxr/src/video.ts`
  - `webxr/src/panels.ts:CameraPanel` (Source material handling)

  **Acceptance Criteria**:
  - [ ] `webxr/src/components/video-stream.ts` exists.
  - [ ] `npm run build` succeeds.

- [x] 4. Create Robot Model Component

  **What to do**:
  - Create `webxr/src/components/robot-model.ts`.
  - Wrap `three/examples/jsm/loaders/URDFLoader`.
  - Logic: Load URDF, apply rotation fix (`-Math.PI/2` on X), update joints from `teleop` system events.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 2)
  - **Blocked By**: Task 1

  **References**:
  - `webxr/src/robot_system.ts` (Source logic)

  **Acceptance Criteria**:
  - [ ] `webxr/src/components/robot-model.ts` exists.
  - [ ] Component handles `robot_state` updates.

- [x] 5. Create UI & Billboard Components

  **What to do**:
  - Create `webxr/src/components/ui-components.ts`.
  - **Billboard Component**:
    - Schema: `{ target: selector }` (default head).
    - Logic: Update position to match parent (controller) + offset, but `lookAt` target.
  - **Main UI**:
    - Create `createTeleopUI()` function that builds the Entity hierarchy (Plane background, Text entities for status/FPS).
    - Hardcoded offsets for layout.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: [`canvas-design`] (Layout)

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 3)
  - **Blocked By**: Task 1

  **References**:
  - `webxr/src/panels.ts`
  - `webxr/ui/teleop.json` (Source layout reference)

  **Acceptance Criteria**:
  - [ ] Billboard logic decouples rotation from parent controller.
  - [ ] UI entities created with `a-text` and `a-plane`.

- [x] 6. Scene Assembly & Entry Point

  **What to do**:
  - Update `webxr/src/index.html`:
    - Add `<script src="https://aframe.io/releases/1.6.0/aframe.min.js"></script>`.
    - Add `<a-scene>` with `teleop` system.
    - Define Rig: Camera, Left Hand, Right Hand (with `tracked-controls`).
  - Rewrite `webxr/src/index.ts`:
    - Import all systems/components.
    - Initialize `console_stream`.

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 3)
  - **Blocked By**: Tasks 2, 3, 4, 5

  **References**:
  - A-Frame Docs (Scene structure)
  - `webxr/index.html`

  **Acceptance Criteria**:
  - [ ] `npm run build` generates valid `dist/`.
  - [ ] `index.html` contains correct A-Frame structure.
