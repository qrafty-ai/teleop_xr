# Plan: Port Vuer URDF Visualization to TeleopXR

## TL;DR

> **Quick Summary**: Port the URDF visualization logic from `vuer` to `teleop_xr` using a modular architecture and modern frontend APIs.
>
> **Deliverables**:
> - Backend: `RobotVisModule` (separate from main Teleop class) handling assets and state.
> - Frontend: `RobotModelSystem` using `urdf-loader` and `three` (modern implementation).
> - Testing: Use `openarm_description` as the test fixture.
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Backend Assets → Frontend Loader → Integration

---

## Context

### Original Request
Port `vuer-ai/vuer`'s URDF viewing capability to `teleop_xr` for visualization only.

### Refinements (Round 2)
- **Test Data**: Use `enactic/openarm_description` instead of dummy URDF.
- **Architecture**: Modularize backend (separate `RobotVisModule` loaded optionally).
- **Assets**: Support both `package://` and relative paths.
- **Frontend**: Use modern `urdf-loader`/`three` APIs; use `vuer-ts` only as a conceptual reference (don't copy stale code).

---

## Work Objectives

### Core Objective
Enable real-time visualization of a URDF-defined robot in the WebXR client, driven by Python state.

### Concrete Deliverables
- [x] `RobotVisConfig` in `teleop_xr/config.py`
- [x] `teleop_xr/robot_vis.py` module (Asset server + State logic)
- [x] `RobotModelSystem` in `webxr/src/robot_system.ts`
- [x] `tests/fixtures/openarm_description` (cloned)
- [x] `demo_robot.py` simulation script

### Definition of Done
- [x] `python -m teleop_xr.demo_robot` launches server.
- [x] WebXR client loads OpenArm URDF with meshes.
- [x] Robot joints move in VR according to Python simulation.

---

## Verification Strategy

### Automated Verification Only
**Frontend/UI**:
- Use `playwright` to load the app and verify the "robot-model" entity exists in the DOM/ECS.

**Backend**:
- Use `curl` to verify `/assets` serves files (both package:// mapped and relative).
- Use `wscat` to verify `robot_state` messages.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Backend & Test Data):
├── Task 1: Setup OpenArm Assets
├── Task 2: Implement Modular Robot Vis Backend
└── Task 3: Backend Joint State Broadcasting

Wave 2 (Frontend):
├── Task 4: Install Dependencies (urdf-loader, three-stdlib)
├── Task 5: Implement RobotModelSystem (Modern Loader)
└── Task 6: Integrate & Register System

Wave 3 (Integration):
└── Task 7: Create Demo Script & Verify
```

---

## TODOs

- [x] 1. Setup OpenArm Assets

  **What to do**:
  - Create `tests/fixtures/` if not exists.
  - Clone `https://github.com/enactic/openarm_description` into `tests/fixtures/openarm_description`.
  - Verify the URDF structure (usually `urdf/openarm.urdf` or similar) and mesh locations.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`bash`]

  **Verification**:
  ```bash
  ls tests/fixtures/openarm_description/urdf/*.urdf
  # Expected: file exists
  ```

- [x] 2. Implement Modular Robot Vis Backend

  **What to do**:
  - Update `teleop_xr/config.py`:
    - Add `RobotVisConfig` model: `urdf_path: str`, `mesh_path: Optional[str]`.
    - Add `robot_vis: Optional[RobotVisConfig]` to `TeleopSettings`.
  - Create `teleop_xr/robot_vis.py`:
    - Define `RobotVisModule` class.
    - **Init**: Accepts `FastAPI` app and `RobotVisConfig`.
    - **Asset Serving**:
      - Mount a custom route/endpoint for `/assets/{file_path:path}`.
      - **Logic**:
        - If `file_path` contains `package://`: Strip prefix, look in `mesh_path` (if configured) or assume standard ROS layout if possible.
        - If `file_path` is relative: Resolve relative to `os.path.dirname(config.urdf_path)`.
        - Return `FileResponse`.
    - **URDF Serving**: Route `/robot.urdf` serving `config.urdf_path`.
  - Update `teleop_xr/__init__.py`:
    - In `Teleop.__init__`:
      - If `settings.robot_vis` is present, instantiate `self.robot_vis = RobotVisModule(self.__app, settings.robot_vis)`.
    - In `websocket_endpoint`:
      - On connect: If `self.robot_vis`, send `robot_config` message (get data from module).

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: [`python`]

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)

  **Verification**:
  ```bash
  # Start dummy server pointing to openarm
  # curl http://localhost:4443/assets/meshes/base_link.stl
  # Expected: File content
  ```

- [x] 3. Implement Joint State Broadcasting

  **What to do**:
  - In `teleop_xr/robot_vis.py`:
    - Add `broadcast_state(self, connection_manager, joints: Dict[str, float])`.
  - In `teleop_xr/__init__.py`:
    - Add `publish_joint_state(self, joints)` to `Teleop`.
    - Implementation: `if self.robot_vis: self.robot_vis.broadcast_state(self.__manager, joints)`.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`python`]

  **Parallelization**:
  - **Can Run In Parallel**: YES (Wave 1)

- [x] 4. Install Frontend Dependencies

  **What to do**:
  - In `webxr/`:
    - `npm install urdf-loader three-stdlib`
    - `npm install --save-dev @types/three`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`bash`]

  **Parallelization**:
  - **Can Run In Parallel**: NO (Blocks Task 5)

- [x] 5. Implement RobotModelSystem (Modern Loader)
- [x] 6. Integrate & Register System
- [x] 7. Create Demo Script & Verify

  **What to do**:
  - Create `teleop_xr/demo_robot.py`.
  - Load OpenArm URDF.
  - Animate joints (sine wave).
  - Verify visualization in browser.

  **Verification**:
  ```bash
  python -m teleop_xr.demo_robot
  ```

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 1 | `test: setup openarm_description fixture` | tests/fixtures/ |
| 2, 3 | `feat(backend): add modular robot visualization` | teleop_xr/ |
| 4 | `chore(webxr): add urdf-loader` | webxr/package.json |
| 5, 6 | `feat(webxr): implement RobotModelSystem` | webxr/src/ |
| 7 | `demo: add robot visualization demo` | teleop_xr/demo_robot.py |
