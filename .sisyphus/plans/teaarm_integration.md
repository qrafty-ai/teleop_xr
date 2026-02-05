# TeaArm Robot Integration & RAM Local Support

## TL;DR

> **Quick Summary**: Extend `ram.py` to support loading robot descriptions from local directories (skipping git clone) and create a `TeaArmRobot` class configured for the `teaarm` dual-arm robot.
>
> **Deliverables**:
> - Updated `teleop_xr/ram.py` with `repo_root` support in `get_resource`.
> - New `teleop_xr/ik/robots/teaarm.py` class.
> - Registered `teaarm` entry point in `pyproject.toml`.
> - Test suite for local RAM functionality.
>
> **Estimated Effort**: Medium
> **Parallel Execution**: Sequential (RAM update blocks Robot creation)
> **Critical Path**: RAM Update → Robot Class → Registry → Verification

---

## Context

### Original Request
Extend `ram` module to support local URDFs and create a new robot class for `/home/cc/codes/tea/ros2_wksp/src/tea-ros2/tea_description/urdf/teaarm.urdf.xacro`. Use xacro options `with_obstacles=false` and `visual_mesh_ext=glb`.

### Interview Summary
**Key Discussions**:
- `ram.py` currently enforces git cloning; needs modification to support local roots.
- `teaarm` is a dual-arm robot; end-effectors are `frame_left_arm_ee` and `frame_right_arm_ee`.
- User requested specific local path hardcoding for this robot class.

### Metis Review
**Identified Gaps** (addressed):
- **Risk**: Writing processed URDFs to user's source directory. **Resolution**: Local mode will write processed files to `~/.cache/ram/processed/` instead of source dir.
- **Risk**: Path traversal. **Resolution**: Validate `path_inside_repo` is relative and contained within `repo_root`.
- **API**: Extend `get_resource` with optional `repo_root` parameter.

---

## Work Objectives

### Core Objective
Enable local robot description loading in `ram` and integrate `TeaArmRobot`.

### Concrete Deliverables
- `teleop_xr/ram.py` (modified)
- `teleop_xr/ik/robots/teaarm.py` (new)
- `pyproject.toml` (modified)
- `tests/test_ram_local.py` (new)

### Definition of Done
- [x] `ram.get_resource` accepts `repo_root` and skips git if set.
- [x] `TeaArmRobot` loads successfully from local path.
- [x] `teleop_xr.demo --list-robots` shows `teaarm`.
- [x] `TeaArmRobot` forward kinematics returns poses for both left and right EEs.
- [x] WebXR frontend supports GLB meshes via `loadMeshCb`.

### Must Have
- Support for `with_obstacles=false`, `visual_mesh_ext=glb` xacro args.
- Safe output handling (no pollution of source directory).

### Must NOT Have (Guardrails)
- Do NOT commit the external `tea-ros2` files.
- Do NOT break existing git-based robot loading (Franka, H1).

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
> ALL verification is executed by the agent using tools.

### Test Decision
- **Infrastructure exists**: YES (`pytest`)
- **Automated tests**: YES (TDD for RAM changes)
- **Framework**: `pytest`

### TDD Workflow
1. **RED**: Write `tests/test_ram_local.py` ensuring `get_resource` with `repo_root` fails (or mocks show git would be called).
2. **GREEN**: Implement `repo_root` logic in `ram.py`.
3. **REFACTOR**: Ensure code handles both git and local paths cleanly.

### Agent-Executed QA Scenarios

#### Scenario 1: RAM Local Resource Loading
- **Tool**: Bash (pytest)
- **Steps**:
    1. Create a dummy local xacro file in a temp dir.
    2. Call `ram.get_resource(repo_root=temp_dir, path_inside_repo="dummy.xacro")`.
    3. Assert returned path exists and is in `~/.cache/ram` (not temp dir).
    4. Assert content matches processed xacro.

#### Scenario 2: TeaArm Robot Instantiation
- **Tool**: Bash (python script)
- **Steps**:
    1. Create script `verify_teaarm.py`:
       ```python
       from teleop_xr.ik.robots.teaarm import TeaArmRobot
       robot = TeaArmRobot()
       print(f"FK Keys: {robot.forward_kinematics(robot.get_config()).keys()}")
       ```
    2. Run `python verify_teaarm.py`.
    3. Assert output contains `FK Keys:` and includes `'left'`, `'right'`.

#### Scenario 3: Registry Check
- **Tool**: Bash
- **Steps**:
    1. Run `pip install -e .` (to refresh entry points).
    2. Run `python -m teleop_xr.demo --list-robots`.
    3. Assert output contains `teaarm`.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Sequential):
├── Task 1: Update RAM (Core Dependency)

Wave 2 (Dependent):
├── Task 2: Create TeaArmRobot
└── Task 3: Update Registry

Wave 3 (Verification):
└── Task 4: Run Tests & Verification
```

---

## TODOs

- [x] 1. Update RAM to Support Local Resources

  **What to do**:
  - Create `tests/test_ram_local.py` to test `get_resource` with `repo_root` (TDD).
  - Modify `teleop_xr/ram.py`: `get_resource` signature to accept `repo_root`.
  - Implement logic: IF `repo_root` set → Skip `get_repo`, validate path, use `repo_root` as base.
  - Implement output redirection: IF local mode → write processed URDF to `_CACHE_DIR / "processed" / <hash>.urdf`.

  **Must NOT do**:
  - Modify `git` logic for existing `repo_url` calls.
  - Write files to `repo_root`.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`code-reviewer`]

  **Parallelization**:
  - **Can Run In Parallel**: NO (Core dependency)

  **References**:
  - `teleop_xr/ram.py:get_resource` - Target function.
  - `tests/test_ram.py` - Reference for testing.

  **Acceptance Criteria**:
  - [x] `pytest tests/test_ram_local.py` PASS.
  - [x] Existing `pytest tests/test_ram.py` PASS.

- [x] 2. Create TeaArmRobot Class

  **What to do**:
  - Create `teleop_xr/ik/robots/teaarm.py`.
  - Class `TeaArmRobot` inherits `BaseRobot`.
  - `__init__`:
    - Defaults: `repo_root="/home/cc/codes/tea/ros2_wksp/src/tea-ros2"`, `path="tea_description/urdf/teaarm.urdf.xacro"`.
    - Call `ram.get_resource` with these local paths and `mappings={'with_obstacles': 'false', 'visual_mesh_ext': 'glb'}`.
    - Initialize `pk.Robot`.
  - `forward_kinematics`:
    - Return `{'left': self.robot.fk(cfg, 'frame_left_arm_ee'), 'right': self.robot.fk(cfg, 'frame_right_arm_ee')}`.
  - `build_costs`:
    - Implement basic position/rotation costs for both EEs (copy pattern from `h1_2.py` or `franka.py`).

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`cartography`]

  **Parallelization**:
  - **Can Run In Parallel**: NO (Depends on Task 1)
  - **Blocked By**: Task 1

  **References**:
  - `teleop_xr/ik/robots/h1_2.py` - Reference implementation (dual arm/humanoid).
  - `teleop_xr/ik/robot.py` - Base class definition.

  **Acceptance Criteria**:
  - [x] File exists: `teleop_xr/ik/robots/teaarm.py`.
  - [x] Class `TeaArmRobot` is importable.

- [x] 3. Register TeaArm Robot

  **What to do**:
  - Edit `pyproject.toml`: Add `teaarm = "teleop_xr.ik.robots.teaarm:TeaArmRobot"` to `[project.entry-points."teleop_xr.robots"]`.

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Parallelization**:
  - **Can Run In Parallel**: NO (Depends on Task 2)
  - **Blocked By**: Task 2

  **Acceptance Criteria**:
  - [x] `grep "teaarm =" pyproject.toml` returns match.

- [x] 4. Verification & Cleanup

  **What to do**:
  - Run `pip install -e .` to register entry points.
  - Run verification script (Scenario 2).
  - Run full tests.

  **Recommended Agent Profile**:
  - **Category**: `quick`

  **Acceptance Criteria**:
  - [x] All tests pass.
  - [x] Verification script prints success keys.

---

## Success Criteria

### Verification Commands
```bash
python -m pytest tests/test_ram_local.py
python -m teleop_xr.demo --list-robots  # Must show teaarm
```

### Final Checklist
- [x] `ram.py` handles local paths safely.
- [x] `TeaArmRobot` works with local xacro.
- [x] No pollution of `/home/cc/codes/tea/...` directory.
