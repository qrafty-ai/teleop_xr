# IK API Modularization

## TL;DR

> **Quick Summary**: Extract the IK logic from teleop_xr to a clean, modularized API with proper exports, making it easy for users to integrate into their own Python projects or ROS2 systems. Add `--mode ik` to the ROS2 node following the demo pattern.
>
> **Deliverables**:
> - Clean `teleop_xr.ik` module with proper `__init__.py` exports
> - ROS2 node with `--mode ik` flag that subscribes to `/joint_states` and publishes `/joint_trajectory`
> - Documentation (docstrings, type hints) for public API
> - TDD tests for IK module functionality
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 2 waves (API cleanup || then ROS2 integration)
> **Critical Path**: Task 1 -> Task 2 -> Task 4 -> Task 5

---

## Context

### Original Request
User wants to fully extract the IK logic to a modularized API so that it's easy for users to integrate into their own projects. The API should support both generic Python usage and ROS2 integration. For ROS2, the plan is to follow the demo program pattern and add a flag to enable IK mode.

### Interview Summary
**Key Discussions**:
- **IK Backend**: Pyroki only (JAX-based), exclude Pinocchio/JacobiRobot
- **Abstraction Level**: Full stack - IKController + Solver + BaseRobot
- **Robot Support**: Single arm + bimanual (optionally with waist/torso)
- **Single-arm Pattern**: Same BaseRobot class, user chooses which controller pose to use
- **Deadman Rule**: Always require both grips squeezed (even for single-arm)
- **Missing Targets**: All targets required - no optional handling
- **ROS2 Topics**: /joint_states (subscribe), /joint_trajectory (publish)
- **Testing**: TDD for IK module, manual testing only for ROS2

**Research Findings**:
- `teleop_xr/ik/__init__.py` is effectively empty - needs proper exports
- `IKController._check_deadman()` already requires both grips - no change needed
- `PyrokiSolver.solve()` signature requires all 3 targets - matches decision
- `demo/__main__.py` has `IKWorker` thread pattern reusable for ROS2
- ROS2 node has no `mode` flag currently

### Metis Review
**Identified Gaps** (addressed):
- Single-arm deadman rule: Decision made (require both grips)
- Optional targets handling: Decision made (all required)
- ROS2 testing: Decision made (manual only)
- JAX JIT + optional targets risk: N/A since all targets required
- API breakage risk: Addressed by keeping interface stable

---

## Work Objectives

### Core Objective
Create a clean, well-documented IK API that users can easily import and integrate into their own Python or ROS2 projects.

### Concrete Deliverables
- `teleop_xr/ik/__init__.py` with proper exports: `BaseRobot`, `PyrokiSolver`, `IKController`
- `teleop_xr/ros2/__main__.py` updated with `--mode ik` flag
- New tests in `tests/test_ik_*.py` for public API surface
- Type hints and docstrings on all public interfaces

### Definition of Done
- [x] `python -c "from teleop_xr.ik import BaseRobot, PyrokiSolver, IKController; print('ok')"` outputs "ok"
- [x] `pytest tests/test_ik_*.py` passes with new tests
- [x] `python -m teleop_xr.ros2 --help` shows `--mode` option
- [x] Existing demo functionality unchanged: `python -m teleop_xr.demo --mode ik` works

### Must Have
- Clean imports: `from teleop_xr.ik import BaseRobot, PyrokiSolver, IKController`
- ROS2 IK mode with `--mode ik` flag
- JointState subscriber for current robot state
- JointTrajectory publisher for IK solutions
- Backward compatibility with existing demo

### Must NOT Have (Guardrails)
- NO Pinocchio/JacobiRobot in the modularized API
- NO new robot implementations (keep only UnitreeH1Robot)
- NO optional target handling (all targets required)
- NO breaking changes to demo/__main__.py teleop mode
- NO ROS2 automated tests (manual testing only)
- NO custom ROS2 message types (use standard messages)
- NO action server interface

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (pytest with coverage)
- **User wants tests**: TDD for IK module, Manual for ROS2
- **Framework**: pytest

### TDD Workflow for IK Module

Each TODO follows RED-GREEN-REFACTOR:

**Task Structure:**
1. **RED**: Write failing test first
   - Test file: `tests/test_ik_*.py`
   - Test command: `pytest tests/test_ik_api.py -v`
   - Expected: FAIL (test exists, implementation doesn't)
2. **GREEN**: Implement minimum code to pass
   - Command: `pytest tests/test_ik_api.py -v`
   - Expected: PASS
3. **REFACTOR**: Clean up while keeping green
   - Command: `pytest tests/test_ik_api.py -v`
   - Expected: PASS (still)

### ROS2 Verification (Manual)
ROS2 IK mode is tested manually. Verification procedure:
```bash
# In a ROS2-sourced terminal:
python -m teleop_xr.ros2 --mode ik --help
# Verify: --mode option exists

python -m teleop_xr.ros2 --mode ik &
# Verify: Node starts without errors
# Verify: ros2 topic list shows /joint_trajectory
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Create IK module public API exports
└── Task 3: Add comprehensive docstrings/type hints

Wave 2 (After Wave 1):
├── Task 2: Add TDD tests for public API
└── Task 4: Add ROS2 --mode ik flag

Wave 3 (After Wave 2):
└── Task 5: Add ROS2 IK worker with JointState/JointTrajectory
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2 | 3 |
| 2 | 1 | 4 | - |
| 3 | None | - | 1 |
| 4 | 2 | 5 | - |
| 5 | 4 | None | - |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 3 | `delegate_task(category="quick", load_skills=[], run_in_background=true)` |
| 2 | 2, 4 | `delegate_task(category="quick", load_skills=[], run_in_background=false)` |
| 3 | 5 | `delegate_task(category="unspecified-low", load_skills=[], run_in_background=false)` |

---

## TODOs

- [x] 1. Create IK Module Public API Exports

  **What to do**:
  - Update `teleop_xr/ik/__init__.py` to export: `BaseRobot`, `PyrokiSolver`, `IKController`
  - Add `__all__` list to explicitly declare public API
  - Ensure clean import: `from teleop_xr.ik import BaseRobot, PyrokiSolver, IKController`

  **Must NOT do**:
  - Export internal/private classes
  - Change any class implementations
  - Add new dependencies

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple file modification, single location, minimal logic
  - **Skills**: `[]`
    - No special skills needed for this simple task

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 3)
  - **Blocks**: Task 2
  - **Blocked By**: None

  **References**:
  - `teleop_xr/ik/__init__.py:1-2` - Current empty init file to modify
  - `teleop_xr/ik/robot.py:9` - BaseRobot class definition to export
  - `teleop_xr/ik/solver.py:9` - PyrokiSolver class definition to export
  - `teleop_xr/ik/controller.py:9` - IKController class definition to export
  - `teleop_xr/__init__.py:1-50` - Example of package __init__ with exports

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  python -c "from teleop_xr.ik import BaseRobot, PyrokiSolver, IKController; print('ok')"
  # Assert: Output is exactly "ok"
  # Assert: Exit code 0

  python -c "from teleop_xr.ik import __all__; print(__all__)"
  # Assert: Output contains "BaseRobot", "PyrokiSolver", "IKController"
  ```

  **Commit**: YES
  - Message: `feat(ik): add public API exports to teleop_xr.ik module`
  - Files: `teleop_xr/ik/__init__.py`
  - Pre-commit: `python -c "from teleop_xr.ik import BaseRobot, PyrokiSolver, IKController"`

---

- [x] 2. Add TDD Tests for Public API Surface

  **What to do**:
  - Create `tests/test_ik_api.py` with tests for public imports
  - Test that BaseRobot is an ABC with required abstract methods
  - Test that PyrokiSolver can be instantiated with a mock robot
  - Test that IKController can be instantiated with a mock robot
  - Follow TDD: write failing tests first, then verify they pass with existing code

  **Must NOT do**:
  - Modify any implementation code
  - Add ROS2-related tests
  - Test internal/private methods

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Writing tests for existing code, clear pattern from existing test files
  - **Skills**: `[]`
    - Existing test patterns in codebase are sufficient

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after Wave 1)
  - **Blocks**: Task 4
  - **Blocked By**: Task 1

  **References**:
  - `tests/test_ik_controller.py:1-150` - Existing IK controller tests (patterns to follow)
  - `tests/test_pyroki_solver.py:1-20` - Existing solver tests (mock robot pattern)
  - `teleop_xr/ik/robot.py:9-66` - BaseRobot ABC definition (test abstract methods exist)

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  pytest tests/test_ik_api.py -v
  # Assert: Exit code 0
  # Assert: Output contains "passed"
  # Assert: Output contains "test_public_imports"
  # Assert: Output contains "test_base_robot_is_abc"
  ```

  **Commit**: YES
  - Message: `test(ik): add tests for public API surface`
  - Files: `tests/test_ik_api.py`
  - Pre-commit: `pytest tests/test_ik_api.py -v`

---

- [x] 3. Add Comprehensive Docstrings and Type Hints

  **What to do**:
  - Add module-level docstring to `teleop_xr/ik/__init__.py` explaining usage
  - Verify all public methods in BaseRobot, PyrokiSolver, IKController have docstrings
  - Ensure type hints are complete on all public method signatures
  - Add usage examples in module docstring

  **Must NOT do**:
  - Change any method implementations
  - Add new methods or classes
  - Modify private/internal methods

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Documentation-only changes, no logic changes
  - **Skills**: `[]`
    - Standard Python documentation

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: None
  - **Blocked By**: None

  **References**:
  - `teleop_xr/ik/robot.py:9-66` - BaseRobot class (verify docstrings)
  - `teleop_xr/ik/solver.py:9-112` - PyrokiSolver class (verify docstrings)
  - `teleop_xr/ik/controller.py:9-176` - IKController class (verify docstrings)
  - `README.md:1-50` - Overall project documentation style

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  python -c "from teleop_xr.ik import BaseRobot; print(BaseRobot.__doc__[:50] if BaseRobot.__doc__ else 'NO DOC')"
  # Assert: Output is NOT "NO DOC"
  # Assert: Output contains meaningful description

  python -c "from teleop_xr import ik; print(ik.__doc__[:50] if ik.__doc__ else 'NO DOC')"
  # Assert: Output is NOT "NO DOC"
  # Assert: Output contains "IK" or "Inverse Kinematics"
  ```

  **Commit**: YES
  - Message: `docs(ik): add comprehensive docstrings to public IK API`
  - Files: `teleop_xr/ik/__init__.py`, `teleop_xr/ik/robot.py`, `teleop_xr/ik/solver.py`, `teleop_xr/ik/controller.py`
  - Pre-commit: `python -c "from teleop_xr.ik import BaseRobot; assert BaseRobot.__doc__"`

---

- [x] 4. Add ROS2 --mode ik Flag

  **What to do**:
  - Add `mode: Literal["teleop", "ik"]` field to `Ros2CLI` dataclass in `teleop_xr/ros2/__main__.py`
  - Default to "teleop" for backward compatibility
  - Add conditional logic in `main()` to handle IK mode (placeholder for Task 5)
  - Follow pattern from `teleop_xr/demo/__main__.py:46` for mode handling

  **Must NOT do**:
  - Implement actual IK logic (that's Task 5)
  - Break existing teleop mode functionality
  - Add ROS2-specific tests

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple CLI flag addition, clear pattern from demo
  - **Skills**: `[]`
    - Standard Python/tyro CLI pattern

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (sequential after Task 2)
  - **Blocks**: Task 5
  - **Blocked By**: Task 2

  **References**:
  - `teleop_xr/ros2/__main__.py:165-181` - Ros2CLI dataclass to modify
  - `teleop_xr/demo/__main__.py:42-48` - DemoCLI mode pattern to follow
  - `teleop_xr/demo/__main__.py:475-481` - Mode conditional logic pattern

  **Acceptance Criteria**:

  ```bash
  # Agent runs:
  python -c "from teleop_xr.ros2.__main__ import Ros2CLI; print(hasattr(Ros2CLI, '__dataclass_fields__') and 'mode' in Ros2CLI.__dataclass_fields__)"
  # Assert: Output is "True"

  python -c "from teleop_xr.ros2.__main__ import Ros2CLI; print(Ros2CLI.__dataclass_fields__['mode'].default)"
  # Assert: Output is "teleop"
  ```

  **Commit**: YES
  - Message: `feat(ros2): add --mode flag to ROS2 node (teleop/ik)`
  - Files: `teleop_xr/ros2/__main__.py`
  - Pre-commit: `python -c "from teleop_xr.ros2.__main__ import Ros2CLI"`

---

- [x] 5. Implement ROS2 IK Worker with JointState/JointTrajectory

  **What to do**:
  - Import IK components: `UnitreeH1Robot`, `PyrokiSolver`, `IKController`
  - In IK mode, initialize robot, solver, and controller (following demo pattern)
  - Add JointState subscriber on `/joint_states` to get current robot state
  - Add JointTrajectory publisher on `/joint_trajectory` for IK solutions
  - Create IKWorker thread (following `demo/__main__.py:334` pattern)
  - Map joint names between ROS message and robot model

  **Must NOT do**:
  - Add new robot implementations
  - Change teleop mode behavior
  - Add automated ROS2 tests
  - Use custom message types

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Moderate complexity, integrating multiple components
  - **Skills**: `[]`
    - ROS2 patterns visible in existing code

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (final task)
  - **Blocks**: None
  - **Blocked By**: Task 4

  **References**:
  - `teleop_xr/demo/__main__.py:334-416` - IKWorker thread pattern to follow
  - `teleop_xr/demo/__main__.py:475-481` - Robot/solver/controller initialization
  - `teleop_xr/ros2/__main__.py:220-227` - ROS2 publisher creation pattern
  - `teleop_xr/ros2/__main__.py:307-360` - XR state callback pattern
  - `teleop_xr/ik/robots/h1_2.py:57` - `self.robot.joints.actuated_names` for joint name mapping

  **Acceptance Criteria**:

  ```bash
  # Agent runs (basic import check, no ROS2 required):
  python -c "from teleop_xr.ros2.__main__ import Ros2CLI; c = Ros2CLI(); print(c.mode)"
  # Assert: Output is "teleop"

  # Manual verification (ROS2 environment):
  # 1. Source ROS2 workspace
  # 2. Run: python -m teleop_xr.ros2 --mode ik
  # 3. Verify: ros2 topic list shows /joint_trajectory
  # 4. Verify: ros2 topic list shows subscription to /joint_states
  # 5. Verify: No errors in console output
  ```

  **Evidence to Capture:**
  - [ ] Terminal output from `python -c` command
  - [ ] Screenshot or log of manual ROS2 verification

  **Commit**: YES
  - Message: `feat(ros2): implement IK mode with JointState/JointTrajectory integration`
  - Files: `teleop_xr/ros2/__main__.py`
  - Pre-commit: `python -c "from teleop_xr.ros2.__main__ import main"`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(ik): add public API exports to teleop_xr.ik module` | `teleop_xr/ik/__init__.py` | `python -c "from teleop_xr.ik import BaseRobot"` |
| 2 | `test(ik): add tests for public API surface` | `tests/test_ik_api.py` | `pytest tests/test_ik_api.py` |
| 3 | `docs(ik): add comprehensive docstrings to public IK API` | `teleop_xr/ik/*.py` | `python -c "from teleop_xr.ik import BaseRobot; assert BaseRobot.__doc__"` |
| 4 | `feat(ros2): add --mode flag to ROS2 node (teleop/ik)` | `teleop_xr/ros2/__main__.py` | `python -c "from teleop_xr.ros2.__main__ import Ros2CLI"` |
| 5 | `feat(ros2): implement IK mode with JointState/JointTrajectory integration` | `teleop_xr/ros2/__main__.py` | Manual ROS2 verification |

---

## Success Criteria

### Verification Commands
```bash
# Public API imports work
python -c "from teleop_xr.ik import BaseRobot, PyrokiSolver, IKController; print('ok')"
# Expected: ok

# Tests pass
pytest tests/test_ik_api.py tests/test_ik_controller.py tests/test_pyroki_solver.py -v
# Expected: All tests pass

# ROS2 mode flag exists
python -c "from teleop_xr.ros2.__main__ import Ros2CLI; print(Ros2CLI.__dataclass_fields__['mode'].default)"
# Expected: teleop

# Existing demo still works
python -m teleop_xr.demo --help
# Expected: Shows --mode option with teleop/ik choices
```

### Final Checklist
- [x] All "Must Have" present:
- [x] Clean imports from teleop_xr.ik
- [x] ROS2 --mode ik flag
- [x] JointState subscriber
- [x] JointTrajectory publisher
- [x] Backward compatibility with demo
- [x] All "Must NOT Have" absent:
- [x] No Pinocchio/JacobiRobot exports
- [x] No new robot implementations
- [x] No optional target handling
- [x] No breaking changes to demo teleop mode
- [x] No ROS2 automated tests
- [x] No custom ROS2 message types
- [x] All tests pass
