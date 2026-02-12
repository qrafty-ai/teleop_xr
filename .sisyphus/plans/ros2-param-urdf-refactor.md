# ROS2 Node Parameter Refactor + URDF Override Uniformity

## TL;DR

> **Quick Summary**: Replace tyro CLI configuration in the ROS2 node with standard ROS2 parameters (`declare_parameter`/`get_parameter`), and unify URDF override logic across all robot classes by lifting common patterns into `BaseRobot` and extending RAM with a `from_string()` function.
>
> **Deliverables**:
> - `TeleopNode(Node)` subclass with ROS2 parameter declarations replacing `Ros2CLI` tyro dataclass
> - `ram.from_string()` function that saves URDF strings to cache files with auto-detected mesh_path
> - Refactored `BaseRobot` with URDF loading/vis_config in base class
> - All 4 robot subclasses updated (H1, Franka, TeaArm, OpenArm)
> - TDD tests for all changes
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 (ram.from_string) → Task 2 (BaseRobot) → Task 3 (robot subclasses) → Task 5 (ROS2 node) → Task 6 (integration)

---

## Context

### Original Request
Refactor the ROS2 node to use ROS2 standard parameters instead of tyro CLI, and make URDF overriding logic more automatic and uniform across all robot classes.

### Interview Summary
**Key Discussions**:
- Node architecture: User chose `Node` subclass over functional `create_node()` style
- `list_robots` action: Remove entirely (was CLI-only, not a runtime parameter)
- `extra_streams` dict: Use JSON string parameter (simple, consistent)
- URDF handling: Extend RAM with `from_string()` to save URDF strings to cache, auto-detect mesh_path from content
- BaseRobot: Full lift of URDF loading and `get_vis_config()` into base class
- Test strategy: TDD (red-green-refactor)
- Demo node: OUT OF SCOPE (stays on tyro, not a ROS2 node)

**Research Findings**:
- Zero `declare_parameter` usage in codebase — all config is via tyro
- `UnitreeH1Robot` violates the loader contract (no `urdf_string` param in `__init__`)
- URDF loading boilerplate duplicated 3x across robots; `get_vis_config()` duplicated 4x
- ROS2 standard: `declare_parameter()` in `__init__`, dot-separated naming, YAML config files

### Metis Review
**Identified Gaps** (addressed):
- Parameter naming scheme needed → Applied: flat ROS2 params, no dot-prefix (simple node, no namespacing needed)
- Backward compatibility stance → Applied: **hard break** — this is a branch feature, no backward compat needed
- Dynamic vs static params → Applied: all parameters are **startup-only** (read once in `__init__`)
- Error handling policy → Applied: invalid JSON → warn + empty dict fallback; invalid URDF → fail fast; missing mesh packages → warn + `mesh_path=None`
- `mesh_path` edge cases (no meshes, multiple packages, too-generic prefix) → Applied: return `None` for mesh_path when no meshes or prefix is too generic (`/`, `/opt`, `/usr`)
- Cache semantics for `ram.from_string()` → Applied: content-hash filename, deterministic, idempotent writes
- URDF override precedence → Applied: topic URDF > robot default (already the pattern, now formalized)

---

## Work Objectives

### Core Objective
Convert the ROS2 node from tyro CLI-driven to ROS2 parameter-driven configuration, and eliminate URDF handling code duplication across robot classes.

### Concrete Deliverables
- `teleop_xr/ros2/__main__.py` — rewritten with `TeleopNode(Node)` subclass
- `teleop_xr/ram.py` — new `from_string()` function added
- `teleop_xr/ik/robot.py` — `BaseRobot` with URDF loading infrastructure
- `teleop_xr/ik/robots/h1_2.py` — updated to new contract
- `teleop_xr/ik/robots/franka.py` — updated to new contract
- `teleop_xr/ik/robots/teaarm.py` — updated to new contract
- `teleop_xr/ik/robots/openarm.py` — updated to new contract
- `teleop_xr/common_cli.py` — retained for demo, but decoupled from ROS2 node
- Test files for each component

### Definition of Done
- [x] `python -m pytest tests/` → all tests pass (including new TDD tests)
- [x] ROS2 node starts with: `python -m teleop_xr.ros2 --ros-args -p mode:=ik -p robot_class:=h1`
- [x] All 4 robot classes accept `urdf_string` in constructor and produce valid `urdf_path`/`mesh_path`
- [x] Demo node still works unchanged: `python -m teleop_xr.demo`
- [x] No tyro dependency remaining in `teleop_xr/ros2/__main__.py`
...
  **Acceptance Criteria**:

  - [x] `python -m pytest tests/ -q` → PASS (exit code 0), all tests green
  - [x] `grep -r "tyro" teleop_xr/ros2/` → no matches
  - [x] `grep -r "get_vis_config" teleop_xr/ik/robots/` → no matches (all removed from subclasses)
  - [x] `python -c "from teleop_xr.demo.__main__ import main; print('OK')"` → OK
  - [x] `python -c "from teleop_xr.ros2.__main__ import TeleopNode; print('OK')"` → OK
...
### Final Checklist
- [x] All "Must Have" present
- [x] All "Must NOT Have" absent
- [x] All tests pass
- [x] No duplicated URDF loading boilerplate across robot classes
- [x] No duplicated get_vis_config() across robot classes
- [x] H1 accepts urdf_string in constructor
- [x] ROS2 node uses only declare_parameter/get_parameter for config
