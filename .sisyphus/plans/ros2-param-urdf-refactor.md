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
- [ ] `python -m pytest tests/` → all tests pass (including new TDD tests)
- [ ] ROS2 node starts with: `python -m teleop_xr.ros2 --ros-args -p mode:=ik -p robot_class:=h1`
- [ ] All 4 robot classes accept `urdf_string` in constructor and produce valid `urdf_path`/`mesh_path`
- [ ] Demo node still works unchanged: `python -m teleop_xr.demo`
- [ ] No tyro dependency remaining in `teleop_xr/ros2/__main__.py`

### Must Have
- All ROS2 node config via `declare_parameter` / `get_parameter`
- `ram.from_string()` with auto mesh_path detection
- Uniform `BaseRobot` URDF handling
- H1 robot fixed to accept `urdf_string`
- TDD tests for each component

### Must NOT Have (Guardrails)
- ❌ Do NOT touch `teleop_xr/demo/__main__.py` — it stays on tyro
- ❌ Do NOT modify RAM git cloning, xacro processing, or dae→glb logic — only add `from_string()`
- ❌ Do NOT add dynamic parameter reconfiguration (callbacks) — all params are startup-only
- ❌ Do NOT add lifecycle node, composition, or QoS redesign — keep scope narrow
- ❌ Do NOT change topic names, message types, frame IDs, or any external ROS2 interface
- ❌ Do NOT change robot-specific IK logic (costs, FK, joint configs)
- ❌ Do NOT add `package://` resolution via `ament_index_python` as a hard dependency — use it opportunistically with graceful fallback
- ❌ Do NOT change visual scale values or orientation while moving `get_vis_config()` to base
- ❌ Do NOT add `list_robots` as a ROS2 service — just remove it cleanly

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.

### Test Decision
- **Infrastructure exists**: YES (pytest + pytest-cov, configured in pyproject.toml)
- **Automated tests**: TDD (red-green-refactor)
- **Framework**: pytest
- **Test command**: `python -m pytest tests/ -x -q`

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

**Verification Tool by Deliverable Type:**

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| Python module | Bash (pytest) | Run tests, assert pass |
| ROS2 node startup | Bash (python -m) | Start node, verify no crash |
| Import compatibility | Bash (python -c) | Import module, verify no error |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — independent foundations):
├── Task 1: ram.from_string() — no dependencies
└── Task 4: ROS2 Node TeleopNode class — no dependencies on robot changes

Wave 2 (After Wave 1):
├── Task 2: BaseRobot refactor — depends on Task 1 (ram.from_string)
└── (Task 4 may continue)

Wave 3 (After Task 2):
├── Task 3: Robot subclass updates — depends on Task 2 (BaseRobot)

Wave 4 (After Tasks 3 + 4):
├── Task 5: ROS2 node URDF integration — depends on Tasks 3 + 4
└── Task 6: Integration tests + cleanup — depends on Task 5

Critical Path: Task 1 → Task 2 → Task 3 → Task 5 → Task 6
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2 | 4 |
| 2 | 1 | 3 | 4 |
| 3 | 2 | 5 | 4 |
| 4 | None | 5 | 1, 2, 3 |
| 5 | 3, 4 | 6 | None |
| 6 | 5 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 4 | task(category="unspecified-high", load_skills=[], run_in_background=false) |
| 2 | 2 | task(category="unspecified-high", load_skills=[], run_in_background=false) |
| 3 | 3 | task(category="unspecified-high", load_skills=[], run_in_background=false) |
| 4 | 5, 6 | task(category="unspecified-high", load_skills=[], run_in_background=false) |

---

## TODOs

### ROS2 Parameter Schema

Before diving into tasks, here is the canonical mapping from tyro fields to ROS2 parameters:

| Old Tyro Field | ROS2 Param Name | Type | Default | Notes |
|---------------|----------------|------|---------|-------|
| `mode` | `mode` | string | `"teleop"` | `"teleop"` or `"ik"` |
| `host` | `host` | string | `"0.0.0.0"` | |
| `port` | `port` | int | `4443` | |
| `input_mode` | `input_mode" | string | `"controller"` | `"controller"`, `"hand"`, `"auto"` |
| `head_topic` | `head_topic` | string | `""` | Empty = disabled |
| `wrist_left_topic` | `wrist_left_topic` | string | `""` | Empty = disabled |
| `wrist_right_topic` | `wrist_right_topic` | string | `""` | Empty = disabled |
| `extra_streams` | `extra_streams_json` | string | `"{}"` | JSON `{"name": "/topic"}` |
| `frame_id` | `frame_id` | string | `"xr_local"` | |
| `publish_hand_tf` | `publish_hand_tf` | bool | `False` | |
| `robot_class` | `robot_class` | string | `""` | Empty = default H1 |
| `robot_args` | `robot_args_json` | string | `"{}"` | JSON for constructor kwargs |
| `urdf_topic` | `urdf_topic` | string | `"/robot_description"` | |
| `urdf_timeout` | `urdf_timeout` | double | `5.0` | |
| `no_urdf_topic` | `no_urdf_topic` | bool | `False` | |
| ~~`list_robots`~~ | — | — | — | **REMOVED** |
| ~~`ros_args`~~ | — | — | — | **REMOVED** (no longer needed) |

---

- [x] 1. Add `ram.from_string()` to Robot Asset Manager
...
- [x] 2. Refactor BaseRobot with URDF Loading Infrastructure
...
- [x] 3. Update All Robot Subclasses to New BaseRobot Contract
...
- [x] 4. Create TeleopNode(Node) with ROS2 Parameters
...
- [x] 5. Wire URDF Override into ROS2 Node
...
- [x] 6. Final Integration Tests + Cleanup

  **What to do**:
  - Verify full test suite passes: `python -m pytest tests/ -x -q`
  - Verify demo node still works: `python -c "from teleop_xr.demo.__main__ import main"`
  - Verify no tyro in ROS2 module: `grep tyro teleop_xr/ros2/__main__.py` → no matches
  - Verify all robot classes conform to contract:
    - All accept `urdf_string` in constructor
    - All have `_load_default_urdf()` method
    - None override `get_vis_config()` (inherited from base)
  - Clean up any unused imports across modified files
  - Remove `list_robots` from any help text or documentation references
  - Verify `pyproject.toml` — `tyro` dependency can stay (demo still uses it), but confirm no new deps needed

  **Must NOT do**:
  - Do NOT remove tyro from `pyproject.toml` dependencies (demo still needs it)
  - Do NOT modify any tests that weren't part of this refactor

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Verification and cleanup — no major implementation
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (final, sequential)
  - **Blocks**: None (final task)
  - **Blocked By**: Task 5

  **References**:

  **Pattern References**:
  - All modified files from Tasks 1-5

  **Acceptance Criteria**:

  - [ ] `python -m pytest tests/ -q` → PASS (exit code 0), all tests green
  - [ ] `grep -r "tyro" teleop_xr/ros2/` → no matches
  - [ ] `grep -r "get_vis_config" teleop_xr/ik/robots/` → no matches (all removed from subclasses)
  - [ ] `python -c "from teleop_xr.demo.__main__ import main; print('OK')"` → OK
  - [ ] `python -c "from teleop_xr.ros2.__main__ import TeleopNode; print('OK')"` → OK

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Complete regression suite
    Tool: Bash (pytest)
    Steps:
      1. python -m pytest tests/ -q --tb=short
      2. Assert: exit code 0
      3. Assert: "passed" in output, "failed" NOT in output
    Expected Result: All tests pass
    Evidence: Full pytest output captured

  Scenario: No tyro in ROS2 module
    Tool: Bash (grep)
    Steps:
      1. grep -rn "tyro" teleop_xr/ros2/
      2. Assert: exit code 1 (no matches)
    Expected Result: Zero tyro references in ROS2 module
    Evidence: grep output

  Scenario: URDF boilerplate eliminated from subclasses
    Tool: Bash (grep)
    Steps:
      1. grep -c "get_vis_config" teleop_xr/ik/robots/h1_2.py teleop_xr/ik/robots/franka.py teleop_xr/ik/robots/teaarm.py teleop_xr/ik/robots/openarm.py
      2. Assert: all counts are 0
    Expected Result: No get_vis_config in any robot subclass
    Evidence: grep output
  ```

  **Commit**: YES
  - Message: `chore: final cleanup and verification for ros2-param-urdf-refactor`
  - Files: any remaining cleanup
  - Pre-commit: `python -m pytest tests/ -q`

---

## Commit Strategy

| After Task | Message | Key Files | Verification |
|------------|---------|-----------|--------------|
| 1 | `feat(ram): add from_string() for URDF string caching with mesh_path auto-detection` | `ram.py`, `test_ram_from_string.py` | `pytest tests/test_ram_from_string.py` |
| 2 | `refactor(robot): lift URDF loading and get_vis_config into BaseRobot` | `robot.py`, `test_base_robot_urdf.py` | `pytest tests/test_ik_api.py tests/test_base_robot_urdf.py` |
| 3 | `refactor(robots): update all robot classes to unified BaseRobot URDF contract` | `h1_2.py`, `franka.py`, `teaarm.py`, `openarm.py` | `pytest tests/test_robots.py tests/test_franka_robot.py tests/test_openarm_robot.py` |
| 4 | `refactor(ros2): replace tyro CLI with ROS2 parameter-based TeleopNode` | `ros2/__main__.py`, `test_ros2_node_params.py` | `pytest tests/test_ros2_node_params.py` |
| 5 | `feat(ros2): wire URDF topic override through BaseRobot._load_urdf` | `ros2/__main__.py` | `pytest tests/` |
| 6 | `chore: final cleanup and verification for ros2-param-urdf-refactor` | misc | `pytest tests/` |

---

## Success Criteria

### Verification Commands
```bash
# Full test suite
python -m pytest tests/ -q                          # Expected: all tests pass

# No tyro in ROS2 module
grep -r "tyro" teleop_xr/ros2/                     # Expected: no matches (exit 1)

# URDF boilerplate eliminated
grep -c "get_vis_config" teleop_xr/ik/robots/*.py   # Expected: all 0

# Demo still works
python -c "from teleop_xr.demo.__main__ import main" # Expected: no error

# ROS2 node importable
python -c "from teleop_xr.ros2.__main__ import TeleopNode"  # Expected: no error (may need ROS2 sourced)
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] No duplicated URDF loading boilerplate across robot classes
- [ ] No duplicated get_vis_config() across robot classes
- [ ] H1 accepts urdf_string in constructor
- [ ] ROS2 node uses only declare_parameter/get_parameter for config
