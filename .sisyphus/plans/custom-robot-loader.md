# Custom Robot Loader for ROS2 Interface

## TL;DR

> **Quick Summary**: Implement a plugin system allowing users to load custom robot classes in the ROS2 interface via CLI arguments, with support for both explicit module paths and entry-point discovery. Robots can declare which frames they support (left/right/head), and URDF loading defaults to the ROS2 `/robot_description` topic.
>
> **Deliverables**:
> - `teleop_xr/ik/loader.py` - Robot class discovery and loading module
> - Enhanced `BaseRobot` with `supported_frames` property
> - Modified `IKController` and `PyrokiSolver` for optional frame handling
> - Updated `ros2/__main__.py` with `--robot-class`, `--robot-args`, `--list-robots`, `--urdf-topic`, `--no-urdf-topic` CLI
> - URDF topic fetching (default ON for IK mode, passes `urdf_string` to robot constructor)
> - Entry points configuration in `pyproject.toml`
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 (loader) -> Task 3 (BaseRobot) -> Task 4 (controller/solver) -> Task 6 (CLI integration)

---

## Context

### Original Request
Implement a system to load user-customized robot classes while still allowing users to use `teleop_xr/ros2/__main__.py` as a ROS node directly. For ROS2, the default URDF loading method should use the `/robot_description` topic.

### Interview Summary
**Key Discussions**:
- Hybrid plugin approach: module path strings (`my_pkg:MyRobot`) + entry points for registered robots
- Flexible frames: robots declare supported frames; missing frames gracefully skipped with warning
- Robot args via CLI: `--robot-args '{"param": "value"}'` JSON format
- Loader returns class, caller instantiates (matches current pattern)
- **URDF loading strategy**: CLI fetches from `/robot_description` topic, passes `urdf_string` to robot constructor
- **URDF default behavior**: Default ON for `--mode ik` - try topic first, fall back to robot's built-in URDF

**Research Findings**:
- `importlib.metadata.entry_points()` is the standard Python plugin discovery mechanism
- Current `BaseRobot` ABC has 5 abstract methods; well-defined contract
- `UnitreeH1Robot` loads URDF from file path; needs ROS2 topic adaptation
- ROS2 commonly uses `/robot_description` topic (std_msgs/String) or parameter

### Metis Review
**Identified Gaps** (addressed):
- URDF source ambiguity: Resolved to use `/robot_description` topic with timeout + clear error
- `--robot-args` encoding: JSON string format with validation
- Missing frames policy: Warn + skip by default, log clearly
- Plugin precedence: explicit `--robot-class` > entry point name > default built-in
- Edge cases: Duplicate entry point names, import failures, timeout handling

---

## Work Objectives

### Core Objective
Enable users to provide custom robot implementations to the ROS2 teleop interface without modifying the core codebase, using either explicit module paths or registered entry points.

### Concrete Deliverables
- `teleop_xr/ik/loader.py` - New module for robot discovery and loading
- `teleop_xr/ik/robot.py` - Enhanced with `supported_frames` property
- `teleop_xr/ik/controller.py` - Adapted for optional frame handling
- `teleop_xr/ik/solver.py` - Adapted `build_costs` signature for optional frames
- `teleop_xr/ros2/__main__.py` - New CLI arguments and robot instantiation logic
- `pyproject.toml` - Entry points configuration

### Definition of Done
- [ ] `python -m teleop_xr.ros2 --list-robots` shows available robots
- [ ] `python -m teleop_xr.ros2 --mode ik --robot-class my_pkg:MyRobot` loads custom robot
- [ ] `python -m teleop_xr.ros2 --mode ik` still works with default H1 robot (backwards compat)
- [ ] Robot with `supported_frames={'left'}` works without crash
- [ ] Invalid `--robot-class` produces clear error message
- [ ] `--mode ik` attempts to fetch URDF from `/robot_description` topic by default
- [ ] `--no-urdf-topic` flag disables URDF topic fetching
- [ ] URDF topic timeout produces warning (not crash) and falls back to built-in

### Must Have
- Backwards compatibility: existing usage without `--robot-class` works unchanged
- Clear error messages for all failure modes (import error, not a BaseRobot, etc.)
- Default timeout for URDF topic retrieval with actionable error
- `supported_frames` defaults to `{"left", "right", "head"}` for existing robots

### Must NOT Have (Guardrails)
- **NO** plugin version negotiation or dependency resolution
- **NO** hot reload of robot classes
- **NO** GUI selectors or complex config file formats
- **NO** full ROS2 URDF infrastructure (xacro processing, TF validation, mesh resolving)
- **NO** broad refactors of IK modules beyond what's needed for optional frames
- **NO** changes to WebXR frontend or video streaming

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (pytest configured in pyproject.toml)
- **User wants tests**: YES (TDD for loader, unit tests for frame handling)
- **Framework**: pytest

### TDD Workflow
Each TODO follows RED-GREEN-REFACTOR where applicable:
1. **RED**: Write failing test first
2. **GREEN**: Implement minimum code to pass
3. **REFACTOR**: Clean up while keeping green

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Create loader.py with discovery + loading
└── Task 2: Add entry points to pyproject.toml

Wave 2 (After Wave 1):
├── Task 3: Enhance BaseRobot with supported_frames
├── Task 4: Adapt IKController for optional frames
└── Task 5: Adapt PyrokiSolver for optional frames

Wave 3 (After Wave 2):
└── Task 6: Integrate loader into ros2/__main__.py CLI

Wave 4 (After Wave 3):
└── Task 7: Integration tests and documentation
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 6 | 2 |
| 2 | None | 6 | 1 |
| 3 | None | 4, 5 | 1, 2 |
| 4 | 3 | 6 | 5 |
| 5 | 3 | 6 | 4 |
| 6 | 1, 2, 4, 5 | 7 | None |
| 7 | 6 | None | None |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1, 2 | quick (straightforward Python/config) |
| 2 | 3, 4, 5 | unspecified-low (incremental changes) |
| 3 | 6 | unspecified-high (integration complexity) |
| 4 | 7 | quick (tests + docs) |

---

## TODOs

- [x] 1. Create Robot Loader Module
- [x] 2. Configure Entry Points in pyproject.toml
- [x] 3. Enhance BaseRobot with supported_frames Property
- [x] 4. Adapt IKController for Optional Frame Handling
- [x] 5. Adapt PyrokiSolver for Optional Frame Targets
- [x] 6. Integrate Robot Loader into ROS2 CLI
- [x] 7. Integration Tests and Documentation

  **What to do**:
  - Create integration test that:
    - Defines a minimal custom robot class
    - Loads it via `--robot-class`
    - Verifies it initializes correctly
  - Update README or add docs section about custom robots
  - Add example custom robot in `examples/` or docs

  **Must NOT do**:
  - Create extensive documentation (keep minimal)
  - Add dependencies for docs

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward test + docs
  - **Skills**: [`crafting-effective-readmes`]

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (final)
  - **Blocks**: None
  - **Blocked By**: Task 6

  **References**:

  **Pattern References**:
  - `tests/` - Existing test structure
  - `README.md` - Existing documentation style

  **Documentation References**:
  - `teleop_xr/ik/robots/h1_2.py` - Reference robot implementation

  **Acceptance Criteria**:

  **Automated Verification**:
  ```bash
  # Integration tests pass
  pytest tests/test_integration_custom_robot.py -v
  # Assert: PASS

  # Docs mention custom robots
  grep -r "robot-class\|custom robot" README.md docs/ 2>/dev/null || grep -r "robot-class\|custom robot" README.md
  # Assert: At least one match
  ```

  **Commit**: YES
  - Message: `docs: add custom robot loading documentation and examples`
  - Files: `tests/test_integration_custom_robot.py`, `README.md` or `docs/`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1+2 | `feat(ik): add robot loader with discovery and validation` | loader.py, tests, pyproject.toml | pytest tests/test_loader.py |
| 3 | `feat(ik): add supported_frames property to BaseRobot` | robot.py | python -c import test |
| 4 | `feat(ik): support optional frames in IKController` | controller.py, tests | pytest tests/test_controller_frames.py |
| 5 | `feat(ik): support optional targets in PyrokiSolver` | solver.py, tests | pytest tests/test_solver_optional.py |
| 6 | `feat(ros2): add --robot-class and --robot-args CLI options` | ros2/__main__.py | python -m teleop_xr.ros2 --list-robots |
| 7 | `docs: add custom robot loading documentation and examples` | tests, docs | pytest + grep |

---

## Success Criteria

### Verification Commands
```bash
# Full test suite passes
pytest tests/ -v

# List robots works
python -m teleop_xr.ros2 --list-robots

# Default mode still works (backwards compat)
python -m teleop_xr.ros2 --mode ik --help

# Custom robot class syntax accepted
python -m teleop_xr.ros2 --mode ik --robot-class teleop_xr.ik.robots.h1_2:UnitreeH1Robot --help
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass
- [ ] Backwards compatibility verified
- [ ] Error messages are clear and actionable
