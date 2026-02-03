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

- [ ] 1. Create Robot Loader Module

  **What to do**:
  - Create `teleop_xr/ik/loader.py` with:
    - `RobotLoadError` custom exception class
    - `load_robot_class(robot_spec: str | None) -> type[BaseRobot]` function
    - `list_available_robots() -> dict[str, str]` function (name -> class path)
  - Loading precedence:
    1. If `robot_spec` is None, return `UnitreeH1Robot`
    2. If `robot_spec` matches an entry point name, load that
    3. If `robot_spec` contains `:`, parse as `module:ClassName`
    4. Otherwise, raise `RobotLoadError` with helpful message
  - Validation: loaded class must be a subclass of `BaseRobot`

  **Must NOT do**:
  - Import robot modules during `list_available_robots()` (metadata only)
  - Execute arbitrary code beyond import
  - Handle version negotiation or dependencies

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file creation with clear specification
  - **Skills**: [`git-master`]
    - `git-master`: For clean atomic commit

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Task 6
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `teleop_xr/ik/robots/__init__.py` - Current robot exports pattern
  - `teleop_xr/ik/robot.py:BaseRobot` - Interface to validate against

  **API/Type References**:
  - Python `importlib.metadata.entry_points(group="teleop_xr.robots")` - Entry point discovery
  - Python `importlib.import_module()` + `getattr()` - Dynamic loading

  **Acceptance Criteria**:

  **TDD Tests** (`tests/test_loader.py`):
  - [ ] Test file created: `tests/test_loader.py`
  - [ ] Test: `load_robot_class(None)` returns `UnitreeH1Robot`
  - [ ] Test: `load_robot_class("teleop_xr.ik.robots.h1_2:UnitreeH1Robot")` returns class
  - [ ] Test: `load_robot_class("nonexistent:Nope")` raises `RobotLoadError`
  - [ ] Test: `load_robot_class("teleop_xr.ik.loader:RobotLoadError")` raises (not a BaseRobot)
  - [ ] Test: `list_available_robots()` returns dict with at least "h1" entry
  - [ ] `pytest tests/test_loader.py -v` -> PASS

  **Commit**: YES
  - Message: `feat(ik): add robot loader with discovery and validation`
  - Files: `teleop_xr/ik/loader.py`, `tests/test_loader.py`
  - Pre-commit: `pytest tests/test_loader.py`

---

- [ ] 2. Configure Entry Points in pyproject.toml

  **What to do**:
  - Add `[project.entry-points."teleop_xr.robots"]` section
  - Register built-in robot: `h1 = "teleop_xr.ik.robots.h1_2:UnitreeH1Robot"`

  **Must NOT do**:
  - Change any other pyproject.toml sections
  - Add robots that don't exist

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single config file edit
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 6
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `pyproject.toml:28-53` - Existing project configuration structure

  **Documentation References**:
  - Python Packaging: entry_points specification

  **Acceptance Criteria**:

  **Automated Verification**:
  ```bash
  # Verify entry point is discoverable after editable install
  uv pip install -e .
  python -c "import importlib.metadata; eps = importlib.metadata.entry_points(group='teleop_xr.robots'); print([ep.name for ep in eps])"
  # Assert: Output contains 'h1'
  ```

  **Commit**: YES (group with Task 1)
  - Message: `feat(ik): add robot loader with discovery and validation`
  - Files: `pyproject.toml`

---

- [ ] 3. Enhance BaseRobot with supported_frames Property

  **What to do**:
  - Add `supported_frames` property to `BaseRobot` class (NOT abstract)
  - Default implementation returns `{"left", "right", "head"}`
  - Add docstring explaining override behavior

  **Must NOT do**:
  - Make it abstract (would break existing implementations)
  - Change any existing abstract methods

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small addition to existing class
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5)
  - **Blocks**: Tasks 4, 5
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `teleop_xr/ik/robot.py:11-79` - BaseRobot class definition

  **Acceptance Criteria**:

  **Automated Verification**:
  ```bash
  python -c "from teleop_xr.ik.robot import BaseRobot; print(BaseRobot.supported_frames.fget.__doc__)"
  # Assert: Docstring is printed (not None)

  python -c "from teleop_xr.ik.robots import UnitreeH1Robot; r = UnitreeH1Robot(); print(r.supported_frames)"
  # Assert: Output is {'left', 'right', 'head'} (set equality)
  ```

  **Commit**: YES
  - Message: `feat(ik): add supported_frames property to BaseRobot`
  - Files: `teleop_xr/ik/robot.py`

---

- [ ] 4. Adapt IKController for Optional Frame Handling

  **What to do**:
  - Modify `_get_device_poses()` to only extract poses for `robot.supported_frames`
  - Modify `step()` to check only supported frames in `has_all_poses`
  - Modify snapshot logic to only snapshot supported frames
  - Modify target computation to skip unsupported frames
  - Add warning log when frames are skipped due to missing support
  - Pass `None` for unsupported frame targets to solver

  **Must NOT do**:
  - Change the public API signature of `step()`
  - Remove support for full 3-frame robots
  - Add hard failure modes for missing frames

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Multiple related changes in one file
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 5)
  - **Blocks**: Task 6
  - **Blocked By**: Task 3

  **References**:

  **Pattern References**:
  - `teleop_xr/ik/controller.py:111-124` - `_get_device_poses()` method
  - `teleop_xr/ik/controller.py:156-232` - `step()` method
  - `teleop_xr/ik/controller.py:171-172` - Current hardcoded `required_keys`

  **Acceptance Criteria**:

  **Unit Test** (`tests/test_controller_frames.py`):
  - [ ] Test: Controller with robot having `supported_frames={'left'}` doesn't crash
  - [ ] Test: Controller only requests poses for supported frames
  - [ ] Test: Full 3-frame robot still works (backwards compat)
  - [ ] `pytest tests/test_controller_frames.py -v` -> PASS

  **Commit**: YES
  - Message: `feat(ik): support optional frames in IKController`
  - Files: `teleop_xr/ik/controller.py`, `tests/test_controller_frames.py`

---

- [ ] 5. Adapt PyrokiSolver for Optional Frame Targets

  **What to do**:
  - Modify `solve()` signature to accept `Optional[jaxlie.SE3]` for each target
  - Modify `_solve_internal()` to skip cost building for `None` targets
  - Update `build_costs()` call to conditionally include costs
  - Update warmup to handle optional targets

  **Must NOT do**:
  - Change the optimization algorithm
  - Add new dependencies
  - Break JIT compilation

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: JAX/JIT-aware changes require care
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 4)
  - **Blocks**: Task 6
  - **Blocked By**: Task 3

  **References**:

  **Pattern References**:
  - `teleop_xr/ik/solver.py:36-85` - `_solve_internal()` method
  - `teleop_xr/ik/solver.py:87-106` - `solve()` method
  - `teleop_xr/ik/robots/h1_2.py:96-146` - `build_costs()` example

  **API/Type References**:
  - `jaxlie.SE3` - Pose type
  - `jaxls.LeastSquaresProblem` - Optimization problem

  **Acceptance Criteria**:

  **Unit Test** (`tests/test_solver_optional.py`):
  - [ ] Test: `solve(target_L, None, None, q)` works (single arm)
  - [ ] Test: `solve(target_L, target_R, target_Head, q)` still works (full)
  - [ ] Test: JIT compilation succeeds with optional targets
  - [ ] `pytest tests/test_solver_optional.py -v` -> PASS

  **Commit**: YES
  - Message: `feat(ik): support optional targets in PyrokiSolver`
  - Files: `teleop_xr/ik/solver.py`, `tests/test_solver_optional.py`

---

- [ ] 6. Integrate Robot Loader into ROS2 CLI

  **What to do**:
  - Add to `Ros2CLI` dataclass:
    - `robot_class: str | None = None` with docstring
    - `robot_args: str | None = None` (JSON string) with docstring
    - `list_robots: bool = False` with docstring
    - `urdf_topic: str = "/robot_description"` with docstring
    - `urdf_timeout: float = 10.0` with docstring
    - `no_urdf_topic: bool = False` to disable topic fetching
  - Create `get_urdf_from_topic(node, topic, timeout) -> str | None` helper function:
    - Subscribe to topic (std_msgs/String)
    - Use `wait_for_message` pattern with timeout
    - Return URDF string or None if timeout
    - Log clear message on success/failure
  - In `main()` for `--mode ik`:
    - If `--list-robots`, print available robots and exit
    - Parse `--robot-args` as JSON dict (with error handling)
    - **URDF Loading (DEFAULT ON for IK mode)**:
      1. Unless `--no-urdf-topic`, attempt to fetch URDF from `--urdf-topic`
      2. If successful, add `urdf_string=<fetched>` to robot_args
      3. If timeout, log warning and proceed (robot uses built-in URDF)
    - Use `load_robot_class(cli.robot_class)` to get robot class
    - Instantiate robot: `robot = RobotCls(**robot_args)`
  - Update IKWorker and state_container initialization to use loaded robot

  **Robot Constructor Contract** (document in loader.py docstring):
  ```python
  # Custom robots MAY accept urdf_string parameter:
  class MyRobot(BaseRobot):
      def __init__(self, urdf_string: str | None = None, **kwargs):
          if urdf_string:
              urdf = yourdfpy.URDF.load_string(urdf_string)
          else:
              # Fall back to built-in/file-based URDF
              urdf = yourdfpy.URDF.load(self.default_urdf_path)
  ```

  **Must NOT do**:
  - Change teleop mode behavior
  - Modify video streaming logic
  - Add complex config file parsing
  - Process xacro files (raw URDF string only)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integration point with multiple dependencies
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (sequential)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 1, 2, 4, 5

  **References**:

  **Pattern References**:
  - `teleop_xr/ros2/__main__.py:172-191` - `Ros2CLI` dataclass
  - `teleop_xr/ros2/__main__.py:271-313` - Mode setup and robot init
  - `teleop_xr/common_cli.py` - CommonCLI base class

  **API/Type References**:
  - `std_msgs.msg.String` - ROS2 message type for robot_description
  - `teleop_xr.ik.loader.load_robot_class` - New loader function

  **Acceptance Criteria**:

  **Automated Verification**:
  ```bash
  # List robots
  python -m teleop_xr.ros2 --list-robots
  # Assert: Exit code 0
  # Assert: Output contains "h1"

  # Help shows new args (including URDF options)
  python -m teleop_xr.ros2 --help 2>&1 | grep -E "(robot-class|robot-args|list-robots|urdf-topic|urdf-timeout|no-urdf-topic)"
  # Assert: All options appear in help

  # Invalid robot class produces clear error
  python -m teleop_xr.ros2 --mode ik --robot-class "nonexistent:Nope" --no-urdf-topic 2>&1
  # Assert: Exit code non-zero
  # Assert: Error message contains "Cannot load" or similar

  # Invalid JSON in robot-args produces clear error
  python -m teleop_xr.ros2 --mode ik --robot-args "not-json" --no-urdf-topic 2>&1
  # Assert: Exit code non-zero
  # Assert: Error message mentions JSON parsing

  # URDF topic timeout produces warning (not error) and continues
  # (This requires ROS2 environment - mark as integration test)
  ```

  **ROS2 Integration Test** (requires sourced ROS2):
  ```bash
  # Test URDF topic fetching with mock publisher
  # In terminal 1: ros2 topic pub --once /robot_description std_msgs/msg/String "data: '<robot name=\"test\"><link name=\"base\"/></robot>'"
  # In terminal 2: python -m teleop_xr.ros2 --mode ik --urdf-timeout 5
  # Assert: Log shows "Loaded URDF from /robot_description"

  # Test timeout fallback (no publisher)
  # python -m teleop_xr.ros2 --mode ik --urdf-timeout 2
  # Assert: Log shows "URDF topic timeout" warning
  # Assert: Robot initializes with built-in URDF (for H1)
  ```

  **Commit**: YES
  - Message: `feat(ros2): add --robot-class, --robot-args, and URDF topic loading`
  - Files: `teleop_xr/ros2/__main__.py`

---

- [ ] 7. Integration Tests and Documentation

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
