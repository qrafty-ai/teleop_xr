# Generic Robot Description Override

## TL;DR

> **Quick Summary**: Refactor robot description handling so that `BaseRobot` owns a `description` property (returning a file path or URDF string), provides a `set_description()` method for runtime overrides that reinitializes URDF-dependent state, and all subclasses + the ROS2 entrypoint adapt to this unified pattern.
>
> **Deliverables**:
> - `teleop_xr/ik/robot.py` - Abstract `description` property + `set_description()` method + `_init_from_description()` hook
> - `teleop_xr/ik/robots/h1_2.py` - Adapted to implement `description` property and use `_init_from_description()`
> - `teleop_xr/ik/robots/franka.py` - Same adaptation, remove `urdf_string` constructor arg
> - `teleop_xr/ik/robots/teaarm.py` - Same adaptation, remove `urdf_string` constructor arg
> - `teleop_xr/ros2/__main__.py` - Use `robot.set_description(urdf_string)` instead of constructor injection
> - Tests updated to use new API
>
> **Estimated Effort**: Medium
> **Critical Path**: Task 1 (BaseRobot) -> Task 2 (subclasses) -> Task 3 (ROS2 entrypoint) -> Task 4 (tests)

---

## Context

### Original Request
Make robot description a property in the robot base class where subclasses implement it. The return type should be either a file path or a URDF string. Create a method to override the description (string or path) and reinitialize all URDF-dependent variables. Revise the ROS2 entrypoint and all robot classes to adapt.

### Current State

**How descriptions are loaded today:**
- `UnitreeH1Robot.__init__()`: Calls `ram.get_resource()` to fetch URDF from a git repo, stores `self.urdf_path`, loads `yourdfpy.URDF`, builds `pk.Robot` and `pk.collision.RobotCollision`.
- `FrankaRobot.__init__(urdf_string=None)`: If `urdf_string` is given, loads from string; otherwise calls `ram.get_resource()` for default URDF.
- `TeaArmRobot.__init__(urdf_string=None)`: Same pattern as Franka — `urdf_string` or `ram.get_resource()`.
- `ros2/__main__.py`: Fetches URDF from `/robot_description` topic, injects it as `robot_args["urdf_string"]` into the robot constructor.

**Problems with current approach:**
1. No unified way to know what description a robot is using.
2. `urdf_string` constructor arg is ad-hoc — only Franka/TeaArm support it, H1 does not.
3. No way to override the description after construction (e.g., when a new URDF arrives on a topic).
4. URDF-dependent state (pyroki `Robot`, collision model, link indices, `urdf_path`, `mesh_path`) is initialized inline in `__init__` with no way to reinitialize.

### Design Decisions

1. **`description` as an abstract property on `BaseRobot`**: Subclasses implement this to return their default description source. Return type is `str` — either a file path or raw URDF XML string.

2. **`RobotDescription` dataclass**: Wraps the description string with a `kind` discriminator (`"path"` or `"urdf_string"`) so callers know how to interpret it unambiguously.

3. **`_init_from_description(description)` abstract method**: Each subclass implements this to (re)initialize all URDF-dependent state from a `RobotDescription`. Called in `__init__` and in `set_description()`.

4. **`set_description(description_or_str, kind=None)` method on `BaseRobot`**: Public API to override the description at runtime. Calls `_init_from_description()` to reinitialize. This replaces the `urdf_string` constructor parameter pattern.

5. **Constructor simplification**: Remove `urdf_string` from subclass constructors. Instead, `__init__` calls `self._init_from_description(self.description)` to load from the default source. Callers use `set_description()` post-construction to override.

6. **ROS2 entrypoint change**: Instead of `robot_args["urdf_string"] = urdf_string`, do `robot = robot_cls(**robot_args)` then `robot.set_description(urdf_string)`.

---

## Work Objectives

### Core Objective
Unify robot description management into a clean base-class pattern where description source is declarative, overridable at runtime, and reinitializes all dependent state.

### Definition of Done
- [ ] `BaseRobot` has abstract `description` property returning `RobotDescription`
- [ ] `BaseRobot` has `set_description()` method that overrides + reinitializes
- [ ] `BaseRobot` has abstract `_init_from_description()` for subclass-specific reinitialization
- [ ] All 3 subclasses (H1, Franka, TeaArm) implement the new pattern
- [ ] No subclass accepts `urdf_string` as a constructor arg
- [ ] `ros2/__main__.py` uses `set_description()` for topic-fetched URDFs
- [ ] All existing tests pass (adapted to new API)
- [ ] `lsp_diagnostics` clean on all modified files

### Must NOT Have (Guardrails)
- **NO** changes to the IK solver, controller, or cost functions
- **NO** changes to RAM module
- **NO** changes to WebXR frontend
- **NO** new dependencies
- **NO** breaking the `get_vis_config()` contract

---

## TODOs

### Task 1: Add description infrastructure to BaseRobot [DONE]

**What to do**:
- Add `RobotDescription` dataclass to `teleop_xr/ik/robot.py`:
  ```python
  @dataclass
  class RobotDescription:
      content: str  # file path or URDF XML string
      kind: Literal["path", "urdf_string"]
  ```
- Add abstract property `description` to `BaseRobot` returning `RobotDescription`
- Add abstract method `_init_from_description(self, description: RobotDescription) -> None`
- Add concrete method `set_description(self, content: str, kind: Literal["path", "urdf_string"] | None = None) -> None`:
  - If `kind` is None, auto-detect: if `content` starts with `<` or contains `<robot`, treat as `urdf_string`; else treat as `path`
  - Wraps into `RobotDescription`, calls `_init_from_description()`

**Must NOT do**:
- Change any existing abstract methods
- Touch subclasses yet

**Files**: `teleop_xr/ik/robot.py`

**Acceptance Criteria**:
```bash
python -c "from teleop_xr.ik.robot import BaseRobot, RobotDescription; print('OK')"
```

---

### Task 2: Adapt all robot subclasses [DONE]

**What to do**:

For each of `UnitreeH1Robot`, `FrankaRobot`, `TeaArmRobot`:

1. Implement `description` property returning `RobotDescription(content=<default_path>, kind="path")` pointing to the default URDF resolved via RAM (or the default loading strategy).
2. Extract all URDF-loading + pyroki initialization into `_init_from_description(self, description: RobotDescription)`:
   - If `description.kind == "path"`: load URDF from file path
   - If `description.kind == "urdf_string"`: load URDF from string via `io.StringIO`
   - Reinitialize: `self.robot`, `self.robot_coll` (if applicable), link indices, `self.urdf_path`, `self.mesh_path`
3. Simplify `__init__`:
   - Remove `urdf_string` parameter from Franka and TeaArm
   - `__init__` resolves the default description via RAM and stores it, then calls `self._init_from_description(self.description)`
   - Keep robot-specific non-URDF state (e.g., `self.leg_joint_names` for H1, `self.L_ee`/`self.R_ee` names, collision data loading for TeaArm)
4. Store `self._description_override: RobotDescription | None = None` so `description` property returns override when set.

**Key pattern** (example for H1):
```python
class UnitreeH1Robot(BaseRobot):
    def __init__(self) -> None:
        self._description_override: RobotDescription | None = None
        # Resolve default path via RAM
        self._default_urdf_path = str(ram.get_resource(...))
        self._default_mesh_path = os.path.dirname(self._default_urdf_path)
        # Robot-specific constants
        self.leg_joint_names = [...]
        self.L_ee = "L_hand_base_link"
        self.R_ee = "R_hand_base_link"
        # Initialize from default description
        self._init_from_description(self.description)

    @property
    def description(self) -> RobotDescription:
        if self._description_override is not None:
            return self._description_override
        return RobotDescription(content=self._default_urdf_path, kind="path")

    def _init_from_description(self, description: RobotDescription) -> None:
        if description.kind == "path":
            self.urdf_path = description.content
            self.mesh_path = os.path.dirname(description.content)
            urdf = yourdfpy.URDF.load(description.content)
        else:
            self.urdf_path = ""
            self.mesh_path = None
            urdf = yourdfpy.URDF.load(io.StringIO(description.content))

        # Freeze legs
        for joint_name in self.leg_joint_names:
            if joint_name in urdf.joint_map:
                urdf.joint_map[joint_name].type = "fixed"

        self.robot = pk.Robot.from_urdf(urdf)
        self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)
        self.L_ee_link_idx = self.robot.links.names.index(self.L_ee)
        self.R_ee_link_idx = self.robot.links.names.index(self.R_ee)
        self.torso_link_idx = self.robot.links.names.index("torso_link")
```

**Must NOT do**:
- Change the IK cost functions or forward kinematics logic
- Change link name strings or collision parameters
- Remove `_load_collision_data` from TeaArm

**Files**: `teleop_xr/ik/robots/h1_2.py`, `teleop_xr/ik/robots/franka.py`, `teleop_xr/ik/robots/teaarm.py`

**Acceptance Criteria**:
```bash
python -c "from teleop_xr.ik.robots.h1_2 import UnitreeH1Robot; r = UnitreeH1Robot(); print(r.description)"
```

---

### Task 3: Revise ROS2 entrypoint [DONE]

**What to do**:
- In `teleop_xr/ros2/__main__.py`, change the IK mode setup:
  - **Before**: Inject `urdf_string` into `robot_args` dict, pass to constructor
  - **After**: Construct robot normally, then call `robot.set_description(urdf_string)` if URDF was fetched from topic
- Remove `urdf_string` from the documented "Robot Constructor Contract" in `loader.py`
- Update loader docstring to reflect new contract: robots no longer need `urdf_string` kwarg

**Current code** (lines ~392-406):
```python
urdf_string = None
if not cli.no_urdf_topic:
    urdf_string = get_urdf_from_topic(node, cli.urdf_topic, cli.urdf_timeout)
if urdf_string:
    robot_args["urdf_string"] = urdf_string
robot = robot_cls(**robot_args)
```

**New code**:
```python
robot = robot_cls(**robot_args)

if not cli.no_urdf_topic:
    urdf_string = get_urdf_from_topic(node, cli.urdf_topic, cli.urdf_timeout)
    if urdf_string:
        robot.set_description(urdf_string)
```

**Files**: `teleop_xr/ros2/__main__.py`, `teleop_xr/ik/loader.py`

**Acceptance Criteria**:
- `python -c "from teleop_xr.ros2.__main__ import main"` doesn't crash
- The `urdf_string` injection pattern is completely gone

---

### Task 4: Update tests [DONE]

**What to do**:
- Update all tests that use `urdf_string=...` constructor arg to use `set_description()` instead
- Add tests for the new `set_description()` API:
  - Test auto-detection of kind (path vs string)
  - Test that `description` property returns override after `set_description()`
  - Test that `_init_from_description` reinitializes robot state
- Verify `test_integration_custom_robot.py` still passes (update the mock robot to implement new abstract methods)

**Files affected**:
- `tests/test_franka_robot.py`
- `tests/test_teaarm_coverage.py`
- `tests/test_teaarm_collision_loading.py`
- `tests/test_robots.py`
- `tests/test_integration_custom_robot.py`

**Acceptance Criteria**:
```bash
pytest tests/ -v
# Assert: all pass
```

---

## Execution Strategy

### Dependency Chain (Sequential)
```
Task 1 (BaseRobot infrastructure)
  -> Task 2 (Subclass adaptation)
    -> Task 3 (ROS2 entrypoint)
      -> Task 4 (Tests)
```

All tasks are sequential because each builds on the previous.

### Risk Mitigation
- **Risk**: Changing constructor signatures breaks downstream code
  - **Mitigation**: Search for all `urdf_string` usages (already done - only in tests and ros2 entrypoint)
- **Risk**: `_init_from_description` called before robot-specific constants set
  - **Mitigation**: Ensure `__init__` sets constants (link names, leg joints) BEFORE calling `_init_from_description()`

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 1 | `feat(ik): add RobotDescription and description override infrastructure to BaseRobot` | `robot.py` |
| 2 | `refactor(ik): adapt all robot subclasses to description property pattern` | `h1_2.py`, `franka.py`, `teaarm.py` |
| 3 | `refactor(ros2): use set_description() instead of urdf_string constructor injection` | `ros2/__main__.py`, `loader.py` |
| 4 | `test: update tests for new robot description API` | `tests/` |
