# Learnings: Custom Robot Loader Implementation

## Robot Loading Strategy
- The loader supports three ways to specify a robot:
    1. **Default (`None`)**: Defaults to `UnitreeH1Robot`.
    2. **Entry Points**: Uses `importlib.metadata` to discover robots registered under the `teleop_xr.robots` group in `pyproject.toml`.
    3. **Explicit Module Path**: Supports `module.path:ClassName` format for loading arbitrary robot classes.

## Precedence Rules
- The loader follows a strict precedence order:
    1. `None` check.
    2. Entry point name matching.
    3. Explicit `:` separator check.

## Validation & Safety
- All loaded classes are verified to be subclasses of `BaseRobot` using `issubclass()`.
- `RobotLoadError` is raised for all failure cases (missing module, missing class, invalid inheritance, etc.).
- `list_available_robots()` avoids importing modules by only reading entry point metadata, which is important for performance and avoiding side effects during discovery.

## Python Compatibility
- Uses `importlib.metadata.entry_points(group=...)` which is the modern standard for entry point discovery in Python 3.10+.
- Since `teleop_xr` requires Python >= 3.10, we can safely use the `EntryPoints.names` property.

## Optional Targets in IK Solver
- JAX/JIT handles `None` arguments by treating them as constants during tracing.
- If the number of targets is small (e.g., 3), recompilation for different combinations of `None` vs. `Not None` (8 combinations) is acceptable and provides a clean way to skip costs.
- Using `if target is None:` inside a JIT-compiled function correctly triggers branch pruning during tracing if the structure of the problem (list of costs) depends on it.

### IKController Optional Frame Handling
- Successfully decoupled IKController from hardcoded "left", "right", "head" requirements.
- Uses `robot.supported_frames` to determine which poses to extract from XRState and which snapshots to take.
- Throttled warnings (warn once) effectively inform about unsupported frames in XRState without flooding logs.
- Passing `None` for unsupported frames to the solver allows flexible robot configurations while maintaining the 3-frame API signature.
### Robot Loader Integration into ROS2 CLI
- Integrated `load_robot_class` and `list_available_robots` into `teleop_xr.ros2.__main__`.
- Added CLI arguments for dynamic robot loading: `--robot-class`, `--robot-args`, `--list-robots`.
- Implemented automatic URDF fetching from ROS2 topic (`/robot_description` by default) when in IK mode.
- URDF fetching uses `TRANSIENT_LOCAL` durability to capture latched messages.
- Updated `UnitreeH1Robot` and `BaseRobot` to support `urdf_string` and `actuated_joint_names`.
- Frontend visualization for robots loaded from URDF topic is disabled by default in this implementation (returns `None` for `get_vis_config`) as the local mesh paths are not known.


## Integration Testing for Custom Robots
- Successfully verified that robots can be loaded from strings using the `module:ClassName` format.
- Integration tests should define mock robot classes that inherit from `BaseRobot` to verify the loading and instantiation logic.
- Using `uv run pytest` ensures that dependencies like `jax` and `jaxlie` are available during testing.

- Encountered issues with pytest-cov missing in default environment; resolved by using `uv run pytest` which uses the project's managed dependencies.
- Pre-commit hooks auto-fixed formatting and linting issues in the new test files (ruff E701, unused imports, trailing whitespace).
- Encountered issues with pytest-cov missing in default environment; resolved by using `uv run pytest` which uses the project's managed dependencies.
- Pre-commit hooks auto-fixed formatting and linting issues in the new test files (ruff E701, unused imports, trailing whitespace).
