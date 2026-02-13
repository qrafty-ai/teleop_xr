
## RAM from_string Implementation
- Implemented `ram.from_string(urdf_content, cache_dir)` to handle URDF content provided as a string.
- Integrated `ament_index_python` for `package://` resolution as a fallback when not in a RAM repo context.
- Added auto-detection for `mesh_path` by finding the common path of all resolved or absolute mesh paths in the URDF.
- Implemented generic path filtering (/, /opt, /usr, /home) to prevent overly broad `mesh_path` detection.
- Used content hashing for deterministic output URDF filenames in the cache.

## ROS2 Parameter Migration
- Migrated from `tyro` CLI parsing to standard ROS2 parameters using `TeleopNode(Node)` subclass.
- Parameters are declared in `__init__` with default values and accessed via properties.
- JSON parameters (`extra_streams_json`, `robot_args_json`) are parsed automatically by properties, providing a cleaner API for the rest of the node.
- Standard ROS2 parameter overrides via CLI (`--ros-args -p mode:=ik`) are now supported natively.
- Tests use mocking for `rclpy` to allow verification in environments without a full ROS2 installation.

## BaseRobot Refactor
- Moved URDF loading infrastructure (`_load_urdf`, `_load_default_urdf`) into `BaseRobot` to eliminate boilerplate in subclasses.
- Concrete `_load_urdf` supports optional `urdf_string` override, using `ram.from_string` to cache and resolve meshes.
- Lifted `get_vis_config` to the base class, making it concrete and driven by `urdf_path`, `mesh_path`, `model_scale`, and `orientation`.
- Introduced `model_scale` property in `BaseRobot` (defaulting to 1.0) to allow subclasses to specify visualization scale.
- Unified `RobotVisConfig.initial_rotation_euler` generation using `self.orientation.as_rpy_radians()`.
- Refactoring `BaseRobot` to include more concrete logic reduces the burden on robot-specific subclasses and ensures a consistent interface for visualization.

## Robot Subclass Standardization
- All robot subclasses (`UnitreeH1Robot`, `FrankaRobot`, `TeaArmRobot`, `OpenArmRobot`) now follow the new `BaseRobot` contract.
- Standardized `__init__(self, urdf_string: str | None = None, **kwargs: Any)` signature across all subclasses.
- Implementation of `_load_default_urdf(self)` for each robot, encapsulating their specific RAM-based loading logic (resource URLs, xacro args, etc.).
- Boilerplate `get_vis_config` removed from all subclasses, leveraging the concrete implementation in `BaseRobot`.
- `model_scale` is now overridden in subclasses (H1=0.5, Franka=0.5, TeaArm/OpenArm=1.0) instead of being hardcoded in `get_vis_config`.
- Standardized use of `@override` decorator for all methods and properties implementing or extending the base contract.

## Task 5: Wire URDF Override into ROS2 Node
- Verified that `teleop_xr/ros2/__main__.py` already had the logic to fetch URDF from topic and pass it to the robot constructor via `urdf_string` keyword argument.
- Confirmed that `TeleopSettings` is populated using `robot.get_vis_config()`, which now correctly handles URDF overrides by serving them from the RAM cache.
- Added `tests/test_ros2_urdf_topic.py` to verify the full integration flow from ROS2 topic to WebXR settings.
- Fixed regressions in existing tests (`tests/test_ik_controller.py`, `tests/test_controller_frames.py`) where mock robots were missing the newly required abstract method `_load_default_urdf`.
- Resolved test isolation issues in ROS2 mocking by using `importlib.reload` to ensure clean state when mocking `rclpy`.

### Integration & Cleanup
- Successfully removed `--list-robots` and `tyro` dependency from `teleop_xr/ros2/__main__.py`.
- Verified that `get_vis_config` overrides are removed from individual robot classes in `teleop_xr/ik/robots/`, relying on the base implementation in `BaseRobot`.
- Confirmed that the ROS2 node correctly uses parameter-driven configuration for robot loading.
- Full test suite passed (237 tests).

## ROS2 Testing without Environment
- To test ROS2-dependent code in a non-ROS environment, patch `sys.modules` with mocks before importing the modules that depend on rclpy.
- These late imports trigger Ruff E402 (Module level import not at top of file).
- Use `# noqa: E402` to suppress these errors since the late import is intentional and necessary.
