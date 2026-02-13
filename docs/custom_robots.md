# Custom Robot Support

TeleopXR supports dynamic loading of custom robot models. For robust management
of robot descriptions (URDF/Xacro) and their dependencies, we use the
**Robot Asset Manager (RAM)**.

## 1. Robot Asset Manager (RAM)

RAM simplifies robot integration by automatically cloning repositories,
processing Xacro files, and resolving asset paths. It ensures that the IK
solver has absolute paths to meshes while the WebXR frontend receives relative
paths dynamically rewritten by the visualization server.

### Initializing from URDF String (ROS 2)

When integrating with ROS 2, the robot's description is often provided as a
string from the `/robot_description` topic. Use `ram.from_string()` to save this
content to a temporary file, resolve any `package://` URIs, and automatically
detect the common mesh directory.

#### Example: Implementation with unified URDF support

```python
from teleop_xr import ram
from teleop_xr.ik.robot import BaseRobot, RobotVisConfig
```

...

When using the ROS2 interface or the demo, you can specify your custom robot:

- `--robot-class`: The robot specification. Can be an entry point name (e.g.,
  `franka`, `h1`) or a full module path (`my_package.robots:MyRobot`).
- `--robot-args`: A JSON string of arguments passed to the robot constructor.
- `--list-robots`: Lists all registered robots.

Example:

```bash
python -m teleop_xr.ros2 --mode ik --robot-class "franka"
```

## 2. BaseRobot Contract

Custom robot classes must inherit from `teleop_xr.ik.robot.BaseRobot` and support
the following constructor signature:

```python
def __init__(self, urdf_string: str | None = None, **kwargs):
    ...
```

- **`urdf_string`**: If provided (e.g., via ROS2 `/robot_description`), the
  robot should prioritize initializing from this string. If `None`, it should
  fallback to RAM or local files.

## 3. CLI Arguments

When using the ROS2 interface or the demo, you can specify your custom robot:

- **`--robot-class`**: The robot specification. Can be an entry point name
  (e.g., `franka`, `h1`) or a full module path (`my_package.robots:MyRobot`).
- **`--robot-args`**: A JSON string of arguments passed to the robot
  constructor.
- **`--list-robots`**: Lists all registered robots.

Example:

```bash
python -m teleop_xr.ros2 --mode ik --robot-class "franka"
```

## 4. Entry Points

Register your robot in `pyproject.toml` to make it discoverable by name:

```toml
[project.entry-points."teleop_xr.robots"]
my-robot = "my_package.robots:MyRobot"
```

## 5. Sphere Collision Support

For advanced robots with complex geometries, TeleopXR supports sphere-based
collision checking. This provides superior performance and differentiable
signed distance fields for IK optimization.

See the [**Sphere Collision Guide**](./sphere_collision.md) for details on how
to generate sphere decompositions and integrate them into your robot
implementation.
