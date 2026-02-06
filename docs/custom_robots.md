# Custom Robot Support

TeleopXR supports dynamic loading of custom robot models. For robust management of robot descriptions (URDF/Xacro) and their dependencies, we use the **Robot Asset Manager (RAM)**.

## 1. Robot Asset Manager (RAM)

RAM simplifies robot integration by automatically cloning repositories, processing Xacro files, and resolving asset paths. It ensures that the IK solver has absolute paths to meshes while the WebXR frontend receives relative paths dynamically rewritten by the visualization server.

**Example: Implementation using RAM**
The `FrankaRobot` implementation uses RAM to fetch the official description:

```python
from teleop_xr import ram
from teleop_xr.ik.robot import BaseRobot, RobotVisConfig

class FrankaRobot(BaseRobot):
    def __init__(self, urdf_string: str | None = None):
        self.mesh_path = None
        if urdf_string:
            # Initialize from ROS2 provided string
            ...
        else:
            repo_url = "https://github.com/frankarobotics/franka_ros.git"
            xacro_path = "franka_description/robots/panda/panda.urdf.xacro"

            # Get resolved URDF for local IK (absolute mesh paths)
            # WebXR visualization automatically rewrites these to relative paths
            self.urdf_path = ram.get_resource(
                repo_url=repo_url,
                path_inside_repo=xacro_path,
                xacro_args={"hand": "true"},
                resolve_packages=True
            )

            # Get repo root to serve meshes
            self.mesh_path = ram.get_repo(repo_url)

    def get_vis_config(self):
        return RobotVisConfig(
            urdf_path=self.urdf_path,
            mesh_path=self.mesh_path
        )
```

## 2. Robot Constructor Contract

Custom robot classes must inherit from `teleop_xr.ik.robot.BaseRobot` and support the following constructor signature:

```python
def __init__(self, urdf_string: str | None = None, **kwargs):
    ...
```

*   **`urdf_string`**: If provided (e.g., via ROS2 `/robot_description`), the robot should prioritize initializing from this string. If `None`, it should fallback to RAM or local files.

## 3. CLI Arguments

When using the ROS2 interface or the demo, you can specify your custom robot:

- **`--robot-class`**: The robot specification. Can be an entry point name (e.g., `franka`, `h1`) or a full module path (`my_package.robots:MyRobot`).
- **`--robot-args`**: A JSON string of arguments passed to the robot constructor.
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

## 5. Advanced Collision Support (Spheres)

TeleopXR uses sphere-based collision checking for high-performance, differentiable inverse kinematics. While basic capsule collision is supported from URDF, sphere decomposition offers better accuracy for complex geometries.

### Workflow

1.  **Generate Collision Data**: Use the interactive tool to generate sphere decompositions and calculate collision ignore pairs.
    ```bash
    python scripts/configure_sphere_collision.py --robot-class "my_robot"
    ```
2.  **Configure Spheres**:
    *   **Allocation**: Set target number of spheres.
    *   **Spherize**: Run the decomposition algorithm.
    *   **Refine**: Optimize sphere positions.

    ![Sphere Generation](./assets/collision_sphere.jpg)

3.  **Calculate Ignore Pairs**:
    *   Go to the **Collision** tab.
    *   Adjust **Settings** (Samples, Threshold) if needed.
    *   Click **Calculate Ignore Pairs** to identify links that never collide or are always colliding (structural).
    *   Use **Manual Overrides** to disable specific links if necessary.

    ![Ignore Pair Calculation](./assets/collision_ignore.jpg)

4.  **Export**: Save to `collision.json` in your robot's asset folder.

### Robot Implementation

Update your robot class to load the collision data:

```python
# ... imports ...
import pyroki as pk

class MyRobot(BaseRobot):
    def __init__(self, ...):
        # ... setup URDF ...

        # Load collision data
        collision_data = self._load_collision_data()
        if collision_data is not None:
            spheres, ignore_pairs = collision_data
            self.robot_coll = pk.collision.RobotCollision.from_sphere_decomposition(
                spheres,
                urdf,
                user_ignore_pairs=ignore_pairs,
                ignore_immediate_adjacents=True,
            )
        else:
            self.robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    @staticmethod
    def _load_collision_data():
        """Helper to load collision.json or fallback to sphere.json"""
        import os, json
        # Adjust path to your asset directory
        asset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "my_robot")

        collision_path = os.path.join(asset_dir, "collision.json")
        if os.path.exists(collision_path):
            with open(collision_path, "r") as f:
                data = json.load(f)
            spheres = data["spheres"]
            ignore_pairs = tuple(tuple(p) for p in data.get("collision_ignore_pairs", []))
            return spheres, ignore_pairs

        return None
```
