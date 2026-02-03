# TeleopXR

**TeleopXR** transforms your VR/AR headset into a powerful, precise robot controller. It provides a lightweight, installation-free teleoperation interface with low-latency video streaming and full WebXR state tracking.

## Key Features

1.  **🚀 Installation-Free Experience**
    Simply open the URL in your headset's browser (Meta Quest, Vision Pro, etc.). No app installation or APK sideloading required.

2.  **🎮 Full-Featured WebXR Streaming**
    Streams complete state data in real-time, including:
    *   6DoF Controller & Hand Poses
    *   Joystick Buttons & Axes
    *   Grip & Trigger Values

3.  **🎥 WebRTC Video Streaming**
    Built-in low-latency WebRTC video streaming for real-time visual feedback from your robot.

4.  **🔌 Modular IK & ROS2 Support**
    High-performance Inverse Kinematics (IK) solver with native ROS2 integration for seamless robot control.

## 🚀 Quick Start (Demo)

Use the built-in demo to verify connectivity and visualize the XR state data in real-time.

```bash
pip install teleop-xr
python -m teleop_xr.demo
```

### Modes

The demo supports two operation modes:

*   **Teleop Mode (Default)**: Visualizes raw XR state data and button events.
    ```bash
    python -m teleop_xr.demo --mode teleop
    ```
*   **IK Mode**: Enables the high-performance IK solver (configured for Unitree H1 by default).
    ```bash
    python -m teleop_xr.demo --mode ik
    ```

### Custom Robot Support

TeleopXR supports dynamic loading of custom robot models. For robust management of robot descriptions (URDF/Xacro) and their dependencies, we use the **Robot Asset Manager (RAM)**.

#### 1. Robot Asset Manager (RAM)

RAM simplifies robot integration by automatically cloning repositories, processing Xacro files, and resolving asset paths. It ensures that the IK solver has absolute paths to meshes while the WebXR frontend receives the standard `package://` URIs.

**Example: Implementation using RAM**
The `FrankaRobot` implementation uses RAM to fetch the official description:

```python
from teleop_xr import ram
from teleop_xr.ik.robot import BaseRobot, RobotVisConfig

class FrankaRobot(BaseRobot):
    def __init__(self, urdf_string: str | None = None):
        if urdf_string:
            # Initialize from ROS2 provided string
            ...
        else:
            repo_url = "https://github.com/frankarobotics/franka_ros.git"
            xacro_path = "franka_description/robots/panda/panda.urdf.xacro"

            # 1. Get resolved URDF for local IK (absolute mesh paths)
            self.urdf_path = ram.get_resource(
                repo_url=repo_url,
                path_inside_repo=xacro_path,
                xacro_args={"hand": "true"},
                resolve_packages=True
            )

            # 2. Get unresolved URDF for WebXR (package:// paths)
            self.vis_urdf_path = ram.get_resource(
                repo_url=repo_url,
                path_inside_repo=xacro_path,
                xacro_args={"hand": "true"},
                resolve_packages=False
            )

            # 3. Get repo root to serve meshes
            self.mesh_path = ram.get_repo(repo_url)

    def get_vis_config(self):
        return RobotVisConfig(
            urdf_path=self.vis_urdf_path,
            mesh_path=self.mesh_path
        )
```

#### 2. Robot Constructor Contract

Custom robot classes must inherit from `teleop_xr.ik.robot.BaseRobot` and support the following constructor signature:

```python
def __init__(self, urdf_string: str | None = None, **kwargs):
    ...
```

*   **`urdf_string`**: If provided (e.g., via ROS2 `/robot_description`), the robot should prioritize initializing from this string. If `None`, it should fallback to RAM or local files.

#### 3. CLI Arguments

When using the ROS2 interface or the demo, you can specify your custom robot:

- **`--robot-class`**: The robot specification. Can be an entry point name (e.g., `franka`, `h1`) or a full module path (`my_package.robots:MyRobot`).
- **`--robot-args`**: A JSON string of arguments passed to the robot constructor.
- **`--list-robots`**: Lists all registered robots.

Example:
```bash
python -m teleop_xr.ros2 --mode ik --robot-class "franka"
```

#### 4. Entry Points

Register your robot in `pyproject.toml` to make it discoverable by name:

```toml
[project.entry-points."teleop_xr.robots"]
my-robot = "my_package.robots:MyRobot"
```

### Usage

1. Open the displayed URL (`https://<ip>:4443`) in your headset.
2. Enter VR mode.
3. Observe the live state data and event logs in your terminal.

## 📖 Documentation

For detailed guides on integrating TeleopXR into your own projects, including the **Generic Python API** and **ROS2 Interface**, please visit our official documentation website:

👉 **[https://qrafty-ai.github.io/teleop_xr/generic/](https://qrafty-ai.github.io/teleop_xr/generic/)**

## Development

For developers contributing to TeleopXR or customizing the frontend:

### Prerequisites
*   [uv](https://github.com/astral-sh/uv) (for Python dependency management)
*   Node.js & npm (for WebXR frontend)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/qrafty-ai/teleop_xr.git
    cd teleop_xr
    ```

2.  **Install Python dependencies:**
    ```bash
    uv sync
    ```

3.  **Build the WebXR frontend:**
    ```bash
    cd webxr
    npm install
    npm run build
    ```
    *(The build output will be used by the Python server)*

4.  **Run from source:**
    ```bash
    # From the root directory
    uv run python -m teleop_xr.demo
    ```

## Acknowledgments

This project is forked from [SpesRobotics/teleop](https://github.com/SpesRobotics/teleop). We are grateful for their foundational work in creating a WebXR-based teleoperation solution.

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](https://github.com/qrafty-ai/teleop_xr/blob/main/LICENSE) file for details.
