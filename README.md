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
