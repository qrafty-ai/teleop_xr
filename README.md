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

4.  **🔌 Generic API & ROS2 Interface**
    Flexible architecture designed for easy integration. Use the provided ROS2 interface or the generic Python API to connect with your own robot projects.

## Quick Start

The easiest way to use TeleopXR is via pip:

```bash
pip install teleop-xr
```

Start the server:

```bash
python -m teleop_xr.basic
```

Then, open the displayed URL (e.g., `https://<your-ip>:4433`) in your headset's browser.

> **Note:** Ensure your headset and computer are on the same network.

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
    uv run python -m teleop_xr.basic
    ```

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
