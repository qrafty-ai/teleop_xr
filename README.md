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

## 🚀 Quick Start (Demo)

Use the built-in demo to verify connectivity and visualize the XR state data and button events in real-time.

```bash
pip install teleop-xr
python -m teleop_xr.demo
```

This will launch a server (default port 4443) and display a **Rich-based TUI** in your terminal with:
- **XR Device State**: Live table showing position, orientation, and inputs for Head, Controllers, and Hands.
- **Event Log**: Real-time display of button gestures (press, release, double-press, long-press).

### Usage

1. Open the displayed URL (https://<ip>:4443) in your headset.
2. Enter VR mode.
3. See the state table update with high-frequency data and the event log capture button interactions.

### CLI Options

```bash
# Customize event detection thresholds
python -m teleop_xr.demo --double-press-ms 250 --long-press-ms 800

# Disable event detection for a simpler view
python -m teleop_xr.demo --no-enable-events

# Configure video streaming from cameras
python -m teleop_xr.demo --head-device 0 --wrist-left-device 1
```

## 🤖 ROS2 Integration

TeleopXR comes with a fully functional ROS2 node wrapper.

### Running the Node

```bash
# Source your ROS2 workspace first
python -m teleop_xr.ros2 --frame-id xr_local
```

### Published Topics

The node publishes state data to the following topics:

| Topic | Type | Description |
| :--- | :--- | :--- |
| `xr/head/pose` | `PoseStamped` | Headset 6DoF pose |
| `xr/controller_{L/R}/pose` | `PoseStamped` | Controller grip pose |
| `xr/controller_{L/R}/joy` | `Joy` | Joystick axes and buttons |
| `xr/controller_{L/R}/joy_touched` | `Joy` | Touch states (capacitive) |
| `xr/hand_{L/R}/joints` | `PoseArray` | 25 hand joint poses (if tracking enabled) |
| `xr/fetch_latency_ms` | `Float64` | Network/Input latency stats |

*Note: All poses are published in the frame specified by `--frame-id` (default: `xr_local`).*

### Video Streaming Configuration

You can map ROS image topics to the VR headset's views using CLI arguments:

```bash
python -m teleop_xr.ros2 \
  --head-topic /camera/head/image_raw \
  --wrist-left-topic /camera/left/image_raw \
  --wrist-right-topic /camera/right/image_raw
```

The node automatically handles image transport and compression (requires `cv_bridge`).

## 🐍 Python API (Custom Integration)

For custom Python projects without ROS, use the `Teleop` class directly.

### Basic Usage

```python
import numpy as np
from teleop_xr import Teleop, TeleopSettings
from teleop_xr.messages import XRState

# 1. Configure Settings
settings = TeleopSettings(
    host="0.0.0.0",
    port=4433,
    input_mode="controller"  # or "hand", "auto"
)

# 2. Initialize Teleop
teleop = Teleop(settings=settings)

# 3. Define a Callback
def on_xr_update(pose: np.ndarray, xr_state_dict: dict):
    # 'pose' is the calculated end-effector pose (4x4 matrix) based on input mode
    # 'xr_state_dict' contains the raw device data

    # Optional: Validate with Pydantic for easier access
    try:
        state = XRState.model_validate(xr_state_dict)
        for device in state.devices:
            print(f"{device.role}: {device.pose.position}")
    except Exception:
        pass

# 4. Subscribe and Run
teleop.subscribe(on_xr_update)
teleop.run()
```

### Event System

TeleopXR includes an event system for handling complex button interactions like double-presses and long-presses.

```python
from teleop_xr import Teleop, TeleopSettings
from teleop_xr.events import EventProcessor, EventSettings, XRButton

# 1. Initialize Processor
event_processor = EventProcessor(EventSettings(
    double_press_threshold_ms=300,
    long_press_threshold_ms=500
))

# 2. Register Callbacks
def on_trigger_down(event):
    print(f"Trigger pressed on {event.controller} hand!")

def on_primary_double_press(event):
    print("Primary button double-pressed!")

# You can filter by button and/or controller
event_processor.on_button_down(button=XRButton.TRIGGER, callback=on_trigger_down)
event_processor.on_double_press(button=XRButton.BUTTON_PRIMARY, callback=on_primary_double_press)

# 3. Subscribe to Teleop
teleop = Teleop(TeleopSettings())
teleop.subscribe(event_processor.process)
teleop.run()
```

#### Available Event Types
*   `BUTTON_DOWN`: Fired when a button is first pressed.
*   `BUTTON_UP`: Fired when a button is released.
*   `DOUBLE_PRESS`: Fired on the second press of a double-click interaction.
*   `LONG_PRESS`: Fired once the button has been held for the configured threshold.

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

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.
