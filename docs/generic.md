# Generic Python API

TeleopXR provides a pure Python API for integrating WebXR teleoperation into non-ROS projects. This allows you to receive head, controller, and hand tracking data directly in your Python application and stream video back to the headset.

## Basic Usage

The core of the API is the `Teleop` class. Here is a minimal example:

```python
from teleop_xr import Teleop
import numpy as np

def on_pose_update(pose: np.ndarray, info: dict):
    """
    Callback for pose updates.

    Args:
        pose: 4x4 transformation matrix (numpy array) of the target pose.
        info: Dictionary containing full XR state (devices, inputs, etc.).
    """
    print(f"Target Pose:\n{pose}")

    # Access raw device data
    xr_data = info.get("devices", [])
    for device in xr_data:
        print(f"Device: {device.get('role')} - {device.get('handedness')}")

# Initialize the server
teleop = Teleop(port=4433)

# Subscribe to updates
teleop.subscribe(on_pose_update)

# Run the server (blocking)
teleop.run()
```

## API Reference

### `Teleop` Class

```python
class Teleop(
    host="0.0.0.0",
    port=4443,
    natural_phone_orientation_euler=None,
    natural_phone_position=None,
    input_mode="controller",
    camera_views=None,
    video_sources=None
)
```

#### Arguments

*   **`host`** *(str)*: Host IP address to bind to. Defaults to `"0.0.0.0"`.
*   **`port`** *(int)*: Port number. Defaults to `4443`.
*   **`input_mode`** *(str)*: Input source mode. Options: `"controller"`, `"hand"`, `"auto"`. Defaults to `"controller"`.
*   **`video_sources`** *(dict)*: Dictionary mapping source IDs to `VideoSource` objects for WebRTC streaming.

#### Methods

*   **`subscribe(callback)`**
    Registers a callback function to be called when new XR data is received.
    *   `callback`: A function with signature `(pose: np.ndarray, info: dict) -> None`.

*   **`run()`**
    Starts the asyncio event loop and runs the server. This method is blocking.

*   **`set_video_streams(payload)`**
    Updates the video stream configuration dynamically.

## Data Structure

The `info` dictionary passed to the callback contains the raw WebXR state:

```json
{
  "devices": [
    {
      "role": "head",
      "pose": { ... }
    },
    {
      "role": "controller",
      "handedness": "right",
      "gripPose": { ... },
      "gamepad": {
        "buttons": [ ... ],
        "axes": [ ... ]
      }
    }
  ]
}
```

## Video Streaming

To stream video from Python to the headset:

```python
from teleop_xr import Teleop
from teleop_xr.video_stream import ExternalVideoSource

# Create a video source
source = ExternalVideoSource()

# Pass it to Teleop
teleop = Teleop(video_sources={"main_camera": source})

# In your capture loop:
# frame = capture_image() # Get a BGR numpy array (OpenCV format)
# source.put_frame(frame)
```
