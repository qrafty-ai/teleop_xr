# Generic Python API

TeleopXR provides a pure Python API for integrating WebXR teleoperation into non-ROS projects. This allows you to receive head, controller, and hand tracking data directly in your Python application and stream video back to the headset.

## Basic Usage

The core of the API is the `Teleop` class. Here is a minimal example:

```python
from teleop_xr import Teleop, TeleopSettings
from teleop_xr.messages import XRState
import numpy as np

def on_pose_update(pose: np.ndarray, info: dict):
    """
    Callback for pose updates.

    Args:
        pose: 4x4 transformation matrix (numpy array) of the target pose.
        info: Dictionary containing full XR state (devices, inputs, etc.).
    """
    print(f"Target Pose:\n{pose}")

    # Optional: Use Pydantic model for cleaner access
    try:
        state = XRState.model_validate(info)
        for device in state.devices:
            print(f"Device: {device.role} - {device.handedness}")
    except Exception:
        pass

# 1. Configure Settings
settings = TeleopSettings(
    port=4433,
    input_mode="controller"
)

# 2. Initialize Teleop
teleop = Teleop(settings=settings)

# 3. Subscribe to updates
teleop.subscribe(on_pose_update)

# 4. Run the server (blocking)
teleop.run()
```

## API Reference

### TeleopSettings

::: teleop_xr.config.TeleopSettings
    options:
        show_root_heading: false

### Teleop Class

::: teleop_xr.Teleop
    options:
        show_root_heading: false

### Data Messages

::: teleop_xr.messages.XRState
    options:
        show_root_heading: false

## Video Streaming

To stream video from Python to the headset:

```python
from teleop_xr import Teleop, TeleopSettings
from teleop_xr.video_stream import ExternalVideoSource

# Create a video source
source = ExternalVideoSource()

# Configure settings with video source
# Note: video_config usage depends on advanced setup,
# typically we use the ROS2 wrapper or custom logic to feed the source.
# ...
```
