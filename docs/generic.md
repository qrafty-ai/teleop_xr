# Generic Python API Guide

TeleopXR provides a pure Python API for integrating WebXR teleoperation into
your own projects. This guide covers how to retrieve raw state data, handle
button events, visualize robot models, and utilize the modular IK stack.

## 1. Raw XRState Value Retrieval

To receive real-time data from the XR headset, use the `Teleop` class. The
state is delivered as a dictionary, which can be validated into a structured
`XRState` object for easy access.

```python
import numpy as np
from teleop_xr import Teleop, TeleopSettings
from teleop_xr.messages import XRState

def on_xr_update(pose: np.ndarray, info: dict):
    """
    Args:
        pose: 4x4 matrix representing the primary end-effector target.
        info: Raw dictionary containing all device poses and inputs.
    """
    try:
        # Convert dictionary to structured Pydantic model
        state = XRState.model_validate(info)

        for device in state.devices:
            print(f"Device: {device.role}")
            print(f"Position: {device.pose.position}")

            if device.gamepad:
                print(f"Buttons: {len(device.gamepad.buttons)}")
                print(f"Axes: {device.gamepad.axes}")
    except Exception as e:
        print(f"Error parsing state: {e}")

# Configure and start
settings = TeleopSettings(port=4443, input_mode="controller")
teleop = Teleop(settings=settings)
teleop.subscribe(on_xr_update)
teleop.run()
```

## 2. Button Event System

TeleopXR includes an `EventProcessor` that detects complex interactions like
double-presses and long-presses, abstracting away the raw boolean polling.

```python
from teleop_xr.events import EventProcessor, EventSettings, XRButton, ButtonEventType

# 1. Initialize Processor
event_settings = EventSettings(
    double_press_threshold_ms=300,
    long_press_threshold_ms=500
)
event_processor = EventProcessor(event_settings)

# 2. Register Callbacks
def on_trigger_down(event):
    print(f"Trigger pressed on {event.controller}!")

def on_primary_double_press(event):
    print("A/X button double-pressed!")

event_processor.on_button_down(button=XRButton.TRIGGER, callback=on_trigger_down)
event_processor.on_double_press(button=XRButton.BUTTON_PRIMARY, callback=on_primary_double_press)

# 3. Integrate with Teleop
teleop = Teleop(TeleopSettings())
teleop.subscribe(event_processor.process)
teleop.run()
```

## 3. Robot Model Visualization

You can stream a URDF-based robot model to the WebXR frontend for real-time
visualization. This allows the operator to see a "ghost" or "digital twin" of
the robot in VR.

```python
from teleop_xr import Teleop, TeleopSettings
from teleop_xr.config import RobotVisConfig

# Configure the robot model
vis_config = RobotVisConfig(
    urdf_path="path/to/my_robot.urdf",
    mesh_path="path/to/meshes",  # Optional: folder containing STLs/OBJs
    model_scale=1.0,
    initial_rotation_euler=[0, 0, 0]
)

settings = TeleopSettings(robot_vis=vis_config)
teleop = Teleop(settings=settings)

# To update the visualizer's joint positions:
# Use teleop.publish_joint_state(joint_positions_dict)
teleop.run()
```

## 4. Camera & Video Streaming

TeleopXR can transmit live camera streams from your robot or workstation into
the WebXR scene. Define `VideoStreamConfig` objects for each camera and pass
them inside `TeleopSettings.video_config` so the frontend automatically
subscribes to the tracks.

```python
from teleop_xr import Teleop, TeleopSettings
from teleop_xr.video_stream import VideoStreamConfig

stream_configs = [
    VideoStreamConfig(id="head", device=0, width=1280, height=720, fps=30),
    VideoStreamConfig(id="hand", device=1, width=1280, height=720, fps=30),
]

settings = TeleopSettings(
    video_config={"streams": [cfg.model_dump() for cfg in stream_configs]}
)
teleop = Teleop(settings=settings)
teleop.run()
```

If you need to feed frames from ROS topics, file playback, or other sources,
subclass `ExternalVideoSource` or call `put_frame` on it directly to inject
numpy arrays into the stream manager.

## 5. IK Control Stack

The modular IK stack maps 6DoF XR poses and explicit end-effector commands to
robot joint configurations using a JAX-powered optimizer.

### Modular Components

- **`BaseRobot`**: Abstract class where you define your robot's kinematics and
  cost functions.
- **`PyrokiSolver`**: High-performance solver that executes the optimization.
- **`IKController`**: Owns teleop engagement, mode switching, relative-motion
  snapshots, and explicit end-effector command execution.

### Control Modes

`IKController` exposes three control modes:

- **`teleop`**: The default live XR path. `controller.step(...)` applies the
  deadman rule and relative-motion teleoperation snapshots.
- **`ee_delta`**: Accepts explicit relative end-effector deltas through
  `controller.submit_ee_delta(...)`.
- **`ee_absolute`**: Accepts explicit absolute end-effector targets through
  `controller.submit_ee_absolute(...)`.

When `Teleop.bind_control_mode_provider(...)` is connected to the controller,
non-`teleop` modes intentionally block live XR teleop input so scripted
commands do not race with headset updates.

### Teleop Example

```python
import numpy as np
from teleop_xr.ik import IKController, PyrokiSolver, BaseRobot

class MyRobotModel(BaseRobot):
    # Implement the abstract interface for your specific URDF
    ...

robot = MyRobotModel()
solver = PyrokiSolver(robot)
controller = IKController(robot, solver)

# Initialize state
q_current = robot.get_default_config()

def teleop_loop(xr_state):
    global q_current

    q_new = controller.step(xr_state, q_current)

    q_current = q_new
    send_to_hardware(q_current)
```

### Explicit Command Example

```python
import numpy as np
from teleop_xr.ik import (
    ControlMode,
    EEAbsoluteCommand,
    EEDeltaCommand,
    IKController,
)

controller.set_mode(ControlMode.EE_DELTA)
q_current = controller.submit_ee_delta(
    EEDeltaCommand(
        frame="right",
        delta_pose={
            "position": {"x": 0.02, "y": 0.0, "z": 0.0},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
    ),
    q_current,
)

controller.set_mode(ControlMode.EE_ABSOLUTE)
q_current = controller.submit_ee_absolute(
    EEAbsoluteCommand(
        frame="right",
        target_pose={
            "position": {"x": 0.45, "y": -0.10, "z": 0.80},
            "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
    ),
    q_current,
)

controller.set_mode(ControlMode.TELEOP)
```

For a full reference of classes and methods, please refer to the [API Reference](api.md).
