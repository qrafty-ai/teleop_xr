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

## Inverse Kinematics (IK) API

TeleopXR provides a modularized IK stack for mapping XR device poses to robot joint configurations. This is useful for controlling humanoid or multi-arm robots.

### Core Components

*   `BaseRobot`: An abstract base class for defining your robot's kinematic model and optimization costs.
*   `PyrokiSolver`: A high-performance, JAX-powered IK solver that optimizes for target poses.
*   `IKController`: A high-level controller that manages teleoperation state, "deadman switch" logic, and coordinate snapshots.

### IK Integration Example

The following example shows how to use the IK API with a custom robot model (inheriting from `BaseRobot`):

```python
import numpy as np
import jax.numpy as jnp
from teleop_xr import Teleop, TeleopSettings
from teleop_xr.ik import IKController, PyrokiSolver, BaseRobot
from teleop_xr.messages import XRState

# 1. Define or use a robot model
# See teleop_xr.ik.robots.h1_2.UnitreeH1Robot for a complete implementation
class MyRobot(BaseRobot):
    # Implement abstract methods: get_vis_config, joint_var_cls,
    # forward_kinematics, get_default_config, build_costs
    ...

robot = MyRobot()
solver = PyrokiSolver(robot)
controller = IKController(robot, solver)

# Initialize current joint state
q_current = np.array(robot.get_default_config())

def on_xr_update(pose: np.ndarray, info: dict):
    global q_current

    # Validate raw state
    state = XRState.model_validate(info)

    # Calculate new joint configuration
    # IKController handles relative motion snapshots and deadman logic automatically
    q_new = controller.step(state, q_current)

    # Update state and send to robot
    q_current = q_new
    print(f"New Joint Command: {q_current}")

# 2. Setup Teleop
teleop = Teleop(TeleopSettings(input_mode="controller"))
teleop.subscribe(on_xr_update)
teleop.run()
```

### API Reference (IK)

#### IKController
::: teleop_xr.ik.controller.IKController
    options:
        show_root_heading: false

#### PyrokiSolver
::: teleop_xr.ik.solver.PyrokiSolver
    options:
        show_root_heading: false

#### BaseRobot
::: teleop_xr.ik.robot.BaseRobot
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
