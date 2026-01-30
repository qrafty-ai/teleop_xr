# Inverse Kinematics (IK) System

TeleopXR includes a powerful Inverse Kinematics system based on [Mink](https://github.com/kevinzakka/mink), a MuJoCo-based IK solver. This system allows you to map WebXR controller or hand poses directly to robot joint configurations in real-time.

## Features

- **MuJoCo-Powered**: High-performance IK solving using MuJoCo's physics engine.
- **Support for URDF and MJCF**: Load robot models in both standard formats.
- **Multi-Task Solving**: Jointly optimize for multiple end-effector targets and posture regularization.
- **Collision Avoidance**: Leverages MuJoCo's collision checking (when configured in the model).

## Configuration

The IK system is configured via frame mappings and weights.

### Frame Mappings

You need to map task names (e.g., `left_hand`, `right_hand`) to specific frame names in your MuJoCo model. Frame names can be `body`, `site`, or `geom` names.

```python
end_effector_frames = {
    "left_hand": "l_wrist_roll_link",
    "right_hand": "r_wrist_roll_link"
}
```

### Weights

Weights determine the priority of each task. Higher weights make the solver try harder to minimize the error for that specific task.

```python
task_weights = {
    "left_hand": 1.0,
    "right_hand": 1.0
}
```

There is also a `posture_weight` (default: `0.01`) which keeps the robot close to its initial/neutral configuration, preventing it from wandering into strange poses when the IK is under-constrained.

## Usage

### Using the Python API

```python
from teleop_xr.ik.mink_solver import MinkIKSolver

# Initialize solver
solver = MinkIKSolver(
    model_path="path/to/your/model.xml",
    end_effector_frames={"ee": "end_effector_link"},
    task_weights={"ee": 1.0}
)

# In your update loop:
target_matrix = ... # 4x4 SE3 matrix from XRState
current_q = ... # Get from robot
new_q = solver.solve({"ee": target_matrix}, current_q)

# Send new_q to robot
```

### Running the Simulator Demo

You can visualize the IK system in action using the built-in simulator:

```bash
uv run python -m teleop_xr.ik_sim
```

To specify a different model and end-effector frames:

```bash
uv run python -m teleop_xr.ik_sim --model /path/to/robot.xml --left-hand-frame L_wrist --right-hand-frame R_wrist
```

### Running with ROS2

TeleopXR provides a ROS2 node that integrates the IK solver:

```bash
uv run python -m teleop_xr.ros2_ik --model /path/to/model.xml
```

## Supported Models

The system uses `urdf2mjcf` to support URDF files. Note that complex URDFs with `package://` mesh URIs might require manual conversion to MJCF if the meshes cannot be resolved automatically at runtime.
