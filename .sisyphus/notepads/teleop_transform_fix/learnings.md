
## Teleop Data Flow Analysis

- Incoming WebXR poses are in **RUB (Right-Up-Back)** coordinate system (+X Right, +Y Up, +Z Back).
- The `Teleop` class in `teleop_xr/__init__.py` intercepts these poses in `__handle_xr_state` and calls `__convert_devices_to_ros`.
- `__convert_pose_to_ros` applies a transformation matrix `TF_RUB2FLU` to convert positions and orientations to **FLU (Front-Left-Up)**:
  - `New X (Front) = -Old Z (Back)`
  - `New Y (Left) = -Old X (Right)`
  - `New Z (Up) = Old Y (Up)`
- This transformation is applied **in-place** to the device dictionaries in the message data.
- Subscribers (like the IK worker in the demo) receive the message containing these **already-converted FLU poses**.
- `IKController` then uses these FLU poses to compute relative motion. It further applies `robot.ros_to_base` to transform this motion into the robot's local frame.
- For the Unitree H1 robot, `ros_to_base` is Identity. For the TeaArm robot, it is a +90 degree rotation around Z.

### Analysis of R_xr_to_robot (ros_to_base)
- The matrix referred to as `R_xr_to_robot` in the task is likely `self.robot.ros_to_base` in `teleop_xr/ik/controller.py`.
- **Current Definition**: In `BaseRobot` and `UnitreeH1Robot`, `orientation` is identity, so `ros_to_base` is also identity.
- **Current Mapping**:
  - XR Forward (-Z) -> ROS Forward (+X) -> Robot +X
  - XR Left (-X) -> ROS Left (+Y) -> Robot +Y
  - XR Up (+Y) -> ROS Up (+Z) -> Robot +Z
- **Mismatch Identified**: "Inherited Wisdom" says moving XR Forward (-Z) makes the robot move Y. This means ROS Forward (+X) is mapping to Robot Y.
- **Root Cause**: `ros_to_base` being identity assumes the robot's Forward is its local +X, but if the robot moves in Y when given +X ROS delta, then the robot's "Forward" (or the direction we want it to move) is its local Y.
- **Proposed Fix**: `ros_to_base` should be a 90-degree rotation around Z (`R_z(90)`) to map ROS +X (Forward) to Robot +Y. This implies the robot's `orientation` (Robot to ROS) should be `R_z(-90)`.

### Teleop Transform Fix Implementation
- Updated `UnitreeH1Robot.orientation` in `teleop_xr/ik/robots/h1_2.py` to `jaxlie.SO3.from_rpy_radians(0.0, 0.0, -1.57079632679)`.
- This rotation (90 deg CW) ensures that ROS Forward (+X) maps correctly to the robot's intended forward motion.
- This change aligns H1_2 with the pattern established in `TeaArmRobot`.
