# Robot Orientation Learnings

## Base Orientation Mechanism
- `BaseRobot` (`teleop_xr/ik/robot.py`) does **not** have a built-in mechanism for base orientation or base transforms in the IK logic.
- Orientation for visualization is handled via `RobotVisConfig.initial_rotation_euler`, which is passed to the WebXR frontend.
- In the WebXR frontend (`webxr/src/xr/robot_system.ts`), the robot model is first rotated by `-Math.PI / 2` around X (to convert Z-up URDF to Y-up Three.js), and then the `initial_rotation_euler` is applied.

## Unitree H1 Orientation
- `UnitreeH1Robot` (`teleop_xr/ik/robots/h1_2.py`) defines `initial_rotation_euler=[0.0, 0.0, 1.57079632679]` (90 degrees around Z) in its `get_vis_config`.
- This suggests that the H1 URDF might be oriented such that its "forward" is along the Y axis, or it simply needs this rotation to align with the frontend's expected viewing direction.
- The IK solver uses the URDF frame directly without additional transforms.

## TeaArm Orientation
- `TeaArmRobot` (`teleop_xr/ik/robots/teaarm.py`) has `initial_rotation_euler=[0.0, 0.0, 0.0]`.
- There is **no** rotation logic in its `forward_kinematics` or `build_costs`.
- If TeaArm is misaligned (not X-forward), it will currently receive IK targets that are rotated incorrectly because the `IKController` assumes a fixed mapping from XR to robot coordinates.

## IK Controller Coordinate Mapping
- `IKController` (`teleop_xr/ik/controller.py`) has a hardcoded rotation matrix `R_xr_to_robot` in `compute_teleop_transform`:
  ```python
  R_xr_to_robot = jaxlie.SO3.from_matrix(
      jnp.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
  )
  ```
- This matrix maps XR forward (-Z) to Robot X, XR right (X) to Robot -Y, and XR up (Y) to Robot Z.
- This assumes all robots use the ROS standard: **X-forward, Y-left, Z-up**.
- Any robot not following this convention will be "misaligned" during teleoperation unless a base transform is introduced.
## Robot Orientation Attribute
- Enhanced `BaseRobot` with an `orientation` property (`jaxlie.SO3`).
- This property represents the rotation from canonical ROS2 frame (X-forward) to the robot's base frame.
- Updated `UnitreeH1Robot`, `TeaArmRobot`, and `FrankaRobot` to use this property for calculating `initial_rotation_euler` in `get_vis_config`.
- `jaxlie.SO3.as_rpy_radians()` returns a `RollPitchYaw` object that is iterable, which facilitates conversion to Euler lists for pydantic models like `RobotVisConfig`.
- TeaArmRobot orientation updated to align with ROS2 X-forward convention using a 90-degree Z-axis rotation (rz=1.57).
## UnitreeH1Robot Orientation Update
- Successfully updated `UnitreeH1Robot` in `teleop_xr/ik/robots/h1_2.py` to use the `orientation` property.
- Hardcoded rotation in `get_vis_config` moved to `orientation` property as requested.
- Verified that `initial_rotation_euler` is derived from `self.orientation.as_rpy_radians()`.
- Existing functionality (90 deg rotation) is preserved using `1.57079632679` radians.
