# Robot Orientation & Control Fix Plan

## Context
The user identified two key issues:
1.  **Control Mismatch**: Moving forward (XR -Z) moves the robot sideways (Robot Y). This is because the IK controller calculates targets in a generic "Canonical ROS" frame (X-forward) but applies them to the robot without accounting for the robot's base rotation (e.g., if the robot is rotated 90°, X-forward becomes Y-forward).
2.  **Orientation Standardization**: Robots are facing inconsistent directions. The goal is to align all robots so their visual "Forward" aligns with the canonical X-axis.

## Objectives
1.  **Fix Control Logic**: Update `teleop_xr/ik/controller.py` to transform teleop deltas from the "Canonical ROS" frame into the "Robot Base" frame using `self.robot.orientation.inverse()`.
2.  **Standardize Robots**:
    *   **Franka**: Rotate -90° (Yaw = -1.57) to face X (currently faces Y).
    *   **H1**: Rotate 0° (Identity) to face X (currently rotated +90° to face Y).
    *   **TeaArm**: Rotate 180° (Yaw = 3.14) to face X (currently faces -X).
3.  **Fix Visualization**: Update `webxr/src/xr/robot_system.ts` to correctly apply the RPY rotation. Since the model is pre-tilted by -90° X, applying RPY blindly is incorrect.

## Steps
- [x] 1. **IK Controller Fix**: Modify `compute_teleop_transform` in `teleop_xr/ik/controller.py`.
    *   Current: `t_delta_robot = R_xr_to_robot @ t_delta_xr`
    *   New: `t_delta_robot = self.robot.orientation.inverse() @ (R_xr_to_canonical @ t_delta_xr)`
    *   Wait, `R_xr_to_robot` *is* `R_xr_to_canonical` (maps XR -Z to X).
    *   So: `t_delta_robot = self.robot.orientation.inverse() @ R_xr_to_robot @ t_delta_xr`.
- [x] 2. **Robot Standardization**:
    *   `teleop_xr/ik/robots/franka.py`: Override `orientation` with `rpy(0, 0, -1.57)`.
    *   `teleop_xr/ik/robots/h1_2.py`: Change `orientation` to `identity()` (0).
    *   `teleop_xr/ik/robots/teaarm.py`: Change `orientation` to `rpy(0, 0, 3.14)`.
- [x] 3. **Visualization Fix**:
    *   Modify `webxr/src/xr/robot_system.ts`.
    *   Instead of setting `rotation.x/y/z` directly from RPY, create a Quaternion from the RPY, then multiply the object's quaternion by it (or apply to a wrapper).
    *   Actually, simpler: `robotObject3D.rotation.set(-Math.PI / 2, 0, 0);` is the base.
    *   We want to apply the RPY *on top* of the canonical frame.
    *   Correction: Create a parent `Object3D` for the tilt, or apply rotations in specific order `Y(Yaw) -> X(Roll) -> Z(Pitch)`.
    *   Let's check `robot_system.ts` logic again.

## Verification
- Run `verify_orientations.py` to check the updated quaternion values.
- Rebuild frontend.

## Learnings
-
