# Robot Orientation Normalization Plan

## Context
The user reports that the TeaArm robot's forward direction is misaligned compared to the Unitree robot. The goal is to enforce the ROS2 canonical coordinate system (X-forward, Y-left, Z-up) across all robots.

## Objectives
1.  **Enhance `BaseRobot`**: Add an `orientation` attribute (quaternion) to define the rotation needed to align the robot's native base frame with the canonical ROS2 frame.
2.  **Refactor `TeaArmRobot`**: Define the correct `orientation` for TeaArm to align it with X-forward.
3.  **Apply Correction**: Ensure this orientation is applied in:
    *   Forward Kinematics (FK) - so returned poses are in the canonical frame.
    *   Visualization - so the mesh is displayed correctly in the frontend.
    *   Inverse Kinematics (IK) - if necessary, though if FK is correct, IK usually follows.

## Steps
- [ ] 1. Analyze existing robot definitions (`h1_2.py`, `teaarm.py`) and `BaseRobot` to understand current coordinate handling.
- [ ] 2. Modify `teleop_xr/ik/robot.py`:
    *   Add `orientation` property (abstract or default to identity).
    *   Update methods if they need to use this (e.g., if `forward_kinematics` is where the base transform happens).
- [ ] 3. Modify `teleop_xr/ik/robots/teaarm.py`:
    *   Determine the required rotation (e.g., +90 deg around Z).
    *   Implement the `orientation` property.
- [ ] 4. Update `teleop_xr/config.py` or `teleop_xr/robot_vis.py` to pass this orientation to the frontend if `initial_rotation_euler` is the mechanism for this.
    *   Currently `RobotVisConfig` has `initial_rotation_euler`. We might need to derive this from the quaternion.
- [ ] 5. Verify the fix.

## Questions/Hypotheses
- Does `pyroki` or `jaxlie` handle base transforms?
- Is `forward_kinematics` returning poses relative to the URDF's `base_link`? Yes.
- If `base_link` in URDF is not X-forward, we need to apply `orientation` *after* FK? No, we treat the robot's base as being rotated in the world.
    - If I put the robot in the world, and I want it to face X, but its URDF says X is "right", I need to rotate the robot instance by +90 deg around Z.
    - So `T_world_base` is the rotation.
    - `FK_world_ee = T_world_base * FK_base_ee`.

## Learnings
-
