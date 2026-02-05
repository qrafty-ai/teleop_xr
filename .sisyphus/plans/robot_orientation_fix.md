# Robot Orientation Revert Plan

## Context
The previous fix applied a 90-degree rotation (`rz=1.57`) to `TeaArmRobot` to match `UnitreeH1Robot`. The user reports this caused the robot to "face left" and misalign with their forward movement. The goal is to ensure the robot faces forward and aligns with the user's forward direction.

## Objectives
1.  **Revert `TeaArmRobot` Orientation**: Set `orientation` back to identity (0 rotation) in `teleop_xr/ik/robots/teaarm.py`.
2.  **Verify H1**: Briefly check if `UnitreeH1Robot`'s rotation is actually desired or if it shares the same issue (though the user specifically flagged the current behavior as the problem).
3.  **Ensure Consistency**: Align Visuals and IK frames. Since `orientation` currently only affects Visuals (`get_vis_config`), removing the rotation should align the Visual X (Forward) with the IK X (Forward).

## Steps
- [ ] 1. Modify `teleop_xr/ik/robots/teaarm.py`:
    *   Change `orientation` to return `jaxlie.SO3.identity()`.
    *   Update `get_vis_config` documentation/comments if needed.
- [ ] 2. Verify `teleop_xr/ik/robots/h1_2.py`:
    *   Read the file to confirm it still has `rz=1.57`.
    *   (Optional) Check if H1 URDF is available to see its native frame.
- [ ] 3. Verify the fix by printing the orientation quaternion/euler.

## Hypothesis
- `TeaArm` URDF is natively X-forward.
- My previous "fix" rotated it +90 deg (to Y-forward/Left).
- Reverting to 0 deg will restore X-forward.
- `IKController` maps User Forward (-Z) to Robot X.
- So User Forward -> Robot X (Visual Forward). This matches the requirement.

## Learnings
-
