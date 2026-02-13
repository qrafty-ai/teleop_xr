# Plan - Robot Orientation Fix 2

## Context

The user reports that despite the previous fixes:

1. The converted forward direction is still +Y (instead of +X).
2. H1 is still facing +Y (needs to face +X).

This implies that:

- My previous "fix" for H1 (identity orientation) might have been incorrect if the
  H1 URDF is naturally Y-forward (or -Y forward).
- The `R_xr_to_robot` transform in `IKController` might still be mapping XR -Z
  (Forward) to Robot Y, or the `orientation` correction isn't working as
  intended.

## Objectives

1. **Re-Verify Transform Chain**: Double-check `R_xr_to_robot` in
   `teleop_xr/ik/controller.py`.
2. **Fix H1 Orientation**: If H1 faces Y with `identity`, and we want X, we need
   to rotate it by -90° (if Y is left of X) or +90° (if Y is right of X). Wait,
   standard ROS: X forward, Y left. So Y is +90 from X. To bring Y to X, we need
   -90° (yaw = -1.57).
   - Previous attempt set H1 to `identity`.
   - User says it faces +Y.
   - Conclusion: H1 URDF is Y-forward.
   - Fix: Set H1 orientation to `rpy(0, 0, -1.57)` (rotate -90 deg).
3. **Investigate Forward Direction**: Why is "converted forward direction" still
   +Y?
   - `compute_teleop_transform` does:
     `t_delta_robot = self.robot.orientation.inverse() @ R_xr_to_robot @ t_delta_xr`.
   - `R_xr_to_robot` maps XR -Z (Forward) to Canonical X.
   - If H1 faces Y (native), its `orientation` should reflect that (e.g.,
     `orientation = rotation_from_canonical_to_native`).
   - If Native = Y-forward, then `orientation` should represent the rotation
     *from* X *to* Y (which is +90 deg).
   - If we set `orientation = +90 deg`, then `orientation.inverse()` is -90 deg.
   - `Canonical X` rotated by -90 deg -> `Native -Y`? No.
   - Let's trace:
     - `Canonical X` (Forward)
     - `Native Frame`: X is Right, Y is Forward (rotated +90 from Canonical).
     - `orientation` (Canonical -> Native) = +90 deg Z.
     - `t_delta_canonical` = [1, 0, 0] (Move Forward).
     - `t_delta_native` = `orientation.inverse() @ t_delta_canonical` =
       `Rot(-90) @ [1, 0, 0]` = `[0, -1, 0]` (Move -Y in native frame).
     - Wait, if Native Y is Forward, we want to move +Y.
     - `[0, 1, 0]` is +Y. `Rot(+90) @ [1, 0, 0]` = `[0, 1, 0]`.
     - So `t_delta_native` should be `orientation.inverse() @ t_delta_canonical`?
     - If `orientation` is `R_canonical_to_native`. Then
       `v_native = R_native_to_canonical.inv() @ v_canonical` ->
       `v_native = R_canonical_to_native.inv() @ v_canonical`. No.
     - `v_native = R_canonical_to_native.T @ v_canonical`?
     - Let's assume `orientation` is `R_world_from_robot`.
     - We want `v_robot`. We have `v_world`.
     - `v_robot = R_world_from_robot.inverse() @ v_world`.
     - If Robot faces Y, `R_world_from_robot` has +90 yaw.
     - `v_world` = [1, 0, 0] (X).
     - `v_robot` = Rot(-90) @ [1, 0, 0] = [0, -1, 0].
     - This moves robot along its native X (which is Right).
     - **Wait.** If Robot faces Y, its "Forward" is its Y axis.
     - So we *want* `v_robot` to be `[0, 1, 0]`.
     - How do we get `[0, 1, 0]` from `[1, 0, 0]`?
     - We need to rotate by +90.
     - So `v_robot = Rot(+90) @ v_world`.
     - But `orientation.inverse()` gives -90.
     - **Hypothesis**: The definition of `orientation` in `BaseRobot` is
       "Rotation from Canonical to Base".
     - If Base is Y-forward, `orientation` = +90.
     - We want to move along Canonical X.
     - Express Canonical X in Base frame:
       - `X_canonical` = `X_base` (if aligned).
       - If Base is rotated +90 (Y is forward), then `X_canonical` (World
         Forward) aligns with `Y_base` (Robot Forward).
       - Wait, no.
       - World X is "North".
       - Robot is rotated 90 deg left (faces West).
       - Robot Forward is -X_world.
       - Let's stick to: Robot Frame X is "Front of Robot" (usually).
       - If H1 URDF has Y as "Front", then H1 URDF is *not* standard.
       - If H1 URDF X is "Front", but we see it facing Y, then the visualizer is
         rotating it, OR the `orientation` we set is rotating it.
       - The user says "H1 still facing +Y".
       - My previous fix set H1 orientation to `identity`.
       - This means we treat H1 URDF as aligned with World.
       - If user sees it facing Y, then **H1 URDF itself is defined with
         Forward = Y**.
       - OR, the visualizer default state is Y-forward.
   - **Action**:
     - Rotate H1 by -90 deg (to align visual X with world X).
     - `orientation = rpy(0, 0, -1.57)`.
     - This means "The robot is rotated -90 deg relative to Canonical".
     - If we want to move "Canonical Forward" (World X), we need to compute that
       vector in Robot Frame.
     - `v_robot = R_world_to_robot @ v_world`
     - `R_world_to_robot = R_robot_to_world.inverse() = orientation.inverse()`.
     - `orientation = -90`. `inverse = +90`.
     - `v_robot = Rot(+90) @ [1, 0, 0] = [0, 1, 0]`.
     - If H1's "Forward" is its X axis, then `v_robot=[0, 1, 0]` moves it Left
       (Y).
     - If H1's "Forward" is its Y axis (as user implies), then `v_robot=[0, 1, 0]`
       moves it Forward.
     - **CRITICAL**: Is H1's "Forward" axis X or Y in its URDF?
     - Standard ROS: X is forward.
     - If H1 faces Y when orientation=identity, it implies **URDF Forward = Y**.
     - To fix "Facing": We must rotate it -90 deg. `orientation` -> -90.
     - To fix "Movement":
       - If we set `orientation = -90`.
       - User inputs Forward (World X).
       - Controller computes `v_robot = Inv(-90) @ X = Rot(+90) @ X = Y`.
       - Command sent to robot: `vx=0, vy=1`.
       - If Robot Forward is Y, it moves Forward.
       - If Robot Forward is X, it moves Left.
       - User said "current converted forward direction is still +y". This
         likely means `v_robot` is along Y.
       - If robot *moves* towards Y (World Y? or Robot Y?), we need to clarify.
       - "robot actually moves toward y axis" (World Y? implied).
       - If I push X, and robot moves Y, then there is a 90 deg error.

## Plan

1. **H1**: User says it faces +Y. Fix: Rotate -90 (Yaw = -1.57).
   - `orientation = rpy(0, 0, -1.57)`.
2. **Controller**:
   - Check `compute_teleop_transform` again.
   - Ensure `R_xr_to_robot` maps XR-Forward (-Z) to World X.
   - Verify `orientation.inverse()` logic.
3. **Visualization**:
   - Verify `robot_system.ts` handles the updated `orientation`.

## Steps

- [ ] 1. Update `teleop_xr/ik/robots/h1_2.py`: Set `orientation` to -90 deg.
- [ ] 2. Update `teleop_xr/ik/controller.py`:
  - Print/Log the transform for debugging? Or just review logic.
  - Check if `R_xr_to_robot` maps -Z -> +Y?
  - Current code:

    ```python
    R_xr_to_robot = jaxlie.SO3.from_matrix(
        jnp.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    )
    ```

    - Col 0 (XR X) -> Robot [0, -1, 0] (-Y).
    - Col 1 (XR Y) -> Robot [0, 0, 1] (+Z).
    - Col 2 (XR Z) -> Robot [-1, 0, 0] (-X).
    - XR Forward is **-Z**.
    - R @ [0, 0, -1] = -1 * [-1, 0, 0] = **[1, 0, 0]** (+X).
    - So XR Forward -> World X. This is correct.
  - If Robot moves Y, then `t_delta_robot` must be Y.
  - `t_delta_robot = Inv(Ori) @ World_X`.
  - If `Ori` is Identity -> `t_delta_robot` = X.
  - If Robot moves Y when receiving X command, then **Robot URDF X points to World
    Y**.
  - This confirms H1 URDF is rotated +90 deg (X points Y).
  - To fix VISUAL: Rotate -90 (`orientation = -1.57`).
  - To fix CONTROL:
    - `Inv(-1.57) @ World_X = Rot(+90) @ X = Y`.
    - Command Y sent to Robot.
    - Robot URDF X points to World Y. Robot URDF Y points to World -X.
    - If we send Y, robot moves along its Y (World -X).
    - **This is wrong.**
    - We want robot to move World X.
    - Robot X axis aligns with World Y. Robot -Y axis aligns with World X.
    - We need to send command along Robot -Y.
    - So we need `v_robot = [0, -1, 0]`.
    - `Inv(Ori) @ [1, 0, 0] = [0, -1, 0]`.
    - `Inv(Ori)` needs to be Rot(-90).
    - `Ori` needs to be Rot(+90).
  - **Contradiction**:
    - Visual Fix requires `Ori = -90` (to turn model Right).
    - Control Fix requires `Ori = +90` (to map World X to Robot -Y).
    - **Wait**: `orientation` in `BaseRobot` defines "Rotation from Canonical to
      Base".
    - If H1 URDF is "Left-facing" (X points Y), then Base IS rotated +90 relative
      to Canonical.
    - So `orientation` SHOULD be +90.
    - **BUT**: User says "H1 still facing +Y".
    - If I set `orientation = +90`, visualizer rotates it +90.
    - If URDF is natively Y-facing, adding +90 makes it face -X?
    - Let's assume URDF is NATIVE.
    - If `orientation = 0`:
      - Visual: Raw URDF. User sees "Facing +Y".
      - Control: Command X sent to Robot X.
      - If "Facing +Y" means "Front of robot points Y", then Robot X axis points
        Y.
      - Sending X command moves robot along its X (World Y).
      - This explains why it moves Y.
    - **Solution**:
      - We need to tell the system: "This robot is rotated +90 deg".
      - `orientation = +90`.
      - `Inv(Ori) = -90`.
      - `Inv(Ori) @ World X = Rot(-90) @ X = -Y`.
      - Command -Y sent.
      - Robot Y axis points World -X. Robot -Y points World +X.
      - Wait:
        - Robot X -> World Y.
        - Robot Y -> World -X.
      - So sending -Y moves along World X.
      - **Success**.
      - **Visual**:
        - `robot_system.ts` applies `orientation` to the model.
        - If we apply +90 to a model that is already +90 (native), we get +180.
        - We need to **un-rotate** the model visually.
        - BUT `BaseRobot.orientation` is defined as `R_canonical_to_base`.
        - IK uses it to transform `v_canonical` to `v_base`.
        - Visual uses it to transform `model_canonical` to `model_base`?
        - No, Visual loads URDF (Base Frame) and places it in World.
        - If we want to align Robot with Canonical, we need to rotate it by
          `Inv(orientation)`.
        - Currently `robot_system.ts` applies `orientation` (rotation).
        - So it displays the robot IN ITS BASE ORIENTATION.
        - If Base is Y-facing, and we define `orientation=+90` (describing that
          state), visualizer applies +90?
        - If visualizer applies +90 to a Y-facing model...
        - **Wait**. Visualizer loads URDF. URDF *is* the mesh.
        - If mesh faces Y, and we apply 0 rotation, it faces Y.
        - If we apply +90 rotation, it faces -X.
        - If we want it to face X (Canonical), we need to rotate it -90.
        - So Visual Rotation must be -90.
      - **Conflict**:
        - Control needs `orientation` to describe the frame (+90).
        - Visual needs rotation to *correct* the frame (-90).
      - **Resolution**:
        - `orientation` property describes "How the robot base is rotated relative
          to World X".
        - H1: Rotated +90. `orientation = +1.57`.
        - Control: `v_robot = Inv(Ori) @ v_world`. Correct.
        - Visual:
          - Currently applies `orientation`.
          - This means it rotates the *container*.
          - If we load H1 URDF (Y-forward), it is already "rotated".
          - We shouldn't rotate it *again*.
          - **Actually**, we want the visual to match the IK.
          - IK assumes `v_robot` moves the mesh.
          - If we command -Y, mesh should move along its -Y axis.
          - If Mesh -Y axis is World X, then visual moves X.
          - So Visual is correct *if* we just display the URDF as is (Identity).
          - **BUT User wants "Robot facing X"**.
          - This means rotating the *entire world* or rotating the robot *mesh* so
            X is forward.
          - If we rotate the mesh -90, then Robot X points World X.
          - Then `orientation` should be 0.
          - And Control works (Send X -> Robot X -> World X).
          - **BUT** does rotating the visual mesh rotate the IK frame?
          - **NO**. IK frame is defined by URDF *structure* (joints/links).
          - Changing `rotation` in Three.js only spins the pixels.
          - It does NOT change "Joint 1 rotates around Axis Z".
          - Unless we rotate the *root* of the calculation.
      - **Real Fix**:
        - We cannot change the URDF frame (hardcoded in physics/kinematics).
        - If URDF X is "Sideways", then IK "Forward" (X) will always be Sideways.
        - We must define a "Virtual Base Frame" that is X-forward.
        - `orientation` is exactly that.
        - If H1 X is Sideways (Y), and we want "Forward" to be X:
          - We define `orientation = +90`.
          - Control logic: `v_robot = Inv(Ori) @ v_world`.
          - Visual logic:
            - We want the robot to *look* X-forward.
            - Current URDF looks Y-forward.
            - We need to rotate visual by -90.
            - `orientation` is +90.
            - `visual_rotation` should be `Inv(orientation)`? or `orientation`?
            - User wants standardization.
            - If we set `orientation = +90`, visualizer rotates +90 -> Faces -X.
            - We want -90.
            - So Visualizer needs `offset`.
    - **Wait, Simpler View**:
      - `orientation` property: "Rotation required to align Robot with World".
      - If Robot faces Y, we need -90 rotation to face X.
      - So `orientation = -90`?
      - If `orientation = -90`:
        - Visual: Rotates -90. Robot (Y-face) -> Rot(-90) -> X-face. **Visual
          Solved**.
        - Control: `v_robot = Inv(-90) @ v_world = Rot(+90) @ X = Y`.
        - Robot receives Y command.
        - Robot (native) has X=Right, Y=Forward (if Y-face).
        - Sending Y moves Forward.
        - **Control Solved**.
    - **Conclusion**:
      - H1 needs `orientation = -90 deg (-1.57)`.
      - TeaArm needs `orientation = 180 deg (3.14)` (if it faces -X).
      - Franka needs `orientation = -90 deg (-1.57)` (if it faces Y).

## Updated Plan

1. **H1**: Set `orientation = -1.57` (Rotate CW 90).
2. **Franka**: Set `orientation = -1.57`.
3. **TeaArm**: Keep `3.14`.
4. **Verify**: If I set `orientation = -1.57`:
   - `robot_system.ts` applies this rotation.
   - `IKController` applies `Inv(-1.57) = +1.57`.
   - `+1.57 @ X = Y`.
   - If Robot Native X=Right, Y=Forward. Sending Y moves Forward. Matches Visual
     X.

## Verification Steps

- [ ] 1. Modify `h1_2.py`: `orientation = rpy(0, 0, -1.57)`.
- [ ] 2. Modify `franka.py`: `orientation = rpy(0, 0, -1.57)`.
- [ ] 3. Modify `teaarm.py`: Confirm `3.14` (Keep as is).
- [ ] 4. Verify logic manually.

## Learnings
