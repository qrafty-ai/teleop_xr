# Teleop Transform Fix Plan

## Context
The user reports:
1.  All robots now correctly face +X.
2.  The **controller pose is still forwarding +Y** (moving controller forward makes robot move Y).
3.  The user states: "teleop class converts pose axis transform automatically to align with ros standard (+x forward, y left), it's just a 90 degree transform around z...".

This implies that the transform in `IKController` (`teleop_xr/ik/controller.py`) or the `TeleopSystem` (`teleop/__init__.py`) is incorrect.

## Analysis
- **Current State**:
    - Robots are standardized to X-forward.
    - `IKController` uses `self.robot.orientation.inverse() @ R_xr_to_robot @ t_delta_xr`.
    - `R_xr_to_robot` maps XR -Z to Robot X.
    - User says "teleop class converts pose axis transform... it's just a 90 degree transform around z".
- **Hypothesis**:
    - The "90 degree transform around Z" mentioned might be what the user *expects* or what is currently implemented and *wrong*.
    - If moving Forward (XR -Z) results in +Y motion, then the effective transform maps -Z to +Y.
    - Currently `R_xr_to_robot` maps -Z to +X.
    - If `orientation` is -90 (-1.57), then `Inverse(-90) = +90`.
    - `+90 @ +X = +Y`.
    - So moving Forward (XR -Z) -> `R_xr_to_robot` -> +X -> `Inv(Ori)` -> +Y.
    - The robot receives a +Y command.
    - Since the robot is visually rotated -90 to face X, its native frame is rotated +90 (X points Y).
    - So native Y axis points -X (Back).
    - Sending +Y command moves it Back.
    - Wait, coordinate frames are hard.
    - **User's Insight**: "teleop class converts... it's just a 90 degree transform around z".
    - Maybe the user implies the *incoming* XR pose is already transformed?
    - Or that we should *add* a 90 deg transform.

## Objectives
1.  **Fix `R_xr_to_robot`**: Adjust the base XR-to-Robot transform in `teleop_xr/ik/controller.py`.
    - Current: XR Forward (-Z) -> Robot X.
    - Problem: Result is Robot +Y (after orientation fix).
    - Goal: Result should be Robot +X (relative to world).
    - If `orientation` rotates the robot frame, we need to transform the *world* delta into the *robot* frame.
    - If Robot Native X points World Y (Left).
    - We want to move World X (Forward).
    - We need to move Robot -Y (Native Right).
    - Current calculation: `+Y`.
    - We need `[0, -1, 0]`.
    - We are getting `[0, 1, 0]`.
    - So we are off by 180 degrees or mirroring?
    - Let's look at `R_xr_to_robot` again.
    - User hint: "teleop class converts pose... it's just a 90 degree transform around z".
    - Maybe the `TeleopSystem` in Python *already* rotates the inputs?
    - Let's check `teleop/__init__.py` or wherever the websocket data comes in.

## Steps
- [x] 1. **Analyze Teleop Data Flow**: Check `teleop_xr/__init__.py` or `teleop_xr/server.py` to see if XR poses are modified before reaching `IKController`.
- [x] 2. **Analyze `R_xr_to_robot`**: Re-evaluate the matrix in `teleop_xr/ik/controller.py`.
- [x] 3. **Apply Fix**:
    - If data is raw XR, adjust `R_xr_to_robot`.
    - If data is pre-transformed, remove the redundant transform.
    - User suggests "it's just a 90 degree transform around z".
    - If I change `R_xr_to_robot` to map -Z to -Y?
    - If -Z (Forward) -> -Y.
    - `Inv(-90) @ -Y = +90 @ -Y = +X`.
    - If we send +X to robot. Robot Native X points Y.
    - Robot moves Y.
    - Still Y.
    - We need the final vector sent to the robot to be `[0, -1, 0]` (Native -Y).
    - `Inv(-90) @ V = [0, -1, 0]`.
    - `V = Rot(-90) @ [0, -1, 0] = [-1, 0, 0]` (-X).
    - So we need `R_xr_to_robot` to map XR Forward (-Z) to World -X?
    - That seems backwards.

## Learnings
-
