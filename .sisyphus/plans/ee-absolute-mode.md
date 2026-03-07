# EE Absolute Mode Plan

## Context
- The latest relevant implementation in this repo is `b6bf9a2`, which adds explicit `ee_delta` control mode.
- There is no literal `ee_relative` mode in the tree today. Relative end-effector behavior currently lives in `ControlMode.TELEOP` and the snapshot-based transform flow in `teleop_xr/ik/controller.py` and `teleop_xr/__init__.py`.
- The simplest safe interpretation is to add `ee_absolute` as a second explicit command-driven mode alongside `ee_delta`, then wire a demo trigger into the TUI the same way the `ee_delta` demo is wired now.

## Relevant Files
- `teleop_xr/ik/control_mode.py`
- `teleop_xr/ik/commands.py`
- `teleop_xr/ik/controller.py`
- `teleop_xr/ik/__init__.py`
- `teleop_xr/__init__.py`
- `teleop_xr/demo/__main__.py`
- `tests/test_ik_controller.py`
- `tests/test_demo_ee_delta.py`
- `tests/test_ws_control_v2.py`

## Goal
Add an explicit `ee_absolute` control interface that:
1. accepts an absolute end-effector target pose for a named frame,
2. runs only while the controller is in `ee_absolute` mode,
3. keeps XR teleop disabled while the absolute command path is active,
4. exposes a demo-app trigger parallel to the existing `ee_delta` demo.

## Plan

### 1. Extend the explicit control-mode surface
- Add `EE_ABSOLUTE = "ee_absolute"` to `ControlMode` in `teleop_xr/ik/control_mode.py`.
- Keep `TELEOP` semantics unchanged so existing relative XR control remains the default mode.

### 2. Add an absolute-pose command schema
- Extend `teleop_xr/ik/commands.py` with a new Pydantic model for the absolute command.
- Reuse the current `position` and `orientation` shape from `DeltaPose` to avoid inventing a second pose payload format.
- Name the payload field clearly as an absolute target, for example `target_pose`, so it is not confused with `delta_pose`.
- Make `target_pose` required rather than defaulting it, so malformed payloads cannot silently command origin plus identity orientation.
- Keep the `frame` field aligned with `EEDeltaCommand` (`left`, `right`, `head`).

### 3. Implement `IKController.submit_ee_absolute`
- Add a pose-to-`jaxlie.SE3` helper that can be shared by both delta and absolute command paths, or add a parallel helper if that keeps the code clearer.
- Add `submit_ee_absolute(command, q_current)` with the same structural flow as `submit_ee_delta`:
  - reject calls unless mode is `EE_ABSOLUTE`,
  - return `q_current` when there is no solver,
  - validate the command model from either dict input or model input,
  - validate that `frame` is supported by the robot,
  - compute current FK and ensure the requested frame exists,
  - start from the current FK targets for `left`, `right`, and `head`,
  - replace only the addressed frame target with the provided absolute pose,
  - call `self.solver.solve(...)`,
  - reuse the existing filter path exactly like `submit_ee_delta`.
- Leave `step()` teleop-only so explicit command modes continue to bypass the XR relative-motion loop.

### 4. Export the new public interface
- Update `teleop_xr/ik/__init__.py` to export the new absolute command model, matching the existing `EEDeltaCommand` export pattern.

### 5. Wire the demo app in parallel with the delta demo
- Add `run_right_ee_absolute_demo(...)` in `teleop_xr/demo/__main__.py` next to `run_right_ee_delta_demo(...)`.
- Mirror the existing demo pattern:
  - save the original mode,
  - switch to `ee_absolute`,
  - submit a sequence of right-arm absolute target poses,
  - publish joint states through `teleop.publish_joint_state(...)` when the teleop loop is active,
  - restore the original mode in `finally`.
- Pick a distinct keybinding for the absolute demo and update `generate_ik_controls_panel()` to show both demos.
- Reuse the current threading/lock pattern so only one scripted EE demo runs at a time.

### 6. Preserve backend gating behavior
- Keep `teleop.bind_control_mode_provider(lambda: controller.get_mode().value)` in place.
- Do not widen `Teleop.__is_teleop_mode_enabled()`; its current behavior is what prevents XR `xr_state` messages from driving the robot while explicit command demos are running.

### 7. Add focused test coverage
- In `tests/test_ik_controller.py`, add tests parallel to the `ee_delta` coverage:
  - requires mode switch,
  - rejects unsupported frames,
  - no-solver returns current config,
  - missing FK frame raises,
  - mode switch/reset behavior still works.
- In the demo tests, add a new file or extend `tests/test_demo_ee_delta.py` with absolute-demo coverage for:
  - mode switch and restore,
  - joint-state publishing when loop is running,
  - restore on error,
  - TUI keybinding dispatch.
- In `tests/test_ws_control_v2.py`, add a mode-gating assertion for `ee_absolute`, mirroring the existing `ee_delta` denial test.

## Implementation Notes
- The main semantic difference from `ee_delta` is target replacement instead of target composition.
- For the demo path, absolute targets should be derived from the robot's current FK before starting the scripted loop, then offset into a small deterministic trajectory. That keeps the demo stable regardless of the robot's current joint configuration.
- Avoid changing the existing teleop relative-motion path; `ee_absolute` should be additive, not a rewrite of the current control model.

## Verification
- Run `lsp_diagnostics` on all modified Python files.
- Run targeted tests for the controller, demo, and websocket gating paths.
- If available in the environment, run the relevant Python test subset covering the new mode.
