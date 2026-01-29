# Camera View Configuration Implementation Plan

## Summary
Add CLI flags to map head/wrist views to device IDs, propagate the config over WebSocket, and route WebXR panels/tracks deterministically. Keep defaults for panel sizing and positioning.

## Steps
1. Add a Python helper module for camera view config.
   - Create `teleop/camera_views.py` with helpers:
     - `parse_device_spec(value: str | int) -> int | str`
     - `build_camera_views_config(head, wrist_left, wrist_right) -> dict`
     - `build_video_streams(camera_views) -> dict` (stream IDs `head`, `wrist_left`, `wrist_right`)
   - Add pytest unit tests under `tests/test_camera_views.py` (TDD) to cover valid/invalid device specs and config assembly.

2. Wire CLI flags into `teleop/basic/__main__.py`.
   - Use `argparse` with explicit flags: `--head-device`, `--wrist-left-device`, `--wrist-right-device`.
   - Build `camera_views` from flags and pass into `Teleop`.
   - Preserve existing callback behavior (note current local change in this file).

3. Extend `Teleop` to accept and broadcast `camera_views`.
   - Update `teleop/__init__.py` to store `camera_views` and include it in the `config` message on WS connect.
   - Use `build_video_streams(camera_views)` to create stream payloads for `set_video_streams`.
   - Ensure missing views disable streams cleanly.

4. Add WebXR config handling for camera views.
   - Create `webxr/src/camera_views.ts` to hold the latest config and provide helpers for visibility decisions.
   - Update `webxr/src/teleop_system.ts` to parse `camera_views` from WS `config` and store it.
   - Update `webxr/src/index.ts` to:
     - toggle visibility of `CameraPanel`/`ControllerCameraPanel` based on config
     - route video tracks by `trackId` (`head`, `wrist_left`, `wrist_right`)
     - retain order fallback if `trackId` missing

5. Add WebXR-side tests.
   - If no TS test runner exists, add `vitest` in `webxr/package.json` with a minimal `vitest.config.ts`.
   - Add tests for `camera_views.ts` helpers and track-to-panel mapping.

6. Clean up and validate.
   - Remove any temporary scaffolding.
   - Run validation commands.

## Validation
- `uv run pytest`
- `npm run build` (from `webxr/`)
