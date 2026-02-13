
## IKWorker Visualization Update
- Added `teleop` and `teleop_loop` to `IKWorker` to enable publishing joint states back to WebXR.
- Used `asyncio.run_coroutine_threadsafe` to bridge the worker thread and the main asyncio loop.
- Ensured that `ik_worker` is updated with the `teleop` instance and the event loop after `Teleop` is initialized in `main()`.
- Joint states are published as a dictionary mapping joint names to float positions.

## Update main() in teleop_xr/ros2/__main__.py
- Wires up the `teleop` instance to `ik_worker` after initialization.
- Added `teleop_xr_state_callback` logic to capture the running `asyncio` loop and set it in `ik_worker`. This ensures that even if `get_event_loop()` in `main()` fails or returns the wrong loop, the callback (which runs within the teleop loop) captures the correct one.
- Updated `joint_state_callback` to publish joint states back to WebXR when IK is not active. This uses `asyncio.run_coroutine_threadsafe` because the callback runs in a ROS spin thread.

## Implementation Verification (2026-02-12)
- Verified `teleop_xr/ros2/__main__.py` implementation.
- Closure capture of `ik_worker` in `joint_state_callback` is safe as it is assigned before ROS spinning starts, and guarded by `if ik_worker` checks.
- `asyncio.run_coroutine_threadsafe` is correctly used to bridge between the IK worker thread (or ROS callback thread) and the main Teleop event loop.
- Loop capture in `teleop_xr_state_callback` provides a fallback to ensure `teleop_loop` is set even if `asyncio.get_event_loop()` in `main()` returns a different loop than what `uvicorn` ends up using.
- Type hints and imports are correct and handle the optional ROS2 environment gracefully.
