# Learnings: IK API Modularization

## Module Exports
- `teleop_xr.ik` now explicitly exports `BaseRobot`, `PyrokiSolver`, and `IKController`.
- Used `__all__` in `teleop_xr/ik/__init__.py` to define the public API.
- Verified exports using `uv run python -c "from teleop_xr.ik import BaseRobot, PyrokiSolver, IKController; print('ok')"`.

## Dependency Management
- Some dependencies like `uvicorn` might be missing in the environment; using `uv sync` ensures all required packages are present.
- `uv run` is helpful to execute commands within the managed virtual environment.

## Testing Public API
- Created `tests/test_ik_api.py` to verify the public API surface.
- Verified that `BaseRobot` is an ABC and requires all abstract methods to be implemented.
- Mocking: Method signatures in mock classes must match the base class exactly to satisfy LSP and type checkers.
- Execution: Running tests with `uv run pytest` ensures that all dependencies (including dev groups) are correctly loaded from the project configuration.

## Docstrings and Type Hints (2026-02-02)
- Added comprehensive docstrings to the public IK API (`BaseRobot`, `PyrokiSolver`, `IKController`).
- Used modern Python type hints (e.g., `| None`, `dict[]`, `list[]`) where applicable.
- Explicitly handled type conversions between NumPy and JAX (`jnp.asarray`, `np.array`) in `IKController` to ensure compatibility and satisfy type checkers.
- Included usage examples in the module-level docstring of `teleop_xr/ik/__init__.py`.

## ROS2 CLI Modularization
- Added `--mode` flag to ROS2 node using `tyro`.
- Supported modes: `teleop` (default) and `ik`.
- Placeholder logic added to `main()` for IK mode.
- Verified CLI changes using `tyro` help output even without ROS2 sourced by temporarily bypassing the `rclpy` import check.

### ROS2 IK Integration
- **Joint Mapping**: Successfully mapped joint names between the `UnitreeH1Robot` model (`actuated_names`) and ROS2 messages (`JointState`, `JointTrajectory`).
- **State Synchronization**: Implemented a `JointState` subscriber that updates the internal configuration (`q`) when IK is not actively engaging, preventing drift before control starts.
- **Worker Pattern**: Reused the `IKWorker` thread pattern from the demo to ensure IK calculations don't block the ROS2 spin or Teleop websocket threads.
- **Event Handling**: Integrated the `EventProcessor` to handle robot reset gestures (double-press on deadman switch) consistently with the demo mode.
- **Dependencies**: Ensuring `jax` is configured for CPU mode at node startup is critical for performance and compatibility in ROS2 environments without GPU access.
