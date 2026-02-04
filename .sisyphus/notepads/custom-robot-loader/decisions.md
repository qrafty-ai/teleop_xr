
## PyrokiSolver Optional Targets
- Decided to allow  for , , and .
- Updated  to accept .
- Implemented cost skipping in  based on  targets.
- Updated  to include various combinations of  targets to pre-compile common use cases.

## PyrokiSolver Optional Targets
- Decided to allow `None` for `target_L`, `target_R`, and `target_Head`.
- Updated `BaseRobot.build_costs` to accept `Optional[jaxlie.SE3]`.
- Implemented cost skipping in `UnitreeH1Robot.build_costs` based on `None` targets.
- Updated `PyrokiSolver._warmup` to include various combinations of `None` targets to pre-compile common use cases.

### Decision: Flexible Frame Handling in IKController
- **Problem**: IKController assumed all robots have "left", "right", and "head" frames.
- **Solution**: Modified controller to dynamically adapt to `robot.supported_frames`.
- **Implementation**:
  - `_get_device_poses` filters by supported frames.
  - `step` activation depends only on supported frames being present.
  - Snapshotting and target computation only process supported frames.
  - Unsupported targets are passed as `None` to the solver.

## Documentation of Custom Robot Contract
- Decided to document the `__init__(self, urdf_string: str | None = None, **kwargs)` contract in both `load_robot_class` docstrings and the main `README.md`.
- This ensures developers know exactly how to implement their robots to be compatible with the ROS2 CLI's automatic URDF fetching and custom argument passing.
