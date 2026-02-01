
## IK Module Initialization
- Created `teleop_xr/ik` package.
- Defined `BaseRobot` in `teleop_xr/ik/robot.py` as an ABC with `forward_kinematics`, `get_default_config`, and `build_costs` methods.
- Used `Cost = Any` as a placeholder for the cost type in `BaseRobot.build_costs` return type hint, as `Cost` is not yet defined in the codebase.

## IKController Design
- **Deadman State Machine**: Implemented a transition-based state machine that snapshots initial states when both squeeze buttons are first pressed.
- **Reference Transform**: Strictly followed the reference implementation for pose delta calculation using `spatialmath` `SE3` and `UnitQuaternion`.
- **Target Poses**: Assumes the robot model (`BaseRobot`) provides FK and takes targets for "left", "right", and "head".
- Added `joint_var_cls` property to `BaseRobot` to allow the solver to create the correct variable types for optimization.
- The solver uses a single timestep (index 0) for IK, matching the expectation for real-time teleoperation.
- JIT compilation is triggered during initialization via a `_warmup` method to ensure low latency during the first real `solve` call.

### H1_2 Robot Setup
- Link Selection: Chose `L_hand_base_link` and `R_hand_base_link` for end-effector targets as they represent the most distal links of the arm chains before the hand/gripper.
- Cost Formulation: Used `pose_cost_analytic_jac` for end-effectors for performance, and `pose_cost` (numerical) for the torso yaw cost to support vector-based orientation weighting.
- Leg Joints: Explicitly identified and masked indices 0-11 for the leg freezing cost to ensure the robot remains stable (feet fixed) during upper-body teleoperation.
- Implemented `demo_ik.py` as the entrypoint for IK-based teleoperation.
- Used `InputMode.CONTROLLER` as default for the IK demo.
- Integrated `RobotVisConfig` for H1 visualization using assets in `teleop_xr/assets/h1_2`.
