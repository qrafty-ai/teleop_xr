# Learnings - Optimization IK Module

## WeightedMovingFilter Implementation

- Implemented in `teleop_xr/utils/filter.py`.
- Uses `np.convolve` for efficient filtering over a window of data.
- Handles multi-dimensional data (e.g., joint positions) by applying the filter
  independently to each dimension.
- Skips duplicate data to avoid filter lag when the input hasn't changed.
- Validates that weights sum to 1.0.

## Dependencies Setup

- pyroki is a local package located at /home/cc/codes/pyroki. Use its path when
  adding with uv.

## IK Module Setup

- Dependency management: Used `uv run` to execute verification scripts, ensuring
  all project dependencies (fastapi, uvicorn, etc.) are available.
- Verification: Confirmed that `BaseRobot` is correctly defined and importable
  from the `teleop_xr.ik.robot` module.

## IKController Implementation

- Implemented in `teleop_xr/ik/controller.py`.
- Implements Deadman logic: requires holding both squeeze buttons (index 1 on
  Meta controllers).
- Takes snapshots of XR tracking poses and robot FK poses on engagement.
- Computes target poses for Left, Right, and Head using
  `compute_teleop_transform`.
- Integrated `WeightedMovingFilter` to filter the joint configuration output if
  weights are provided.
- Verification:
  `uv run python -c "from teleop_xr.ik.controller import IKController; print('ok')"`
  passed.
- Pyroki uses `jaxls` for optimization, which requires variables to be
  explicitly passed to `LeastSquaresProblem`.
- `jaxls.LeastSquaresProblem.solve()` takes an `initial_vals` argument which is
  a `VarValues` object.
- Initial values can be constructed using
  `jaxls.VarValues.make([var.with_value(val), ...])`.
- For single-step IK, we use a single timestep index (usually 0) for variables.

## Unitree H1_2 Robot Implementation

- H1_2 URDF structure: Leg joints occupy indices 0-11 in the actuated joint
  list.
- Link identification: `L_hand_base_link` and `R_hand_base_link` serve as
  effective end-effector frames.
- Head-Waist alignment: Torso yaw can be biased towards head yaw using a
  weighted orientation cost on `torso_link` with a weight mask
  `jnp.array([0, 0, W])`.
- Leg freezing: Achieved by applying a high-weight `rest_cost` to the 12 leg
  joints, effectively locking them in their default configuration.

## Pyroki and jaxlie Integration

- Pyroki's `forward_kinematics` returns parameters in `wxyz_xyz` format (shape
  `(N, 7)`), which can be directly converted to `jaxlie.SE3` using the
  constructor: `jaxlie.SE3(fk[idx])`.
- `jaxlie` uses `wxyz` (scalar-first) order for quaternions, consistent with
  standard JAX conventions.
- When computing teleop transforms with `jaxlie`, rotation and translation
  deltas should be computed separately and recombined for consistency with
  common teleop practices.
- Use `tyro` for CLI consistent with other demos in the repository.
- `Teleop.subscribe` callbacks are synchronous; use
  `asyncio.get_running_loop().create_task()` to call async methods like
  `publish_joint_state` from within them.
- `UnitreeH1Robot` is located in `teleop_xr.ik.robots.h1_2`.
- Always convert JAX arrays to NumPy arrays before passing them to components
  that expect NumPy.
