# Disable Collision for Adjacent Links and Support Custom Ignore Pairs

## Context
The user requested to "disable collision checking between adjacent links".
While `pyroki` supports `ignore_immediate_adjacents=True` by default, this plan ensures it is explicitly configured and adds support for additional robot-specific ignore pairs (e.g. links separated by 2 joints that still overlap, or links physically adjacent but not in the URDF tree).

## Analysis
- `pyroki`'s `RobotCollision` has a `user_ignore_pairs` parameter.
- `ignore_immediate_adjacents` is enabled by default in `pyroki` but should be explicitly passed in `teleop_xr` for clarity.
- Current robot implementations (`h1_2.py`, `teaarm.py`) do not expose a way to define custom ignore pairs.
- Some robots (like H1) might have links that are physically adjacent but not directly connected by a joint in the simplified URDF used for IK.

## Objectives
1.  **Expose Ignore Pairs**: Add a `collision_ignore_pairs` property/method to `BaseRobot` and its subclasses.
2.  **Explicit Configuration**: Explicitly pass `ignore_immediate_adjacents=True` when instantiating `RobotCollision`.
3.  **H1 & TeaArm Updates**: Populate ignore pairs for `h1_2` and `teaarm` based on observation or standard robot configs.
4.  **Verification**: Add a test to verify that adjacent links and custom pairs are indeed ignored.

## Steps
- [x] 1. **Update `BaseRobot`**: Add an empty `collision_ignore_pairs` property to `BaseRobot` in `teleop_xr/ik/robot.py`.
- [x] 2. **Update `UnitreeH1Robot`**:
    - Explicitly pass `ignore_immediate_adjacents=True`.
    - Implement `collision_ignore_pairs` with any known overlapping pairs (if any identified).
- [x] 3. **Update `TeaArmRobot`**:
    - Explicitly pass `ignore_immediate_adjacents=True`.
    - Implement `collision_ignore_pairs`.
- [x] 4. **Add Verification Test**: Create `tests/test_collision_ignore_logic.py` to verify that the ignore list is correctly applied to `RobotCollision.active_idx_i/j`.

## Acceptance Criteria
- [x] `UnitreeH1Robot` and `TeaArmRobot` instantiate `RobotCollision` with explicit `ignore_immediate_adjacents=True`.
- [x] `user_ignore_pairs` from the robot classes are passed to `RobotCollision`.
- [x] Test proves that parent-child pairs are NOT in the active collision list.
- [x] Test proves that custom ignore pairs are NOT in the active collision list.
