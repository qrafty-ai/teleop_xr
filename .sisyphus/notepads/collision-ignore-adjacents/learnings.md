Added collision_ignore_pairs to BaseRobot as an empty tuple by default. This property will be used to filter out self-collision pairs in subclasses.
Updated UnitreeH1Robot to use collision_ignore_pairs and explicitly enable ignore_immediate_adjacents.
Currently collision_ignore_pairs is empty for H1_2, but the mechanism is now in place for future tuning.
Added @override decorators to all overridden methods in UnitreeH1Robot to satisfy lsp_diagnostics.
### TeaArmRobot Update
- Updated `TeaArmRobot` in `teleop_xr/ik/robots/teaarm.py`.
- Implemented `collision_ignore_pairs` property returning an empty tuple.
- Updated `RobotCollision.from_sphere_decomposition` and `RobotCollision.from_urdf` to use `ignore_immediate_adjacents=True` and `user_ignore_pairs=self.collision_ignore_pairs`.
- Verified changes with `lsp_diagnostics`.
### Verification Test
- Created `tests/test_collision_ignore_logic.py` to verify the collision ignore mechanism.
- Confirmed that `ignore_immediate_adjacents=True` correctly filters out parent-child link pairs from the active collision indices.
- Confirmed that `user_ignore_pairs` correctly filters out custom specified link pairs.
- The implementation in `pyroki.collision.RobotCollision.from_sphere_decomposition` handles both mechanisms simultaneously, allowing for broad adjacent filtering combined with surgical ignores for specific non-adjacent overlaps.
