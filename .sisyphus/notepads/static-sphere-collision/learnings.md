
## Asset Path Definition (2026-02-06)
- Established the base asset directory structure at `teleop_xr/ik/robots/assets/`.
- Created subdirectories for `h1_2` and `teaarm` robots.
- This structure supports the shift from dynamic caching to local, committed asset files.

## BaseRobot Asset Loading
- Added `name` abstract property to `BaseRobot` to facilitate asset lookups.
- Implemented `load_sphere_decomposition` in `BaseRobot` to automatically load `sphere.json` from `teleop_xr/ik/robots/assets/{name}/sphere.json`.
- `BaseRobot.__init__` now calls this loader and stores it in `self.sphere_decomposition`.
- Standardized return type to `dict[str, Any] | None`.

## Robot Subclass Updates for Static Sphere Loading
- UnitreeH1Robot and TeaArmRobot were updated to leverage the new BaseRobot sphere loading mechanism.
- Both robots now implement the `name` property ("h1_2" and "teaarm" respectively).
- Calling `super().__init__()` in the subclass constructor triggers `load_sphere_decomposition()` in the base class.
- Dynamic sphere generation and caching code was removed from `TeaArmRobot`.
- Both robots now use `self.sphere_decomposition` to initialize `self.robot_coll` via `pk.collision.RobotCollision.from_sphere_decomposition`.
- Fallback to `RobotCollision.from_urdf` is maintained if no sphere decomposition is found.

### Sphere Generation Script
- Implemented `scripts/generate_spheres.py` using `tyro`.
- Reused `teleop_xr.ik.loader.load_robot_class` for robot loading.
- Fixed JSON serialization issue in `teleop_xr.ik.collision.generate_collision_spheres` where `np.float32` was being returned.
- Assets are saved in `teleop_xr/ik/robots/assets/{robot.name}/sphere.json`.


- Moved sphere generation logic from `teleop_xr/ik/collision.py` to `scripts/generate_spheres.py`.
- Deleted `teleop_xr/ik/collision.py` and `teleop_xr/ik/collision_sphere_cache.py` to remove dynamic calculation and caching from the main package.
- Deleted related tests: `tests/test_collision_sphere_generation.py`, `tests/test_collision_sphere_cache.py`, and `tests/test_collision_sphere_contract.py`.
- The package now relies on static assets for sphere decomposition, which are generated using the standalone script.
- Verified that `scripts/generate_spheres.py` works correctly with `uv run`.

### FrankaRobot Update
- Implemented `name` property returning "franka".
- Added `super().__init__()` to trigger sphere decomposition loading in `BaseRobot`.
- Added `@override` decorators to all overriding methods to satisfy LSP and follow project standards.
- Updated `MockBaseRobot` in tests to include `name` property and proper overrides.
