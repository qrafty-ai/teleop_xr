# Switch to Static Sphere Collision Loading

## Context
The user wants to remove all dynamic sphere calculation features and instead load a pre-generated `sphere.json` file for each robot. Additionally, an interactive script should be provided to ease the generation of these files, leveraging the `--robot-class` mechanism for robot description management.

## Analysis
- **Current State**: `teleop_xr` uses a dynamic `ballpark`-based engine with a file-based cache to generate collision spheres at runtime.
- **Desired State**:
    - Remove `teleop_xr/ik/collision.py` and `teleop_xr/ik/collision_sphere_cache.py` from runtime dependencies.
    - Robot classes should load `sphere.json` from a predictable location.
    - An interactive script `scripts/generate_spheres.py` should be provided to generate these files.

## Technical Decisions
1.  **Storage**: Static `sphere.json` files will be stored in `teleop_xr/ik/robots/assets/{robot_name}/sphere.json`.
2.  **Base Class**: Add a `sphere_decomposition` property to `BaseRobot` that loads the JSON file if it exists.
3.  **Generation Script**: Provide `scripts/generate_spheres.py` that:
    - Uses `teleop_xr.ik.loader.load_robot_class` to find the robot.
    - Uses `ballpark` for interactive/automated spherization.
    - Saves to the correct asset directory.

## Steps
- [x] 1. **Define Asset Paths**: Establish the `teleop_xr/ik/robots/assets/` directory structure.
- [x] 2. **Update `BaseRobot`**:
    - Add a helper to load `sphere.json` from the asset directory based on the robot name.
    - Modify the constructor to attempt loading this file.
- [x] 3. **Update `UnitreeH1Robot` & `TeaArmRobot`**:
    - Remove dynamic calculation and caching code.
    - Use the base class mechanism to load spheres.
- [x] 3b. **Update `FrankaRobot` & Tests**:
    - Implement `name` property for `FrankaRobot` to satisfy `BaseRobot` abstraction.
    - Update mock classes in `tests/test_robots.py`.
- [x] 4. **Implement `scripts/generate_spheres.py`**:
    - Re-use `teleop_xr.ik.loader`.
    - Interface with `ballpark` (similar to their interactive script).
- [x] 5. **Cleanup**: Remove `teleop_xr/ik/collision.py` (except parts needed for script) and `teleop_xr/ik/collision_sphere_cache.py`.
- [x] 6. **Verification**:
    - Verify robots load spheres from JSON.
    - Verify script can generate a valid JSON.

## Acceptance Criteria
- [x] No `ballpark` or `trimesh` calls at robot runtime (unless in script).
- [x] Robots correctly initialize `RobotCollision` from a local `sphere.json` file.
- [x] `scripts/generate_spheres.py --robot-class UnitreeH1Robot` works.
