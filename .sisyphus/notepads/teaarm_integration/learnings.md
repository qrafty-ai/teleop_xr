# TeaArm Integration Learnings

## RAM Local Support
- `teleop_xr/ram.py` handles URDF/Xacro fetching and processing.
- `_resolve_package` and `_resolve_uri` are key for path resolution.
- `process_xacro` is the core conversion function.

## Robot Implementation
- `BaseRobot` in `teleop_xr/ik/robot.py` is the template.
- Dual-arm robots (like H1) override `forward_kinematics`.
- EE link names for TeaArm: `frame_left_arm_ee`, `frame_right_arm_ee`.

## Registry
- Uses `pyproject.toml` entry points `teleop_xr.robots`.

### RAM Local Resource Loading
- Supported local URDF/Xacro loading via `repo_root` parameter in `get_resource`.
- Implemented unique caching for local resources in `~/.cache/ram/processed` using SHA256 hashes of absolute path, relative path, and xacro arguments. This prevents polluting the local workspace and handles collisions between different local repositories.
- Enforced relative paths for `path_inside_repo` to maintain consistency with the `repo_root` / `repo_url` logic.
- Ensured mutual exclusivity between `repo_url` and `repo_root`.
