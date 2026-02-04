# Plan: Replace Unitree H1_2 model with external repository version using RAM

## TL;DR
> **Quick Summary**: Replace the local `h1_2.urdf` model with the version from `unitreerobotics/xr_teleoperate` GitHub repository using the newly created `ram` module. This removes the need for local asset storage in the repo.
>
> **Deliverables**:
> - Modified `teleop_xr/ik/robots/h1_2.py`: Use `ram.get_resource` to fetch URDF.
> - Removed `teleop_xr/assets/h1_2/`: Clean up local assets.

---

## Work Objectives
1. Update `UnitreeH1Robot` to fetch its URDF and meshes from the specified GitHub repository via the `ram` module.
2. Verify that the robot visualization and IK still work with the external assets.
3. Remove the local assets to reduce repository size and follow the "no dump" policy.

## TODOs
- [x] 1. Update `teleop_xr/ik/robots/h1_2.py`
  - Import `teleop_xr.ram`.
  - Use `ram.get_resource` with:
    - `repo_url="https://github.com/unitreerobotics/xr_teleoperate.git"`
    - `path_inside_repo="assets/h1_2/h1_2.urdf"`
  - Ensure `self.mesh_path` points to the directory containing the fetched URDF (which should be in the RAM cache).

- [x] 2. Verify with tests
  - Run `pytest tests/test_ik_api.py` and `tests/test_robot_vis.py` to ensure no regressions.

- [x] 3. Cleanup local assets
  - `rm -rf teleop_xr/assets/h1_2`

---

## Verification Strategy
- Run existing tests that load the H1_2 robot.
- Verify `UnitreeH1Robot` can be instantiated and returns valid paths from RAM.
