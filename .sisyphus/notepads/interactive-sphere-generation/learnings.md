
## Refactored generate_spheres.py
- Integrated `viser` for interactive sphere generation.
- Ported GUI components from `ballpark` reference script.
- Used `teleop_xr.ik.loader.load_robot_class` to support various robot specifications.
- Export logic correctly targets `teleop_xr/ik/robots/assets/{robot_name}/sphere.json`.
- Fixed type hint issues with `viser.GuiInputHandle` by adding `[Any]`.
- Used `robot_inst.get_vis_config()` to retrieve `urdf_path` safely.
