# Interactive Sphere Generation Script

## Context
The user wants `scripts/generate_spheres.py` to be interactive, providing a GUI to visualize and tune the sphere decomposition before saving. This should be similar to the `spherize_robot_interactive.py` script in the `ballpark` repository.

## Analysis
- **Current State**: `scripts/generate_spheres.py` is a non-interactive CLI that saves spheres to a fixed path.
- **Desired State**:
    - Launch a `viser` server for visualization.
    - Provide sliders and checkboxes for `ballpark` parameters (`target_spheres`, `padding`, `refine`, etc.).
    - Use `teleop_xr` robot loader to load robots via `--robot-class`.
    - "Export" button in the GUI saves to the standard asset path.

## Technical Decisions
1.  **GUI Framework**: Use `viser` as it is the framework used by `ballpark` and the reference script.
2.  **Logic Port**: Port the `_SpheresGui` and `_SphereVisuals` helper classes from the `ballpark` reference script.
3.  **Robot Loading Integration**:
    - Load the robot class using `load_robot_class(robot_class)`.
    - Extract the URDF (either from `urdf_path` or `urdf_string`).
    - Create `ballpark.Robot` instance for the GUI logic.

## Steps
- [x] 1. **Add `viser` to `pyproject.toml`**: (Already done by Orchestrator, but subagent should verify).
- [x] 2. **Refactor `scripts/generate_spheres.py`**:
    - Implement `viser` server setup.
    - Port interactive GUI components from `ballpark`.
    - Integrate `teleop_xr` robot loading.
    - Ensure "Export" logic targets `teleop_xr/ik/robots/assets/{robot.name}/sphere.json`.
- [x] 3. **Verification**:
    - Run the script and verify the GUI launches.
    - Verify that exporting from the GUI saves the file to the correct location.

## Acceptance Criteria
- [x] `uv run python scripts/generate_spheres.py --robot-class h1` launches a `viser` server.
- [x] The GUI provides interactive controls for sphere decomposition.
- [x] Exporting from the GUI correctly saves `sphere.json` in the robot's asset directory.
