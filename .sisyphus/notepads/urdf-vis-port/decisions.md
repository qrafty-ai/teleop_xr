# Architectural Decisions

## Robot Visualization Module
- **Encapsulation**: Created `RobotVisModule` in `teleop_xr/robot_vis.py` to handle all visualization-related tasks (asset serving, URDF serving, state broadcasting).
- **Integration**: Integrated into `Teleop` class as an optional component (`settings.robot_vis`). This ensures that users who don't need visualization aren't affected.
- **Asset Serving Strategy**:
  - `package://` paths: Mapped to a configured `mesh_path`. If `mesh_path` is missing, we log a warning and try the path as-is (though it likely won't work), rather than guessing system paths, for security and predictability.
  - Relative paths: Resolved relative to the directory containing the URDF file.
- **State Broadcasting**: Added `publish_joint_state` to `Teleop` which delegates to `RobotVisModule.broadcast_state`. This allows the main application loop to easily push updates.
## Modular Robot Visualization Backend
- **RobotVisModule**: Decoupled from main Teleop class to allow optional loading.
- **Asset Serving**: Custom route `/assets/{file_path:path}` handles both `package://` (via `mesh_path`) and relative paths (relative to URDF directory).
- **URDF Serving**: Dedicated `/robot.urdf` endpoint.
- **WebSocket Protocol**: New `robot_config` message on connection and `robot_state` for joint updates.

## Robot Model System (Frontend)
- **Modern Loader**: Implemented `RobotModelSystem` in `webxr/src/robot_system.ts` using `URDFLoader` and `three-stdlib`.
- **Mesh Loading**: Integrated `STLLoader`, `ColladaLoader`, and `OBJLoader` via `LoadingManager` to support standard robot mesh formats.
- **URL Mapping**: Configured `URDFLoader.packages` to map `package://` URLs to `/assets/package://`, aligning with the backend `RobotVisModule` route.
- **Coordinate System Fix**: Applied a -90deg rotation on the X-axis to the loaded robot model to convert from URDF's Z-up to Three.js's Y-up coordinate system.
- **ECS Integration**:
    - The system is registered in `webxr/src/index.ts`.
    - `TeleopSystem` was updated to dispatch `robot_config` and `robot_state` messages to the `RobotModelSystem`.
## Renaming Robot Asset Route
- Date: 2026-01-31
- Decision: Renamed the robot asset route from `/assets/` to `/robot_assets/` in both the backend (`teleop_xr/robot_vis.py`) and the frontend (`webxr/src/robot_system.ts`).
- Rationale: Avoid collision with Vite's static asset directory while maintaining functionality for loading robot meshes and URDFs.
