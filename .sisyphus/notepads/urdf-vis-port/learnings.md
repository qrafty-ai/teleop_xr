
## OpenArm Assets Setup
- Main URDF file: tests/fixtures/openarm_description/urdf/openarm_description.urdf
- Repository: https://github.com/enactic/openarm_description cloned into tests/fixtures/openarm_description

## Frontend Dependencies
- Installed `urdf-loader` and `three-stdlib` for URDF visualization in WebXR.
- Added `@types/three` as dev dependency for TypeScript support.

## URDF Loading in WebXR
- `URDFLoader` from `urdf-loader` works seamlessly with `three-stdlib` loaders when they are registered in the same `LoadingManager` using `addHandler`.
- Mapping `package://` URLs can be done by providing a function to `loader.packages`, which is more flexible than a static mapping if the backend uses a custom asset route.
- `@iwsdk/core` systems can be retrieved from the world using `this.world.getSystem(SystemClass)`, which is useful for cross-system communication.

## Demo Script Implementation
- When running a background task alongside `Teleop.run()` (which is blocking and runs uvicorn), using a separate `threading.Thread` with its own `asyncio` event loop is a viable approach to call `publish_joint_state`.
- Ensure absolute paths are used for URDF and mesh files to avoid issues when running the script from different directories.
- Using `uv run` ensures that all project dependencies (like `uvicorn`, `fastapi`) are available in the execution environment.
