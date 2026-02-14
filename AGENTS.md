# TELEOP_XR PROJECT KNOWLEDGE BASE

**Generated:** 2026-02-07 21:59:22 EST
**Commit:** 7f5675c
**Branch:** agent/init_knowledge

## OVERVIEW
TeleopXR transforms VR/AR headsets into precision robot controllers via WebXR. **Hybrid monorepo**: Python backend (FastAPI/WebSocket server, JAX-powered IK) + Next.js/TypeScript frontend (ECS-based WebXR app using @iwsdk/core).

**Core Stack:** Python 3.10+, Next.js 16, React 19, Three.js (super-three fork), JAX, Pyroki (IK), Hatch (build), uv (deps)

## STRUCTURE
```
teleop_xr-worktrees/agent-init_knowledge/
├── teleop_xr/          # Python backend: FastAPI server, IK system, ROS2 bridge
│   ├── ik/             # JAX-based IK solver, robot models (Franka, H1, TeaArm)
│   ├── demo/           # CLI demo with TUI
│   └── ros2/           # ROS2 integration
├── webxr/              # Next.js frontend: WebXR app (ECS architecture)
│   ├── src/xr/         # Core XR logic (systems, panels, video streaming)
│   ├── src/components/ # UI components (dashboard, settings)
│   └── ui/             # .uikitml custom markup (VR panels)
├── tests/              # Python test suite (pytest + anyio)
├── scripts/            # Build utilities, collision GUI
└── docs/               # MkDocs documentation
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| WebSocket protocol | `teleop_xr/__init__.py` | 817L monolith; control claiming, video sessions |
| XR state streaming | `webxr/src/xr/teleop_system.ts` | Sends poses/buttons → Python |
| IK solver | `teleop_xr/ik/solver.py` | JAX-compiled, uses pyroki/jaxls |
| Robot definitions | `teleop_xr/ik/robots/` | BaseRobot subclasses (FK, costs, URDFs) |
| Video routing | `webxr/src/xr/track_routing.ts` | Maps WebRTC tracks → panels |
| UI panels | `webxr/ui/*.uikitml` | Custom VR panel markup |
| Debug console | `webxr/src/xr/console_stream.ts` | Quest logs → Python terminal |
| CLI config | `teleop_xr/common_cli.py` | Shared flags (host, port, speed) |

## CODE MAP

**Python Entry Points:**
- `teleop_xr/__init__.py::Teleop` — Main server class (FastAPI + WebSocket)
- `teleop_xr/demo/__main__.py` — Interactive demo (TUI, IK mode)
- `teleop_xr/ros2/__main__.py` — ROS2 integration

**WebXR Entry Points:**
- `webxr/src/xr/index.ts::initWorld()` — World initialization, system registration
- `webxr/src/app/page.tsx` — Next.js root page

**Critical Systems:**
- `TeleopSystem` — Input gathering, WebSocket streaming
- `RobotModelSystem` — URDF visualization, joint state sync
- `ControllerCameraPanelSystem` — Wrist-mounted camera panels

## CONVENTIONS

### Build & Deploy
- **Python**: `uv sync` (dev), `hatch build` (package)
- **WebXR**: `cd webxr && npm run build` → output to `webxr/out/`
- **Hybrid Build**: `hatch_build.py` bundles frontend into Python package

### Testing
- **Python**: `pytest` with `@pytest.mark.anyio` for async
- **WebXR**: `vitest` (config: `webxr/vitest.config.ts`)
- **Coverage**: Python (pytest-cov), WebXR (v8 provider, 80% threshold)

### Dependency Management
- **Python**: `pyproject.toml` + `uv.lock`
  - IK deps: `pyroki`, `ballpark` (from GitHub, not PyPI)
- **WebXR**: `webxr/package.json` + `package-lock.json`
  - Forked Three.js: `npm:super-three@0.177.0`

### Coordinate Systems
- **WebXR**: RUB (Right-Up-Back, standard WebXR)
- **Python**: FLU (Forward-Left-Up, ROS2 standard)
- **Transform**: Applied in `teleop_xr/__init__.py::__convert_pose_to_ros`

### WebXR Patterns
- **VR Simulation**: Uses `immersive-ar` + opaque skybox (no `immersive-vr`)
- **ECS**: Systems via `createSystem`, entities via `world.createTransformEntity()`
- **UI Decoupling**: Handles/panels = separate entities (manual transform sync)

## ANTI-PATTERNS (THIS PROJECT)

**Orchestration:**
- ❌ NEVER commit without `.sisyphus/` plan/notepad files
- ✅ ALWAYS ensure `.sisyphus/boulder.json` reflects active plan

**Python:**
- ❌ NEVER suppress type errors (`# type: ignore`, `as any`)
- ❌ NEVER require full ROS installation for RAM module
- ✅ ALWAYS use PC timestamps for ROS TF (avoid headset drift)
- ✅ ALWAYS squeeze both grips for IK (Deadman Rule)

**WebXR:**
- ❌ NEVER parent panels to handle entities in ECS
- ❌ NEVER use `<style>` blocks in new UI (except state classes)
- ✅ ALWAYS use `immersive-ar` + `LocalFloor` for sessions
- ✅ ALWAYS import `* as horizonKit` from `@pmndrs/uikit-horizon`

**Build:**
- ⚠️  CI runs duplicate frontend builds (`typescript-ci.yml` + `ci.yml`)
- ⚠️  `typescript-ci.yml` linter has `continue-on-error: true`

## UNIQUE STYLES

### Console Log Streaming
Quest VR headset lacks accessible browser console. Debugging:
1. `webxr/src/xr/console_stream.ts` intercepts `console.log/warn/error`
2. Logs sent as `{type: "console_log", data: {level, message}}` via WebSocket
3. Python server (`teleop_xr/__init__.py`) prints `[WebXR:level] message` to terminal

### Resource Asset Manager (RAM)
Lightweight package: URDF/mesh files fetched on-demand from GitHub.
- **Module**: `teleop_xr/ram.py`
- **Usage**: `get_resource("franka/panda.urdf")` → caches to `~/.cache/teleop_xr/`
- **Custom Robots**: Use `assets/` subfolder (e.g., `custom_robot/collision.json`)

### UIKitML
Custom markup for VR panels (XML-like, compiled to JSON).
- **Location**: `webxr/ui/*.uikitml` (outside `src/`)
- **Build**: Vite plugin `@iwsdk/vite-plugin-uikitml`
- **Runtime**: Loaded as JSON, rendered via `@pmndrs/uikit`

## COMMANDS

```bash
# Dev (Python)
uv sync                          # Install deps
uv run python -m teleop_xr.demo  # Run demo
pytest                           # Run tests

# Dev (WebXR)
cd webxr
npm install                      # Install deps
npm run dev                      # Dev server
npm run build                    # Production build
npm run test                     # Run Vitest

# Build Package
npm run build --prefix webxr     # Build frontend first
hatch build                      # Build Python package (includes frontend)

# ROS2 Integration
uv run python -m teleop_xr.ros2  # Launch ROS2 bridge
```

## NOTES

### Gotchas
- **SSL Certs**: `cert.pem`/`key.pem` in `teleop_xr/` (bundled in package)
- **URDF Outlier**: `teleop_xr/utils/lite6.urdf` breaks asset convention (should be in `ik/robots/assets/`)
- **WebXR Test Exclusions**: `teleop_system.ts`, `robot_system.ts` excluded from coverage (XR hardware dependencies)
- **Fragmented Tests**: WebXR tests in multiple `__tests__/` dirs (not consolidated)

### Performance
- **IK Solver**: JAX JIT-compiled, ~15 iterations per frame
- **Video Streaming**: WebRTC via `aiortc`, multi-track support
- **Control Timeout**: 5s inactivity → auto-release control

### Architecture Decisions
- **Hybrid Monorepo**: Python root + Next.js subfolder (no workspace orchestration)
- **ECS Over React-Three-Fiber**: Game engine pattern for XR (not web standard)
- **Relative Motion**: IK targets = `Robot_Init + (XR_Current - XR_Init)` (prevents jumping)
