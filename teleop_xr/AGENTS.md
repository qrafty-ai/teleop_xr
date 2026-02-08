# teleop_xr/ — Python Backend

Main Python package providing WebSocket server, IK solver, and ROS2 integration.

## OVERVIEW
FastAPI/WebSocket server that bridges WebXR frontend to robot control. Handles pose transformation (RUB→FLU), video streaming (WebRTC), and joint state synchronization.

## STRUCTURE
```
teleop_xr/
├── __init__.py         # Teleop class (WebSocket handler, video, control)
├── ik/                 # JAX-based IK system
├── demo/               # Interactive TUI demo
├── ros2/               # ROS2 publisher/subscriber
├── utils/              # Filters, transform limiters
├── ram.py              # Resource Asset Manager (URDF fetcher)
└── cert.pem, key.pem   # SSL certificates (bundled)
```

## WHERE TO LOOK

| Task | File | Line Range |
|------|------|------------|
| WebSocket lifecycle | `__init__.py` | 512-785 (websocket_endpoint) |
| Pose conversion (RUB→FLU) | `__init__.py` | 277-313 (__convert_pose_to_ros) |
| Control claiming | `__init__.py` | 610-644 (check_or_claim_control) |
| Console log receiver | `__init__.py` | 754-764 (console_log handling) |
| Video session mgmt | `__init__.py` | 346-378 (_start_video_session) |
| CLI defaults | `common_cli.py` | All |
| URDF caching | `ram.py` | All |

## CONVENTIONS

### Coordinate Systems
- **Input**: WebXR RUB (Right-Up-Back)
- **Output**: ROS2 FLU (Forward-Left-Up)
- **Transform**: `TF_RUB2FLU` matrix applied in `__convert_pose_to_ros`

### Async Patterns
- Uses `asyncio.Lock` for control claim synchronization
- `anyio` in tests for async WebSocket testing

### Resource Loading
- URDFs fetched via `ram.get_resource()` (caches to `~/.cache/teleop_xr/`)
- Custom robots: assets in `ik/robots/assets/`

## ANTI-PATTERNS (teleop_xr)

- ❌ NEVER use headset timestamps for ROS TF (use PC time)
- ❌ NEVER require rospack for RAM module
- ✅ ALWAYS validate pose jumps (>5cm lin or >35° ang)

## NOTES

### Gotchas
- `teleop_xr/__init__.py` is 817 lines (largest file, refactor candidate)
- SSL certs bundled in package (not env-specific)
- `utils/lite6.urdf` breaks convention (should be in `ik/robots/assets/`)
