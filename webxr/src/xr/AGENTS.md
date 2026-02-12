# webxr/src/xr/ — Core WebXR Logic

ECS-based XR systems using @iwsdk/core. Handles input, rendering, video, and robot visualization.

## OVERVIEW
Systems registered with `World` manage entity lifecycle and data flow. Custom ECS pattern (not React-Three-Fiber).

## STRUCTURE
```
xr/
├── index.ts                   # World init, system registration
├── teleop_system.ts           # Input → WebSocket streaming
├── robot_system.ts            # URDF loading, joint sync
├── controller_camera_system.ts # Wrist camera panels
├── panels.ts                  # Draggable UI panels (handle/panel decoupling)
├── video.ts                   # WebRTC video client
├── console_stream.ts          # Quest log → Python server
├── track_routing.ts           # Video track → panel mapping
└── __tests__/                 # Vitest tests (excluded from coverage)
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add new system | `index.ts` | Register in `initWorld()` |
| Input gathering | `teleop_system.ts` | Sends XR state at update rate |
| Video panel logic | `panels.ts` | CameraPanel, ControllerCameraPanel |
| Debug logging | `console_stream.ts` | Intercepts console.log |

## CONVENTIONS

### ECS Patterns
- Systems: `class X extends createSystem({})`
- Components: `createComponent` with typed data
- Entities: `world.createTransformEntity()`

### UI Decoupling
- **Handle**: Grabbable entity
- **Panel**: UI entity (not parented to handle)
- **Sync**: Manual transform copy in system

### VR-in-AR
- Always use `SessionMode.ImmersiveAR`
- VR mode: add opaque skybox to hide passthrough

## ANTI-PATTERNS (xr)

- ❌ NEVER parent panel to handle entity
- ❌ NEVER access ECS queries during UI events (freezes)
- ✅ ALWAYS use GlobalRefs for external access

## NOTES

### Test Exclusions
`teleop_system.ts`, `robot_system.ts` excluded from coverage (require XR hardware)
