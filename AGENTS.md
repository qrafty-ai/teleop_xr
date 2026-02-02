# Teleop Project Context

## WebXR App Debugging

### Console Log Streaming
Quest VR headset doesn't have accessible browser console. To debug:
1. WebXR app intercepts console.log/warn/error via `console_stream.ts`
2. Logs are sent as `{type: "console_log", data: {level, message}}` over WebSocket
3. Python server (`teleop/__init__.py`) receives and prints `[WebXR:level] message` to terminal

### Key Architecture
- `webxr/src/` - TypeScript WebXR app using @iwsdk/core (ECS framework on Three.js)
- `webxr/src/panels.ts` - DraggablePanel, CameraPanel, ControllerCameraPanel classes
- `webxr/src/controller_camera_system.ts` - System for controller-attached panels with billboard behavior
- `webxr/src/console_stream.ts` - Console log interceptor for Quest debugging
- `webxr/src/video.ts` - VideoClient for WebRTC video streaming
- `webxr/src/teleop_system.ts` - XR input gathering and WebSocket streaming

### Orchestration & Planning
- **Always commit `.sisyphus/` files**: Plan files (`.sisyphus/plans/`) and notepads (`.sisyphus/notepads/`) must be included in commits to preserve work history and context.
- **Boulder State**: Ensure `.sisyphus/boulder.json` reflects the current active plan.
