Refactored `webxr/src/xr/camera_config.ts` to use the global Zustand store from `../lib/store`.

Summary of changes:
- Removed local `currentConfig` and `handlers`.
- `getCameraEnabled` now delegates to `useAppStore.getState().getCameraEnabled`.
- `setCameraEnabled` now delegates to `useAppStore.getState().setCameraEnabled`.
- `onCameraConfigChanged` now uses `useAppStore.subscribe` with a state comparison check to only trigger the handler when `cameraConfig` actually changes, and returns the unsubscribe function directly.
- Maintained immediate execution of handler with current config in `onCameraConfigChanged` for backward compatibility.
