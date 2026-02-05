# Robot Vis Axis Integration Plan

## Context
The user wants to add a 3D axis helper to the WebXR robot visualization for debugging purposes. This requires modifying the WebXR frontend to render axes at the robot's origin and ideally providing a UI toggle for it.

## Objectives
1.  **Add `showAxes` Setting**: Update the app store to include a persistent setting for toggling the axes visibility.
2.  **Implement Axes Rendering**: Modify `RobotModelSystem` to create and manage a `THREE.AxesHelper` attached to the robot's root object.
3.  **Add UI Toggle**: Update `AdvancedSettingsPanel` to expose the `showAxes` switch to the user.

## Steps
- [ ] 1. Update `webxr/src/lib/store.ts`:
    - Add `showAxes: boolean` to `AdvancedSettings` type.
    - Set default to `false`.
- [ ] 2. Update `webxr/src/components/dashboard/AdvancedSettingsPanel.tsx`:
    - Add a "Show Axes" switch in the Visualization section.
- [ ] 3. Update `webxr/src/xr/robot_system.ts`:
    - Import `AxesHelper`.
    - Create `axesHelper` instance attached to `robotObject3D`.
    - Subscribe to `useAppStore` to toggle visibility based on `showAxes`.
- [ ] 4. Rebuild Frontend:
    - Run `npm run build` in `webxr/` to verify no errors.

## Verification
- Can be verified by running the build.
- Visual verification would require running the app (not possible for agent), but code review of the logic is sufficient.

## Learnings
-
