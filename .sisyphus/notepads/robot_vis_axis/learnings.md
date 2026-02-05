### 2026-02-05: Added showAxes to AdvancedSettings

- Updated  with  field in  type.
- Set default value to  in .
- Verified with .
### 2026-02-05: Added showAxes to AdvancedSettings

- Updated `webxr/src/lib/store.ts` with `showAxes: boolean` field in `AdvancedSettings` type.
- Set default value to `false` in `defaultAdvancedSettings`.
- Verified with `lsp_diagnostics`.

### 2026-02-05: Added UI toggle for showAxes

- The `useAppStore` in `webxr/src/lib/store.ts` already contained the `showAxes` property in `AdvancedSettings`, so no store update was needed.
- `AdvancedSettingsPanel` uses a straightforward pattern of destructuring state and creating handler functions for each setting.
- Followed the existing pattern for `robotVisible` to implement `showAxes`, ensuring consistency in the UI and code structure.
- Placed the "Show Axes" switch immediately after "Robot Visibility" in the "Visualization" section as they are related visual toggles.

### 2026-02-05: Implemented AxesHelper in RobotModelSystem

- Implemented `AxesHelper` toggle functionality in `RobotModelSystem`.
- Subscribed to `advancedSettings.showAxes` in `init()` to dynamically update visibility.
- Created `AxesHelper` in `onRobotConfig()` and attached it to `robotObject3D`.
- Ensured consistent state synchronization with the store.
