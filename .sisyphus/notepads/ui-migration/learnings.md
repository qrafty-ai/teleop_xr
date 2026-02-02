- Replaced custom CSS/div structure in `webxr/ui/camera.uikitml` with `<Panel>` and `<Text>` from Horizon Kit.
- Preserved `id="camera-label"` on the `<Text>` element to ensure compatibility with `webxr/src/panels.ts` which updates the label text dynamically.
- Noted that video overlay is handled via Three.js Mesh attachment in `panels.ts`, so it is independent of the internal HTML structure of the panel, provided the panel entity exists.
## Horizon Kit Configuration
- Always use 'import * as horizonKit' when importing from @pmndrs/uikit-horizon for kit registration in World.create.
- Icons from @pmndrs/uikit-lucide can be imported selectively and passed as an object in the kits array to optimize bundle size.
- Example kits registration:
```typescript
import * as horizonKit from '@pmndrs/uikit-horizon';
import { LogInIcon } from '@pmndrs/uikit-lucide';

World.create(container, {
  features: {
    spatialUI: {
      kits: [horizonKit, { LogInIcon }],
    },
  },
});
```

## Welcome UI Migration
- Refactored `webxr/ui/welcome.uikitml` to use Horizon Kit components.
- Replaced HTML structure with `<Panel>`, `<Button>`, and `<Text>`.
- Utilized `<ButtonIcon>` and `<LogInIcon>` for the button.
- Confirmed that removing `<style>` blocks works as Horizon Kit components provide default styling.
- Verified compilation success with `npm run build`.

## Camera Settings UI Migration
- Refactored `webxr/ui/camera_settings.uikitml` to use Horizon Kit components `<Panel>`, `<Text>`, and `<Button>`.
- Preserved row IDs (`row-0`..`row-5`), label IDs (`label-0`..`label-5`), and button IDs (`btn-0`..`btn-5`) for `camera_settings_system.ts` compatibility.
- Kept `.hidden` and `.active` CSS classes in `<style>` block as they are dynamically toggled by the system logic (for visibility and active state visual feedback).
- Used flexbox layout on `div` elements for rows to ensure `display: flex` / `display: none` toggling works correctly.
- Verified compilation success with `npm run build`.

## Teleop UI Migration
- Refactored `webxr/ui/teleop.uikitml` to use Horizon Kit components.
- Replaced custom CSS containers with `<Panel>` and `<Container>` (using flex props).
- Replaced buttons with `<Button>` and `<ButtonIcon>` (using `VideoIcon`, `SettingsIcon`, `XIcon`).
- Preserved IDs (`status-text`, `fps-text`, `latency-text`, `camera-button`, etc.) for `teleop_system.ts`.
- Maintained `.connected` and `.status-value` classes via minimal `<style>` block, as `teleop_system.ts` toggles them.

## 2026-02-02 Task: ui-migration (Final Update)
The UI migration to Horizon Kit is complete.

### Key Conclusions
- **Horizon Kit Components**: Elements like `<Panel>`, `<Button>`, and `<Text>` provide a consistent glass-like spatial UI.
- **ID Preservation**: Keeping original IDs is critical for TypeScript systems that use `getElementById` to drive UI state (e.g., `status-text`, `xr-button`).
- **Layouting**: Standard flexbox properties (`flexDirection`, `justifyContent`, `alignItems`) work as expected in UIKitML containers.
- **State Classes**: Functional CSS classes for visibility (`.hidden`) or status (`.connected`) should be preserved in small `<style>` blocks to minimize changes to system logic.
- **Icon Kits**: Selective imports from `@pmndrs/uikit-lucide` keep the bundle size small while providing high-quality spatial icons.
