# webxr/src/components/ — UI Components

React components for dashboard, settings, and XR scene integration.

## OVERVIEW
Next.js App Router components. Split between dashboard (2D UI) and XR (3D scene).

## STRUCTURE
```
components/
├── dashboard/               # 2D UI panels (React)
│   ├── TeleopPanel.tsx
│   ├── RobotSettingsPanel.tsx
│   └── CameraSettingsPanel.tsx
├── ui/                      # Radix UI + Tailwind components
│   ├── button.tsx
│   ├── slider.tsx
│   └── switch.tsx
└── xr/                      # XR scene wrapper
    └── XRScene.tsx
```

## WHERE TO LOOK

| Task | File | Notes |
|------|------|-------|
| Add dashboard panel | `dashboard/` | Use Radix UI components |
| Modify UI primitives | `ui/` | Tailwind + class-variance-authority |
| Integrate XR scene | `xr/XRScene.tsx` | Calls `initWorld()` |

## CONVENTIONS

### Styling
- **Tailwind**: All styling via utility classes
- **No `<style>` blocks**: Except state classes
- **cn()**: Merge classes with `webxr/src/lib/utils.ts`

### State Management
- **Zustand**: `webxr/src/lib/store.ts`
- **Selectors**: Use specific slices (avoid global)

## ANTI-PATTERNS (components)

- ❌ NEVER use `<style>` blocks (new UI)
- ✅ ALWAYS use `cn()` for conditional classes

## NOTES

**Fragmented Tests**: Tests in `dashboard/`, `ui/`, and `xr/` (not consolidated)
