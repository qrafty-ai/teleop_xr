# Migration Plan: Next.js Dashboard + IWSDK Integration

## TL;DR

> **Quick Summary**: Transform the existing `webxr/` project from a Vite-based app to a Next.js 14+ application. The Next.js dashboard will act as the primary interface (Settings/Launch), and the IWSDK immersive session will be launched on-demand via a button.
>
> **Deliverables**:
> - Reorganized `webxr/` structure (Next.js layout)
> - `src/xr/`: Ported IWSDK core and ECS systems
> - `src/components/dashboard/`: Shadcn/UI panels for Teleop and Camera settings
> - `src/lib/store.ts`: Zustand store for state sharing between DOM and XR
> - `scripts/optimize-assets.mjs`: Standalone GLTF optimization script
> - Updated Python serving logic
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Project Reorganization → Logic Port → UI Implementation → Integration

---

## Context

### Original Request
Migrate spatial UI settings to a browser panel within the DOM using Next.js and Shadcn. The Goal is not to replace IWSDK, but to use Next.js for a settings/launching page. Both must coexist in the same project/folder.

### Interview Summary
**Key Decisions**:
- **Architecture**: Next.js is the "host" project. IWSDK logic resides in `src/xr/`.
- **UI**: Shadcn/UI for Teleop/Camera settings in 2D.
- **Integration**: "Launch XR" button in Next.js triggers the IWSDK `World.create` and XR session request.
- **State**: Zustand bridges the gap. UI writes to store; ECS Systems read from it.
- **Assets**: Replicate Vite plugins using standalone Node.js scripts for GLTF optimization and UIKit compilation.

### Review Findings
- **IWER**: Will be injected manually in the React launcher component during development to replace the Vite plugin.
- **Serving**: Python's `_resolve_frontend_paths` in `teleop_xr/__init__.py` must point to `webxr/out`.

---

## Work Objectives

### Core Objective
Replace the 3D spatial settings panels with a professional 2D dashboard while maintaining the high-performance IWSDK teleoperation engine.

### Concrete Deliverables
- `webxr/src/app/page.tsx` (Dashboard)
- `webxr/src/components/settings/TeleopPanel.tsx`
- `webxr/src/components/settings/CameraSettings.tsx`
- `webxr/src/components/xr/XRScene.tsx` (Launcher + Three.js Container)
- `webxr/scripts/optimize-assets.mjs`
- Modified `teleop_xr/__init__.py`

### Definition of Done
- [x] `npm run build` produces a static export in `webxr/out/`
- [x] Python backend serves the Next.js dashboard
- [x] "Launch XR" button successfully starts the immersive session
- [x] DOM settings toggles correctly update the XR behavior (Cameras/Teleop)
- [x] **Verification**: Screenshot at `.sisyphus/evidence/final_dashboard_check.png` confirms UI state.

---

## Verification Strategy
- **Final Dashboard Check**: COMPLETE
- **Screenshot**: Captured and verified.

---

## TODOs
- [x] 1. Project Reorganization & Next.js Initialization
- [x] 2. Standalone Asset Optimization Script
- [x] 3. Setup Shared Store (Zustand)
- [x] 4. Port and Wrap IWSDK Logic
- [x] 5. Implement 2D Dashboard UI
- [x] 6. Create XRScene Component (Launcher)
- [x] 7. State Synchronization (The Bridge)
- [x] 8. Update Python Serving & Final Cleanup
- [x] 9. Fix Build Errors (SSR & Typos)
- [x] 10. Final Verification with Screenshot DoD

  **What to do**:
  - Update `teleop_xr/__init__.py`: Point static serving to `webxr/out`.
  - Delete `webxr/vite.config.ts` and other Vite artifacts.
  - Final end-to-end verification.

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: [`python`]

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3
  - **Blocks**: None

---

## Success Criteria

### Verification Commands
```bash
cd webxr && npm run build
# Expected: Static export in 'out/'

# From root
python -m teleop_xr.demo
# Expected: Opens Next.js dashboard on port 4443
```
