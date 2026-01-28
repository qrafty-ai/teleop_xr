# Teleop Logic & UI Fix Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the Teleop WebXR implementation to match the logic of `teleop/index.html` (controller support, correct state format, latency tracking) and fix the UI update issue (Panel showing but stats not updating).

**Worktree:** `/home/cc/codes/teleop/.worktrees/webxr-fix`

---

### Task 1: Fix TeleopSystem Logic (Controllers & State)

**Files:**
- Modify: `webxr/src/teleop_system.ts`

**Step 1: Update `gatherXRState` to match Python protocol**
- Use `timestamp_unix_ms`.
- Iterate `world.app.xr.session.inputSources` (if available via iwsdk or access raw session).
- Implement controller tracking (targetRaySpace, gamepad).
- Add `fetch_latency_ms`.

**Step 2: Update `execute` loop**
- Implement 10ms rate limit for sending.
- Update UI stats (Pose/FPS/Latency).

**Code Skeleton (Mental Draft):**
```typescript
// Need to access raw XRSession from iwsdk World/App if possible, or via navigator.xr (but session is managed by iwsdk).
// iwsdk exposes `world.app.xr` (Three.js WebXRManager).
// `world.app.renderer.xr.getSession()`

gatherXRState(time: number, frame: XRFrame) {
  const session = this.world.app.renderer.xr.getSession();
  if (!session) return null;
  
  // Logic from teleop/index.html...
  // ...
}
```

### Task 2: Fix UI Updates

**Files:**
- Modify: `webxr/src/teleop_system.ts`
- Modify: `webxr/ui/teleop.uikitml` (ensure IDs match)

**Step 1: Debug UI Query**
- Ensure `this.statusText`, `this.fpsText` are actually found.
- The previous code used `document.getElementById`. In `uikitml`, we might need `document.querySelector` or specific ID lookup.
- Verify `updateStatus` and `updateLocalStats` actually set properties.

### Task 3: Build and Verify

**Files:**
- None

**Step 1: Build**
`npm run build`

**Step 2: Push**
Commit and push to update PR.

