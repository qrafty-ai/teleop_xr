# Teleop Config Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve teleop configuration by adding a menu button toggle for the config panel and ensuring the camera toggle button closes all camera views (including wrist cameras).

**Architecture:** 
- Update `GlobalRefs` to store references to all relevant panels (Teleop, Wrist Left/Right).
- Populate these references in `index.ts` upon instantiation.
- Modify `TeleopSystem` to handle the camera button click (toggling all refs) and monitor the left controller's menu button (toggling the teleop panel).

**Tech Stack:** TypeScript, Three.js (for Object3D visibility), WebXR Gamepad API.

---

### Task 1: Update GlobalRefs

**Files:**
- Modify: `webxr/src/global_refs.ts:4-6`

**Step 1: Update GlobalRefs object**

Add fields for `teleopPanelRoot`, `leftWristPanelRoot`, and `rightWristPanelRoot`.

```typescript
export const GlobalRefs = {
  cameraPanelRoot: null as any,
  teleopPanelRoot: null as any,
  leftWristPanelRoot: null as any,
  rightWristPanelRoot: null as any,
};
```

**Step 2: Commit**

```bash
git add webxr/src/global_refs.ts
git commit -m "refactor: add global refs for teleop and wrist panels"
```

---

### Task 2: Populate GlobalRefs in index.ts

**Files:**
- Modify: `webxr/src/index.ts`

**Step 1: Assign GlobalRefs**

In `webxr/src/index.ts`, after creating `teleopPanel`, `leftControllerPanel`, and `rightControllerPanel`, assign their `entity.object3D` to `GlobalRefs`.

Around line 130 (Teleop):
```typescript
  teleopPanel.setPosition(0, 1.29, -1.9);
  if (teleopPanel.entity.object3D) {
    GlobalRefs.teleopPanelRoot = teleopPanel.entity.object3D;
  }
```

Around line 144 (Wrist):
```typescript
  const leftControllerPanel = new ControllerCameraPanel(world, "left");
  if (leftControllerPanel.entity.object3D) {
    GlobalRefs.leftWristPanelRoot = leftControllerPanel.entity.object3D;
  }
  const rightControllerPanel = new ControllerCameraPanel(world, "right");
  if (rightControllerPanel.entity.object3D) {
    GlobalRefs.rightWristPanelRoot = rightControllerPanel.entity.object3D;
  }
```

**Step 2: Commit**

```bash
git add webxr/src/index.ts
git commit -m "feat: populate global refs for panels"
```

---

### Task 3: Implement Camera Toggle Logic

**Files:**
- Modify: `webxr/src/teleop_system.ts`

**Step 1: Modify camera button listener**

In `init()` (around line 52), update the click listener to toggle all camera panels.

```typescript
      const cameraButton = document.getElementById("camera-button");
      if (cameraButton) {
        cameraButton.addEventListener("click", () => {
          // Master toggle based on head camera state (or any visible state)
          // If head camera is visible, close everything.
          // If head camera is hidden, open everything (or just head? Requirement: "toggle camera now closes all")
          // Interpretation: Switch state. If we are turning ON, turn ON. If OFF, turn OFF all.
          
          // Let's use cameraPanelRoot as the master state
          if (GlobalRefs.cameraPanelRoot) {
             const newState = !GlobalRefs.cameraPanelRoot.visible;
             GlobalRefs.cameraPanelRoot.visible = newState;
             
             if (GlobalRefs.leftWristPanelRoot) GlobalRefs.leftWristPanelRoot.visible = newState;
             if (GlobalRefs.rightWristPanelRoot) GlobalRefs.rightWristPanelRoot.visible = newState;
          }
        });
      }
```

**Step 2: Verify**

(Since we can't easily unit test DOM/GlobalRefs without mocking, we rely on the code structure. Manual verification in VR would be needed later).

**Step 3: Commit**

```bash
git add webxr/src/teleop_system.ts
git commit -m "feat: update camera toggle to control all camera views"
```

---

### Task 4: Implement Menu Button Toggle

**Files:**
- Modify: `webxr/src/teleop_system.ts`

**Step 1: Add state for debounce**

Add `private menuButtonState = false;` to `TeleopSystem` class.

**Step 2: Implement toggle logic in gatherInputState**

In `gatherInputState` (around line 251 where `leftDevice` is built), check for button press.

```typescript
    // Inside gatherInputState, after getting leftDevice
    const leftGamepad = input?.gamepads?.left;
    if (leftGamepad && leftGamepad.buttons) {
        // Button 4 ('X' or Menu on some profiles) or 5 ('Y')
        // We'll try index 4 and 5 (Menu/X/Y)
        // WebXR standard mapping: 4 is X, 5 is Y. System menu is usually reserved.
        // But often 'X' is used for menu in apps.
        const menuButton = leftGamepad.buttons[4] || leftGamepad.buttons[5]; 
        
        if (menuButton) {
            if (menuButton.pressed) {
                if (!this.menuButtonState) {
                    // Button just pressed
                    this.menuButtonState = true;
                    if (GlobalRefs.teleopPanelRoot) {
                        GlobalRefs.teleopPanelRoot.visible = !GlobalRefs.teleopPanelRoot.visible;
                    }
                }
            } else {
                // Button released
                this.menuButtonState = false;
            }
        }
    }
```

*Note: Accessing `input` directly in `gatherInputState` is correct as it's passed in.*

**Step 3: Commit**

```bash
git add webxr/src/teleop_system.ts
git commit -m "feat: toggle teleop config panel with left controller menu button"
```
