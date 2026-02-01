# Spatial UI Implementation Notes

## Overview

This document describes the spatial glass UI implementation for TeleopXR, including known limitations and verification instructions.

## Features Implemented

### 1. Spatial Glass Panels
- **Component**: `spatial-panel.ts`
- **Technology**: Three.js `MeshPhysicalMaterial` with transmission properties
- **Visual Effect**: Frosted glass appearance with realistic light transmission
- **Properties**:
  - Transmission: 0.95 (high transparency)
  - Roughness: 0.05 (smooth surface)
  - Thickness: 0.5 (glass depth simulation)
  - IOR: 1.5 (index of refraction)
  - Opacity: 0.2 (base transparency)

### 2. Robot Settings Panel
- **Location**: `webxr/index.html` - `#robot-settings-panel`
- **Functionality**: Toggleable settings panel with spatial glass background
- **Content**: Displays robot configuration and status information
- **Interaction**: Controlled via `teleop-dashboard` component

### 3. Camera Settings Panel
- **Integration**: Part of the dashboard system
- **Purpose**: Configure video stream settings for head and wrist cameras
- **Visual**: Uses spatial glass material for consistent UI appearance

## Known Limitations

### Headless Rendering Issue

**Problem**: When running the WebXR app in headless mode (e.g., using Puppeteer or Playwright for automated screenshots), the spatial glass panels render as **opaque** instead of transparent.

**Root Cause**:
- The `MeshPhysicalMaterial` with transmission requires WebGL features that may not be fully supported in headless rendering contexts
- Environment map generation (using `PMREMGenerator` and `RoomEnvironment`) may not function correctly without a full GPU context
- Headless browsers often use software rendering or limited GPU acceleration

**Impact**:
- Automated screenshots show solid panels instead of glass effect
- Visual regression testing cannot verify the glass appearance
- Documentation screenshots require manual capture

**Workaround**:
- **Manual verification on Quest headset is required** to confirm the glass effect works correctly
- Use real device testing for visual QA
- Consider using device screenshots (via `adb` or Quest's built-in screenshot feature) for documentation

## Verification Instructions

### On Meta Quest Headset

1. **Deploy the app**:
   ```bash
   python -m teleop_xr.demo
   ```

2. **Access from Quest**:
   - Open the browser on your Quest headset
   - Navigate to `https://<server-ip>:4443`
   - Enter VR mode

3. **Verify spatial glass**:
   - Look at the Robot Settings panel (toggle via dashboard)
   - Confirm the panel has a frosted glass appearance
   - Check that background objects are visible through the panel with slight distortion
   - Verify text remains readable on top of the glass surface

4. **Test interactions**:
   - Toggle panel visibility
   - Verify billboard behavior (panel faces user)
   - Check that glass effect persists during movement

### Expected Appearance

- **Glass panels should**:
  - Be semi-transparent with a frosted effect
  - Show subtle reflections from the environment
  - Allow background visibility with slight blur
  - Maintain consistent appearance across different viewing angles

- **Text on panels should**:
  - Remain fully opaque and readable
  - Contrast well against the glass background
  - Not be affected by the transmission effect

## Technical Details

### Environment Map Setup

The spatial glass effect requires an environment map for realistic reflections:

```typescript
const pmremGenerator = new PMREMGenerator(renderer);
scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
```

This is automatically set up once per scene in the `spatial-panel` component's `init()` method.

### Material Application

The component intelligently applies the glass material only to background planes, preserving text materials:

```typescript
const isBackgroundPlane = node.geometry?.type === 'PlaneGeometry' ||
                          node.name?.toLowerCase().includes('background') ||
                          node.name?.toLowerCase().includes('panel');
```

This ensures text remains visible while the panel background has the glass effect.

## Future Improvements

- Investigate headless rendering solutions for automated testing
- Consider fallback materials for non-VR contexts
- Add user-configurable glass intensity settings
- Implement dynamic environment map updates for changing scenes
