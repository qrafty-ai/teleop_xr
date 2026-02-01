# WebGL Headless Rendering Investigation - Findings

## Executive Summary

Investigated blank screenshot issue in WebXR app using Playwright automation. **WebGL rendering works correctly in headed browser mode** but fails completely in headless mode with current Chromium flags.

## Test Results

### Headless Mode (FAILED)
- **Flags tested:** `--use-gl=swiftshader`, `--disable-gpu-sandbox`, `--disable-software-rasterizer`
- **Result:** WebGL context creation failed
- **Error:** `Could not create a WebGL context, GL_VENDOR = Disabled, GL_RENDERER = Disabled`
- **Screenshot:** White canvas, HTML overlay visible, no 3D rendering

### Headed Mode (SUCCESS)
- **Flags tested:** `--use-gl=egl`, `--enable-webgl`, `--ignore-gpu-blocklist`
- **Result:** Full WebGL rendering working
- **Screenshot:** Grey background + red debug box + 3D shading visible
- **Viewports tested:** Desktop (1280x720) and Mobile/Quest-like (Pixel 5 emulation)

## Key Findings

1. **Quest Browser is NOT Headless**
   - Quest browser has full GPU access
   - The blank screenshot issue is NOT a WebGL rendering failure

2. **Likely Root Causes for Quest Screenshot Issue:**
   - **Camera positioning:** Objects may be outside camera frustum
   - **Timing:** Screenshot captured before scene fully loads
   - **Asset loading:** 404 error detected during tests (resource missing)

3. **Technical Issues Identified:**
   - A-Frame CustomElementRegistry warning: `"a-node" has already been used`
   - Missing resource causing 404 (needs investigation)
   - Multiple Three.js instances imported (performance concern)

## Test Artifacts

- `tests/debug-render.spec.ts` - Main WebGL rendering test
- `tests/quest-screenshot.spec.ts` - Mobile viewport test
- `debug-screenshot.png` - Desktop headed mode screenshot (1280x720)
- `quest-screenshot.png` - Mobile headed mode screenshot (393x851)

## Recommendations

### Immediate Actions
1. **Investigate 404 Error**
   - Identify missing resource
   - Check asset paths in production build

2. **Camera Position Analysis**
   - Verify default camera position/rotation
   - Check if objects are in view frustum
   - Test with camera at `position="0 1.6 0" rotation="-20 0 0"`

3. **Screenshot Timing**
   - Add scene load detection before screenshot
   - Wait for `a-scene` `loaded` event
   - Verify all entities are initialized

### Future Improvements
- Fix CustomElementRegistry duplicate registration
- Resolve multiple Three.js instance warnings
- Add proper asset loading error handling

## Conclusion

**The Quest screenshot issue is NOT a WebGL rendering problem.** The rendering pipeline works correctly when GPU is available. Focus investigation on:
1. Scene initialization timing
2. Camera/object positioning
3. Missing asset resolution
