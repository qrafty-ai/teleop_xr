### Environment Setup & Dependency Migration
- Successfully removed @iwsdk dependencies from package.json and vite.config.ts.
- Added aframe ^1.6.0 and @types/aframe ^1.2.9.
- Created aframe-types.d.ts to handle global AFRAME and THREE types.
- A-Frame 1.6.0 bundles Three.js r164, so removed super-three alias.
- npm install completed successfully.

### A-Frame System Testing
- To test A-Frame systems with Vitest without a full DOM environment:
  - Export the system definition object (e.g., `export const MySystemDef = { ... }`).
  - In tests, instantiate it manually: `const system = { ...MySystemDef };`.
  - Mock `this.el` and `this.sceneEl` (and `this.el.sceneEl` circular ref if needed).
  - Mock global `AFRAME.registerSystem` if the file calls it at the top level.
- `WebSocket` mocking requires static constants (`OPEN`, `CLOSED`) if the code checks `ws.readyState === WebSocket.OPEN`.
- A-Frame's `tick(time, delta)` provides time in milliseconds.
- Final assembly successfully integrated Teleop system, Robot Model, and Video Streaming in A-Frame.
- A-Frame's system 'tick' and component 'tick' were used for real-time input gathering and billboard behavior.
- Event-based communication (scene.emit -> entity.addEventListener/window.addEventListener) works well for decoupled systems.
- Console streaming is essential for debugging on Quest VR headsets.
