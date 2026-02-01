## Video Stream Architecture
- **System**: Implemented `video-stream` system to manage the single `VideoClient` connection and track distribution. This prevents multiple WebSocket connections when multiple video components are used.
- **Component**: Implemented `video-stream` component that registers with the system and updates the entity's mesh/texture when a track is received.
- **Dependencies**: Used `AFRAME` global variable (typed as `any` in project) but defined local interfaces for better type safety within the file.
- **Texture**: Used `VideoTexture` from Three.js applied to either existing mesh or a new PlaneGeometry (defaulting to 4:3 aspect ratio).

## Task 5: UI & Billboard Components
- **Billboard Logic**: Implemented `billboard-follow` component. It allows an entity to follow another entity's position (like a controller) with an offset, but maintain its own rotation (looking at the camera). This decouples the panel rotation from the controller rotation, solving the readability issue.
- **UI Implementation**: Created `teleop-dashboard` component. Instead of porting the old complex `PanelUI` system (which depended on `uikit`), we used native A-Frame primitives (`a-plane`, `a-text`) to create a lightweight dashboard.
- **Event Handling**: The UI component listens for `teleop-status` and `teleop-stats` events on the scene element (`this.el.sceneEl`), ensuring loose coupling with the Teleop system.
