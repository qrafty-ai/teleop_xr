import { CameraPanel, ControllerCameraPanel } from "./panels";

// Global references store - populated at creation time, read by systems
// This avoids ECS query access during UI events which causes freezes

export const GlobalRefs = {
  cameraPanels: new Map<string, CameraPanel>(),
  teleopPanelRoot: null as any,
  cameraSettingsPanel: null as any,
  leftWristPanel: null as ControllerCameraPanel | null,
  rightWristPanel: null as ControllerCameraPanel | null,
};
