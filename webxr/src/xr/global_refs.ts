import type {
	CameraPanel,
	ControllerCameraPanel,
	DraggablePanel,
} from "./panels";

// Global references store - populated at creation time, read by systems
// This avoids ECS query access during UI events which causes freezes

export const GlobalRefs = {
	cameraPanels: new Map<string, CameraPanel>(),
	teleopPanelRoot: null as DraggablePanel | null,
	cameraSettingsPanel: null as DraggablePanel | null,
	leftWristPanel: null as ControllerCameraPanel | null,
	rightWristPanel: null as ControllerCameraPanel | null,
};
