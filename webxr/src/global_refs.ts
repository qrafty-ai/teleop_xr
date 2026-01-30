// Global references store - populated at creation time, read by systems
// This avoids ECS query access during UI events which causes freezes

export const GlobalRefs = {
  cameraPanels: new Map<string, any>(),
  teleopPanelRoot: null as any,
  leftWristPanelRoot: null as any,
  rightWristPanelRoot: null as any,
};
