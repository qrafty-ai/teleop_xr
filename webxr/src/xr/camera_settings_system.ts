import { createSystem } from "@iwsdk/core";
import { type CameraConfig, useAppStore } from "../lib/store";
import { GlobalRefs } from "./global_refs";

export class CameraSettingsSystem extends createSystem({}) {
	init() {
		this.updatePanelVisibility(useAppStore.getState().cameraConfig);
		useAppStore.subscribe((state, prevState) => {
			if (state.cameraConfig !== prevState?.cameraConfig) {
				this.updatePanelVisibility(state.cameraConfig);
			}
		});
	}

	private updatePanelVisibility(cameraConfig: CameraConfig) {
		for (const [key, panel] of GlobalRefs.cameraPanels.entries()) {
			const enabled = cameraConfig[key] !== false;
			if (panel?.entity?.object3D) {
				panel.entity.object3D.visible = enabled;
			}
		}
	}
}
