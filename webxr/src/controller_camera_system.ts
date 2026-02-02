import { createSystem } from "@iwsdk/core";
import { type Object3D, Vector3 } from "three";
import type { ControllerCameraPanel, DraggablePanel } from "./panels";

type ControllerCameraPanelRef = {
	panel: ControllerCameraPanel | DraggablePanel; // ControllerCameraPanel instance
	controllerObject: Object3D; // raySpace Object3D
	getControllerObject?: () => Object3D; // Getter for dynamic raySpace resolution
};

export class ControllerCameraPanelSystem extends createSystem({}) {
	private panels: ControllerCameraPanelRef[] = [];
	private tempPosition = new Vector3();
	private tempControllerPosition = new Vector3();
	private tempHeadPosition = new Vector3();

	// Offset above controller (15cm up, 5cm forward)
	private offset = new Vector3(0, 0.15, -0.05);

	registerPanel(
		panel: ControllerCameraPanel | DraggablePanel,
		controllerObject: Object3D,
	) {
		this.panels.push({ panel, controllerObject });
	}

	registerPanelWithGetter(
		panel: ControllerCameraPanel | DraggablePanel,
		getControllerObject: () => Object3D,
	) {
		this.panels.push({ panel, controllerObject: null, getControllerObject });
	}

	update() {
		const player = this.world.player;
		if (!player?.head) return;
		const input = this.world.input;

		// Get head position for billboard
		player.head.getWorldPosition(this.tempHeadPosition);

		for (const ref of this.panels) {
			const { panel, getControllerObject } = ref;
			const handedness = panel?.handedness as "left" | "right" | undefined;

			// Resolve controller object based on primary/secondary controller spaces
			let controllerObject: Object3D | undefined;
			if (handedness) {
				const primaryGrip = player.gripSpaces?.[handedness];
				const secondaryGrip = player.secondaryGripSpaces?.[handedness];
				const primaryRay = player.raySpaces?.[handedness];
				const secondaryRay = player.secondaryRaySpaces?.[handedness];
				const isPrimary = input?.isPrimary?.("controller", handedness);
				const preferredPrimary = primaryGrip ?? primaryRay;
				const preferredSecondary = secondaryGrip ?? secondaryRay;
				controllerObject =
					isPrimary === false ? preferredSecondary : preferredPrimary;
			}

			// Fallback to getter or direct reference
			if (!controllerObject) {
				controllerObject = getControllerObject
					? getControllerObject()
					: ref.controllerObject;
			}

			if (!controllerObject) {
				// Controller not available yet (XR session not started or controller not connected)
				continue;
			}

			if (!panel?.entity?.object3D) continue;

			// Get controller world position
			controllerObject.getWorldPosition(this.tempControllerPosition);
			this.tempPosition.copy(this.tempControllerPosition);

			// Apply offset in WORLD space (not controller space)
			// This keeps the panel stable when controller rotates
			this.tempPosition.add(this.offset);

			// Set panel position
			panel.entity.object3D.position.copy(this.tempPosition);

			// Billboard: rotate to face head (full 3-axis)
			// Use Object3D.lookAt to correctly point +Z (front face) at target
			panel.entity.object3D.lookAt(this.tempHeadPosition);
		}
	}
}
