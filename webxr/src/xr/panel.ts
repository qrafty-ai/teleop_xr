import {
	createSystem,
	eq,
	PanelDocument,
	PanelUI,
	type UIKit,
	type UIKitDocument,
	VisibilityState,
} from "@iwsdk/core";

export class PanelSystem extends createSystem({
	teleopPanel: {
		required: [PanelUI, PanelDocument],
		where: [eq(PanelUI, "config", "./ui/teleop.json")],
	},
}) {
	init() {
		this.queries.teleopPanel.subscribe("qualify", (entity) => {
			const document = PanelDocument.data.document[
				entity.index
			] as UIKitDocument;
			if (!document) {
				return;
			}

			const xrButton = document.getElementById(
				"xr-button",
			) as UIKit.Text | null;
			if (!xrButton) {
				return;
			}
			xrButton.addEventListener("click", () => {
				if (this.world.visibilityState.value === VisibilityState.NonImmersive) {
					this.world.launchXR();
				} else {
					this.world.exitXR();
				}
			});
			this.world.visibilityState.subscribe((visibilityState) => {
				if (visibilityState === VisibilityState.NonImmersive) {
					xrButton.setProperties({ text: "Enter XR" });
				} else {
					xrButton.setProperties({ text: "Exit to Browser" });
				}
			});
		});
	}
}
