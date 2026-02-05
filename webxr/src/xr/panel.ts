import {
	createSystem,
	eq,
	PanelDocument,
	PanelUI,
	type UIKit,
	type UIKitDocument,
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

			const exitButton = document.getElementById(
				"exit-xr-button",
			) as UIKit.Text | null;
			if (exitButton) {
				exitButton.addEventListener("click", () => {
					console.log("[PanelSystem] Exit XR button clicked");
					this.world.exitXR();
				});
			}
		});
	}
}
