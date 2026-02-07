import { beforeEach, describe, expect, it } from "vitest";

import { GlobalRefs } from "../global_refs";

describe("GlobalRefs", () => {
	beforeEach(() => {
		// Clear all references before each test
		GlobalRefs.cameraPanels.clear();
		GlobalRefs.teleopPanelRoot = null;
		GlobalRefs.cameraSettingsPanel = null;
		GlobalRefs.leftWristPanel = null;
		GlobalRefs.rightWristPanel = null;
		GlobalRefs.panelEntities.clear();
	});

	it("initializes with empty state", () => {
		expect(GlobalRefs.cameraPanels.size).toBe(0);
		expect(GlobalRefs.teleopPanelRoot).toBeNull();
		expect(GlobalRefs.cameraSettingsPanel).toBeNull();
		expect(GlobalRefs.leftWristPanel).toBeNull();
		expect(GlobalRefs.rightWristPanel).toBeNull();
		expect(GlobalRefs.panelEntities.size).toBe(0);
	});

	it("stores camera panels in map", () => {
		const mockPanel = {
			id: "head",
		} as unknown as typeof GlobalRefs.cameraPanels extends Map<string, infer T>
			? T
			: never;
		GlobalRefs.cameraPanels.set("head", mockPanel);

		expect(GlobalRefs.cameraPanels.has("head")).toBe(true);
		expect(GlobalRefs.cameraPanels.get("head")).toBe(mockPanel);
		expect(GlobalRefs.cameraPanels.size).toBe(1);
	});

	it("stores panel root references", () => {
		const mockPanel = {
			type: "teleop",
		} as unknown as typeof GlobalRefs.teleopPanelRoot;
		GlobalRefs.teleopPanelRoot = mockPanel;

		expect(GlobalRefs.teleopPanelRoot).toBe(mockPanel);
	});

	it("stores panel entities in map", () => {
		const mockEntity = {
			id: 123,
		} as unknown as typeof GlobalRefs.panelEntities extends Map<number, infer T>
			? T
			: never;
		GlobalRefs.panelEntities.set(123, mockEntity);

		expect(GlobalRefs.panelEntities.has(123)).toBe(true);
		expect(GlobalRefs.panelEntities.get(123)).toBe(mockEntity);
		expect(GlobalRefs.panelEntities.size).toBe(1);
	});

	it("handles multiple camera panels", () => {
		const panel1 = {
			id: "head",
		} as unknown as typeof GlobalRefs.cameraPanels extends Map<string, infer T>
			? T
			: never;
		const panel2 = {
			id: "wrist_left",
		} as unknown as typeof GlobalRefs.cameraPanels extends Map<string, infer T>
			? T
			: never;

		GlobalRefs.cameraPanels.set("head", panel1);
		GlobalRefs.cameraPanels.set("wrist_left", panel2);

		expect(GlobalRefs.cameraPanels.size).toBe(2);
		expect(GlobalRefs.cameraPanels.get("head")).toBe(panel1);
		expect(GlobalRefs.cameraPanels.get("wrist_left")).toBe(panel2);
	});
});
