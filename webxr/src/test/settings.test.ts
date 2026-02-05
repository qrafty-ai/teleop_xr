import { beforeEach, describe, expect, it, vi } from "vitest";
import { useAppStore } from "../lib/store";

describe("useAppStore", () => {
	beforeEach(() => {
		useAppStore.setState({
			cameraConfig: {},
			availableCameras: [],
			teleopSettings: {
				speed: 1,
				turnSpeed: 1,
				precisionMode: false,
			},
			advancedSettings: {
				updateRate: 100,
				logLevel: "info",
				robotVisible: true,
				spawnDistance: 1.0,
				spawnHeight: -0.3,
			},
			robotResetTrigger: 0,
			teleopTelemetry: {
				fps: 0,
				latencyMs: 0,
			},
			connectionStatus: "disconnected",
		});
	});

	it("should have correct default values", () => {
		const state = useAppStore.getState();
		expect(state.advancedSettings).toEqual({
			updateRate: 100,
			logLevel: "info",
			robotVisible: true,
			spawnDistance: 1.0,
			spawnHeight: -0.3,
		});
		expect(state.robotResetTrigger).toBe(0);
	});

	it("should update advancedSettings via setAdvancedSettings", () => {
		useAppStore
			.getState()
			.setAdvancedSettings({ updateRate: 60, logLevel: "error" });
		const state = useAppStore.getState();
		expect(state.advancedSettings.updateRate).toBe(60);
		expect(state.advancedSettings.logLevel).toBe("error");
		expect(state.advancedSettings.robotVisible).toBe(true);
	});

	it("should update robotResetTrigger via setRobotResetTrigger", () => {
		const now = Date.now();
		useAppStore.getState().setRobotResetTrigger(now);
		expect(useAppStore.getState().robotResetTrigger).toBe(now);
	});
});
