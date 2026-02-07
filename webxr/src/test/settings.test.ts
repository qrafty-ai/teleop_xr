import { beforeEach, describe, expect, it } from "vitest";
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
			},
			robotSettings: {
				robotVisible: true,
				showAxes: false,
				spawnDistance: 1.0,
				spawnHeight: -0.3,
				distanceGrabEnabled: false,
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
		});
		expect(state.robotResetTrigger).toBe(0);
		expect(state.connectionStatus).toBe("disconnected");
	});

	it("should update advancedSettings via setAdvancedSettings", () => {
		useAppStore
			.getState()
			.setAdvancedSettings({ updateRate: 60, logLevel: "error" });
		const state = useAppStore.getState();
		expect(state.advancedSettings.updateRate).toBe(60);
		expect(state.advancedSettings.logLevel).toBe("error");
	});

	it("should update robotResetTrigger via setRobotResetTrigger", () => {
		const now = Date.now();
		useAppStore.getState().setRobotResetTrigger(now);
		expect(useAppStore.getState().robotResetTrigger).toBe(now);
	});

	it("should get camera enabled status", () => {
		useAppStore.setState({
			cameraConfig: { head: true, wrist_left: false },
		});

		const state = useAppStore.getState();
		expect(state.getCameraEnabled("head")).toBe(true);
		expect(state.getCameraEnabled("wrist_left")).toBe(false);
		expect(state.getCameraEnabled("unknown")).toBe(true); // Default is true
	});

	it("should set camera enabled status", () => {
		useAppStore.getState().setCameraEnabled("head", true);
		useAppStore.getState().setCameraEnabled("wrist_left", false);

		const state = useAppStore.getState();
		expect(state.cameraConfig.head).toBe(true);
		expect(state.cameraConfig.wrist_left).toBe(false);
	});

	it("should toggle all cameras on", () => {
		useAppStore.setState({
			availableCameras: ["head", "wrist_left", "wrist_right"],
		});

		useAppStore.getState().toggleAllCameras(true);

		const state = useAppStore.getState();
		expect(state.cameraConfig.head).toBe(true);
		expect(state.cameraConfig.wrist_left).toBe(true);
		expect(state.cameraConfig.wrist_right).toBe(true);
	});

	it("should toggle all cameras off", () => {
		useAppStore.setState({
			availableCameras: ["head", "wrist_left", "wrist_right"],
			cameraConfig: { head: true, wrist_left: true, wrist_right: true },
		});

		useAppStore.getState().toggleAllCameras(false);

		const state = useAppStore.getState();
		expect(state.cameraConfig.head).toBe(false);
		expect(state.cameraConfig.wrist_left).toBe(false);
		expect(state.cameraConfig.wrist_right).toBe(false);
	});

	it("should set available cameras", () => {
		useAppStore.getState().setAvailableCameras(["cam1", "cam2", "cam3"]);

		const state = useAppStore.getState();
		expect(state.availableCameras).toEqual(["cam1", "cam2", "cam3"]);
	});

	it("should update teleop settings", () => {
		useAppStore.getState().setTeleopSettings({ speed: 2, precisionMode: true });

		const state = useAppStore.getState();
		expect(state.teleopSettings.speed).toBe(2);
		expect(state.teleopSettings.precisionMode).toBe(true);
		expect(state.teleopSettings.turnSpeed).toBe(1); // Unchanged
	});

	it("should update robot settings", () => {
		useAppStore
			.getState()
			.setRobotSettings({ robotVisible: false, showAxes: true });

		const state = useAppStore.getState();
		expect(state.robotSettings.robotVisible).toBe(false);
		expect(state.robotSettings.showAxes).toBe(true);
		expect(state.robotSettings.spawnDistance).toBe(1.0); // Unchanged
	});

	it("should update teleop telemetry", () => {
		useAppStore.getState().setTeleopTelemetry({ fps: 60, latencyMs: 50 });

		const state = useAppStore.getState();
		expect(state.teleopTelemetry.fps).toBe(60);
		expect(state.teleopTelemetry.latencyMs).toBe(50);
	});

	it("should update connection status", () => {
		useAppStore.getState().setConnectionStatus("connected");
		expect(useAppStore.getState().connectionStatus).toBe("connected");

		useAppStore.getState().setConnectionStatus("connecting");
		expect(useAppStore.getState().connectionStatus).toBe("connecting");

		useAppStore.getState().setConnectionStatus("disconnected");
		expect(useAppStore.getState().connectionStatus).toBe("disconnected");
	});
});
