import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

export type CameraConfig = Record<string, boolean>;

export type TeleopSettings = {
	speed: number;
	turnSpeed: number;
	precisionMode: boolean;
};

export type AdvancedSettings = {
	updateRate: number;
	logLevel: "info" | "warn" | "error";
};

export type RobotSettings = {
	robotVisible: boolean;
	showAxes: boolean;
	spawnDistance: number;
	spawnHeight: number;
	distanceGrabEnabled: boolean;
};

export type TeleopTelemetry = {
	fps: number;
	latencyMs: number;
};

export type ConnectionStatus = "connected" | "disconnected" | "connecting";

export type AppState = {
	cameraConfig: CameraConfig;
	availableCameras: string[];
	teleopSettings: TeleopSettings;
	advancedSettings: AdvancedSettings;
	robotSettings: RobotSettings;
	robotResetTrigger: number;
	teleopTelemetry: TeleopTelemetry;
	connectionStatus: ConnectionStatus;
	getCameraEnabled: (key: string) => boolean;
	setCameraEnabled: (key: string, enabled: boolean) => void;
	toggleAllCameras: (enabled: boolean) => void;
	setAvailableCameras: (keys: string[]) => void;
	setTeleopSettings: (settings: Partial<TeleopSettings>) => void;
	setAdvancedSettings: (settings: Partial<AdvancedSettings>) => void;
	setRobotSettings: (settings: Partial<RobotSettings>) => void;
	setRobotResetTrigger: (trigger: number) => void;
	setTeleopTelemetry: (telemetry: TeleopTelemetry) => void;
	setConnectionStatus: (status: ConnectionStatus) => void;
};

const defaultTeleopSettings: TeleopSettings = {
	speed: 1,
	turnSpeed: 1,
	precisionMode: false,
};

const defaultAdvancedSettings: AdvancedSettings = {
	updateRate: 100,
	logLevel: "info",
};

const defaultRobotSettings: RobotSettings = {
	robotVisible: true,
	showAxes: false,
	spawnDistance: 1.0,
	spawnHeight: -0.3,
	distanceGrabEnabled: false,
};

const defaultTeleopTelemetry: TeleopTelemetry = {
	fps: 0,
	latencyMs: 0,
};

export const useAppStore = create<AppState>()(
	persist(
		(set, get) => ({
			cameraConfig: {},
			availableCameras: [],
			teleopSettings: defaultTeleopSettings,
			advancedSettings: defaultAdvancedSettings,
			robotSettings: defaultRobotSettings,
			robotResetTrigger: 0,
			teleopTelemetry: defaultTeleopTelemetry,
			connectionStatus: "disconnected",
			getCameraEnabled: (key) => get().cameraConfig[key] !== false,
			setCameraEnabled: (key, enabled) => {
				set((state) => ({
					cameraConfig: { ...state.cameraConfig, [key]: enabled },
				}));
			},
			toggleAllCameras: (enabled) => {
				const { availableCameras } = get();
				const newConfig: CameraConfig = {};
				for (const key of availableCameras) {
					newConfig[key] = enabled;
				}
				set({ cameraConfig: newConfig });
			},
			setAvailableCameras: (keys) => {
				set({ availableCameras: keys });
			},
			setTeleopSettings: (settings) => {
				set((state) => ({
					teleopSettings: { ...state.teleopSettings, ...settings },
				}));
			},
			setAdvancedSettings: (settings) => {
				set((state) => ({
					advancedSettings: { ...state.advancedSettings, ...settings },
				}));
			},
			setRobotSettings: (settings) => {
				set((state) => ({
					robotSettings: { ...state.robotSettings, ...settings },
				}));
			},
			setRobotResetTrigger: (trigger) => {
				set({ robotResetTrigger: trigger });
			},
			setTeleopTelemetry: (telemetry) => {
				set({ teleopTelemetry: telemetry });
			},
			setConnectionStatus: (status) => {
				set({ connectionStatus: status });
			},
		}),
		{
			name: "teleop-xr-store",
			storage: createJSONStorage(() => localStorage),
			partialize: (state) => ({
				cameraConfig: state.cameraConfig,
				teleopSettings: state.teleopSettings,
				advancedSettings: state.advancedSettings,
				robotSettings: state.robotSettings,
			}),
		},
	),
);
