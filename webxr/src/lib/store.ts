import { create } from "zustand";

export type CameraConfig = Record<string, boolean>;

export type TeleopSettings = {
	speed: number;
	turnSpeed: number;
	precisionMode: boolean;
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
	teleopTelemetry: TeleopTelemetry;
	connectionStatus: ConnectionStatus;
	isPassthroughEnabled: boolean;
	isImmersiveActive: boolean;
	getCameraEnabled: (key: string) => boolean;
	setCameraEnabled: (key: string, enabled: boolean) => void;
	toggleAllCameras: (enabled: boolean) => void;
	setAvailableCameras: (keys: string[]) => void;
	setTeleopSettings: (settings: Partial<TeleopSettings>) => void;
	setTeleopTelemetry: (telemetry: TeleopTelemetry) => void;
	setConnectionStatus: (status: ConnectionStatus) => void;
	setIsPassthroughEnabled: (enabled: boolean) => void;
	setIsImmersiveActive: (active: boolean) => void;
};

const defaultTeleopSettings: TeleopSettings = {
	speed: 1,
	turnSpeed: 1,
	precisionMode: false,
};

const defaultTeleopTelemetry: TeleopTelemetry = {
	fps: 0,
	latencyMs: 0,
};

export const useAppStore = create<AppState>((set, get) => ({
	cameraConfig: {},
	availableCameras: [],
	teleopSettings: defaultTeleopSettings,
	teleopTelemetry: defaultTeleopTelemetry,
	connectionStatus: "disconnected",
	isPassthroughEnabled: true,
	isImmersiveActive: false,
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
	setTeleopTelemetry: (telemetry) => {
		set({ teleopTelemetry: telemetry });
	},
	setConnectionStatus: (status) => {
		set({ connectionStatus: status });
	},
	setIsPassthroughEnabled: (enabled) => {
		set({ isPassthroughEnabled: enabled });
	},
	setIsImmersiveActive: (active) => {
		set({ isImmersiveActive: active });
	},
}));
