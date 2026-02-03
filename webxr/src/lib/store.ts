import { create } from "zustand";

import type { CameraViewKey } from "../xr/track_routing";

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
	teleopSettings: TeleopSettings;
	teleopTelemetry: TeleopTelemetry;
	connectionStatus: ConnectionStatus;
	getCameraEnabled: (key: CameraViewKey) => boolean;
	setCameraEnabled: (key: CameraViewKey, enabled: boolean) => void;
	setTeleopSettings: (settings: Partial<TeleopSettings>) => void;
	setTeleopTelemetry: (telemetry: TeleopTelemetry) => void;
	setConnectionStatus: (status: ConnectionStatus) => void;
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
	teleopSettings: defaultTeleopSettings,
	teleopTelemetry: defaultTeleopTelemetry,
	connectionStatus: "disconnected",
	getCameraEnabled: (key) => get().cameraConfig[key] !== false,
	setCameraEnabled: (key, enabled) => {
		set((state) => ({
			cameraConfig: { ...state.cameraConfig, [key]: enabled },
		}));
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
}));
