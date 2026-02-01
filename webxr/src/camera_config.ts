import { CameraViewKey } from "./track_routing";

export interface CameraSettings {
  width: number;
  height: number;
  fps: number;
  deviceId: string;
  enabled: boolean;
}

export type CameraConfig = Record<CameraViewKey, CameraSettings>;

const DEFAULT_SETTINGS: CameraSettings = {
  width: 1280,
  height: 720,
  fps: 30,
  deviceId: "",
  enabled: true,
};

let currentConfig: CameraConfig = {};
const handlers: ((config: CameraConfig) => void)[] = [];

export function getCameraSettings(key: CameraViewKey): CameraSettings {
  return currentConfig[key] || { ...DEFAULT_SETTINGS };
}

export function setCameraSettings(key: CameraViewKey, settings: Partial<CameraSettings>): void {
  currentConfig[key] = { ...getCameraSettings(key), ...settings };
  notify();
}

export function getCameraEnabled(key: CameraViewKey): boolean {
  return getCameraSettings(key).enabled;
}

export function setCameraEnabled(key: CameraViewKey, enabled: boolean): void {
  setCameraSettings(key, { enabled });
}

export function onCameraConfigChanged(
  handler: (config: CameraConfig) => void,
): () => void {
  handlers.push(handler);
  handler({ ...currentConfig });
  return () => {
    const index = handlers.indexOf(handler);
    if (index !== -1) {
      handlers.splice(index, 1);
    }
  };
}

function notify(): void {
  const configCopy = { ...currentConfig };
  handlers.forEach((h) => h(configCopy));
}
