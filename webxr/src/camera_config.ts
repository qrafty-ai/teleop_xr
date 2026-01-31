import { CameraViewKey } from "./track_routing";

export type CameraConfig = Record<CameraViewKey, boolean>;

let currentConfig: CameraConfig = {};
const handlers: ((config: CameraConfig) => void)[] = [];

export function getCameraEnabled(key: CameraViewKey): boolean {
  return currentConfig[key] !== false;
}

export function setCameraEnabled(key: CameraViewKey, enabled: boolean): void {
  currentConfig[key] = enabled;
  notify();
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
