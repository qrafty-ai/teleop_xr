export type CameraView = {
  device: string;
};

export type CameraViewsConfig = Record<string, CameraView>;

let currentConfig: CameraViewsConfig = {};
const handlers: ((config: CameraViewsConfig) => void)[] = [];

export function setCameraViewsConfig(config: CameraViewsConfig | null): void {
  currentConfig = config || {};
  handlers.forEach(h => h(currentConfig));
}

export function getCameraViewsConfig(): CameraViewsConfig {
  return currentConfig;
}

export function isViewEnabled(key: string): boolean {
  return !!currentConfig[key];
}

export function onCameraViewsChanged(handler: (config: CameraViewsConfig) => void): void {
  handlers.push(handler);
  handler(currentConfig);
}
