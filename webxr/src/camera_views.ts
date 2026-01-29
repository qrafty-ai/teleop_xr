export type CameraView = {
  device: string;
};

export type CameraViewsConfig = {
  head?: CameraView;
  wrist_left?: CameraView;
  wrist_right?: CameraView;
};

let currentConfig: CameraViewsConfig = {};
const handlers: ((config: CameraViewsConfig) => void)[] = [];

export function setCameraViewsConfig(config: CameraViewsConfig | null): void {
  currentConfig = config || {};
  handlers.forEach(h => h(currentConfig));
}

export function getCameraViewsConfig(): CameraViewsConfig {
  return currentConfig;
}

export function isViewEnabled(key: keyof CameraViewsConfig): boolean {
  return !!currentConfig[key];
}

export function onCameraViewsChanged(handler: (config: CameraViewsConfig) => void): void {
  handlers.push(handler);
  handler(currentConfig);
}
