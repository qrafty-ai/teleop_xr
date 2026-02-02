export type CameraView = {
	device: string;
};

export type CameraViewsConfig = Record<string, CameraView>;

let currentConfig: CameraViewsConfig = {};
const handlers: ((config: CameraViewsConfig) => void)[] = [];

export function setCameraViewsConfig(config: CameraViewsConfig | null): void {
	currentConfig = config || {};
	console.log(
		"[CameraViews] setCameraViewsConfig called, keys:",
		Object.keys(currentConfig),
		"handlers:",
		handlers.length,
	);
	handlers.forEach((h, i) => {
		console.log("[CameraViews] Calling handler", i);
		try {
			h(currentConfig);
			console.log("[CameraViews] Handler", i, "completed");
		} catch (e) {
			console.error("[CameraViews] Handler", i, "threw error:", e);
		}
	});
	console.log("[CameraViews] All handlers completed");
}

export function getCameraViewsConfig(): CameraViewsConfig {
	return currentConfig;
}

export function isViewEnabled(key: string): boolean {
	return !!currentConfig[key];
}

export function onCameraViewsChanged(
	handler: (config: CameraViewsConfig) => void,
): void {
	handlers.push(handler);
	console.log("[CameraViews] Handler added, total handlers:", handlers.length);
	handler(currentConfig);
}
