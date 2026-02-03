import { useAppStore } from "../lib/store";
import type { CameraViewKey } from "./track_routing";

export type CameraConfig = Record<CameraViewKey, boolean>;

export function getCameraEnabled(key: CameraViewKey): boolean {
	return useAppStore.getState().getCameraEnabled(key);
}

export function setCameraEnabled(key: CameraViewKey, enabled: boolean): void {
	useAppStore.getState().setCameraEnabled(key, enabled);
}

export function onCameraConfigChanged(
	handler: (config: CameraConfig) => void,
): () => void {
	let currentConfig = useAppStore.getState().cameraConfig;
	const unsubscribe = useAppStore.subscribe((state) => {
		if (state.cameraConfig !== currentConfig) {
			currentConfig = state.cameraConfig;
			handler(currentConfig as CameraConfig);
		}
	});
	handler(useAppStore.getState().cameraConfig as CameraConfig);
	return unsubscribe;
}
