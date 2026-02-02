import { describe, expect, it, vi } from "vitest";
import {
	getCameraEnabled,
	onCameraConfigChanged,
	setCameraEnabled,
} from "../camera_config";

describe("camera_config", () => {
	it("all cameras are enabled by default", () => {
		expect(getCameraEnabled("head")).toBe(true);
		expect(getCameraEnabled("wrist_left")).toBe(true);
		expect(getCameraEnabled("random")).toBe(true);
	});

	it("can disable and re-enable a camera", () => {
		setCameraEnabled("head", false);
		expect(getCameraEnabled("head")).toBe(false);

		setCameraEnabled("head", true);
		expect(getCameraEnabled("head")).toBe(true);
	});

	it("notifies listeners on change", () => {
		const handler = vi.fn();
		const unsubscribe = onCameraConfigChanged(handler);

		expect(handler).toHaveBeenCalledTimes(1);

		setCameraEnabled("wrist_left", false);
		expect(handler).toHaveBeenCalledTimes(2);
		expect(handler).toHaveBeenLastCalledWith(
			expect.objectContaining({ wrist_left: false }),
		);

		unsubscribe();
		setCameraEnabled("wrist_left", true);
		expect(handler).toHaveBeenCalledTimes(2);
	});

	it("multiple listeners can be registered", () => {
		const h1 = vi.fn();
		const h2 = vi.fn();

		const u1 = onCameraConfigChanged(h1);
		const u2 = onCameraConfigChanged(h2);

		expect(h1).toHaveBeenCalledTimes(1);
		expect(h2).toHaveBeenCalledTimes(1);

		setCameraEnabled("head", false);
		expect(h1).toHaveBeenCalledTimes(2);
		expect(h2).toHaveBeenCalledTimes(2);

		u1();
		setCameraEnabled("head", true);
		expect(h1).toHaveBeenCalledTimes(2);
		expect(h2).toHaveBeenCalledTimes(3);

		u2();
	});
});
