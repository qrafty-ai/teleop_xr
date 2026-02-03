"use client";

import { useCallback, useEffect, useRef } from "react";
import type { WebGLRenderer } from "three";
import { Button } from "@/components/ui/button";
import { initWorld } from "@/xr";

type XRSceneProps = {
	onExitAction: () => void;
};

type XRWorld = Awaited<ReturnType<typeof initWorld>>;

export function XRScene({ onExitAction }: XRSceneProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const worldRef = useRef<XRWorld | null>(null);
	const onExitRef = useRef(onExitAction);
	const exitTriggeredRef = useRef(false);

	useEffect(() => {
		onExitRef.current = onExitAction;
	}, [onExitAction]);

	const triggerExit = useCallback(() => {
		if (exitTriggeredRef.current) return;
		exitTriggeredRef.current = true;
		onExitRef.current();
	}, []);

	useEffect(() => {
		const container = containerRef.current;
		if (!container) return;

		let isMounted = true;
		let renderer: unknown = null;

		const setup = async () => {
			try {
				// Prevent double initialization if possible, though strict mode handles cleanup
				if (worldRef.current) return;

				const world = await initWorld(container);

				if (!isMounted) {
					console.log("XRScene unmounted before world init completed");
					cleanupWorld(world);
					return;
				}

				worldRef.current = world;
				renderer = world.renderer;
				const xrManager = (renderer as WebGLRenderer | null)?.xr;

				if (xrManager?.addEventListener) {
					xrManager.addEventListener("sessionend", triggerExit);
				}
			} catch (err) {
				console.error("Failed to initialize XR world:", err);
			}
		};

		setup();

		return () => {
			isMounted = false;
			const xrManager = (renderer as WebGLRenderer | null)?.xr;
			if (xrManager?.removeEventListener) {
				xrManager.removeEventListener("sessionend", triggerExit);
			}

			const sessionActive =
				xrManager?.isPresenting || Boolean(xrManager?.getSession?.());
			if (sessionActive) {
				triggerExit();
			}

			if (worldRef.current) {
				console.log("Disposing XR world");
				cleanupWorld(worldRef.current);
				worldRef.current = null;
			}
		};
	}, [triggerExit]);

	return (
		<div className="fixed inset-0 z-0" data-testid="xr-scene">
			<div ref={containerRef} className="absolute inset-0" />
		</div>
	);
}

function cleanupWorld(world: any) {
	try {
		if (typeof world.dispose === "function") {
			world.dispose();
		} else {
			// Manual cleanup if dispose() is missing
			if (world.renderer) {
				world.renderer.setAnimationLoop(null);
				world.renderer.dispose();
			}
			if (world.exitXR) {
				world.exitXR();
			}
		}
	} catch (err) {
		console.warn("Error during world cleanup:", err);
	}
}
