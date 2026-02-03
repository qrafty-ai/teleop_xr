"use client";

import { useEffect, useRef } from "react";
import { initWorld } from "@/xr";

export function XRScene({ onExit }: { onExit?: () => void }) {
	const containerRef = useRef<HTMLDivElement>(null);
	const worldRef = useRef<Awaited<ReturnType<typeof initWorld>> | null>(null);

	useEffect(() => {
		const container = containerRef.current;
		if (!container) return;

		let isMounted = true;

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

				if (world.renderer?.xr) {
					world.renderer.xr.addEventListener("sessionend", () => {
						console.log("XR Session ended");
						onExit?.();
					});
				}
			} catch (err) {
				console.error("Failed to initialize XR world:", err);
			}
		};

		setup();

		return () => {
			isMounted = false;
			if (worldRef.current) {
				console.log("Disposing XR world");
				cleanupWorld(worldRef.current);
				worldRef.current = null;
			}
		};
	}, [onExit]);

	return (
		<div
			ref={containerRef}
			className="fixed inset-0 z-50 bg-black"
			data-testid="xr-scene"
		/>
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
