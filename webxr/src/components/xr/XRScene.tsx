"use client";

import { useCallback, useEffect, useRef } from "react";
import { useAppStore } from "@/lib/store";
import { initWorld } from "@/xr";

type XRSceneProps = {
	onExitAction: () => void;
	isImmersiveRequested: boolean;
	passthrough: boolean;
};

type XRWorld = Awaited<ReturnType<typeof initWorld>>;

export function XRScene({
	onExitAction,
	isImmersiveRequested,
	passthrough,
}: XRSceneProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const worldRef = useRef<XRWorld | null>(null);
	const onExitRef = useRef(onExitAction);
	const setIsImmersiveActive = useAppStore(
		(state) => state.setIsImmersiveActive,
	);

	useEffect(() => {
		onExitRef.current = onExitAction;
	}, [onExitAction]);

	const triggerExit = useCallback(() => {
		onExitRef.current();
		setIsImmersiveActive(false);
	}, [setIsImmersiveActive]);

	useEffect(() => {
		const world = worldRef.current;
		if (!world || !world.renderer || !world.renderer.xr) return;

		const xrManager = world.renderer.xr;
		const session = xrManager.getSession();
		const isSessionActive = Boolean(session);

		if (isImmersiveRequested && !isSessionActive) {
			const mode = passthrough ? "immersive-ar" : "immersive-vr";
			const sessionInit: XRSessionInit = {
				requiredFeatures: ["local-floor"],
				optionalFeatures: ["hand-tracking", "layers", "anchors"],
			};

			if (navigator.xr) {
				navigator.xr
					.requestSession(mode, sessionInit)
					.then((newSession) => {
						xrManager.setSession(newSession);
						setIsImmersiveActive(true);
						newSession.addEventListener("end", triggerExit);
					})
					.catch((err) => {
						console.error("Failed to request XR session:", err);
						triggerExit();
					});
			}
		} else if (!isImmersiveRequested && isSessionActive) {
			session?.end().catch(console.error);
		}
	}, [isImmersiveRequested, passthrough, triggerExit, setIsImmersiveActive]);

	useEffect(() => {
		const container = containerRef.current;
		if (!container) return;

		let isMounted = true;

		const setup = async () => {
			try {
				// Prevent double initialization if possible
				if (worldRef.current) return;

				const world = await initWorld(container);

				if (!isMounted) {
					console.log("XRScene unmounted before world init completed");
					cleanupWorld(world);
					return;
				}

				worldRef.current = world;
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
	}, []);

	return (
		<div className="fixed inset-0 z-0" data-testid="xr-scene">
			<div ref={containerRef} className="absolute inset-0" />
		</div>
	);
}

function cleanupWorld(world: XRWorld) {
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
