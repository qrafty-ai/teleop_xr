"use client";

import { ReferenceSpaceType, SessionMode } from "@iwsdk/core";
import { useCallback, useEffect, useRef } from "react";
import type { XRMode } from "@/app/page";
import { initWorld } from "@/xr";

type XRSceneProps = {
	mode: XRMode;
	onError?: (message: string) => void;
	onExit?: () => void;
};

type XRWorld = Awaited<ReturnType<typeof initWorld>>;

export function XRScene({ mode, onError, onExit }: XRSceneProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const worldRef = useRef<XRWorld | null>(null);
	const sessionRef = useRef<XRSession | null>(null);
	const onErrorRef = useRef<XRSceneProps["onError"]>(onError);
	const onExitRef = useRef<XRSceneProps["onExit"]>(onExit);

	useEffect(() => {
		onErrorRef.current = onError;
		onExitRef.current = onExit;
	}, [onError, onExit]);

	const reportError = useCallback((message: string) => {
		console.error(`[XRScene] ${message}`);
		onErrorRef.current?.(message);
	}, []);

	const cleanupAndExit = useCallback(() => {
		const world = worldRef.current;
		if (world) {
			cleanupWorld(world);
			worldRef.current = null;
		}
		sessionRef.current = null;
		onExitRef.current?.();
	}, []);

	useEffect(() => {
		const container = containerRef.current;
		if (!container || !mode) return;

		let isMounted = true;

		const setup = async () => {
			try {
				const isPassthrough = mode === "passthrough";
				console.log(
					`[XRScene] Initializing world with mode: ${mode} (passthrough: ${isPassthrough})`,
				);

				const world = await initWorld(container, isPassthrough);
				if (!isMounted) {
					cleanupWorld(world);
					return;
				}
				worldRef.current = world;

				// Always use immersive-ar to ensure stability and consistent reference space
				// We simulate VR by adding a skybox in initWorld if passthrough is false
				const sessionMode = SessionMode.ImmersiveAR;
				const optionalFeatures = [
					"local-floor",
					"hand-tracking",
					"anchors",
					"layers",
					"dom-overlay",
				];
				const sessionInit: XRSessionInit = {
					optionalFeatures,
					domOverlay: { root: document.body },
				};

				console.log(`[XRScene] Requesting session: ${sessionMode}`);
				const xr = navigator.xr;
				if (!xr) {
					throw new Error("WebXR not available");
				}

				const session = await xr.requestSession(sessionMode, sessionInit);
				if (!isMounted) {
					await session.end();
					return;
				}

				sessionRef.current = session;
				// Always use LocalFloor for AR-based session
				world.renderer.xr.setReferenceSpaceType(ReferenceSpaceType.LocalFloor);
				await world.renderer.xr.setSession(session);
				world.session = session;

				session.addEventListener(
					"end",
					() => {
						console.log("[XRScene] XR session ended");
						if (isMounted) {
							cleanupAndExit();
						}
					},
					{ once: true },
				);

				console.log(`[XRScene] Session started: ${sessionMode}`);
			} catch (err) {
				console.error("[XRScene] Failed to initialize:", err);
				reportError(
					err instanceof Error ? err.message : "Failed to initialize XR",
				);
				cleanupAndExit();
			}
		};

		void setup();

		return () => {
			isMounted = false;
			const session = sessionRef.current;
			if (session) {
				session.end().catch(() => {});
			}
			if (worldRef.current) {
				cleanupWorld(worldRef.current);
				worldRef.current = null;
			}
		};
	}, [mode, reportError, cleanupAndExit]);

	return (
		<div className="fixed inset-0 z-0" data-testid="xr-scene">
			<div ref={containerRef} className="absolute inset-0" />
		</div>
	);
}

function cleanupWorld(world: unknown) {
	try {
		if (isDisposable(world)) {
			world.dispose();
			return;
		}

		if (hasRenderer(world)) {
			world.renderer.setAnimationLoop(null);
			world.renderer.dispose();
		}

		if (hasExitXR(world)) {
			world.exitXR();
		}
	} catch (err) {
		console.warn("[XRScene] Error during world cleanup:", err);
	}
}

function isDisposable(value: unknown): value is { dispose: () => void } {
	return (
		typeof value === "object" &&
		value !== null &&
		"dispose" in value &&
		typeof (value as { dispose?: unknown }).dispose === "function"
	);
}

function hasRenderer(value: unknown): value is {
	renderer: { setAnimationLoop: (cb: null) => void; dispose: () => void };
} {
	if (typeof value !== "object" || value === null) return false;
	if (!("renderer" in value)) return false;
	const renderer = (value as { renderer?: unknown }).renderer;
	return (
		typeof renderer === "object" &&
		renderer !== null &&
		"setAnimationLoop" in renderer &&
		"dispose" in renderer &&
		typeof (renderer as { setAnimationLoop?: unknown }).setAnimationLoop ===
			"function" &&
		typeof (renderer as { dispose?: unknown }).dispose === "function"
	);
}

function hasExitXR(value: unknown): value is { exitXR: () => void } {
	return (
		typeof value === "object" &&
		value !== null &&
		"exitXR" in value &&
		typeof (value as { exitXR?: unknown }).exitXR === "function"
	);
}
