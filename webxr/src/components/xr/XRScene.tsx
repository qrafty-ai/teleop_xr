"use client";

import {
	forwardRef,
	useCallback,
	useEffect,
	useImperativeHandle,
	useRef,
} from "react";
import { useAppStore } from "@/lib/store";
import { initWorld } from "@/xr";

export type XRSceneHandle = {
	enterXR: (options: { passthrough: boolean }) => Promise<void>;
	exitXR: () => Promise<void>;
};

type XRSceneProps = {
	onError?: (message: string) => void;
};

type XRWorld = Awaited<ReturnType<typeof initWorld>>;

export const XRScene = forwardRef<XRSceneHandle, XRSceneProps>(function XRScene(
	{ onError },
	ref,
) {
	const containerRef = useRef<HTMLDivElement>(null);
	const worldRef = useRef<XRWorld | null>(null);
	const sessionRef = useRef<XRSession | null>(null);
	const onErrorRef = useRef<XRSceneProps["onError"]>(onError);
	const setIsImmersiveActive = useAppStore(
		(state) => state.setIsImmersiveActive,
	);
	const setXrActions = useAppStore((state) => state.setXrActions);

	useEffect(() => {
		onErrorRef.current = onError;
	}, [onError]);

	const reportError = useCallback((message: string) => {
		console.error(`[XRScene] ${message}`);
		onErrorRef.current?.(message);
	}, []);

	const onSessionEnd = useCallback(() => {
		sessionRef.current = null;
		setIsImmersiveActive(false);
	}, [setIsImmersiveActive]);

	const exitXR = useCallback(async () => {
		const world = worldRef.current;
		const session = world?.renderer?.xr?.getSession?.() ?? null;
		if (!session) {
			sessionRef.current = null;
			setIsImmersiveActive(false);
			return;
		}

		try {
			await session.end();
		} catch (err) {
			console.warn("[XRScene] Failed to end XR session", err);
			onSessionEnd();
		}
	}, [onSessionEnd, setIsImmersiveActive]);

	const enterXR = useCallback(
		async ({ passthrough }: { passthrough: boolean }) => {
			const world = worldRef.current;
			if (!world?.renderer?.xr) {
				const msg = "XR world is not initialized yet";
				reportError(msg);
				throw new Error(msg);
			}

			const mode: XRSessionMode = passthrough ? "immersive-ar" : "immersive-vr";
			console.log(
				`[XRScene] enterXR requested: ${mode} (passthrough: ${passthrough})`,
			);

			const xr = navigator.xr;
			if (!xr?.requestSession) {
				const msg = `WebXR is not available (navigator.xr missing) - requested ${mode}`;
				reportError(msg);
				throw new Error(msg);
			}
			const currentSession = world.renderer.xr.getSession();
			if (currentSession) {
				sessionRef.current = currentSession;
				setIsImmersiveActive(true);
				return;
			}

			const sessionInit: XRSessionInit = {
				// Keep this list minimal to maximize cross-device stability.
				optionalFeatures: ["local-floor", "dom-overlay", "hand-tracking"],
				domOverlay: { root: document.body },
			};

			try {
				console.log(
					`[XRScene] Requesting session mode: ${mode} (passthrough: ${passthrough})`,
				);
				const newSession = await xr.requestSession(mode, sessionInit);
				world.renderer.xr.setSession(newSession);
				sessionRef.current = newSession;
				setIsImmersiveActive(true);
				newSession.addEventListener("end", onSessionEnd, { once: true });
			} catch (err) {
				console.error("[XRScene] Failed to request XR session", {
					mode,
					passthrough,
					optionalFeatures: sessionInit.optionalFeatures ?? [],
					error: formatError(err),
				});
				sessionRef.current = null;
				setIsImmersiveActive(false);
				if (err instanceof DOMException) {
					throw new Error(`${err.name}: ${err.message}`);
				}
				throw err instanceof Error
					? err
					: new Error("Failed to start XR session");
			}
		},
		[onSessionEnd, reportError, setIsImmersiveActive],
	);

	useImperativeHandle(
		ref,
		() => ({
			enterXR,
			exitXR,
		}),
		[enterXR, exitXR],
	);

	useEffect(() => {
		setXrActions({ enterXR, exitXR });
		return () => setXrActions(null);
	}, [enterXR, exitXR, setXrActions]);

	useEffect(() => {
		const container = containerRef.current;
		if (!container) return;

		let isMounted = true;

		const setup = async () => {
			try {
				if (worldRef.current) return;
				const world = await initWorld(container);
				if (!isMounted) {
					cleanupWorld(world);
					return;
				}
				worldRef.current = world;
			} catch (err) {
				console.error("Failed to initialize XR world:", err);
				reportError(
					err instanceof Error ? err.message : "Failed to initialize XR world",
				);
			}
		};

		void setup();

		return () => {
			isMounted = false;
			void exitXR();
			if (worldRef.current) {
				cleanupWorld(worldRef.current);
				worldRef.current = null;
			}
		};
	}, [exitXR, reportError]);

	return (
		<div className="fixed inset-0 z-0" data-testid="xr-scene">
			<div ref={containerRef} className="absolute inset-0" />
		</div>
	);
});

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
		console.warn("Error during world cleanup:", err);
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

function formatError(err: unknown): { name?: string; message: string } {
	if (err instanceof DOMException) {
		return { name: err.name, message: err.message };
	}
	if (err instanceof Error) {
		return { name: err.name, message: err.message };
	}
	return { message: String(err) };
}
