import { type AssetManifest, AssetType, SessionMode, World } from "@iwsdk/core";
import { Container, Content, Image, Text } from "@pmndrs/uikit";
import * as horizonKit from "@pmndrs/uikit-horizon";
import {
	CheckIcon,
	LogInIcon,
	SettingsIcon,
	SignalIcon,
	SignalZeroIcon,
	VideoIcon,
	XIcon,
} from "@pmndrs/uikit-lucide";
import { getCameraEnabled, onCameraConfigChanged } from "./camera_config.js";
import { CameraSettingsSystem } from "./camera_settings_system.js";
import {
	getCameraViewsConfig,
	isViewEnabled,
	onCameraViewsChanged,
} from "./camera_views.js";
import { initConsoleStream } from "./console_stream.js";
import { ControllerCameraPanelSystem } from "./controller_camera_system.js";
import { GlobalRefs } from "./global_refs.js";
import { PanelSystem } from "./panel.js";
import {
	CameraPanel,
	CameraPanelSystem,
	ControllerCameraPanel,
	DraggablePanel,
	PanelHoverSystem,
} from "./panels.js";
import { RobotModelSystem } from "./robot_system.js";
import { TeleopSystem } from "./teleop_system.js";
import { type CameraViewKey, resolveTrackView } from "./track_routing.js";
import { VideoClient } from "./video.js";

// Initialize console streaming for Quest VR debugging
initConsoleStream();

const disableHeadCameraPanel = true;

const assets: AssetManifest = {
	chimeSound: {
		url: "./audio/chime.mp3",
		type: AssetType.Audio,
		priority: "background",
	},
};

World.create(document.getElementById("scene-container") as HTMLDivElement, {
	assets,
	xr: {
		sessionMode: SessionMode.ImmersiveAR,
		offer: "always",
		// Optional structured features; layers/local-floor are offered by default
		features: {
			handTracking: true,
			anchors: true,
			hitTest: false,
			planeDetection: false,
			meshDetection: false,
			layers: true,
		},
	},
	features: {
		locomotion: false,
		grabbing: true,
		physics: false,
		sceneUnderstanding: true,
		spatialUI: {
			kits: [
				{ ...horizonKit, Toggle: horizonKit.Toggle },
				{
					Container,
					Image,
					Text,
					Content,
					CheckIcon,
					LogInIcon,
					SettingsIcon,
					XIcon,
					VideoIcon,
					SignalIcon,
					SignalZeroIcon,
				},
			],
		},
	},
}).then((world) => {
	const { camera } = world;

	camera.position.set(0, 1, 0.5);

	const teleopPanel = new DraggablePanel(world, "./ui/teleop.json", {
		maxHeight: 0.8,
		maxWidth: 1.6,
	});
	teleopPanel.setPosition(0, 1.29, -1.9);
	teleopPanel.faceUser();
	if (teleopPanel.entity.object3D) {
		GlobalRefs.teleopPanelRoot = teleopPanel.entity.object3D;
	}

	const cameraSettingsPanel = new DraggablePanel(
		world,
		"./ui/camera_settings.json?v=4",
		{
			maxHeight: 0.6,
			maxWidth: 1.2,
		},
	);
	cameraSettingsPanel.setPosition(0.8, 1.29, -1.7);
	cameraSettingsPanel.faceUser();
	if (cameraSettingsPanel.entity.object3D) {
		GlobalRefs.cameraSettingsPanel = cameraSettingsPanel;
		cameraSettingsPanel.entity.object3D.visible = false;
	}

	const cameraPanels = new Map<string, CameraPanel>();

	// Controller-attached camera panels (for wrist cameras)
	const leftControllerPanel = new ControllerCameraPanel(world, "left");
	GlobalRefs.leftWristPanel = leftControllerPanel;

	const rightControllerPanel = new ControllerCameraPanel(world, "right");
	GlobalRefs.rightWristPanel = rightControllerPanel;

	// Pending tracks queue for tracks that arrive before config
	const pendingTracks: Array<{
		track: MediaStreamTrack;
		trackId: string;
		index: number;
	}> = [];
	let configReceived = false;

	const getFallbackOrder = (): string[] => {
		const config = getCameraViewsConfig();
		const keys = Object.keys(config);

		if (keys.length === 0) {
			const defaultKeys: string[] = [];
			if (!disableHeadCameraPanel) {
				defaultKeys.push("head");
			}
			defaultKeys.push("wrist_left", "wrist_right");
			return defaultKeys;
		}

		keys.sort();
		const prioritized = ["head", "wrist_left", "wrist_right"];
		const result: string[] = [];

		for (const key of prioritized) {
			if (keys.includes(key)) {
				result.push(key);
			}
		}

		for (const key of keys) {
			if (!prioritized.includes(key)) {
				result.push(key);
			}
		}

		return result;
	};

	const assignTrackToPanel = (
		track: MediaStreamTrack,
		trackId: string,
		trackIndex: number,
	) => {
		const targetView = resolveTrackView(
			trackId,
			trackIndex,
			getFallbackOrder(),
		);

		console.log(
			`[Video] Assigning track. ID: ${trackId}, Index: ${trackIndex}, Target: ${targetView}`,
		);

		if (!targetView) {
			return;
		}

		if (targetView === "wrist_left") {
			leftControllerPanel.setVideoTrack(track);
		} else if (targetView === "wrist_right") {
			rightControllerPanel.setVideoTrack(track);
		} else {
			const panel = cameraPanels.get(targetView);
			if (panel) {
				if (targetView !== "head" || !disableHeadCameraPanel) {
					panel.setVideoTrack(track);
				}
			} else {
				console.warn(
					`[Video] Target view '${targetView}' not found`,
					Array.from(cameraPanels.keys()),
				);
			}
		}
	};

	const processPendingTracks = () => {
		if (pendingTracks.length === 0) return;
		console.log(`[Video] Processing ${pendingTracks.length} pending tracks`);
		for (const { track, trackId, index } of pendingTracks) {
			assignTrackToPanel(track, trackId, index);
		}
		pendingTracks.length = 0;
	};

	onCameraViewsChanged((config) => {
		try {
			if (leftControllerPanel.entity.object3D) {
				leftControllerPanel.entity.object3D.visible =
					isViewEnabled("wrist_left") && getCameraEnabled("wrist_left");
			}
			if (rightControllerPanel.entity.object3D) {
				rightControllerPanel.entity.object3D.visible =
					isViewEnabled("wrist_right") && getCameraEnabled("wrist_right");
			}

			const allKeys = Object.keys(config);
			const reserved = ["wrist_left", "wrist_right"];
			const floatingKeys = allKeys.filter((k) => !reserved.includes(k)).sort();

			console.log(
				"[Video] Updating camera panels. Config keys:",
				allKeys,
				"Floating:",
				floatingKeys,
			);

			floatingKeys.forEach((key, index) => {
				let panel = cameraPanels.get(key);
				if (!panel) {
					console.log(`[Video] Creating new CameraPanel for key: ${key}`);
					panel = new CameraPanel(world);
					cameraPanels.set(key, panel);

					GlobalRefs.cameraPanels.set(key, panel);
					const shouldHide =
						(key === "head" && disableHeadCameraPanel) ||
						!getCameraEnabled(key as CameraViewKey);
					if (panel.entity?.object3D) {
						panel.entity.object3D.visible = !shouldHide;
					}
				}

				panel.setLabel(`CAMERA: ${key.toUpperCase()}`);

				const x = 1.2 + index * 0.9;
				if (panel && typeof panel.setPosition === "function") {
					panel.setPosition(x, 1.3, -1.5);
					panel.faceUser();
				} else {
					console.error(`[Video] panel.setPosition is missing for key: ${key}`);
				}
			});

			for (const [key, panel] of Array.from(cameraPanels.entries())) {
				if (!floatingKeys.includes(key)) {
					console.log(`[Video] Disposing panel for key: ${key}`);
					panel.dispose();
					cameraPanels.delete(key);
					GlobalRefs.cameraPanels.delete(key);
				} else if (panel.entity?.object3D) {
					panel.entity.object3D.visible =
						isViewEnabled(key) && getCameraEnabled(key as CameraViewKey);
				}
			}

			if (allKeys.length > 0 && !configReceived) {
				configReceived = true;
				setTimeout(() => processPendingTracks(), 50);
			}
		} catch (err) {
			console.error("[Video] Error in onCameraViewsChanged handler:", err);
			if (err instanceof Error) {
				console.error("[Video] Stack trace:", err.stack);
			} else {
				console.error("[Video] Error detail:", JSON.stringify(err));
			}
		}
	});

	onCameraConfigChanged((_config) => {
		if (leftControllerPanel.entity.object3D) {
			leftControllerPanel.entity.object3D.visible =
				isViewEnabled("wrist_left") && getCameraEnabled("wrist_left");
		}
		if (rightControllerPanel.entity.object3D) {
			rightControllerPanel.entity.object3D.visible =
				isViewEnabled("wrist_right") && getCameraEnabled("wrist_right");
		}
		for (const [key, panel] of cameraPanels.entries()) {
			if (panel.entity.object3D) {
				panel.entity.object3D.visible =
					isViewEnabled(key) && getCameraEnabled(key as CameraViewKey);
			}
		}
	});

	// Video connection
	const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
	const videoWsUrl = `${protocol}//${window.location.host}/ws`;

	let trackCount = 0;
	const _videoClient = new VideoClient(
		videoWsUrl,
		(_stats) => {},
		(track, trackId) => {
			console.log(
				`[Video] New track received. ID: ${trackId}, Index: ${trackCount}, ConfigReceived: ${configReceived}`,
			);

			if (!configReceived) {
				pendingTracks.push({ track, trackId, index: trackCount });
				console.log(`[Video] Queued track ${trackCount} (awaiting config)`);
			} else {
				assignTrackToPanel(track, trackId, trackCount);
			}
			trackCount++;
		},
	);

	world.registerSystem(PanelSystem);
	world.registerSystem(TeleopSystem);
	world.registerSystem(RobotModelSystem);
	world.registerSystem(ControllerCameraPanelSystem);
	world.registerSystem(CameraSettingsSystem);
	world.registerSystem(CameraPanelSystem);
	world.registerSystem(PanelHoverSystem);

	// Register controller panels with their raySpaces once XR session starts
	// The system will handle waiting for raySpaces to be available
	const controllerCameraSystem = world.getSystem(ControllerCameraPanelSystem);
	if (controllerCameraSystem) {
		// Register with callbacks that resolve raySpaces dynamically
		const getLeftRaySpace = () => world.player?.raySpaces?.left;
		const getRightRaySpace = () => world.player?.raySpaces?.right;

		// The system expects controllerObject - we'll modify the system to handle getters
		// For now, pass a reference that will be resolved each frame
		// biome-ignore lint/suspicious/noExplicitAny: Temporary hack for dynamic getter
		(controllerCameraSystem as any).registerPanelWithGetter(
			leftControllerPanel,
			getLeftRaySpace,
		);
		// biome-ignore lint/suspicious/noExplicitAny: Temporary hack for dynamic getter
		(controllerCameraSystem as any).registerPanelWithGetter(
			rightControllerPanel,
			getRightRaySpace,
		);
	}
});
