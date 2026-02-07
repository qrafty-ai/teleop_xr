import {
	type AssetManifest,
	AssetType,
	type Camera,
	type Object3D,
	SessionMode,
	World,
} from "@iwsdk/core";
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
import {
	BackSide,
	Euler,
	GridHelper,
	Mesh,
	MeshBasicMaterial,
	SphereGeometry,
} from "three";
import { getCameraEnabled, onCameraConfigChanged } from "./camera_config";
import { CameraSettingsSystem } from "./camera_settings_system";
import {
	getCameraViewsConfig,
	isViewEnabled,
	onCameraViewsChanged,
} from "./camera_views";
import { initConsoleStream } from "./console_stream";
import { ControllerCameraPanelSystem } from "./controller_camera_system";
import { GlobalRefs } from "./global_refs";
import { PanelSystem } from "./panel";
import {
	CameraPanel,
	CameraPanelSystem,
	ControllerCameraPanel,
	PanelHoverSystem,
} from "./panels";
import { RobotModelSystem } from "./robot_system";
import { TeleopSystem } from "./teleop_system";
import { type CameraViewKey, resolveTrackView } from "./track_routing";
import { VideoClient } from "./video";

const disableHeadCameraPanel = false;

const assets: AssetManifest = {
	chimeSound: {
		url: "./audio/chime.mp3",
		type: AssetType.Audio,
		priority: "background",
	},
};

const placeRelative = (
	obj: Object3D,
	cam: Camera,
	x: number,
	y: number,
	z: number,
) => {
	const yaw = new Euler().setFromQuaternion(cam.quaternion, "YXZ").y;
	obj.position.copy(cam.position);
	obj.rotation.set(0, yaw, 0);
	obj.translateX(x);
	obj.translateY(y); // relative to eye level
	obj.translateZ(z);
};

export const initWorld = async (
	container: HTMLElement,
	initialPassthrough = true,
) => {
	initConsoleStream();

	// Always initialize as AR to support consistent session features
	// We simulate VR by adding an opaque background if passthrough is disabled
	const initialMode = SessionMode.ImmersiveAR;

	console.log(
		`[initWorld] Creating world with sessionMode: ${initialMode} (passthrough: ${initialPassthrough})`,
	);

	const world = await World.create(container as HTMLDivElement, {
		assets,
		xr: {
			sessionMode: initialMode,
			offer: "none",
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
			locomotion: true, // Enable locomotion to ensure Player rig is initialized
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
	});

	const { camera } = world;

	// If in "VR Mode" (passthrough disabled), add a skybox to hide the real world
	// This allows us to use immersive-ar consistently while simulating VR
	if (!initialPassthrough) {
		const skyGeo = new SphereGeometry(100, 32, 32);
		const skyMat = new MeshBasicMaterial({
			color: 0x080808,
			side: BackSide,
			depthWrite: false,
		});
		const sky = new Mesh(skyGeo, skyMat);
		const skyEntity = world.createTransformEntity();
		if (skyEntity.object3D) {
			skyEntity.object3D.add(sky);

			const grid = new GridHelper(100, 50, 0x00ff00, 0x333333);
			// Lower the grid slightly to prevent z-fighting
			grid.position.y = -0.01;
			skyEntity.object3D.add(grid);
		}
	}

	camera.position.set(0, 1, 0.5);

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

				// Calculate offset in arc or line
				const spacing = 0.9;
				const totalWidth = (floatingKeys.length - 1) * spacing;
				const startX = -totalWidth / 2;
				const xOffset = startX + index * spacing;

				if (panel.entity.object3D) {
					const cam = world.camera;
					// Place relative to current camera
					placeRelative(panel.entity.object3D, cam, xOffset, 0, -1.5);
					panel.faceUser();
				} else {
					console.error(
						`[Video] panel entity/object3D missing for key: ${key}`,
					);
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
	// Assign to a variable that can be cleaned up
	const videoClient = new VideoClient(
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

	// Attach video client to world for cleanup
	// biome-ignore lint/suspicious/noExplicitAny: Temporary attachment for cleanup
	(world as any)._videoClient = videoClient;

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

	return world;
};
