import {
	createSystem,
	eq,
	PanelDocument,
	PanelUI,
	Quaternion,
	type UIKitDocument,
	Vector3,
} from "@iwsdk/core";
import { getCameraEnabled } from "./camera_config";
import { setCameraViewsConfig } from "./camera_views";
import { getClientId } from "./client_id";
import { GlobalRefs } from "./global_refs";
import { RobotModelSystem } from "./robot_system";
import type { CameraViewKey } from "./track_routing";

type DevicePose = {
	position: { x: number; y: number; z: number };
	orientation: { x: number; y: number; z: number; w: number };
};

export class TeleopSystem extends createSystem({
	teleopPanel: {
		required: [PanelUI, PanelDocument],
		where: [eq(PanelUI, "config", "./ui/teleop.json")],
	},
}) {
	private ws: WebSocket | null = null;
	private clientId = getClientId();
	private inControl = false;
	private controlPollTimer: number | null = null;
	// biome-ignore lint/suspicious/noExplicitAny: legacy
	private statusText: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: legacy
	private fpsText: any = null;
	// biome-ignore lint/suspicious/noExplicitAny: legacy
	private latencyText: any = null;
	private frameCount = 0;
	private lastFpsTime = 0;
	private currentFps = 0;
	private lastSendTime = 0;
	private tempPosition = new Vector3();
	private tempQuaternion = new Quaternion();
	private menuButtonState = false;
	private loggedStats = false;
	// biome-ignore lint/suspicious/noExplicitAny: legacy
	private xrButtonText: any = null;
	private wasPresenting = false;
	private lastConnectionError: string | null = null;

	init() {
		this.connectWS();

		this.queries.teleopPanel.subscribe("qualify", (entity) => {
			const document = PanelDocument.data.document[
				entity.index
			] as UIKitDocument;
			if (!document) {
				return;
			}

			this.statusText = document.getElementById("status-text");
			this.fpsText = document.getElementById("fps-text");
			this.latencyText = document.getElementById("latency-text");
			this.xrButtonText = document.getElementById("xr-button-text");

			console.log("[TeleopSystem] Found elements:", {
				status: !!this.statusText,
				fps: !!this.fpsText,
				latency: !!this.latencyText,
				xrBtn: !!this.xrButtonText,
			});

			const cameraButton = document.getElementById("camera-button");
			if (cameraButton) {
				cameraButton.addEventListener("click", () => {
					// Use GlobalRefs - populated by index.ts at creation time
					// DO NOT access ECS queries during click events (causes freeze)

					// Determine target state based on currently visible AND enabled panels
					const floatingPanels = Array.from(
						GlobalRefs.cameraPanels.entries(),
					).map(([key, panel]) => ({ key, panel }));
					const wristPanels = [
						{ key: "wrist_left", panel: GlobalRefs.leftWristPanel },
						{ key: "wrist_right", panel: GlobalRefs.rightWristPanel },
					];

					const allPanelRefs = [...floatingPanels, ...wristPanels].filter(
						(p) => !!p.panel,
					);

					// Find if any enabled panel is currently visible
					const anyEnabledVisible = allPanelRefs.some(({ key, panel }) => {
						const enabled = getCameraEnabled(key as CameraViewKey);
						return (
							enabled &&
							panel.entity &&
							panel.entity.object3D &&
							panel.entity.object3D.visible
						);
					});

					const targetVisible = !anyEnabledVisible;

					// Apply targetVisible ONLY to enabled panels. Disabled panels stay hidden.
					allPanelRefs.forEach(({ key, panel }) => {
						if (panel.entity?.object3D) {
							const enabled = getCameraEnabled(key as CameraViewKey);
							if (targetVisible && enabled) {
								// Only show if it has an active video track
								if (
									typeof panel.hasVideoTrack === "function" &&
									panel.hasVideoTrack()
								) {
									panel.entity.object3D.visible = true;
								}
							} else {
								// Always hide if target is hidden OR if panel is disabled
								panel.entity.object3D.visible = false;
							}
						}
					});
				});
			}

			const cameraSettingsBtn = document.getElementById("camera-settings-btn");
			if (cameraSettingsBtn) {
				cameraSettingsBtn.addEventListener("click", () => {
					const panel = GlobalRefs.cameraSettingsPanel;
					if (panel?.entity.object3D) {
						panel.entity.object3D.visible = !panel.entity.object3D.visible;
					}
				});
			}

			const xrButton = document.getElementById("xr-button");
			if (xrButton) {
				xrButton.addEventListener("click", async () => {
					const renderer = this.world.renderer;
					if (renderer.xr.isPresenting) {
						renderer.xr.getSession()?.end();
					} else {
						// Attempt to enter XR
						if ("xr" in navigator) {
							try {
								// biome-ignore lint/suspicious/noExplicitAny: navigator.xr
								const session = await (navigator as any).xr.requestSession(
									"immersive-ar",
									{
										requiredFeatures: ["local-floor"],
										optionalFeatures: ["hand-tracking", "anchors", "layers"],
									},
								);
								renderer.xr.setSession(session);
							} catch (e) {
								console.error("Failed to enter XR", e);
							}
						}
					}
				});
			}

			const isConnected = this.ws && this.ws.readyState === WebSocket.OPEN;
			this.updateStatus(
				isConnected ? "Connected" : "Disconnected",
				!!isConnected,
			);
		});
	}

	connectWS() {
		const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
		const wsUrl = `${protocol}//${window.location.host}/ws`;

		this.ws = new WebSocket(wsUrl);

		this.ws.onopen = () => {
			this.lastConnectionError = null;
			this.inControl = false;
			this.updateStatus("Connected", true);
			this.startControlPolling();
			this.sendControlCheck();
		};

		this.ws.onclose = () => {
			this.stopControlPolling();
			if (this.lastConnectionError) {
				this.updateStatus(this.lastConnectionError, false);
			} else {
				this.updateStatus("Disconnected", false);
			}
			const delayMs = this.lastConnectionError ? 500 : 3000;
			setTimeout(() => this.connectWS(), delayMs);
		};

		this.ws.onerror = (error) => {
			console.error("WS Error", error);
		};

		this.ws.onmessage = (event) => {
			try {
				const message = JSON.parse(event.data);
				if (message.type === "deny") {
					this.inControl = false;
					this.lastConnectionError =
						message.data?.reason === "not_in_control"
							? "Connected (no control)"
							: "Connected (denied)";
					this.updateStatus(this.lastConnectionError, true);
					this.startControlPolling();
				} else if (message.type === "control_status") {
					const inControl = Boolean(message.data?.in_control);
					this.inControl = inControl;
					if (inControl) {
						this.lastConnectionError = null;
						this.updateStatus("Connected (in control)", true);
						this.stopControlPolling();
					} else {
						this.updateStatus("Connected (waiting for control)", true);
						this.startControlPolling();
					}
				} else if (message.type === "config") {
					if (message.data?.input_mode) {
						this.inputMode = message.data.input_mode;
					}
					setCameraViewsConfig(message.data?.camera_views ?? null);
				} else if (message.type === "robot_config") {
					const robotSystem = this.world.getSystem(RobotModelSystem);
					if (robotSystem) {
						robotSystem.onRobotConfig(message.data);
					}
				} else if (message.type === "robot_state") {
					const robotSystem = this.world.getSystem(RobotModelSystem);
					if (robotSystem) {
						robotSystem.onRobotState(message.data);
					}
				}
			} catch (error) {
				console.warn("Failed to parse WS message", error);
			}
		};
	}

	private startControlPolling() {
		if (this.controlPollTimer !== null) {
			return;
		}
		this.controlPollTimer = window.setInterval(() => {
			this.sendControlCheck();
		}, 1000);
	}

	private stopControlPolling() {
		if (this.controlPollTimer === null) {
			return;
		}
		window.clearInterval(this.controlPollTimer);
		this.controlPollTimer = null;
	}

	private sendControlCheck() {
		if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
			return;
		}
		this.ws.send(
			JSON.stringify({
				type: "control_check",
				client_id: this.clientId,
				data: {},
			}),
		);
	}

	updateStatus(text: string, connected: boolean) {
		if (!this.statusText) {
			return;
		}

		this.statusText.setProperties({ text });
		if (this.statusText.classList) {
			if (connected) {
				this.statusText.classList.add("connected");
			} else {
				this.statusText.classList.remove("connected");
			}
		}
	}

	// biome-ignore lint/suspicious/noExplicitAny: legacy
	poseFromObject(object: any): DevicePose | null {
		if (!object?.getWorldPosition || !object?.getWorldQuaternion) {
			return null;
		}
		object.getWorldPosition(this.tempPosition);
		object.getWorldQuaternion(this.tempQuaternion);
		return {
			position: {
				x: this.tempPosition.x,
				y: this.tempPosition.y,
				z: this.tempPosition.z,
			},
			orientation: {
				x: this.tempQuaternion.x,
				y: this.tempQuaternion.y,
				z: this.tempQuaternion.z,
				w: this.tempQuaternion.w,
			},
		};
	}

	buildControllerDevice(
		handedness: "left" | "right",
		// biome-ignore lint/suspicious/noExplicitAny: legacy
		raySpace: any,
		// biome-ignore lint/suspicious/noExplicitAny: legacy
		gamepad: any,
		isHandPrimary: boolean,
	) {
		if (isHandPrimary) {
			return null;
		}

		const pose = this.poseFromObject(raySpace);
		if (!pose) {
			return null;
		}

		const device: {
			role: string;
			handedness: string;
			gripPose: DevicePose;
			gamepad?: {
				buttons: Array<{ pressed: boolean; touched: boolean; value: number }>;
				axes: number[];
			};
		} = {
			role: "controller",
			handedness,
			gripPose: pose,
		};

		const rawGamepad = gamepad?.gamepad;
		if (rawGamepad) {
			device.gamepad = {
				// biome-ignore lint/suspicious/noExplicitAny: legacy
				buttons: Array.from(rawGamepad.buttons).map((button: any) => ({
					pressed: button.pressed,
					touched: button.touched,
					value: button.value,
				})),
				axes: Array.from(rawGamepad.axes),
			};
		}

		return device;
	}

	update(_delta: number, time: number) {
		const isPresenting = this.world.renderer.xr.isPresenting;
		if (isPresenting !== this.wasPresenting) {
			this.wasPresenting = isPresenting;
			if (this.xrButtonText) {
				this.xrButtonText.setProperties({
					text: isPresenting ? "Exit XR" : "Enter XR",
				});
			}
		}

		if (this.lastFpsTime === 0) {
			this.lastFpsTime = time;
		}

		this.frameCount += 1;
		if (time - this.lastFpsTime >= 1.0) {
			this.currentFps = Math.round(this.frameCount / (time - this.lastFpsTime));
			this.frameCount = 0;
			this.lastFpsTime = time;
		}

		if (time - this.lastSendTime <= 0.01) {
			return;
		}
		this.lastSendTime = time;
		// biome-ignore lint/suspicious/noExplicitAny: legacy
		const input = (this as any).input ?? this.world.input;

		const state = this.gatherInputState(input);

		const latency = state ? state.fetch_latency_ms : 0;
		const head = state?.devices.find((device) => device.role === "head");
		this.updateLocalStats(head?.pose ?? null, this.currentFps, latency);

		if (!state || state.devices.length === 0) {
			return;
		}

		if (this.ws && this.ws.readyState === WebSocket.OPEN && this.inControl) {
			this.ws.send(
				JSON.stringify({
					type: "xr_state",
					client_id: this.clientId,
					data: state,
				}),
			);
		}
	}

	updateLocalStats(_pose: DevicePose | null, fps: number, latency: number) {
		if (!this.loggedStats) {
			console.log(
				`[TeleopSystem] First updateLocalStats: fps=${fps}, latency=${latency}, fpsText=${!!this
					.fpsText}`,
			);
			this.loggedStats = true;
		}

		if (this.fpsText) {
			this.fpsText.setProperties({ text: `${fps}`, color: "#18181b" });
		}

		if (this.latencyText) {
			const latencyValue = Number.isFinite(latency) ? latency : 0;
			this.latencyText.setProperties({
				text: `${latencyValue.toFixed(1)}ms`,
				color: "#18181b",
			});
		}
	}

	// biome-ignore lint/suspicious/noExplicitAny: legacy
	gatherInputState(input: any) {
		const leftGamepad = input?.gamepads?.left?.gamepad;
		if (leftGamepad?.buttons && leftGamepad.buttons.length > 0) {
			// The menu button is the last item of the left gamepad button array
			const menuButton = leftGamepad.buttons[leftGamepad.buttons.length - 1];

			if (menuButton) {
				if (menuButton.pressed) {
					if (!this.menuButtonState) {
						this.menuButtonState = true;
						if (GlobalRefs.teleopPanelRoot) {
							GlobalRefs.teleopPanelRoot.visible =
								!GlobalRefs.teleopPanelRoot.visible;
						}
					}
				} else {
					this.menuButtonState = false;
				}
			}
		}

		const fetchStart = performance.now();
		const timestamp_unix_ms = Date.now();
		const devices: Array<{
			role: string;
			handedness: string;
			pose?: DevicePose;
			gripPose?: DevicePose;
			gamepad?: {
				buttons: Array<{ pressed: boolean; touched: boolean; value: number }>;
				axes: number[];
			};
		}> = [];

		// biome-ignore lint/suspicious/noExplicitAny: legacy
		const player = (this as any).player ?? this.world.player;
		const headPose = this.poseFromObject(player?.head);
		if (headPose) {
			devices.push({
				role: "head",
				handedness: "none",
				pose: headPose,
			});
		}

		const leftDevice = this.buildControllerDevice(
			"left",
			player?.raySpaces?.left,
			input?.gamepads?.left,
			Boolean(input?.isPrimary?.("hand", "left")),
		);
		if (leftDevice) {
			devices.push(leftDevice);
		}

		const rightDevice = this.buildControllerDevice(
			"right",
			player?.raySpaces?.right,
			input?.gamepads?.right,
			Boolean(input?.isPrimary?.("hand", "right")),
		);
		if (rightDevice) {
			devices.push(rightDevice);
		}

		if (devices.length === 0) {
			return null;
		}

		const fetch_latency_ms = performance.now() - fetchStart;

		return {
			timestamp_unix_ms,
			devices,
			fps: this.currentFps,
			fetch_latency_ms,
		};
	}
}
