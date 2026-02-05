import { createSystem, Quaternion, Vector3, Visibility } from "@iwsdk/core";
import { getClientId } from "../client_id";
import {
	type TeleopSettings,
	type TeleopTelemetry,
	useAppStore,
} from "../lib/store";
import { setCameraViewsConfig } from "./camera_views";
import { GlobalRefs } from "./global_refs";
import { RobotModelSystem } from "./robot_system";

type DevicePose = {
	position: { x: number; y: number; z: number };
	orientation: { x: number; y: number; z: number; w: number };
};

export class TeleopSystem extends createSystem({}) {
	private ws: WebSocket | null = null;
	private frameCount = 0;
	private lastFpsTime = 0;
	private currentFps = 0;
	private lastSendTime = 0;
	private updateInterval = 0.01;
	private tempPosition = new Vector3();
	private tempQuaternion = new Quaternion();
	private menuButtonState = false;
	public inputMode: string | null = null;
	private clientId = getClientId();

	init() {
		this.connectWS();

		useAppStore.subscribe((state) => {
			this.updateInterval = 1 / state.advancedSettings.updateRate;
		});
		this.updateInterval =
			1 / useAppStore.getState().advancedSettings.updateRate;
	}

	connectWS() {
		useAppStore.getState().setConnectionStatus("connecting");
		const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
		const wsUrl = `${protocol}//${window.location.host}/ws`;

		this.ws = new WebSocket(wsUrl);

		this.ws.onopen = () => {
			this.updateStatus("Connected", true);
		};

		this.ws.onclose = () => {
			this.updateStatus("Disconnected", false);
			setTimeout(() => this.connectWS(), 3000);
		};

		this.ws.onerror = (error) => {
			console.error("WS Error", error);
		};

		this.ws.onmessage = (event) => {
			try {
				const message = JSON.parse(event.data);
				if (message.type === "config") {
					if (message.data?.input_mode) {
						this.inputMode = message.data.input_mode;
					}
					const cameraViews = message.data?.camera_views ?? null;
					const availableCameraKeys = cameraViews
						? Object.keys(cameraViews)
						: [];
					useAppStore.getState().setAvailableCameras(availableCameraKeys);
					setCameraViewsConfig(cameraViews);
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

	updateStatus(_text: string, connected: boolean) {
		useAppStore
			.getState()
			.setConnectionStatus(connected ? "connected" : "disconnected");
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
		if (this.lastFpsTime === 0) {
			this.lastFpsTime = time;
		}

		this.frameCount += 1;
		if (time - this.lastFpsTime >= 1.0) {
			this.currentFps = Math.round(this.frameCount / (time - this.lastFpsTime));
			this.frameCount = 0;
			this.lastFpsTime = time;
		}

		if (time - this.lastSendTime <= this.updateInterval) {
			return;
		}
		this.lastSendTime = time;
		// biome-ignore lint/suspicious/noExplicitAny: legacy
		const input = (this as any).input ?? this.world.input;

		const appState = useAppStore.getState();
		const { speed, turnSpeed, precisionMode } = appState.teleopSettings;
		const teleopSettings: TeleopSettings = {
			speed,
			turnSpeed,
			precisionMode,
		};
		const state = this.gatherInputState(input, teleopSettings);

		const latency = state ? state.fetch_latency_ms : 0;
		const head = state?.devices.find((device) => device.role === "head");
		this.updateLocalStats(
			head?.pose ?? null,
			this.currentFps,
			latency,
			appState.setTeleopTelemetry,
		);

		if (!state || state.devices.length === 0) {
			return;
		}

		if (this.ws && this.ws.readyState === WebSocket.OPEN) {
			this.ws.send(
				JSON.stringify({
					type: "xr_state",
					client_id: this.clientId,
					data: state,
				}),
			);
		}
	}

	updateLocalStats(
		_pose: DevicePose | null,
		fps: number,
		latency: number,
		setTeleopTelemetry: (telemetry: TeleopTelemetry) => void,
	) {
		const latencyMsValue = Number.isFinite(latency) ? latency : 0;
		setTeleopTelemetry({ fps, latencyMs: latencyMsValue });
	}

	// biome-ignore lint/suspicious/noExplicitAny: legacy
	gatherInputState(input: any, teleopSettings: TeleopSettings) {
		const leftGamepad = input?.gamepads?.left?.gamepad;
		if (leftGamepad?.buttons && leftGamepad.buttons.length > 0) {
			// The menu button is the last item of the left gamepad button array
			const menuButton = leftGamepad.buttons[leftGamepad.buttons.length - 1];

			if (menuButton) {
				if (menuButton.pressed) {
					if (!this.menuButtonState) {
						this.menuButtonState = true;
						const teleopPanelRoot = GlobalRefs.teleopPanelRoot;
						if (teleopPanelRoot?.entity?.hasComponent(Visibility)) {
							const currentVisibility = teleopPanelRoot.entity.getValue(
								Visibility,
								"isVisible",
							);
							teleopPanelRoot.entity.setValue(
								Visibility,
								"isVisible",
								!currentVisibility,
							);
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
			teleop_settings: teleopSettings,
		};
	}
}
