import { createSystem, type World } from "@iwsdk/core";
import {
	CanvasTexture,
	Mesh,
	MeshBasicMaterial,
	PlaneGeometry,
	Vector3,
} from "three";
import { type TeleopLifecycle, useAppStore } from "../lib/store";
import { ControllerCameraPanelSystem } from "./controller_camera_system";

type XRTransformEntity = ReturnType<World["createTransformEntity"]>;

type LifecycleDisplay = {
	label: string;
	color: string;
	background: string;
};

const lifecycleDisplay: Record<TeleopLifecycle, LifecycleDisplay> = {
	disconnected: {
		label: "Disconnected",
		color: "#dc2626",
		background: "rgba(34, 8, 8, 0.86)",
	},
	connecting: {
		label: "Connecting",
		color: "#f59e0b",
		background: "rgba(36, 24, 8, 0.86)",
	},
	connected: {
		label: "Connected",
		color: "#3b82f6",
		background: "rgba(10, 18, 38, 0.86)",
	},
	loading_robot: {
		label: "Loading robot",
		color: "#f97316",
		background: "rgba(36, 16, 8, 0.86)",
	},
	ready: {
		label: "Ready",
		color: "#16a34a",
		background: "rgba(8, 30, 12, 0.86)",
	},
	reconnecting: {
		label: "Reconnecting",
		color: "#f59e0b",
		background: "rgba(36, 24, 8, 0.86)",
	},
	error: {
		label: "Connection error",
		color: "#ef4444",
		background: "rgba(40, 10, 12, 0.86)",
	},
};

class WristConnectionStatusPanel {
	public entity: XRTransformEntity;
	public handedness: "right" = "right";
	public anchorOffset = new Vector3(0.08, -0.1, -0.035);

	private canvas: HTMLCanvasElement;
	private context: CanvasRenderingContext2D;
	private texture: CanvasTexture;

	constructor(world: World) {
		this.entity = world.createTransformEntity();

		this.canvas = document.createElement("canvas");
		this.canvas.width = 512;
		this.canvas.height = 160;

		const context = this.canvas.getContext("2d");
		if (!context) {
			throw new Error("Connection HUD canvas context unavailable");
		}
		this.context = context;

		this.texture = new CanvasTexture(this.canvas);
		this.texture.needsUpdate = true;

		const material = new MeshBasicMaterial({
			map: this.texture,
			transparent: true,
			depthWrite: false,
		});
		const geometry = new PlaneGeometry(0.2, 0.0625);
		const mesh = new Mesh(geometry, material);
		mesh.renderOrder = 2;

		if (this.entity.object3D) {
			this.entity.object3D.add(mesh);
			this.entity.object3D.visible = true;
		}
	}

	setLifecycle(lifecycle: TeleopLifecycle) {
		const display = lifecycleDisplay[lifecycle];

		this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
		this.drawRoundedRect(
			this.context,
			8,
			8,
			this.canvas.width - 16,
			this.canvas.height - 16,
			26,
			display.background,
		);

		this.context.fillStyle = display.color;
		this.context.beginPath();
		this.context.arc(54, this.canvas.height / 2, 15, 0, Math.PI * 2);
		this.context.fill();

		this.context.fillStyle = "#f4f4f5";
		this.context.font = "600 40px sans-serif";
		this.context.textBaseline = "middle";
		this.context.fillText(display.label, 86, this.canvas.height / 2);

		this.texture.needsUpdate = true;
	}

	private drawRoundedRect(
		context: CanvasRenderingContext2D,
		x: number,
		y: number,
		width: number,
		height: number,
		radius: number,
		fillStyle: string,
	) {
		context.fillStyle = fillStyle;
		context.beginPath();
		context.moveTo(x + radius, y);
		context.lineTo(x + width - radius, y);
		context.quadraticCurveTo(x + width, y, x + width, y + radius);
		context.lineTo(x + width, y + height - radius);
		context.quadraticCurveTo(
			x + width,
			y + height,
			x + width - radius,
			y + height,
		);
		context.lineTo(x + radius, y + height);
		context.quadraticCurveTo(x, y + height, x, y + height - radius);
		context.lineTo(x, y + radius);
		context.quadraticCurveTo(x, y, x + radius, y);
		context.closePath();
		context.fill();
	}
}

export class ConnectionHudSystem extends createSystem({}) {
	private panel: WristConnectionStatusPanel | null = null;
	private lastLifecycle: TeleopLifecycle | null = null;

	init() {
		this.panel = new WristConnectionStatusPanel(this.world as World);
		const lifecycle = useAppStore.getState().teleopLifecycle;
		this.lastLifecycle = lifecycle;
		this.panel.setLifecycle(lifecycle);

		const controllerCameraSystem = this.world.getSystem(
			ControllerCameraPanelSystem,
		);
		if (controllerCameraSystem) {
			controllerCameraSystem.registerPanelWithGetter(this.panel, () => {
				return this.world.player?.raySpaces?.right;
			});
		}
	}

	update() {
		const lifecycle = useAppStore.getState().teleopLifecycle;
		if (!this.panel || lifecycle === this.lastLifecycle) {
			return;
		}

		this.lastLifecycle = lifecycle;
		this.panel.setLifecycle(lifecycle);
	}
}
