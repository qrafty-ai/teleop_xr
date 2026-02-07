import {
	createComponent,
	createSystem,
	DistanceGrabbable,
	eq,
	Hovered,
	Interactable,
	MovementMode,
	PanelDocument,
	PanelUI,
	Types,
	type UIKitDocument,
	Visibility,
	type World,
} from "@iwsdk/core";

export type Entity = ReturnType<World["createTransformEntity"]>;

import {
	BoxGeometry,
	DoubleSide,
	type Material,
	Mesh,
	MeshBasicMaterial,
	type Object3D,
	PlaneGeometry,
	VideoTexture,
} from "three";
import { GlobalRefs } from "./global_refs";

export const CameraPanelInfo = createComponent("CameraPanelInfo", {
	label: { type: Types.String, default: "" },
});

export const PanelHandle = createComponent("PanelHandle", {
	originalPosZ: { type: Types.Float32, default: 0 },
	originalScaleX: { type: Types.Float32, default: 1 },
	originalScaleY: { type: Types.Float32, default: 1 },
	originalScaleZ: { type: Types.Float32, default: 1 },
	originalColorR: { type: Types.Float32, default: 1 },
	originalColorG: { type: Types.Float32, default: 1 },
	originalColorB: { type: Types.Float32, default: 1 },
	visualState: { type: Types.Boolean, default: false },
	cooldown: { type: Types.Float32, default: 0 },
	panelEntityId: { type: Types.Float32, default: -1 },
	panelOffsetY: { type: Types.Float32, default: 0 },
});

export class DraggablePanel {
	public entity: Entity;
	public panelEntity: Entity;

	constructor(
		protected world: World,
		configPath: string,
		options: {
			maxWidth?: number;
			maxHeight?: number;
			[key: string]: unknown;
		} = {},
	) {
		const width = options.maxWidth || 0.8;
		const height = options.maxHeight || 0.6;
		const handleHeight = 0.05;
		const gap = 0.02;

		// 1. Create Handle (Root) - Interactable and Grabbable
		// Pre-add Visibility component to enable safe toggling at runtime
		this.entity = world
			.createTransformEntity()
			.addComponent(Interactable)
			.addComponent(Visibility, { isVisible: true })
			.addComponent(DistanceGrabbable, {
				movementMode: MovementMode.MoveFromTarget,
			});

		// Handle Visuals - Styling aligned with uikit panel
		const handleWidth = width * 0.5;
		const handleGeo = new BoxGeometry(handleWidth, handleHeight, 0.05);
		const handleMat = new MeshBasicMaterial({
			color: 0xe4e4e7, // Light grey
			transparent: true,
			opacity: 0.5,
		});
		const handleMesh = new Mesh(handleGeo, handleMat);
		if (this.entity.object3D) {
			this.entity.object3D.add(handleMesh);
		}

		// Panel Y - height/2 = handleHeight/2 + gap
		const panelY = height / 2 + handleHeight / 2 + gap;

		// 2. Create Panel (Child) - Interactable but NOT Grabbable
		this.panelEntity = world.createTransformEntity().addComponent(PanelUI, {
			config: configPath,
			...options,
		});

		// NOTE: We do NOT parent the panel to the handle entity.
		// Instead, we link them via ID and sync their transforms in the system.
		// This prevents "grabbing the panel" from triggering "grabbing the handle".

		GlobalRefs.panelEntities.set(this.panelEntity.index, this.panelEntity);

		this.entity.addComponent(PanelHandle, {
			originalPosZ: handleMesh.position.z,
			originalScaleX: handleMesh.scale.x,
			originalScaleY: handleMesh.scale.y,
			originalScaleZ: handleMesh.scale.z,
			originalColorR: handleMat.color.r,
			originalColorG: handleMat.color.g,
			originalColorB: handleMat.color.b,
			panelEntityId: this.panelEntity.index,
			panelOffsetY: panelY,
		});
	}

	setPosition(x: number, y: number, z: number) {
		if (this.entity.object3D) {
			this.entity.object3D.position.set(x, y, z);
		}
	}

	faceUser() {
		if (this.entity.object3D) {
			const head = this.world.camera;
			if (head) {
				this.entity.object3D.lookAt(head.position);
			}
		}
	}

	dispose() {
		if (this.panelEntity) {
			GlobalRefs.panelEntities.delete(this.panelEntity.index);
			if (typeof this.panelEntity.destroy === "function") {
				this.panelEntity.destroy();
			}
		}
		if (this.entity && typeof this.entity.destroy === "function") {
			this.entity.destroy();
		}
	}
}

export class CameraPanel extends DraggablePanel {
	private videoMesh: Mesh | null = null;
	private videoElement: HTMLVideoElement | null = null;
	private _hasVideoTrack = false;

	constructor(world: World) {
		super(world, "./ui/camera.json", {
			maxHeight: 0.6,
			maxWidth: 0.8,
		});
	}

	public hasVideoTrack(): boolean {
		return this._hasVideoTrack;
	}

	setLabel(text: string) {
		if (this.panelEntity.hasComponent(CameraPanelInfo)) {
			this.panelEntity.setValue(CameraPanelInfo, "label", text);
		} else {
			this.panelEntity.addComponent(CameraPanelInfo, { label: text });
		}
	}

	dispose() {
		if (this.videoElement) {
			this.videoElement.pause();
			this.videoElement.srcObject = null;
			this.videoElement.remove();
			this.videoElement = null;
		}

		if (this.videoMesh) {
			this.videoMesh.geometry.dispose();
			if (Array.isArray(this.videoMesh.material)) {
				this.videoMesh.material.forEach((m) => {
					m.dispose();
				});
			} else {
				(this.videoMesh.material as Material).dispose();
			}
			if (this.panelEntity?.object3D) {
				this.panelEntity.object3D.remove(this.videoMesh);
			}
			this.videoMesh = null;
		}

		this._hasVideoTrack = false;
		super.dispose();
	}

	setVideoTrack(track: MediaStreamTrack) {
		if (this.videoMesh) return; // Already set

		this._hasVideoTrack = true;
		const stream = new MediaStream([track]);
		this.videoElement = document.createElement("video");
		this.videoElement.srcObject = stream;
		this.videoElement.playsInline = true;
		this.videoElement.muted = true; // Required for autoplay
		this.videoElement.style.display = "none";
		document.body.appendChild(this.videoElement);

		this.videoElement.play().catch((e) => {
			console.error(`Video play error: ${e}`);
		});

		const texture = new VideoTexture(this.videoElement);
		// Aspect ratio adjusted to fit panel
		const geometry = new PlaneGeometry(0.76, 0.45);
		const material = new MeshBasicMaterial({ map: texture, side: DoubleSide });
		this.videoMesh = new Mesh(geometry, material);

		// Position it slightly in front of the panel to avoid z-fighting
		this.videoMesh.position.z = 0.01;
		// Adjust y to be centered below the header
		this.videoMesh.position.y = -0.05;

		// Attach to panelEntity, not the handle/root
		if (this.panelEntity.object3D) {
			this.panelEntity.object3D.add(this.videoMesh);
		}
	}
}

export class ControllerCameraPanel {
	public entity: Entity;
	public handedness: "left" | "right";
	private videoMesh: Mesh | null = null;
	private videoElement: HTMLVideoElement | null = null;
	private _hasVideoTrack = false;

	constructor(world: World, handedness: "left" | "right") {
		this.handedness = handedness;

		// Create a simple transform entity (no grabbable, no panel UI)
		this.entity = world.createTransformEntity();

		// Create a background plane for the video
		const bgGeo = new PlaneGeometry(0.2, 0.15);
		const bgMat = new MeshBasicMaterial({
			color: 0x1a1a1a,
			transparent: true,
			opacity: 0.9,
			depthWrite: false,
			side: DoubleSide,
		});
		const bgMesh = new Mesh(bgGeo, bgMat);
		bgMesh.position.z = -0.002;
		bgMesh.renderOrder = 0;
		if (this.entity.object3D) {
			this.entity.object3D.add(bgMesh);
		}

		// Default to hidden if no track
		if (this.entity.object3D) {
			this.entity.object3D.visible = false;
		}
	}

	public hasVideoTrack(): boolean {
		return this._hasVideoTrack;
	}

	setVideoTrack(track: MediaStreamTrack) {
		if (this.videoMesh) return; // Already set

		this._hasVideoTrack = true;
		if (this.entity.object3D) {
			this.entity.object3D.visible = true;
		}

		const stream = new MediaStream([track]);
		this.videoElement = document.createElement("video");
		this.videoElement.srcObject = stream;
		this.videoElement.playsInline = true;
		this.videoElement.muted = true;
		this.videoElement.style.display = "none";
		document.body.appendChild(this.videoElement);

		this.videoElement.play().catch((e) => {
			console.error(`Video play error: ${e}`);
		});

		const texture = new VideoTexture(this.videoElement);
		const geometry = new PlaneGeometry(0.18, 0.135); // Slightly smaller than bg
		const material = new MeshBasicMaterial({ map: texture, side: DoubleSide });
		this.videoMesh = new Mesh(geometry, material);
		this.videoMesh.position.z = 0.001; // Slightly in front of bg
		this.videoMesh.renderOrder = 1;
		if (this.entity.object3D) {
			this.entity.object3D.add(this.videoMesh);
		}
	}
}

export class CameraPanelSystem extends createSystem({
	cameraPanels: {
		required: [PanelUI, PanelDocument, CameraPanelInfo],
		where: [eq(PanelUI, "config", "./ui/camera.json")],
	},
}) {
	init() {
		this.queries.cameraPanels.subscribe("qualify", (entity) => {
			const document = PanelDocument.data.document[
				entity.index
			] as UIKitDocument;
			const label = CameraPanelInfo.data.label[entity.index];
			if (document && label) {
				const el = document.getElementById("camera-label");
				if (el) {
					console.log(
						`[CameraPanelSystem] Setting label for entity ${entity.index} to: ${label}`,
					);
					el.setProperties({ text: label });
				}
			}
		});
	}
}

export class PanelHoverSystem extends createSystem({
	handles: {
		required: [PanelHandle, Interactable],
	},
}) {
	update(delta: number) {
		this.queries.handles.entities.forEach((entity) => {
			const isCurrentlyHovered = entity.hasComponent(Hovered);
			let visualState = PanelHandle.data.visualState[entity.index];
			let cd = PanelHandle.data.cooldown[entity.index];

			if (isCurrentlyHovered) {
				visualState = 1;
				cd = 0.1; // Cooldown of 0.1s to prevent flicker
			} else if (cd > 0) {
				cd -= delta;
				if (cd <= 0) {
					visualState = 0;
					cd = 0;
				}
			} else {
				visualState = 0;
			}

			PanelHandle.data.visualState[entity.index] = visualState;
			PanelHandle.data.cooldown[entity.index] = cd;

			// Sync Panel Position
			const panelId = PanelHandle.data.panelEntityId[entity.index];
			const offsetY = PanelHandle.data.panelOffsetY[entity.index];

			if (panelId !== -1 && entity.object3D) {
				const panelEntity = GlobalRefs.panelEntities.get(panelId);
				if (panelEntity?.object3D) {
					panelEntity.object3D.position.copy(entity.object3D.position);
					panelEntity.object3D.quaternion.copy(entity.object3D.quaternion);
					// Apply local offset Y in rotated space (along local Up)
					panelEntity.object3D.translateY(offsetY);
				}
			}

			if (!entity.object3D) return;

			const mesh = entity.object3D.children.find(
				(c: Object3D) => (c as Mesh).isMesh,
			) as Mesh;
			if (!mesh) return;

			const mat = mesh.material as MeshBasicMaterial;
			const origZ = PanelHandle.data.originalPosZ[entity.index];
			const origSX = PanelHandle.data.originalScaleX[entity.index];
			const origSY = PanelHandle.data.originalScaleY[entity.index];
			const origSZ = PanelHandle.data.originalScaleZ[entity.index];
			const origR = PanelHandle.data.originalColorR[entity.index];
			const origG = PanelHandle.data.originalColorG[entity.index];
			const origB = PanelHandle.data.originalColorB[entity.index];

			if (visualState) {
				mesh.position.z = origZ + 0.04;
				mesh.scale.set(origSX * 1.2, origSY * 1.2, origSZ * 1.2);
				mat.color.setRGB(origR * 0.4, origG * 0.4, origB * 0.4);
			} else {
				mesh.position.z = origZ;
				mesh.scale.set(origSX, origSY, origSZ);
				mat.color.setRGB(origR, origG, origB);
			}
		});
	}
}
