import {
	createSystem,
	DistanceGrabbable,
	Interactable,
	MovementMode,
	type World,
} from "@iwsdk/core";
import {
	AmbientLight,
	AxesHelper,
	DirectionalLight,
	Group,
	LoadingManager,
	Mesh,
	type Object3D,
	SRGBColorSpace,
	Texture,
	Vector3,
} from "three";
import { ColladaLoader, GLTFLoader, STLLoader } from "three-stdlib";
import URDFLoader from "urdf-loader";
import { useAppStore } from "../lib/store";
import type { Entity } from "./panels";

interface URDFRobot extends Object3D {
	joints: Record<string, { setJointValue: (v: number) => void }>;
}

export class RobotModelSystem extends createSystem({}) {
	private loader!: URDFLoader;
	private robotEntity: Entity | null = null;
	private robotModel: Object3D | null = null;
	private axesHelper: AxesHelper | null = null;

	init() {
		this.loader = new URDFLoader();
		this.loader.packages = (pkg: string) => `/robot_assets/${pkg}`;
		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		// biome-ignore lint/suspicious/noExplicitAny: URDFLoader type doesn't expose loadMeshCb
		(this.loader as any).loadMeshCb = (
			path: string,
			manager: LoadingManager,
			// biome-ignore lint/suspicious/noExplicitAny: callback error type is unspecified in URDFLoader
			done: (mesh: Object3D | null, err?: any) => void,
		) => {
			const ext = path.split(".").pop()?.toLowerCase();
			if (ext === "glb" || ext === "gltf") {
				new GLTFLoader(manager).load(
					path,
					(gltf) => done(gltf.scene),
					undefined,
					(err) => done(null, err),
				);
			} else if (ext === "stl") {
				new STLLoader(manager).load(
					path,
					(geom) => done(new Mesh(geom)),
					undefined,
					(err) => done(null, err),
				);
			} else if (ext === "dae") {
				new ColladaLoader(manager).load(
					path,
					(collada) => done(collada.scene),
					undefined,
					(err) => done(null, err),
				);
			} else {
				// biome-ignore lint/suspicious/noExplicitAny: fallback to default loader if available
				const loader = this.loader as any;
				if (loader.defaultMeshLoader) {
					loader.defaultMeshLoader(path, manager, done);
				} else {
					done(null, new Error(`No loader available for extension "${ext}"`));
				}
			}
		};

		let lastRobotVisible = useAppStore.getState().robotSettings.robotVisible;
		let lastShowAxes = useAppStore.getState().robotSettings.showAxes;
		let lastResetTrigger = useAppStore.getState().robotResetTrigger;
		let lastDistanceGrab =
			useAppStore.getState().robotSettings.distanceGrabEnabled;

		useAppStore.subscribe((state) => {
			if (state.robotSettings.robotVisible !== lastRobotVisible) {
				lastRobotVisible = state.robotSettings.robotVisible;
				if (this.robotEntity?.object3D) {
					this.robotEntity.object3D.visible = lastRobotVisible;
				}
			}

			if (state.robotSettings.showAxes !== lastShowAxes) {
				lastShowAxes = state.robotSettings.showAxes;
				if (this.axesHelper) {
					this.axesHelper.visible = lastShowAxes;
				}
			}

			if (state.robotSettings.distanceGrabEnabled !== lastDistanceGrab) {
				lastDistanceGrab = state.robotSettings.distanceGrabEnabled;
				if (this.robotEntity) {
					const hasGrab = this.robotEntity.hasComponent(DistanceGrabbable);

					if (lastDistanceGrab && !hasGrab) {
						if (!this.robotEntity.hasComponent(Interactable)) {
							this.robotEntity.addComponent(Interactable);
						}
						this.robotEntity.addComponent(DistanceGrabbable, {
							movementMode: MovementMode.MoveFromTarget,
						});
					} else if (!lastDistanceGrab && hasGrab) {
						this.robotEntity.removeComponent(DistanceGrabbable);
						if (this.robotEntity.hasComponent(Interactable)) {
							this.robotEntity.removeComponent(Interactable);
						}
					}
				}
			}

			if (state.robotResetTrigger !== lastResetTrigger) {
				lastResetTrigger = state.robotResetTrigger;
				if (this.robotEntity?.object3D) {
					this.positionRobotInFront(
						this.robotEntity.object3D,
						this.world.camera,
					);
				}
			}
		});
	}

	async onRobotConfig(data: {
		urdf_url: string;
		model_scale?: number;
		initial_rotation_euler?: number[];
	}) {
		console.log(
			"[RobotModelSystem] Loading robot from",
			data.urdf_url,
			"Scale:",
			data.model_scale,
		);

		try {
			const robot = await new Promise<Object3D>((resolve, reject) => {
				const manager = new LoadingManager();
				let _robot: Object3D;

				manager.onLoad = () => {
					console.log("[RobotModelSystem] All meshes loaded");
					if (_robot) {
						this.processRobotMaterials(_robot);
						resolve(_robot);
					}
				};

				manager.onError = (url) => {
					console.error(
						"[RobotModelSystem] LoadingManager error for URL:",
						url,
					);
					reject(new Error(`Failed to load: ${url}`));
				};

				this.loader.manager = manager;

				this.loader.load(
					data.urdf_url,
					(result) => {
						_robot = result as Object3D;
						console.log(
							"[RobotModelSystem] URDF parsed, waiting for meshes...",
						);
					},
					(progress) => {
						if (progress?.total) {
							console.log(
								`[RobotModelSystem] Loading progress: ${Math.round((progress.loaded / progress.total) * 100)}%`,
							);
						}
					},
					(err) => {
						console.error("[RobotModelSystem] Loader error details:", err);
						if (err instanceof ErrorEvent) {
							console.error(
								"[RobotModelSystem] ErrorEvent message:",
								err.message,
							);
						}
						reject(err);
					},
				);
			});

			if (this.robotEntity && typeof this.robotEntity.destroy === "function") {
				this.robotEntity.destroy();
				this.robotEntity = null;
				this.robotModel = null;
			}

			const tiltNode = new Group();
			tiltNode.rotation.x = -Math.PI / 2;
			tiltNode.rotation.z = Math.PI / 2;
			let rx = 0;
			let ry = 0;
			let rz = 0;
			if (
				data.initial_rotation_euler &&
				data.initial_rotation_euler.length === 3
			) {
				[rx, ry, rz] = data.initial_rotation_euler;
				// Apply rotation to the robot itself (in Z-up space)
				robot.rotation.set(rx, ry, rz);
			}

			const scale = data.model_scale || 1.0;
			robot.scale.set(scale, scale, scale);

			tiltNode.add(robot);

			this.robotModel = robot;

			// Create interactable entity
			this.robotEntity = (this.world as World).createTransformEntity();

			if (useAppStore.getState().robotSettings.distanceGrabEnabled) {
				this.robotEntity.addComponent(Interactable);
				this.robotEntity.addComponent(DistanceGrabbable, {
					movementMode: MovementMode.MoveFromTarget,
				});
			}

			if (this.robotEntity?.object3D) {
				this.robotEntity.object3D.visible =
					useAppStore.getState().robotSettings.robotVisible;
			}

			const robotObject = this.robotEntity?.object3D;
			if (!robotObject) {
				console.warn("[RobotModelSystem] Robot entity has no object3D");
				return;
			}
			const robotObject3D: Object3D = robotObject;
			robotObject3D.add(tiltNode);

			this.axesHelper = new AxesHelper(1.0);
			this.axesHelper.visible = useAppStore.getState().robotSettings.showAxes;
			tiltNode.add(this.axesHelper);

			// Add lights to ensure textures are visible
			const ambientLight = new AmbientLight(0xffffff, 0.6);
			const dirLight = new DirectionalLight(0xffffff, 0.8);
			dirLight.position.set(1, 2, 1);
			robotObject3D.add(ambientLight);
			robotObject3D.add(dirLight);

			const camera = this.world.camera;
			if (camera) {
				this.positionRobotInFront(robotObject3D, camera);
			}

			console.log("[RobotModelSystem] Robot model loaded and added to scene");
		} catch (error: unknown) {
			if (error instanceof Error) {
				console.error(
					"[RobotModelSystem] Failed to load robot:",
					error.message,
				);
				console.error("[RobotModelSystem] Stack trace:", error.stack);
			} else {
				console.error("[RobotModelSystem] Failed to load robot:", error);
			}
		}
	}

	private positionRobotInFront(
		robotObject: Object3D,
		camera: Object3D | null | undefined,
	) {
		if (!camera || !camera.position) {
			return;
		}
		const { spawnDistance, spawnHeight } = useAppStore.getState().robotSettings;
		const cameraPosition = camera.position;
		const forward = new Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
		forward.y = 0; // Keep horizontal
		forward.normalize();
		const spawnPos = cameraPosition
			.clone()
			.add(forward.multiplyScalar(spawnDistance));
		if (!robotObject.position || !robotObject.lookAt) {
			return;
		}
		// Keep somewhat level with camera but not too high/low
		// If we assume floor is 0, maybe fixed height?
		// But user might be sitting/standing. Let's use camera height minus a bit, or just keep camera height.
		// Let's drop it slightly below eye level (e.g. 30cm down) so it's comfortable to look at
		spawnPos.y = Math.max(0.5, cameraPosition.y + spawnHeight);
		robotObject.position.copy(spawnPos);
		robotObject.rotation.set(0, 0, 0);
	}

	private processRobotMaterials(robot: Object3D) {
		robot.traverse((child) => {
			if (child instanceof Mesh) {
				const mesh = child as Mesh;
				const material = mesh.material;

				if (material) {
					const mats = Array.isArray(material) ? material : [material];

					for (const m of mats) {
						// Ensure texture encoding is correct for standard materials
						const textureMap = this.getTextureMap(m);
						if (textureMap) {
							textureMap.colorSpace = SRGBColorSpace;
							textureMap.needsUpdate = true;
						}
						m.needsUpdate = true;
					}
				}
			}
		});
	}

	private getTextureMap(material: unknown): Texture | null {
		if (typeof material !== "object" || material === null) {
			return null;
		}
		if (!("map" in material)) {
			return null;
		}
		const map = (material as { map?: unknown }).map;
		return map instanceof Texture ? map : null;
	}

	onRobotState(data: { joints: Record<string, number> }) {
		if (!this.robotModel || !data.joints) return;

		const robot = this.robotModel as unknown as URDFRobot;
		if (robot.joints) {
			for (const [name, value] of Object.entries(data.joints)) {
				if (
					robot.joints[name] &&
					typeof robot.joints[name].setJointValue === "function"
				) {
					robot.joints[name].setJointValue(value);
				}
			}
		}
	}
}
