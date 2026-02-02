import { createSystem, type World } from "@iwsdk/core";
import { LoadingManager, type Object3D } from "three";
import URDFLoader from "urdf-loader";
import type { Entity } from "./panels";

interface URDFRobot extends Object3D {
	joints: Record<string, { setJointValue: (v: number) => void }>;
}

export class RobotModelSystem extends createSystem({}) {
	private loader!: URDFLoader;
	private robotEntity: Entity | null = null;
	private robotModel: Object3D | null = null;

	init() {
		this.loader = new URDFLoader();
		this.loader.packages = (pkg: string) => `/robot_assets/${pkg}`;
	}

	async onRobotConfig(data: { urdf_url: string }) {
		console.log("[RobotModelSystem] Loading robot from", data.urdf_url);

		try {
			const robot = await new Promise<Object3D>((resolve, reject) => {
				const manager = new LoadingManager();

				manager.onLoad = () => {
					console.log("[RobotModelSystem] All meshes loaded");
					resolve(robot);
				};

				manager.onError = (url) => {
					console.error(
						"[RobotModelSystem] LoadingManager error for URL:",
						url,
					);
					reject(new Error(`Failed to load: ${url}`));
				};

				this.loader.manager = manager;

				let _robot: Object3D;
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

			// Rotation fix: -90deg on X (Z-up to Y-up)
			robot.rotation.x = -Math.PI / 2;

			// Scale down (URDF is in meters, scale to 0.1% for better VR viewing)
			//robot.scale.set(0.001, 0.001, 0.001);

			this.robotModel = robot;
			this.robotEntity = (this.world as World).createTransformEntity();
			this.robotEntity?.object3D.add(robot);

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
