import "aframe";
import URDFLoader from "urdf-loader";
import { LoadingManager, Object3D } from "three";

interface RobotConfigDetail {
  urdf_url: string;
}

interface RobotStateDetail {
  joints: Record<string, number>;
}

AFRAME.registerComponent("robot-model", {
  schema: {},

  init: function () {
    const self = this as any;
    self.loader = new URDFLoader();
    self.loader.packages = (pkg: string) => `/robot_assets/${pkg}`;

    self.robotModel = null;

    self.onRobotConfig = this.onRobotConfig.bind(this);
    self.onRobotState = this.onRobotState.bind(this);

    window.addEventListener("robot-config", self.onRobotConfig);
    window.addEventListener("robot-state", self.onRobotState);

    const placeholderEntity = document.createElement("a-box");
    placeholderEntity.setAttribute("color", "blue");
    placeholderEntity.setAttribute("width", "0.5");
    placeholderEntity.setAttribute("height", "0.5");
    placeholderEntity.setAttribute("depth", "0.5");
    placeholderEntity.setAttribute("position", "0 0.25 0");
    this.el.appendChild(placeholderEntity);
    self.placeholderEntity = placeholderEntity;

    console.log("[RobotModelComponent] Initialized with blue placeholder box");
  },

  remove: function () {
    const self = this as any;
    window.removeEventListener("robot-config", self.onRobotConfig);
    window.removeEventListener("robot-state", self.onRobotState);

    if (self.robotModel) {
      this.el.object3D.remove(self.robotModel);
    }
  },

  onRobotConfig: function (event: Event) {
    const self = this as any;
    const detail = (event as CustomEvent<RobotConfigDetail>).detail;
    if (!detail || !detail.urdf_url) {
      console.warn("[RobotModelComponent] Received robot-config without urdf_url");
      return;
    }

    const urdfUrl = detail.urdf_url;
    console.log("[RobotModelComponent] Loading robot from", urdfUrl);

    const manager = new LoadingManager();
    self.loader.manager = manager;

    manager.onLoad = () => {
      console.log("[RobotModelComponent] All meshes loaded");
    };

    manager.onError = (url) => {
      console.error("[RobotModelComponent] LoadingManager error for URL:", url);
    };

      self.loader.load(
      urdfUrl,
      (result: Object3D) => {
        const robot = result;
        console.log("[RobotModelComponent] URDF parsed, waiting for meshes...");

        if (self.robotModel) {
          this.el.object3D.remove(self.robotModel);
        }

        if (self.placeholderEntity && self.placeholderEntity.parentNode) {
          self.placeholderEntity.parentNode.removeChild(self.placeholderEntity);
          self.placeholderEntity = null;
        }

        // Rotation fix: -90deg on X (Z-up to Y-up)
        robot.rotation.x = -Math.PI / 2;

        self.robotModel = robot;
        this.el.object3D.add(robot);
        console.log("[RobotModelComponent] Robot model loaded and added to scene");
      },
      (progress: ProgressEvent) => {
        if (progress?.total) {
        }
      },
      (err: ErrorEvent) => {
        console.error("[RobotModelComponent] Loader error details:", err);
      }
    );
  },

  onRobotState: function (event: Event) {
    const self = this as any;
    if (!self.robotModel) return;

    const detail = (event as CustomEvent<RobotStateDetail>).detail;
    if (!detail || !detail.joints) return;

    const robot = self.robotModel;
    if (robot.joints) {
      for (const [name, value] of Object.entries(detail.joints)) {
        if (robot.joints[name] && typeof robot.joints[name].setJointValue === "function") {
          robot.joints[name].setJointValue(value);
        }
      }
    }
  }
});
