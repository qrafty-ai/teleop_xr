import {
  createSystem,
  PanelDocument,
  PanelUI,
  eq,
  UIKitDocument,
} from "@iwsdk/core";
import { GlobalRefs } from "./global_refs";
import { getRobotConfig, setRobotConfig, onRobotConfigChanged } from "./robot_config";

export class RobotSettingsSystem extends createSystem({
  robotSettingsPanel: {
    required: [PanelUI, PanelDocument],
    where: [eq(PanelUI, "config", "./ui/robot_settings.json")],
  },
}) {
  private initialized = false;
  private urdfInput: any = null;
  private meshInput: any = null;

  init() {
    this.queries.robotSettingsPanel.subscribe("qualify", (entity) => {
      const document = PanelDocument.data.document[entity.index] as UIKitDocument;
      console.log("[RobotSettings] Panel qualified");

      if (!document || this.initialized) {
        return;
      }
      this.initialized = true;

      // Register in GlobalRefs so TeleopSystem can toggle it
      GlobalRefs.robotSettingsPanel = { entity };

      this.urdfInput = document.getElementById("input-urdf");
      this.meshInput = document.getElementById("input-mesh");
      const saveBtn = document.getElementById("save-btn");
      const closeBtn = document.getElementById("close-btn");

      if (closeBtn) {
        closeBtn.addEventListener("click", () => {
          if (entity.object3D) {
            entity.object3D.visible = false;
          }
        });
      }

      if (saveBtn) {
        saveBtn.addEventListener("click", () => {
          this.handleSave();
        });
      }

      this.updateInputs(getRobotConfig());

      onRobotConfigChanged((config) => {
        this.updateInputs(config);
      });
    });
  }

  private handleSave() {
    if (!this.urdfInput || !this.meshInput) return;

    const urdfPath = this.urdfInput.value ?? this.urdfInput.properties?.value ?? "robot.urdf";
    const meshPath = this.meshInput.value ?? this.meshInput.properties?.value ?? "";

    console.log("[RobotSettings] Saving config:", { urdfPath, meshPath });
    setRobotConfig({ urdfPath, meshPath });
  }

  private updateInputs(config: { urdfPath: string; meshPath: string }) {
    if (this.urdfInput) {
      this.urdfInput.setProperties({ value: config.urdfPath });
    }
    if (this.meshInput) {
      this.meshInput.setProperties({ value: config.meshPath });
    }
  }
}
