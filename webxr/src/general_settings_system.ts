import {
  createSystem,
  PanelDocument,
  PanelUI,
  eq,
  UIKitDocument,
} from "@iwsdk/core";
import { GlobalRefs } from "./global_refs";
import { getGeneralConfig, setGeneralConfig, onGeneralConfigChanged, GeneralConfig } from "./general_config";

export class GeneralSettingsSystem extends createSystem({
  generalSettingsPanel: {
    required: [PanelUI, PanelDocument],
    where: [eq(PanelUI, "config", "./ui/general_settings.json")],
  },
}) {
  private initialized = false;
  private inputModeInput: any = null;
  private posXInput: any = null;
  private posYInput: any = null;
  private posZInput: any = null;
  private oriXInput: any = null;
  private oriYInput: any = null;
  private oriZInput: any = null;

  init() {
    this.queries.generalSettingsPanel.subscribe("qualify", (entity) => {
      const document = PanelDocument.data.document[entity.index] as UIKitDocument;
      console.log("[GeneralSettings] Panel qualified");

      if (!document || this.initialized) {
        return;
      }
      this.initialized = true;

      // Register in GlobalRefs so TeleopSystem can toggle it
      GlobalRefs.generalSettingsPanel = { entity };

      this.inputModeInput = document.getElementById("input-mode");
      this.posXInput = document.getElementById("pos-x");
      this.posYInput = document.getElementById("pos-y");
      this.posZInput = document.getElementById("pos-z");
      this.oriXInput = document.getElementById("ori-x");
      this.oriYInput = document.getElementById("ori-y");
      this.oriZInput = document.getElementById("ori-z");

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

      this.updateInputs(getGeneralConfig());

      onGeneralConfigChanged((config) => {
        this.updateInputs(config);
      });
    });
  }

  private handleSave() {
    const config: GeneralConfig = {
      input_mode: this.inputModeInput?.value ?? this.inputModeInput?.properties?.value ?? "controller",
      natural_phone_position: [
        parseFloat(this.posXInput?.value ?? this.posXInput?.properties?.value ?? "0"),
        parseFloat(this.posYInput?.value ?? this.posYInput?.properties?.value ?? "0"),
        parseFloat(this.posZInput?.value ?? this.posZInput?.properties?.value ?? "0"),
      ],
      natural_phone_orientation_euler: [
        parseFloat(this.oriXInput?.value ?? this.oriXInput?.properties?.value ?? "0"),
        parseFloat(this.oriYInput?.value ?? this.oriYInput?.properties?.value ?? "0"),
        parseFloat(this.oriZInput?.value ?? this.oriZInput?.properties?.value ?? "0"),
      ],
    };

    console.log("[GeneralSettings] Saving config:", config);
    setGeneralConfig(config);
  }

  private updateInputs(config: GeneralConfig) {
    if (this.inputModeInput) {
      this.inputModeInput.setProperties({ value: config.input_mode });
    }
    if (this.posXInput) {
      this.posXInput.setProperties({ value: config.natural_phone_position[0].toString() });
    }
    if (this.posYInput) {
      this.posYInput.setProperties({ value: config.natural_phone_position[1].toString() });
    }
    if (this.posZInput) {
      this.posZInput.setProperties({ value: config.natural_phone_position[2].toString() });
    }
    if (this.oriXInput) {
      this.oriXInput.setProperties({ value: config.natural_phone_orientation_euler[0].toString() });
    }
    if (this.oriYInput) {
      this.oriYInput.setProperties({ value: config.natural_phone_orientation_euler[1].toString() });
    }
    if (this.oriZInput) {
      this.oriZInput.setProperties({ value: config.natural_phone_orientation_euler[2].toString() });
    }
  }
}
