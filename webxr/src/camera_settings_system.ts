import {
  createSystem,
  PanelDocument,
  PanelUI,
  eq,
  UIKitDocument,
} from "@iwsdk/core";
import { GlobalRefs } from "./global_refs";
import { onCameraViewsChanged } from "./camera_views";
import { getCameraEnabled, setCameraEnabled, getCameraSettings, setCameraSettings, CameraSettings } from "./camera_config";
import { CameraViewKey } from "./track_routing";

const MAX_ROWS = 6;

export class CameraSettingsSystem extends createSystem({
  cameraSettingsPanel: {
    required: [PanelUI, PanelDocument],
    where: [eq(PanelUI, "config", "./ui/camera_settings.json")],
  },
}) {
  private initialized = false;
  private keyToRowIndex = new Map<string, number>();
  private rows: any[] = [];
  private labels: any[] = [];
  private buttons: any[] = [];

  // New input references
  private targetInput: any = null;
  private widthInput: any = null;
  private heightInput: any = null;
  private fpsInput: any = null;
  private deviceIdInput: any = null;

  init() {
    this.queries.cameraSettingsPanel.subscribe("qualify", (entity) => {
      const document = PanelDocument.data.document[entity.index] as UIKitDocument;
      console.log("[CameraSettings] Panel qualified, document:", !!document);
      if (!document || this.initialized) {
        return;
      }
      this.initialized = true;

      const closeBtn = document.getElementById("close-btn");
      if (closeBtn) {
        closeBtn.addEventListener("click", () => {
          const panel = GlobalRefs.cameraSettingsPanel;
          if (panel && panel.entity.object3D) {
            panel.entity.object3D.visible = false;
          }
        });
      }

      // Bind new inputs
      this.targetInput = document.getElementById("input-target-cam");
      this.widthInput = document.getElementById("input-width");
      this.heightInput = document.getElementById("input-height");
      this.fpsInput = document.getElementById("input-fps");
      this.deviceIdInput = document.getElementById("input-device-id");

      const updateBtn = document.getElementById("update-btn");
      if (updateBtn) {
        updateBtn.addEventListener("click", () => {
          this.handleUpdateConfig();
        });
      }

      // Pre-fetch all element references
      for (let i = 0; i < MAX_ROWS; i++) {
        const row = document.getElementById(`row-${i}`);
        const label = document.getElementById(`label-${i}`);
        const btn = document.getElementById(`btn-${i}`);

        this.rows.push(row);
        this.labels.push(label);
        this.buttons.push(btn);

        if (btn) {
          btn.addEventListener("click", () => {
            this.handleToggleClick(i);
          });
        }

        // Click on label to select camera for editing
        if (label) {
          label.addEventListener("click", () => {
            this.handleSelectCamera(i);
          });
        }
      }

      onCameraViewsChanged((config) => {
        console.log("[CameraSettings] Config changed, keys:", Object.keys(config));
        this.updateRows(config);
      });
    });
  }

  private handleSelectCamera(rowIndex: number) {
    let targetKey: string | null = null;
    for (const [key, idx] of this.keyToRowIndex.entries()) {
      if (idx === rowIndex) {
        targetKey = key;
        break;
      }
    }

    if (!targetKey) return;

    console.log("[CameraSettings] Selected camera:", targetKey);
    const settings = getCameraSettings(targetKey as CameraViewKey);

    if (this.targetInput) this.targetInput.setProperties({ value: targetKey });
    if (this.widthInput) this.widthInput.setProperties({ value: settings.width.toString() });
    if (this.heightInput) this.heightInput.setProperties({ value: settings.height.toString() });
    if (this.fpsInput) this.fpsInput.setProperties({ value: settings.fps.toString() });
    if (this.deviceIdInput) this.deviceIdInput.setProperties({ value: settings.deviceId });
  }

  private handleUpdateConfig() {
    if (!this.targetInput) return;

    const key = this.targetInput.value ?? this.targetInput.properties?.value;
    if (!key) return;

    const widthStr = this.widthInput?.value ?? this.widthInput?.properties?.value ?? "1280";
    const heightStr = this.heightInput?.value ?? this.heightInput?.properties?.value ?? "720";
    const fpsStr = this.fpsInput?.value ?? this.fpsInput?.properties?.value ?? "30";
    const deviceId = this.deviceIdInput?.value ?? this.deviceIdInput?.properties?.value ?? "";

    const newSettings: Partial<CameraSettings> = {
      width: parseInt(widthStr, 10),
      height: parseInt(heightStr, 10),
      fps: parseInt(fpsStr, 10),
      deviceId: deviceId
    };

    console.log("[CameraSettings] Updating config for", key, newSettings);
    setCameraSettings(key as CameraViewKey, newSettings);
  }

  private handleToggleClick(rowIndex: number) {
    let targetKey: string | null = null;
    for (const [key, idx] of this.keyToRowIndex.entries()) {
      if (idx === rowIndex) {
        targetKey = key;
        break;
      }
    }

    if (!targetKey) return;

    const newState = !getCameraEnabled(targetKey as CameraViewKey);
    setCameraEnabled(targetKey as CameraViewKey, newState);

    const btn = this.buttons[rowIndex];
    if (btn) {
      btn.setProperties({
        text: newState ? "ON" : "OFF",
      });
      if (btn.classList) {
        if (newState) {
          btn.classList.add("active");
        } else {
          btn.classList.remove("active");
        }
      }
    }
  }

  private updateRows(config: Record<string, any>) {
    const keys = Object.keys(config).sort();
    console.log("[CameraSettings] updateRows, keys:", keys);
    this.keyToRowIndex.clear();

    for (let i = 0; i < MAX_ROWS; i++) {
      const row = this.rows[i];
      const label = this.labels[i];
      const btn = this.buttons[i];

      if (i < keys.length) {
        const key = keys[i];
        this.keyToRowIndex.set(key, i);
        const enabled = getCameraEnabled(key as CameraViewKey);

        if (row) {
          // Use display property instead of className
          row.setProperties({ display: "flex" });
          if (row.classList) {
            row.classList.remove("hidden");
            row.classList.add("toggle-row");
          }
        }
        if (label) {
          label.setProperties({ text: key.toUpperCase() });
        }
        if (btn) {
          btn.setProperties({
            text: enabled ? "ON" : "OFF",
          });
          if (btn.classList) {
            if (enabled) {
              btn.classList.add("active");
            } else {
              btn.classList.remove("active");
            }
          }
        }
      } else {
        if (row) {
          row.setProperties({ display: "none" });
          if (row.classList) {
            row.classList.add("hidden");
          }
        }
      }
    }
  }
}
