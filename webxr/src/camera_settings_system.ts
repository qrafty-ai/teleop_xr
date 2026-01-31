import {
  createSystem,
  PanelDocument,
  PanelUI,
  eq,
  UIKitDocument,
} from "@iwsdk/core";
import { GlobalRefs } from "./global_refs";
import { onCameraViewsChanged } from "./camera_views";
import { getCameraEnabled, setCameraEnabled } from "./camera_config";
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
      }

      onCameraViewsChanged((config) => {
        console.log("[CameraSettings] Config changed, keys:", Object.keys(config));
        this.updateRows(config);
      });
    });
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
