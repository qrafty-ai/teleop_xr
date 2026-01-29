import {
  createSystem,
  PanelDocument,
  PanelUI,
  eq,
  UIKitDocument,
} from "@iwsdk/core";
import { Quaternion, Vector3 } from "@iwsdk/core";
import { Robot } from "./robot";
import { GlobalRefs } from "./global_refs";
import { setCameraViewsConfig } from "./camera_views";

type DevicePose = {
  position: { x: number; y: number; z: number };
  orientation: { x: number; y: number; z: number; w: number };
};

export class TeleopSystem extends createSystem({
  teleopPanel: {
    required: [PanelUI, PanelDocument],
    where: [eq(PanelUI, "config", "./ui/teleop.json")],
  },
  robot: {
    required: [Robot],
  },
}) {
  private ws: WebSocket | null = null;
  private statusText: any = null;
  private fpsText: any = null;
  private latencyText: any = null;
  private frameCount = 0;
  private lastFpsTime = 0;
  private currentFps = 0;
  private lastSendTime = 0;
  private inputMode = "auto";
  private tempPosition = new Vector3();
  private tempQuaternion = new Quaternion();

  init() {
    this.connectWS();

    this.queries.teleopPanel.subscribe("qualify", (entity) => {
      const document = PanelDocument.data.document[entity.index] as UIKitDocument;
      if (!document) {
        return;
      }

      this.statusText = document.getElementById("status-text");
      this.fpsText = document.getElementById("fps-text");
      this.latencyText = document.getElementById("latency-text");

      const cameraButton = document.getElementById("camera-button");
      if (cameraButton) {
        cameraButton.addEventListener("click", () => {
          // Use GlobalRefs - populated by index.ts at creation time
          // DO NOT access ECS queries during click events (causes freeze)
          if (GlobalRefs.cameraPanelRoot) {
            GlobalRefs.cameraPanelRoot.visible = !GlobalRefs.cameraPanelRoot.visible;
          }
        });
      }

      const isConnected = this.ws && this.ws.readyState === WebSocket.OPEN;
      this.updateStatus(isConnected ? "Connected" : "Disconnected", !!isConnected);
    });
  }

  connectWS() {
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
          setCameraViewsConfig(message.data?.camera_views ?? null);
        }
      } catch (error) {
        console.warn("Failed to parse WS message", error);
      }
    };
  }

  updateStatus(text: string, connected: boolean) {
    if (!this.statusText) {
      return;
    }

    this.statusText.setProperties({
      text,
      className: connected ? "status-value connected" : "status-value",
    });
  }

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
    raySpace: any,
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

  update(delta: number, time: number) {
    if (this.lastFpsTime === 0) {
      this.lastFpsTime = time;
    }

    this.frameCount += 1;
    if (time - this.lastFpsTime >= 1.0) {
      this.currentFps = Math.round(
        this.frameCount / (time - this.lastFpsTime),
      );
      this.frameCount = 0;
      this.lastFpsTime = time;
    }

    if (time - this.lastSendTime <= 0.01) {
      return;
    }
    this.lastSendTime = time;
    const input = (this as any).input ?? this.world.input;

    const state = this.gatherInputState(input);
    if (!state || state.devices.length === 0) {
      return;
    }

    const head = state.devices.find((device) => device.role === "head");
    this.updateLocalStats(head?.pose ?? null, this.currentFps, state.fetch_latency_ms);

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(
        JSON.stringify({
          type: "xr_state",
          data: state,
        }),
      );
    }
  }

  updateLocalStats(pose: DevicePose | null, fps: number, latency: number) {
    if (this.fpsText) {
      this.fpsText.setProperties({ text: `${fps}` });
    }

    if (this.latencyText) {
      const latencyValue = Number.isFinite(latency) ? latency : 0;
      this.latencyText.setProperties({ text: `${latencyValue.toFixed(1)}ms` });
    }
  }

  gatherInputState(input: any) {
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
    };
  }
}
