import {
  createSystem,
  PanelDocument,
  PanelUI,
  eq,
  UIKitDocument,
} from "@iwsdk/core";

type DevicePose = {
  position: { x: number; y: number; z: number };
  orientation: { x: number; y: number; z: number; w: number };
};

export class TeleopSystem extends createSystem({
  teleopPanel: {
    required: [PanelUI, PanelDocument],
    where: [eq(PanelUI, "config", "./ui/teleop.json")],
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
  private timeOrigin = 0;
  private lastSession: XRSession | null = null;
  private inputMode = "auto";

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
        if (message.type === "config" && message.data?.input_mode) {
          this.inputMode = message.data.input_mode;
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

  update(delta: number, time: number) {
    const renderer = (this.world as any).app?.renderer;
    const frame = this.xrFrame ?? renderer?.xr?.getFrame?.();
    if (!frame) {
      return;
    }

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

    const session = (this.world as any).session ?? renderer?.xr?.getSession?.();
    const referenceSpace = renderer?.xr?.getReferenceSpace?.();
    if (!session || !referenceSpace) {
      return;
    }

    if (this.lastSession !== session) {
      this.lastSession = session;
      this.timeOrigin = Date.now() - performance.now();
    }

    const state = this.gatherXRState(frame, referenceSpace, session);
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

  gatherXRState(
    frame: XRFrame,
    referenceSpace: XRReferenceSpace,
    session: XRSession,
  ) {
    const fetchStart = performance.now();
    const timestamp_unix_ms = Math.round(
      this.timeOrigin + frame.predictedDisplayTime,
    );

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

    const viewerPose = frame.getViewerPose(referenceSpace);
    if (viewerPose) {
      const pos = viewerPose.transform.position;
      const ori = viewerPose.transform.orientation;
      devices.push({
        role: "head",
        handedness: "none",
        pose: {
          position: { x: pos.x, y: pos.y, z: pos.z },
          orientation: { x: ori.x, y: ori.y, z: ori.z, w: ori.w },
        },
      });
    }

    for (const inputSource of session.inputSources) {
      if (inputSource.hand) {
        continue;
      }

      const pose = frame.getPose(inputSource.targetRaySpace, referenceSpace);
      if (!pose) {
        continue;
      }

      const pos = pose.transform.position;
      const ori = pose.transform.orientation;
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
        handedness: inputSource.handedness,
        gripPose: {
          position: { x: pos.x, y: pos.y, z: pos.z },
          orientation: { x: ori.x, y: ori.y, z: ori.z, w: ori.w },
        },
      };

      if (inputSource.gamepad) {
        device.gamepad = {
          buttons: Array.from(inputSource.gamepad.buttons).map((button) => ({
            pressed: button.pressed,
            touched: button.touched,
            value: button.value,
          })),
          axes: Array.from(inputSource.gamepad.axes),
        };
      }

      devices.push(device);
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
