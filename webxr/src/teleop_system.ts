import { createSystem, PanelUI, PanelDocument, eq, UIKitDocument, UIKit } from "@iwsdk/core";

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
  private timeOrigin = Date.now() - performance.now();

  init() {
    this.connectWS();

    this.queries.teleopPanel.subscribe("qualify", (entity) => {
      const document = PanelDocument.data.document[entity.index] as UIKitDocument;
      if (document) {
        this.statusText = document.getElementById("status-text");
        this.fpsText = document.getElementById("fps-text");
        this.latencyText = document.getElementById("latency-text");

        // Sync initial state
        const isConnected = this.ws && this.ws.readyState === WebSocket.OPEN;
        this.updateStatus(isConnected ? "Connected" : "Disconnected", !!isConnected);
      }
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
      setTimeout(() => this.connectWS(), 1000); // Reconnect
    };

    this.ws.onerror = (e) => {
      console.error("WS Error", e);
    };

    this.ws.onmessage = (msg) => {
      // Handle config/latency logic if needed
    };
  }

  updateStatus(text: string, connected: boolean) {
    if (this.statusText) {
      this.statusText.setProperties({
        text: text,
        className: connected ? "status-value connected" : "status-value"
      });
    }
  }

  execute(delta: number, time: number) {
    // 1. Calculate FPS
    this.frameCount++;
    if (time - this.lastFpsTime >= 1.0) {
      this.currentFps = Math.round(this.frameCount / (time - this.lastFpsTime));
      this.frameCount = 0;
      this.lastFpsTime = time;
    }

    // 2. Send XR State (Rate limit: 10ms)
    if (time - this.lastSendTime > 0.01) {
      this.lastSendTime = time;
      const state = this.gatherXRState(time);
      if (state) {
        // 3. Update local stats
        const head = state.devices.find((d: any) => d.role === "head");
        this.updateLocalStats(head?.pose, this.currentFps, state.fetch_latency_ms);

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
          this.ws.send(JSON.stringify({
            type: "xr_state",
            data: state,
          }));
        }
      }
    }
  }

  updateLocalStats(pose: any, fps: number, latency: number) {
    if (this.fpsText) {
      this.fpsText.setProperties({ text: `${fps}` });
    }
    if (this.latencyText) {
      this.latencyText.setProperties({ text: `${latency.toFixed(1)}ms` });
    }
  }

  gatherXRState(time: number) {
    const fetchStart = performance.now();
    const renderer = (this.world as any).app.renderer;
    const session = renderer.xr.getSession();
    if (!session) return null;

    const frame = renderer.xr.getFrame();
    const referenceSpace = renderer.xr.getReferenceSpace();
    if (!frame || !referenceSpace) return null;

    const timestamp_unix_ms = Math.round(this.timeOrigin + frame.predictedDisplayTime);

    const devices: any[] = [];

    // Head pose
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

    // Controllers
    for (const inputSource of session.inputSources) {
      if (inputSource.hand) continue; // Skip hand tracking

      const pose = frame.getPose(inputSource.targetRaySpace, referenceSpace);
      if (!pose) continue;

      const pos = pose.transform.position;
      const ori = pose.transform.orientation;
      const device: any = {
        role: "controller",
        handedness: inputSource.handedness,
        gripPose: {
          position: { x: pos.x, y: pos.y, z: pos.z },
          orientation: { x: ori.x, y: ori.y, z: ori.z, w: ori.w },
        },
      };

      // Full gamepad input
      if (inputSource.gamepad) {
        device.gamepad = {
          buttons: Array.from(inputSource.gamepad.buttons).map((b: any) => ({
            pressed: b.pressed,
            touched: b.touched,
            value: b.value,
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
      fetch_latency_ms,
    };
  }
}
