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

  init() {
    this.connectWS();

    this.queries.teleopPanel.subscribe("qualify", (entity) => {
      const document = PanelDocument.data.document[entity.index] as UIKitDocument;
      if (document) {
        this.statusText = document.getElementById("status-text");
        this.fpsText = document.getElementById("fps-text");
        this.latencyText = document.getElementById("latency-text");
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
      this.statusText.setProperties({ text: text });
      // Change color based on connection?
      // this.statusText.style.color = connected ? "#22c55e" : "#ef4444"; 
      // uikitml might support classes, but setProperties is safer for text content.
      // For color, we might need to access style or classes if exposed.
      // For now just text.
    }
  }

  execute(delta: number, time: number) {
    // 1. Calculate FPS
    this.frameCount++;
    if (time - this.lastFpsTime >= 1.0) {
      const fps = Math.round(this.frameCount / (time - this.lastFpsTime));
      if (this.fpsText) this.fpsText.setProperties({ text: `${fps}` });
      this.frameCount = 0;
      this.lastFpsTime = time;
    }

    // 2. Send XR State
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const state = this.gatherXRState(time);
      this.ws.send(JSON.stringify(state));
    }
  }

  gatherXRState(time: number) {
    const head = this.world.camera.transform; // Main camera is head

    return {
      timestamp: time,
      devices: [
        {
          role: "head",
          pose: {
            position: { x: head.position.x, y: head.position.y, z: head.position.z },
            orientation: {
              x: head.rotation.x,
              y: head.rotation.y,
              z: head.rotation.z,
              w: head.rotation.w,
            },
          },
        },
        // TODO: Add controllers iterating inputs
      ],
    };
  }
}
