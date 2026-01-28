import { System, World, XRInputSource } from "@iwsdk/core";

export class TeleopSystem extends System {
  private ws: WebSocket | null = null;
  private statusText: any = null;
  private fpsText: any = null;
  private latencyText: any = null;
  private lastTime = 0;
  private frameCount = 0;
  private lastFpsTime = 0;

  constructor(world: World) {
    super(world);
  }

  init() {
    this.connectWS();
    // Query UI elements
    const ui = this.world.entityManager.getComponents("PanelUI")[0]; // Assuming one panel
    if (ui && ui.root) {
      this.statusText = ui.root.querySelector("#status-text");
      this.fpsText = ui.root.querySelector("#fps-text");
      this.latencyText = ui.root.querySelector("#latency-text");
    }
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
      // Simple echo for latency check usually, or server sends stats
    };
  }

  updateStatus(text: string, connected: boolean) {
    if (this.statusText) {
      this.statusText.text = text;
      // Note: uikitml might not support classList toggle easily via JS yet,
      // so we might just set color directly or text.
      // Assuming direct text update works.
    }
  }

  execute(delta: number, time: number) {
    // 1. Calculate FPS
    this.frameCount++;
    if (time - this.lastFpsTime >= 1.0) {
      const fps = Math.round(this.frameCount / (time - this.lastFpsTime));
      if (this.fpsText) this.fpsText.text = `${fps}`;
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
    const inputs = this.world.input.sources; // Abstracted input access
    // This part requires checking iwsdk input API.
    // Fallback to standard WebXR if needed, but iwsdk abstracts it.
    // For now, assume we can access raw XRFrame or sources via world.app.xr
    
    // Simplification: Send head pose
    const head = this.world.camera.transform; // Main camera is head
    
    return {
      timestamp: time,
      devices: [
        {
          role: "head",
          pose: {
            position: { x: head.position.x, y: head.position.y, z: head.position.z },
            orientation: { x: head.rotation.x, y: head.rotation.y, z: head.rotation.z, w: head.rotation.w }
          }
        }
        // TODO: Add controllers iterating inputs
      ]
    };
  }
}
