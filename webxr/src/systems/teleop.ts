import type { System } from 'aframe';
import { Vector3, Quaternion, Object3D } from 'three';

type DevicePose = {
  position: { x: number; y: number; z: number };
  orientation: { x: number; y: number; z: number; w: number };
};

interface TeleopSystemDef extends System {
  ws: WebSocket | null;
  inputMode: string;
  frameCount: number;
  lastFpsTime: number;
  currentFps: number;
  lastSendTime: number;
  menuButtonState: boolean;
  tempPosition: Vector3;
  tempQuaternion: Quaternion;

  connectWS(): void;
  updateStatus(text: string, connected: boolean): void;
  poseFromObject(object: Object3D): DevicePose | null;
  buildControllerDevice(
    handedness: "left" | "right",
    object3D: Object3D,
    gamepad: Gamepad | undefined,
    isHandPrimary: boolean
  ): any;
  gatherInputState(time: number): any;
}

export const TeleopSystemDefinition = {
  schema: {},

  init: function(this: TeleopSystemDef) {
    this.ws = null;
    this.inputMode = 'auto';
    this.frameCount = 0;
    this.lastFpsTime = 0;
    this.currentFps = 0;
    this.lastSendTime = 0;
    this.menuButtonState = false;
    this.tempPosition = new Vector3();
    this.tempQuaternion = new Quaternion();

    this.connectWS();
  },

  connectWS: function(this: TeleopSystemDef) {
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
          this.el.emit('teleop-config', message.data);
        } else if (message.type === "robot_config") {
          this.el.emit('robot-config', message.data);
        } else if (message.type === "robot_state") {
          this.el.emit('robot-state', message.data);
        }
      } catch (error) {
        console.warn("Failed to parse WS message", error);
      }
    };
  },

  updateStatus: function(this: TeleopSystemDef, text: string, connected: boolean) {
    this.el.emit('teleop-status', { text, connected });
  },

  poseFromObject: function(this: TeleopSystemDef, object: Object3D): DevicePose | null {
    if (!object) {
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
  },

  buildControllerDevice: function(
    this: TeleopSystemDef,
    handedness: "left" | "right",
    object3D: Object3D,
    gamepad: Gamepad | undefined,
    isHandPrimary: boolean
  ) {
    if (isHandPrimary) {
      return null;
    }

    const pose = this.poseFromObject(object3D);
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

    if (gamepad) {
      device.gamepad = {
        buttons: Array.from(gamepad.buttons).map((button: any) => ({
          pressed: button.pressed,
          touched: button.touched,
          value: button.value,
        })),
        axes: Array.from(gamepad.axes),
      };
    }

    return device;
  },

  tick: function(this: TeleopSystemDef, time: number, delta: number) {
    const timeSec = time / 1000;

    if (this.lastFpsTime === 0) {
      this.lastFpsTime = timeSec;
    }

    this.frameCount += 1;
    if (timeSec - this.lastFpsTime >= 1.0) {
      this.currentFps = Math.round(
        this.frameCount / (timeSec - this.lastFpsTime),
      );
      this.frameCount = 0;
      this.lastFpsTime = timeSec;
    }

    if (timeSec - this.lastSendTime <= 0.01) {
      return;
    }
    this.lastSendTime = timeSec;

    const state = this.gatherInputState(time);
    if (!state || state.devices.length === 0) {
      return;
    }

    this.el.emit('teleop-stats', {
        fps: this.currentFps,
        latency: state.fetch_latency_ms
    });

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(
        JSON.stringify({
          type: "xr_state",
          data: state,
        }),
      );
    }
  },

  gatherInputState: function(this: TeleopSystemDef, timeMs: number) {
    const fetchStart = performance.now();
    const timestamp_unix_ms = Date.now();
    const devices: Array<any> = [];

    const camera = this.el.sceneEl?.camera;
    if (camera) {
       const headPose = this.poseFromObject(camera);
       if (headPose) {
         devices.push({
           role: "head",
           handedness: "none",
           pose: headPose,
         });
       }
    }

    const trackedControls = this.el.sceneEl?.querySelectorAll('[tracked-controls]');

    if (trackedControls) {
      trackedControls.forEach((el: any) => {
        const component = el.components['tracked-controls'];
        if (!component || !component.controller) return;

        const gamepad = component.controller as Gamepad;
        const handedness = gamepad.hand as "left" | "right";

        const isHandPrimary = false;

        if (handedness === 'left' && gamepad.buttons && gamepad.buttons.length > 0) {
           const menuButton = gamepad.buttons[gamepad.buttons.length - 1];
           if (menuButton) {
             if (menuButton.pressed) {
               if (!this.menuButtonState) {
                 this.menuButtonState = true;
                 this.el.emit('teleop-menu-toggle');
               }
             } else {
               this.menuButtonState = false;
             }
           }
        }

        const device = this.buildControllerDevice(
          handedness,
          el.object3D,
          gamepad,
          isHandPrimary
        );

        if (device) {
          devices.push(device);
        }
      });
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
};

AFRAME.registerSystem('teleop', TeleopSystemDefinition);
