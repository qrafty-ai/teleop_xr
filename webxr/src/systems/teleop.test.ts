import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Vector3, Quaternion, Object3D } from 'three';

const registerSystem = vi.fn();
global.AFRAME = {
  registerSystem,
} as any;

const mockWebSocket = {
  send: vi.fn(),
  close: vi.fn(),
  readyState: 1,
  onopen: null,
  onclose: null,
  onerror: null,
  onmessage: null,
};

const MockWebSocketConstructor = vi.fn();
class MockWebSocket {
  static OPEN = 1;
  static CONNECTING = 0;
  static CLOSING = 2;
  static CLOSED = 3;

  constructor(url: string) {
    MockWebSocketConstructor(url);
    return mockWebSocket;
  }
}
global.WebSocket = MockWebSocket as any;

global.window = {
  location: {
    protocol: 'http:',
    host: 'localhost:8080',
  }
} as any;

describe('TeleopSystem', () => {
  let TeleopSystemDefinition: any;
  let system: any;
  let sceneEl: any;

  beforeEach(async () => {
    vi.clearAllMocks();
    mockWebSocket.readyState = 1;

    const module = await import('./teleop');
    TeleopSystemDefinition = module.TeleopSystemDefinition;

    sceneEl = {
      camera: new Object3D(),
      querySelectorAll: vi.fn().mockReturnValue([]),
      emit: vi.fn(),
    };

    system = { ...TeleopSystemDefinition };
    system.el = {
      sceneEl,
      emit: vi.fn(),
    };

    system.init = system.init.bind(system);
    system.connectWS = system.connectWS.bind(system);
    system.tick = system.tick.bind(system);
    system.gatherInputState = system.gatherInputState.bind(system);
    system.buildControllerDevice = system.buildControllerDevice.bind(system);
    system.poseFromObject = system.poseFromObject.bind(system);
    system.updateStatus = system.updateStatus.bind(system);

    system.ws = null;
    system.inputMode = 'auto';
    system.frameCount = 0;
    system.lastFpsTime = 0;
    system.currentFps = 0;
    system.lastSendTime = 0;
    system.menuButtonState = false;
    system.tempPosition = new Vector3();
    system.tempQuaternion = new Quaternion();
  });

  it('should register the system', () => {
    expect(registerSystem).toHaveBeenCalledWith('teleop', expect.anything());
  });

  it('should connect to WebSocket on init', () => {
    system.init();
    expect(MockWebSocketConstructor).toHaveBeenCalledWith('ws://localhost:8080/ws');
    expect(system.ws).toBe(mockWebSocket);
  });

  it('should handle incoming config messages', () => {
    system.init();
    const onMessage = mockWebSocket.onmessage as any;

    onMessage({
      data: JSON.stringify({
        type: 'config',
        data: { input_mode: 'hand', camera_views: {} }
      })
    });

    expect(system.inputMode).toBe('hand');
    expect(system.el.emit).toHaveBeenCalledWith('teleop-config', expect.objectContaining({ input_mode: 'hand' }));
  });

  it('should gather input and send xr_state in tick', () => {
    system.init();

    sceneEl.camera.position.set(1, 2, 3);
    sceneEl.camera.quaternion.set(0, 0, 0, 1);
    sceneEl.camera.getWorldPosition = (v: Vector3) => v.copy(sceneEl.camera.position);
    sceneEl.camera.getWorldQuaternion = (q: Quaternion) => q.copy(sceneEl.camera.quaternion);

    system.tick(1000, 16);

    expect(mockWebSocket.send).toHaveBeenCalled();
    const sentData = JSON.parse(mockWebSocket.send.mock.calls[0][0]);
    expect(sentData.type).toBe('xr_state');
    expect(sentData.data.devices).toHaveLength(1);
    expect(sentData.data.devices[0].role).toBe('head');
    expect(sentData.data.devices[0].pose.position.x).toBe(1);
  });

  it('should handle controller input', () => {
    system.init();

    const controllerEl = {
      object3D: new Object3D(),
      components: {
        'tracked-controls': {
          controller: {
            hand: 'left',
            buttons: [],
            axes: [0, 0, 0, 0]
          }
        }
      }
    };
    controllerEl.object3D.position.set(-0.5, 1, -0.5);
    controllerEl.object3D.getWorldPosition = (v: Vector3) => v.copy(controllerEl.object3D.position);
    controllerEl.object3D.getWorldQuaternion = (q: Quaternion) => q.copy(controllerEl.object3D.quaternion);

    sceneEl.querySelectorAll.mockReturnValue([controllerEl]);

    system.tick(1000, 16);

    expect(mockWebSocket.send).toHaveBeenCalled();
    const sentData = JSON.parse(mockWebSocket.send.mock.calls[0][0]);
    const leftController = sentData.data.devices.find((d: any) => d.role === 'controller' && d.handedness === 'left');
    expect(leftController).toBeDefined();
    expect(leftController.gripPose.position.x).toBe(-0.5);
  });
});
