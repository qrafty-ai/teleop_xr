import {
  AssetManifest,
  AssetType,
  Mesh,
  MeshBasicMaterial,
  PlaneGeometry,
  SessionMode,
  SRGBColorSpace,
  AssetManager,
  World,
} from "@iwsdk/core";

import {
  AudioSource,
  DistanceGrabbable,
  MovementMode,
  Interactable,
  PanelUI,
  PlaybackMode,
} from "@iwsdk/core";

import { EnvironmentType, LocomotionEnvironment } from "@iwsdk/core";

import { PanelSystem } from "./panel.js";

import { TeleopSystem } from "./teleop_system.js";

import { VideoClient } from "./video.js";

import { DraggablePanel, CameraPanel, ControllerCameraPanel } from "./panels.js";

import { ControllerCameraPanelSystem } from "./controller_camera_system.js";

import { GlobalRefs } from "./global_refs.js";

import { initConsoleStream } from "./console_stream.js";
import { onCameraViewsChanged, isViewEnabled, getCameraViewsConfig } from "./camera_views.js";
import { resolveTrackView, type CameraViewKey } from "./track_routing.js";

// Initialize console streaming for Quest VR debugging
initConsoleStream();

const disableHeadCameraPanel = true;

const assets: AssetManifest = {
  chimeSound: {
    url: "./audio/chime.mp3",
    type: AssetType.Audio,
    priority: "background",
  },
};

World.create(document.getElementById("scene-container") as HTMLDivElement, {
  assets,
  xr: {
    sessionMode: SessionMode.ImmersiveAR,
    offer: "always",
    // Optional structured features; layers/local-floor are offered by default
    features: {
      handTracking: true,
      anchors: true,
      hitTest: false,
      planeDetection: false,
      meshDetection: false,
      layers: true,
    },
  },
  features: {
    locomotion: false,
    grabbing: true,
    physics: false,
    sceneUnderstanding: true,
  },
}).then((world) => {
  const { camera } = world;

  camera.position.set(0, 1, 0.5);

  const teleopPanel = new DraggablePanel(world, "./ui/teleop.json", {
    maxHeight: 0.8,
    maxWidth: 1.6,
  });
  teleopPanel.setPosition(0, 1.29, -1.9);
  if (teleopPanel.entity.object3D) {
    GlobalRefs.teleopPanelRoot = teleopPanel.entity.object3D;
  }

  // Camera panels map
  const cameraPanels = new Map<string, CameraPanel>();

  // Controller-attached camera panels (for wrist cameras)
  const leftControllerPanel = new ControllerCameraPanel(world, "left");
  if (leftControllerPanel.entity.object3D) {
    GlobalRefs.leftWristPanelRoot = leftControllerPanel.entity.object3D;
  }
  const rightControllerPanel = new ControllerCameraPanel(world, "right");
  if (rightControllerPanel.entity.object3D) {
    GlobalRefs.rightWristPanelRoot = rightControllerPanel.entity.object3D;
  }

  onCameraViewsChanged((config) => {
    if (leftControllerPanel.entity.object3D) {
      leftControllerPanel.entity.object3D.visible = isViewEnabled("wrist_left");
    }
    if (rightControllerPanel.entity.object3D) {
      rightControllerPanel.entity.object3D.visible = isViewEnabled("wrist_right");
    }

    const allKeys = Object.keys(config);
    const reserved = ["wrist_left", "wrist_right"];
    const floatingKeys = allKeys.filter((k) => !reserved.includes(k)).sort();

    floatingKeys.forEach((key, index) => {
      let panel = cameraPanels.get(key);
      if (!panel) {
        panel = new CameraPanel(world);
        cameraPanels.set(key, panel);

        if (panel.entity.object3D) {
          GlobalRefs.cameraPanels.set(key, panel.entity.object3D);
          panel.entity.object3D.visible = !disableHeadCameraPanel;
        }
      }

      const x = 1.2 + index * 0.9;
      panel.setPosition(x, 1.3, -1.5);
    });

    for (const [key, panel] of cameraPanels.entries()) {
      if (!floatingKeys.includes(key)) {
        panel.dispose();
        cameraPanels.delete(key);
        GlobalRefs.cameraPanels.delete(key);
      }
    }
  });

  const getFallbackOrder = (): string[] => {
    const config = getCameraViewsConfig();
    const keys = Object.keys(config);

    if (keys.length === 0) {
      const defaultKeys: string[] = [];
      if (!disableHeadCameraPanel) {
        defaultKeys.push("head");
      }
      defaultKeys.push("wrist_left", "wrist_right");
      return defaultKeys;
    }

    // Sort keys alphabetically but prioritize head, then wrists
    keys.sort();

    const prioritized = ["head", "wrist_left", "wrist_right"];
    const result: string[] = [];

    // Add prioritized keys if they exist in config
    for (const key of prioritized) {
      if (keys.includes(key)) {
        result.push(key);
      }
    }

    // Add remaining keys
    for (const key of keys) {
      if (!prioritized.includes(key)) {
        result.push(key);
      }
    }

    return result;
  };

  // Video connection
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const videoWsUrl = `${protocol}//${window.location.host}/ws`;

  // Track assignment: use trackId mapping with order-based fallback
  let trackCount = 0;
  const videoClient = new VideoClient(
    videoWsUrl,
    (stats) => {},
    (track, trackId) => {
      const targetView = resolveTrackView(trackId, trackCount, getFallbackOrder());

      if (!targetView) {
        trackCount++;
        return;
      }

      if (targetView === "wrist_left") {
        leftControllerPanel.setVideoTrack(track);
      } else if (targetView === "wrist_right") {
        rightControllerPanel.setVideoTrack(track);
      } else {
        const panel = cameraPanels.get(targetView);
        if (panel && !disableHeadCameraPanel) {
          panel.setVideoTrack(track);
        }
      }
      trackCount++;
    },
  );

  world.registerSystem(PanelSystem);
  world.registerSystem(TeleopSystem);
  world.registerSystem(ControllerCameraPanelSystem);

  // Register controller panels with their raySpaces once XR session starts
  // The system will handle waiting for raySpaces to be available
  const controllerCameraSystem = world.getSystem(ControllerCameraPanelSystem);
  if (controllerCameraSystem) {
    // Register with callbacks that resolve raySpaces dynamically
    const getLeftRaySpace = () => world.player?.raySpaces?.left;
    const getRightRaySpace = () => world.player?.raySpaces?.right;

    // The system expects controllerObject - we'll modify the system to handle getters
    // For now, pass a reference that will be resolved each frame
    (controllerCameraSystem as any).registerPanelWithGetter(leftControllerPanel, getLeftRaySpace);
    (controllerCameraSystem as any).registerPanelWithGetter(rightControllerPanel, getRightRaySpace);
  }
});
