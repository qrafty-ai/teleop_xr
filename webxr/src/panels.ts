import { World, PanelUI, Interactable, DistanceGrabbable, MovementMode, Visibility } from "@iwsdk/core";
import { Mesh, PlaneGeometry, MeshBasicMaterial, VideoTexture, DoubleSide, BoxGeometry, Object3D, Vector3, Quaternion } from "three";

export class DraggablePanel {
  public entity: any;
  public panelEntity: any;

  constructor(protected world: World, configPath: string, options: any = {}) {
    const width = options.maxWidth || 0.8;
    const height = options.maxHeight || 0.6;
    const handleHeight = 0.05;
    const gap = 0.02;

    // 1. Create Handle (Root) - Interactable and Grabbable
    // Pre-add Visibility component to enable safe toggling at runtime
    this.entity = world.createTransformEntity()
      .addComponent(Interactable)
      .addComponent(Visibility, { isVisible: true })
      .addComponent(DistanceGrabbable, {
        movementMode: MovementMode.MoveFromTarget
      });

    // Handle Visuals - Styling aligned with uikit panel
    const handleWidth = width * 0.5;
    const handleGeo = new BoxGeometry(handleWidth, handleHeight, 0.05);
    const handleMat = new MeshBasicMaterial({ 
      color: 0xe4e4e7, // Light grey
      transparent: true, 
      opacity: 0.5 
    });
    const handleMesh = new Mesh(handleGeo, handleMat);
    this.entity.object3D.add(handleMesh);

    // 2. Create Panel (Child) - Interactable but NOT Grabbable
    this.panelEntity = world.createTransformEntity()
      .addComponent(PanelUI, {
        config: configPath,
        ...options
      })
      .addComponent(Interactable)
      // Stop grab event bubbling by consuming it with a locked DistanceGrabbable
      .addComponent(DistanceGrabbable, {
        movementMode: MovementMode.MoveFromTarget,
        translate: false,
        rotate: false,
        scale: false
      });

    // Parent Panel to Handle
    this.entity.object3D.add(this.panelEntity.object3D);

    // Offset Panel above Handle
    // Panel Y - height/2 = handleHeight/2 + gap
    const panelY = (height / 2) + (handleHeight / 2) + gap;
    this.panelEntity.object3D.position.set(0, panelY, 0);
  }

  setPosition(x: number, y: number, z: number) {
    this.entity.object3D.position.set(x, y, z);
  }
}

export class CameraPanel extends DraggablePanel {
  private videoMesh: Mesh | null = null;
  private videoElement: HTMLVideoElement | null = null;

  constructor(world: World) {
    super(world, "./ui/camera.uikitml", {
      maxHeight: 0.6,
      maxWidth: 0.8,
    });
  }

  setVideoTrack(track: MediaStreamTrack) {
    if (this.videoMesh) return; // Already set

    const stream = new MediaStream([track]);
    this.videoElement = document.createElement("video");
    this.videoElement.srcObject = stream;
    this.videoElement.playsInline = true;
    this.videoElement.muted = true; // Required for autoplay
    this.videoElement.style.display = "none";
    document.body.appendChild(this.videoElement);

    this.videoElement.play().catch(e => {
        console.error(`Video play error: ${e}`);
    });

    const texture = new VideoTexture(this.videoElement);
    // Aspect ratio 1.5 roughly
    const geometry = new PlaneGeometry(0.6, 0.4); 
    const material = new MeshBasicMaterial({ map: texture, side: DoubleSide });
    this.videoMesh = new Mesh(geometry, material);
    
    // Position it slightly in front of the panel to avoid z-fighting
    this.videoMesh.position.z = 0.02; 
    // Adjust y to be centered or below title
    this.videoMesh.position.y = -0.1;

    // Attach to panelEntity, not the handle/root
    this.panelEntity.object3D.add(this.videoMesh);
  }
}

export class ControllerCameraPanel {
  public entity: any;
  public handedness: "left" | "right";
  private videoMesh: Mesh | null = null;
  private videoElement: HTMLVideoElement | null = null;

  constructor(world: World, handedness: "left" | "right") {
    this.handedness = handedness;
    
    // Create a simple transform entity (no grabbable, no panel UI)
    this.entity = world.createTransformEntity();
    
    // Create a background plane for the video
    const bgGeo = new PlaneGeometry(0.2, 0.15);
    const bgMat = new MeshBasicMaterial({ 
      color: 0x1a1a1a,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
      side: DoubleSide
    });
    const bgMesh = new Mesh(bgGeo, bgMat);
    bgMesh.position.z = -0.002;
    bgMesh.renderOrder = 0;
    this.entity.object3D.add(bgMesh);
  }

  setVideoTrack(track: MediaStreamTrack) {
    if (this.videoMesh) return; // Already set

    const stream = new MediaStream([track]);
    this.videoElement = document.createElement("video");
    this.videoElement.srcObject = stream;
    this.videoElement.playsInline = true;
    this.videoElement.muted = true;
    this.videoElement.style.display = "none";
    document.body.appendChild(this.videoElement);

    this.videoElement.play().catch(e => {
      console.error(`Video play error: ${e}`);
    });

    const texture = new VideoTexture(this.videoElement);
    const geometry = new PlaneGeometry(0.18, 0.135); // Slightly smaller than bg
    const material = new MeshBasicMaterial({ map: texture, side: DoubleSide });
    this.videoMesh = new Mesh(geometry, material);
    this.videoMesh.position.z = 0.001; // Slightly in front of bg
    this.videoMesh.renderOrder = 1;
    this.entity.object3D.add(this.videoMesh);
  }
}
