import { VideoClient, VideoStats } from '../video';
import { VideoTexture, MeshBasicMaterial, Mesh, PlaneGeometry, DoubleSide } from 'three';
import { getCameraSettings } from '../camera_config';

// Global AFRAME is declared as 'any' in aframe-types.d.ts
// We declare it here to satisfy the compiler if needed, but it should be global.
declare const AFRAME: any;

interface VideoStreamSystem {
  videoClient: VideoClient | null;
  tracks: Map<string, MediaStreamTrack>;
  components: Map<string, Set<VideoStreamComponent>>;

  registerComponent(component: VideoStreamComponent): void;
  unregisterComponent(component: VideoStreamComponent): void;
  connect(): void;
}

interface VideoStreamComponent {
  data: { trackId: string };
  el: any; // HTMLEntity with object3D
  system: VideoStreamSystem;

  videoElement: HTMLVideoElement | null;
  mesh: Mesh | null;
  setVideoTrack(track: MediaStreamTrack): void;
}

AFRAME.registerSystem('video-stream', {
  schema: {},

  init: function() {
    this.videoClient = null;
    this.tracks = new Map();
    this.components = new Map();
    this.connect();
  },

  connect: function() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    console.log(`[VideoSystem] Connecting to ${wsUrl}`);

    this.videoClient = new VideoClient(
      wsUrl,
      (stats: VideoStats) => {
      },
      (track: MediaStreamTrack, trackId: string) => {
        console.log(`[VideoSystem] Received track: ${trackId}`);
        this.tracks.set(trackId, track);

        const waitingComponents = this.components.get(trackId);
        if (waitingComponents) {
          waitingComponents.forEach((comp: VideoStreamComponent) => comp.setVideoTrack(track));
        }
      }
    );
  },

  registerComponent: function(component: VideoStreamComponent) {
    const trackId = component.data.trackId;
    if (!trackId) return;

    if (!this.components.has(trackId)) {
      this.components.set(trackId, new Set());
    }
    this.components.get(trackId)!.add(component);

    if (this.tracks.has(trackId)) {
      component.setVideoTrack(this.tracks.get(trackId)!);
    }
  },

  unregisterComponent: function(component: VideoStreamComponent) {
    const trackId = component.data.trackId;
    if (trackId && this.components.has(trackId)) {
      this.components.get(trackId)!.delete(component);
    }
  }
});

AFRAME.registerComponent('video-stream', {
  schema: {
    trackId: { type: 'string', default: '' }
  },

  init: function() {
    this.videoElement = null;
    this.mesh = null;

    if (this.data.trackId) {
      (this.system as VideoStreamSystem).registerComponent(this as any);
    }
  },

  update: function(oldData: any) {
    if (oldData.trackId !== this.data.trackId) {
      (this.system as VideoStreamSystem).unregisterComponent(this as any);
      if (this.data.trackId) {
        (this.system as VideoStreamSystem).registerComponent(this as any);
      }
    }
  },

  remove: function() {
    (this.system as VideoStreamSystem).unregisterComponent(this as any);
    if (this.videoElement) {
      this.videoElement.pause();
      this.videoElement.srcObject = null;
      this.videoElement.remove();
      this.videoElement = null;
    }

    // Three.js texture disposal is manual.
    if (this.mesh && this.mesh.material) {
        if (Array.isArray(this.mesh.material)) {
            this.mesh.material.forEach((m: any) => {
                if (m.map) m.map.dispose();
            });
        } else {
            const mat = this.mesh.material as MeshBasicMaterial;
            if (mat.map) mat.map.dispose();
        }
    }
  },

  setVideoTrack: function(track: MediaStreamTrack) {
    if (this.videoElement) return;

    console.log(`[VideoStream] Setting track for ${this.data.trackId}`);

    const stream = new MediaStream([track]);
    this.videoElement = document.createElement("video");
    this.videoElement.srcObject = stream;
    this.videoElement.playsInline = true;
    this.videoElement.muted = true; // Required for autoplay
    this.videoElement.style.display = "none";
    document.body.appendChild(this.videoElement);

    this.videoElement.play().catch(e => {
        console.error(`[VideoStream] Video play error: ${e}`);
    });

    const texture = new VideoTexture(this.videoElement);

    const mesh = this.el.getObject3D('mesh') as Mesh;
    if (mesh) {
        if (Array.isArray(mesh.material)) {
             (mesh.material[0] as MeshBasicMaterial).map = texture;
             (mesh.material[0] as MeshBasicMaterial).needsUpdate = true;
        } else {
            (mesh.material as MeshBasicMaterial).map = texture;
            (mesh.material as MeshBasicMaterial).needsUpdate = true;
        }
    } else {
        const settings = getCameraSettings(this.data.trackId);
        const aspect = settings.width / settings.height || 4/3;
        const width = 1.0;
        const height = width / aspect;
        const geometry = new PlaneGeometry(width, height);
        const material = new MeshBasicMaterial({ map: texture, side: DoubleSide });
        this.mesh = new Mesh(geometry, material);
        this.el.setObject3D('mesh', this.mesh);
    }
  }
});
