declare const AFRAME: any;
import { MeshPhysicalMaterial, Mesh, PMREMGenerator } from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';

AFRAME.registerComponent('spatial-panel', {
  schema: {
    color: { type: 'color', default: '#ffffff' },
    opacity: { type: 'number', default: 0.2 },
    transmission: { type: 'number', default: 0.95 },
    roughness: { type: 'number', default: 0.05 },
    thickness: { type: 'number', default: 0.5 },
    ior: { type: 'number', default: 1.5 },
  },

  init: function() {
    // Setup environment map for realistic glass reflections
    const scene = this.el.sceneEl?.object3D;
    if (scene && !scene.environment) {
      const renderer = this.el.sceneEl?.renderer;
      if (renderer) {
        const pmremGenerator = new PMREMGenerator(renderer);
        pmremGenerator.compileEquirectangularShader();
        scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
        pmremGenerator.dispose();
      }
    }

    // Cache the material once
    this.material = new MeshPhysicalMaterial({
      color: this.data.color,
      metalness: 0,
      roughness: this.data.roughness,
      transmission: this.data.transmission,
      thickness: this.data.thickness,
      ior: this.data.ior,
      opacity: this.data.opacity,
      transparent: true,
    });
    this.materialApplied = false;

    this.applyMaterial = this.applyMaterial.bind(this);
    this.el.addEventListener('model-loaded', this.applyMaterial);
    this.el.addEventListener('object3dset', this.applyMaterial);
    this.applyMaterial();
  },

  update: function() {
    this.applyMaterial();
  },

  applyMaterial: function() {
    const mesh = this.el.getObject3D('mesh') as Mesh;
    if (!mesh) return;

    mesh.traverse((node: any) => {
      if (node.isMesh) {
        // Only apply glass material to the background plane, not text meshes
        const isBackgroundPlane = node.geometry?.type === 'PlaneGeometry' ||
                                  node.name?.toLowerCase().includes('background') ||
                                  node.name?.toLowerCase().includes('panel');

        if (isBackgroundPlane) {
          node.material = new MeshPhysicalMaterial({
            color: this.data.color,
            metalness: 0,
            roughness: this.data.roughness,
            transmission: this.data.transmission,
            thickness: this.data.thickness,
            ior: this.data.ior,
            opacity: this.data.opacity,
            transparent: true,
          });
        }
        // Text meshes keep their original material (visible/white)
      }
    });
  },

  tick: function() {
    if (this.materialApplied) return;

    const mesh = this.el.getObject3D('mesh') as Mesh;
    if (!mesh) return;

    let applied = false;
    mesh.traverse((node: any) => {
      if (node.isMesh) {
        const isBackgroundPlane = node.geometry?.type === 'PlaneGeometry' ||
                                  node.name?.toLowerCase().includes('background') ||
                                  node.name?.toLowerCase().includes('panel');

        if (isBackgroundPlane && node.material !== this.material) {
          node.material = this.material;
          applied = true;
        }
      }
    });

    if (applied) {
      this.materialApplied = true;
    }
  }
});
