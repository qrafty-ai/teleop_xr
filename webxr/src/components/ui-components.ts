import type { Entity, Component } from 'aframe';
import { Vector3, Quaternion, Object3D } from 'three';

/**
 * Billboard Follow Component
 *
 * Follows a target entity (like a controller) with a positional offset,
 * but maintains its own rotation to face a target (like the camera).
 * This allows for "floating" UI panels attached to controllers that always face the user.
 */
export const BillboardFollowComponent = {
  schema: {
    target: { type: 'selector', default: '[camera]' }, // The entity to look at (head)
    follow: { type: 'selector', default: null },       // The entity to follow (controller)
    offset: { type: 'vec3', default: { x: 0, y: 0.15, z: -0.05 } } // Offset from the followed entity
  },

  init: function(this: any) {
    this.targetPos = new Vector3();
    this.followPos = new Vector3();
    this.followQuat = new Quaternion();
    this.offsetVec = new Vector3();
    this.lookAtPos = new Vector3();
  },

  tick: function(this: any) {
    const data = this.data;
    const targetEl = data.target;
    const followEl = data.follow;

    // If requirements aren't met, do nothing
    if (!targetEl || !targetEl.object3D || !followEl || !followEl.object3D) {
      return;
    }

    // 1. Get 'follow' position and rotation (controller)
    followEl.object3D.getWorldPosition(this.followPos);
    followEl.object3D.getWorldQuaternion(this.followQuat);

    // 2. Calculate offset position
    // Apply the local offset vector rotated by the controller's orientation
    this.offsetVec.copy(data.offset).applyQuaternion(this.followQuat);

    // Set this entity's position to controller pos + rotated offset
    this.el.object3D.position.copy(this.followPos).add(this.offsetVec);

    // 3. Look at the target (camera/head)
    targetEl.object3D.getWorldPosition(this.lookAtPos);
    this.el.object3D.lookAt(this.lookAtPos);
  }
};

/**
 * Teleop Dashboard Component
 *
 * Creates the main UI dashboard for TeleopXR (Status, FPS, Latency).
 * Uses A-Frame primitives (a-plane, a-text).
 */
export const TeleopDashboardComponent = {
  schema: {},

  init: function(this: any) {
    this.createUI();
    this.bindEvents();
  },

  createUI: function(this: any) {
    const el = this.el;

    // 1. Background Panel
    const bg = document.createElement('a-plane');
    bg.setAttribute('width', '1.2');
    bg.setAttribute('height', '0.6');
    bg.setAttribute('color', '#111');
    bg.setAttribute('opacity', '0.8');
    bg.setAttribute('material', 'shader: flat');
    el.appendChild(bg);

    // 2. Title
    const title = document.createElement('a-text');
    title.setAttribute('value', 'TeleopXR');
    title.setAttribute('align', 'center');
    title.setAttribute('position', '0 0.2 0.01');
    title.setAttribute('width', '3');
    el.appendChild(title);

    // 3. Status Text
    this.statusText = document.createElement('a-text');
    this.statusText.setAttribute('value', 'Status: Connecting...');
    this.statusText.setAttribute('align', 'center');
    this.statusText.setAttribute('position', '0 0.05 0.01');
    this.statusText.setAttribute('width', '2');
    this.statusText.setAttribute('color', '#ffff00'); // Yellow
    el.appendChild(this.statusText);

    // 4. FPS Text
    this.fpsText = document.createElement('a-text');
    this.fpsText.setAttribute('value', 'FPS: 0');
    this.fpsText.setAttribute('align', 'left');
    this.fpsText.setAttribute('position', '-0.5 -0.15 0.01');
    this.fpsText.setAttribute('width', '1.5');
    this.fpsText.setAttribute('color', '#aaa');
    el.appendChild(this.fpsText);

    // 5. Latency Text
    this.latencyText = document.createElement('a-text');
    this.latencyText.setAttribute('value', 'Latency: 0ms');
    this.latencyText.setAttribute('align', 'left');
    this.latencyText.setAttribute('position', '0.1 -0.15 0.01');
    this.latencyText.setAttribute('width', '1.5');
    this.latencyText.setAttribute('color', '#aaa');
    el.appendChild(this.latencyText);
  },

  bindEvents: function(this: any) {
    const scene = this.el.sceneEl;

    // Listen for status updates
    scene.addEventListener('teleop-status', (evt: CustomEvent) => {
      if (this.statusText) {
        const { text, connected } = evt.detail;
        this.statusText.setAttribute('value', `Status: ${text}`);
        this.statusText.setAttribute('color', connected ? '#00ff00' : '#ff0000');
      }
    });

    // Listen for stats updates
    scene.addEventListener('teleop-stats', (evt: CustomEvent) => {
      if (this.fpsText && this.latencyText) {
        const { fps, latency } = evt.detail;
        this.fpsText.setAttribute('value', `FPS: ${Math.round(fps)}`);
        this.latencyText.setAttribute('value', `Latency: ${Math.round(latency)}ms`);
      }
    });
  }
};

// Register components if AFRAME is available
if (typeof AFRAME !== 'undefined') {
  AFRAME.registerComponent('billboard-follow', BillboardFollowComponent);
  AFRAME.registerComponent('teleop-dashboard', TeleopDashboardComponent);
}
