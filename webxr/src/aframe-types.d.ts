import 'aframe';
import * as THREE from 'three';

declare global {
  const AFRAME: any;
  interface Window {
    AFRAME: any;
  }
}

declare module 'aframe' {
  export import THREE = THREE;
}

export {};
