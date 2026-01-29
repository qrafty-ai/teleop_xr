import { describe, it, expect } from "vitest";

import {
  getCameraViewsConfig,
  isViewEnabled,
  onCameraViewsChanged,
  setCameraViewsConfig,
} from "../camera_views";

describe("camera views config", () => {
  it("stores config and enables views", () => {
    setCameraViewsConfig({ head: { device: 0 }, wrist_left: { device: 1 } });

    expect(getCameraViewsConfig()).toEqual({
      head: { device: 0 },
      wrist_left: { device: 1 },
    });
    expect(isViewEnabled("head")).toBe(true);
    expect(isViewEnabled("wrist_left")).toBe(true);
    expect(isViewEnabled("wrist_right")).toBe(false);
  });

  it("notifies handlers immediately and on updates", () => {
    setCameraViewsConfig(null);

    let calls = 0;
    let lastConfig: unknown = null;

    onCameraViewsChanged((config) => {
      calls += 1;
      lastConfig = config;
    });

    expect(calls).toBe(1);
    expect(lastConfig).toEqual({});

    setCameraViewsConfig({ wrist_right: { device: "/dev/video2" } });

    expect(calls).toBe(2);
    expect(lastConfig).toEqual({ wrist_right: { device: "/dev/video2" } });
  });

  it("clears config when null", () => {
    setCameraViewsConfig({ head: { device: 0 } });
    setCameraViewsConfig(null);

    expect(getCameraViewsConfig()).toEqual({});
    expect(isViewEnabled("head")).toBe(false);
  });
});
