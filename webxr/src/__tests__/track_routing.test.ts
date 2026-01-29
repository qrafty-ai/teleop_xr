import { describe, it, expect } from "vitest";

import { resolveTrackView } from "../track_routing";

describe("resolveTrackView", () => {
  it("uses explicit trackId mapping", () => {
    expect(resolveTrackView("head", 2)).toBe("head");
    expect(resolveTrackView("wrist_left", 0)).toBe("wrist_left");
    expect(resolveTrackView("wrist_right", 1)).toBe("wrist_right");
  });

  it("falls back by order for unknown or missing ids", () => {
    expect(resolveTrackView(undefined, 0)).toBe("head");
    expect(resolveTrackView(null, 1)).toBe("wrist_left");
    expect(resolveTrackView("", 2)).toBe("wrist_right");
    expect(resolveTrackView("unknown", 1)).toBe("wrist_left");
  });

  it("maps numeric track ids to enabled order", () => {
    const order = ["wrist_left", "wrist_right"] as const;

    expect(resolveTrackView("0", 0, order)).toBe("wrist_left");
    expect(resolveTrackView("1", 0, order)).toBe("wrist_right");
    expect(resolveTrackView("2", 0, order)).toBeNull();
  });

  it("returns null for extra tracks", () => {
    expect(resolveTrackView("unknown", 3)).toBeNull();
  });
});
