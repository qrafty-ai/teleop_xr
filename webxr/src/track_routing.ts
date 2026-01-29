export type CameraViewKey = "head" | "wrist_left" | "wrist_right";

const defaultOrder: CameraViewKey[] = ["head", "wrist_left", "wrist_right"];

export function resolveTrackView(
  trackId: string | null | undefined,
  trackCount: number,
  order: readonly CameraViewKey[] = defaultOrder,
): CameraViewKey | null {
  if (trackId === "head" || trackId === "wrist_left" || trackId === "wrist_right") {
    return trackId;
  }

  const fallbackOrder = order.length > 0 ? order : defaultOrder;
  if (typeof trackId === "string" && /^[0-9]+$/.test(trackId)) {
    const index = Number(trackId);
    if (Number.isInteger(index) && index >= 0 && index < fallbackOrder.length) {
      return fallbackOrder[index];
    }
    return null;
  }

  if (trackCount >= 0 && trackCount < fallbackOrder.length) {
    return fallbackOrder[trackCount];
  }
  return null;
}
