#!/usr/bin/env python3
# pyright: basic

from __future__ import annotations

import numpy as np
import pyroki as pk
import trimesh
import yourdfpy

from teleop_xr.ik.robots.teaarm import TeaArmRobot


def _sample_points(mesh: trimesh.Trimesh, total_samples: int = 1000) -> np.ndarray:
    if mesh.is_empty:
        return np.zeros((0, 3), dtype=np.float32)

    n_surface = total_samples // 2
    n_volume = total_samples - n_surface

    surface_result = trimesh.sample.sample_surface(mesh, n_surface)
    surface_points = np.asarray(surface_result[0])

    volume_points = np.zeros((0, 3), dtype=np.float32)
    if n_volume > 0:
        try:
            volume_points = trimesh.sample.volume_mesh(mesh, n_volume)
        except Exception:
            volume_points = np.zeros((0, 3), dtype=np.float32)

    if volume_points.shape[0] > 0:
        points = np.concatenate([surface_points, volume_points], axis=0)
    else:
        points = surface_points
    return points.astype(np.float32)


def _occupancy_ratio(
    points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> float:
    if points.shape[0] == 0:
        return 1.0
    if centers.shape[0] == 0:
        return 0.0

    diff = points[:, None, :] - centers[None, :, :]
    inside = np.sum(diff * diff, axis=-1) <= (radii[None, :] ** 2)
    return float(np.mean(np.any(inside, axis=1)))


def main() -> None:
    robot = TeaArmRobot()
    urdf = yourdfpy.URDF.load(robot.urdf_path)
    coll = robot.multi_sphere_coll

    centers_all = np.asarray(coll.sphere_centers_local)
    radii_all = np.asarray(coll.sphere_radii)
    link_idx_all = np.asarray(coll.sphere_link_indices)

    print("\nSphere Occupancy by Link")
    print("=" * 90)
    print(f"{'Link':36} {'Spheres':>7} {'Samples':>8} {'Occupied':>8} {'Ratio':>8}")
    print("-" * 90)

    arm_ratios: list[float] = []
    for link_idx, link_name in enumerate(robot.robot.links.names):
        mesh = pk.collision.RobotCollision._get_trimesh_collision_geometries(
            urdf, link_name
        )
        points = _sample_points(mesh, total_samples=1000)

        mask = link_idx_all == link_idx
        centers = centers_all[mask]
        radii = radii_all[mask]

        ratio = _occupancy_ratio(points, centers, radii)
        occupied_count = int(round(ratio * points.shape[0]))

        if "_arm_l" in link_name:
            arm_ratios.append(ratio)

        print(
            f"{link_name:36} {centers.shape[0]:7d} {points.shape[0]:8d} {occupied_count:8d} {ratio:8.3%}"
        )

    print("-" * 90)
    if arm_ratios:
        min_arm_ratio = min(arm_ratios)
        avg_arm_ratio = float(np.mean(arm_ratios))
        print(f"Arm link occupancy min: {min_arm_ratio:.3%}")
        print(f"Arm link occupancy avg: {avg_arm_ratio:.3%}")
        print(f"Arm occupancy > 90%: {min_arm_ratio > 0.90}")
    print("=" * 90)


if __name__ == "__main__":
    main()
