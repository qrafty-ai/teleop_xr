# ruff: noqa: F722, F821
# pyright: basic
from __future__ import annotations

from typing import Protocol, cast
import xml.etree.ElementTree as ET

import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
import pyroki as pk
import trimesh
import yourdfpy
from jaxtyping import Array, Float, Int
from pyroki._robot_urdf_parser import RobotURDFParser


class _RobotLinks(Protocol):
    names: tuple[str, ...]


class RobotLike(Protocol):
    links: _RobotLinks

    def forward_kinematics(
        self, cfg: Float[Array, "*batch actuated_count"]
    ) -> Float[Array, "*batch L 7"]: ...


@jdc.pytree_dataclass
class MultiSphereCollision:
    """Flattened multi-sphere self-collision model for a robot."""

    num_primitives: jdc.Static[int]
    num_links: jdc.Static[int]
    link_names: jdc.Static[tuple[str, ...]]

    sphere_centers_local: Float[Array, "P 3"]
    sphere_radii: Float[Array, "P"]
    sphere_link_indices: Int[Array, "P"]

    pair_i: Int[Array, "K"]
    pair_j: Int[Array, "K"]

    def at_config(
        self, robot: RobotLike, cfg: Float[Array, "*batch actuated_count"]
    ) -> tuple[Float[Array, "*batch P 3"], Float[Array, "P"]]:
        """Return sphere centers in world frame and radii at configuration."""
        assert self.link_names == robot.links.names, (
            "Link name mismatch between MultiSphereCollision and robot kinematics."
        )

        link_poses = jaxlie.SE3(robot.forward_kinematics(cfg))
        sphere_link_poses = jaxlie.SE3(
            link_poses.wxyz_xyz[..., self.sphere_link_indices, :]
        )
        world_centers = sphere_link_poses.apply(self.sphere_centers_local)
        return world_centers, self.sphere_radii

    def compute_self_collision_distance(
        self, robot: RobotLike, cfg: Float[Array, "*batch actuated_count"]
    ) -> Float[Array, "*batch K"]:
        """Compute signed pairwise sphere distances for active primitive pairs."""
        world_centers, sphere_radii = self.at_config(robot, cfg)

        c_i = world_centers[..., self.pair_i, :]
        c_j = world_centers[..., self.pair_j, :]
        r_i = sphere_radii[self.pair_i]
        r_j = sphere_radii[self.pair_j]

        dist = cast(
            Float[Array, "*batch K"],
            jnp.linalg.norm(c_i - c_j, axis=-1) - (r_i + r_j),
        )
        return dist


def _default_spheres_for_link(link_name: str, default_n_spheres: int) -> int:
    if link_name == "base_link":
        return 2
    if link_name in {"waist_link", "torso_link"}:
        return 4
    if "_arm_l" in link_name or "_arm_r" in link_name:
        return 4
    if link_name.startswith("frame_") or "ee" in link_name:
        return 1
    return default_n_spheres


def _fit_radii_along_centerline(
    vertices: onp.ndarray,
    c1: onp.ndarray,
    c2: onp.ndarray,
    centers: list[onp.ndarray],
    sidelength: float,
    radius_margin: float,
    min_radius: float,
) -> list[float]:
    if len(centers) == 0:
        return []
    if vertices.shape[0] == 0:
        return [min_radius for _ in centers]

    axis = c2 - c1
    length = float(onp.linalg.norm(axis))
    if length < 1e-9:
        return [
            max(
                min_radius,
                float(onp.linalg.norm(vertices - c, axis=1).max()) + radius_margin,
            )
            for c in centers
        ]

    axis_dir = axis / length
    rel = vertices - c1[None, :]
    t = rel @ axis_dir
    perp = onp.linalg.norm(rel - t[:, None] * axis_dir[None, :], axis=1)
    fallback_max_perp = float(perp.max())

    radii: list[float] = []
    for center in centers:
        tc = float((center - c1) @ axis_dir)
        t_min = tc - sidelength
        t_max = tc + sidelength
        mask = (t >= t_min) & (t <= t_max)
        if onp.any(mask):
            max_perp = float(perp[mask].max())
        else:
            max_perp = fallback_max_perp
        radius = float(onp.sqrt(sidelength**2 + max_perp**2)) + radius_margin
        radii.append(max(min_radius, radius))

    return radii


def _points_covered_by_spheres(
    points: onp.ndarray,
    centers: list[onp.ndarray],
    radii: list[float],
) -> onp.ndarray:
    if points.shape[0] == 0:
        return onp.zeros((0,), dtype=bool)
    if len(centers) == 0:
        return onp.zeros((points.shape[0],), dtype=bool)

    centers_arr = onp.asarray(centers, dtype=onp.float32)
    radii_arr = onp.asarray(radii, dtype=onp.float32)
    diff = points[:, None, :] - centers_arr[None, :, :]
    inside = onp.sum(diff * diff, axis=-1) <= (radii_arr[None, :] ** 2)
    return onp.any(inside, axis=1)


def parse_srdf_ignore_pairs(srdf_path: str) -> tuple[tuple[str, str], ...]:
    if not srdf_path:
        return ()

    try:
        root = ET.parse(srdf_path).getroot()
    except (FileNotFoundError, ET.ParseError, OSError):
        return ()

    pairs: set[tuple[str, str]] = set()
    for entry in root.findall(".//disable_collisions"):
        link1 = entry.attrib.get("link1")
        link2 = entry.attrib.get("link2")
        if link1 is None or link2 is None or link1 == link2:
            continue
        pair = (link1, link2) if link1 < link2 else (link2, link1)
        pairs.add(pair)

    return tuple(sorted(pairs))


def build_multi_sphere_collision(
    urdf: yourdfpy.URDF,
    spheres_per_link: dict[str, int] | None = None,
    default_n_spheres: int = 3,
    radius_margin: float = 0.01,
    user_ignore_pairs: tuple[tuple[str, str], ...] = (),
) -> MultiSphereCollision:
    _, link_info = RobotURDFParser.parse(urdf)
    link_names = link_info.names
    num_links = link_info.num_links

    per_link_override = spheres_per_link or {}
    min_radius = 0.005

    child_joint_origins: dict[str, list[onp.ndarray]] = {
        name: [] for name in link_names
    }
    for joint in urdf.joint_map.values():
        if joint.parent in child_joint_origins:
            if joint.origin is None:
                child_joint_origins[joint.parent].append(
                    onp.zeros(3, dtype=onp.float32)
                )
            else:
                child_joint_origins[joint.parent].append(
                    onp.asarray(joint.origin[:3, 3], dtype=onp.float32)
                )

    all_centers: list[onp.ndarray] = []
    all_radii: list[float] = []
    sphere_link_indices: list[int] = []

    for link_idx, link_name in enumerate(link_names):
        mesh = pk.collision.RobotCollision._get_trimesh_collision_geometries(
            urdf, link_name
        )
        vertices = (
            onp.asarray(mesh.vertices, dtype=onp.float32)
            if isinstance(mesh, trimesh.Trimesh) and mesh.vertices is not None
            else onp.zeros((0, 3), dtype=onp.float32)
        )

        child_origins = child_joint_origins.get(link_name, [])
        link_centers: list[onp.ndarray] = []
        link_radii: list[float] = []

        if len(child_origins) == 0:
            if vertices.shape[0] == 0:
                continue
            center = onp.zeros(3, dtype=onp.float32)
            radius = (
                float(onp.linalg.norm(vertices - center, axis=1).max()) + radius_margin
            )
            link_centers.append(center)
            link_radii.append(max(min_radius, radius))
        else:
            for child_origin in child_origins:
                c1 = onp.zeros(3, dtype=onp.float32)
                c2 = onp.asarray(child_origin, dtype=onp.float32)
                segment = c2 - c1
                length = float(onp.linalg.norm(segment))

                n_spheres = per_link_override.get(
                    link_name,
                    _default_spheres_for_link(link_name, default_n_spheres),
                )
                n_spheres = max(1, int(n_spheres))

                if length < 1e-9:
                    center = 0.5 * (c1 + c2)
                    if vertices.shape[0] > 0:
                        radius = (
                            float(onp.linalg.norm(vertices - center, axis=1).max())
                            + radius_margin
                        )
                    else:
                        radius = min_radius
                    link_centers.append(center.astype(onp.float32))
                    link_radii.append(max(min_radius, radius))
                    continue

                direction = segment / length
                sidelength = length / (2.0 * n_spheres)
                centers = [
                    c1 + direction * ((2.0 * i + 1.0) * sidelength)
                    for i in range(n_spheres)
                ]
                radii = _fit_radii_along_centerline(
                    vertices=vertices,
                    c1=c1,
                    c2=c2,
                    centers=centers,
                    sidelength=sidelength,
                    radius_margin=radius_margin,
                    min_radius=min_radius,
                )

                for center, radius in zip(centers, radii):
                    link_centers.append(center.astype(onp.float32))
                    link_radii.append(radius)

        if vertices.shape[0] > 0:
            covered = _points_covered_by_spheres(vertices, link_centers, link_radii)
            if not bool(onp.all(covered)):
                uncovered_vertices = vertices[~covered]
                fallback_center = onp.zeros(3, dtype=onp.float32)
                fallback_radius = (
                    float(
                        onp.linalg.norm(
                            uncovered_vertices - fallback_center, axis=1
                        ).max()
                    )
                    + radius_margin
                )
                link_centers.append(fallback_center)
                link_radii.append(max(min_radius, fallback_radius))

        for center, radius in zip(link_centers, link_radii):
            all_centers.append(center.astype(onp.float32))
            all_radii.append(radius)
            sphere_link_indices.append(link_idx)

    num_primitives = len(all_centers)
    if num_primitives == 0:
        sphere_centers_local = jnp.zeros((0, 3), dtype=jnp.float32)
        sphere_radii = jnp.zeros((0,), dtype=jnp.float32)
        sphere_link_indices_arr = jnp.zeros((0,), dtype=jnp.int32)
    else:
        sphere_centers_local = jnp.asarray(
            onp.stack(all_centers, axis=0), dtype=jnp.float32
        )
        sphere_radii = jnp.asarray(
            onp.asarray(all_radii, dtype=onp.float32), dtype=jnp.float32
        )
        sphere_link_indices_arr = jnp.asarray(
            onp.asarray(sphere_link_indices, dtype=onp.int32), dtype=jnp.int32
        )

    link_name_to_idx = {name: idx for idx, name in enumerate(link_names)}
    ignore_set: set[tuple[int, int]] = set()
    for joint in urdf.joint_map.values():
        if joint.parent in link_name_to_idx and joint.child in link_name_to_idx:
            i = link_name_to_idx[joint.parent]
            j = link_name_to_idx[joint.child]
            ignore_set.add((min(i, j), max(i, j)))
    for name_i, name_j in user_ignore_pairs:
        if name_i in link_name_to_idx and name_j in link_name_to_idx:
            i = link_name_to_idx[name_i]
            j = link_name_to_idx[name_j]
            ignore_set.add((min(i, j), max(i, j)))

    pair_i: list[int] = []
    pair_j: list[int] = []
    for i in range(num_primitives):
        li = sphere_link_indices[i]
        for j in range(i + 1, num_primitives):
            lj = sphere_link_indices[j]
            if li == lj:
                continue
            if (min(li, lj), max(li, lj)) in ignore_set:
                continue
            pair_i.append(i)
            pair_j.append(j)

    return MultiSphereCollision(
        num_primitives=num_primitives,
        num_links=num_links,
        link_names=link_names,
        sphere_centers_local=sphere_centers_local,
        sphere_radii=sphere_radii,
        sphere_link_indices=sphere_link_indices_arr,
        pair_i=jnp.asarray(onp.asarray(pair_i, dtype=onp.int32), dtype=jnp.int32),
        pair_j=jnp.asarray(onp.asarray(pair_j, dtype=onp.int32), dtype=jnp.int32),
    )
