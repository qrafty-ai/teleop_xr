# ruff: noqa: F722, F821
from __future__ import annotations

from typing import Protocol, cast

import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
from jaxtyping import Array, Float, Int


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
