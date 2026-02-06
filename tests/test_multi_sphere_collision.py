# pyright: basic

from typing import cast

import jax
import jax.numpy as jnp
import jaxlie
import pytest

from teleop_xr.ik.collision import MultiSphereCollision, RobotLike
from teleop_xr.ik.robots.teaarm import TeaArmRobot
from teleop_xr.ik.solver import PyrokiSolver


@pytest.fixture(scope="module")
def teaarm_robot() -> TeaArmRobot:
    return TeaArmRobot()


@pytest.fixture(scope="module")
def multi_sphere_coll(teaarm_robot: TeaArmRobot) -> MultiSphereCollision:
    return teaarm_robot.multi_sphere_coll


def test_construction(
    teaarm_robot: TeaArmRobot, multi_sphere_coll: MultiSphereCollision
) -> None:
    coll = multi_sphere_coll
    assert coll.num_primitives > 19
    assert bool(jnp.all(coll.sphere_radii > 0.0))
    assert bool(
        jnp.all(
            (coll.sphere_link_indices >= 0)
            & (coll.sphere_link_indices < coll.num_links)
        )
    )
    assert coll.pair_i.shape == coll.pair_j.shape

    pair_link_i = coll.sphere_link_indices[coll.pair_i]
    pair_link_j = coll.sphere_link_indices[coll.pair_j]
    assert bool(jnp.all(pair_link_i != pair_link_j))
    assert coll.link_names == teaarm_robot.robot.links.names


def test_jit_compatible(
    teaarm_robot: TeaArmRobot, multi_sphere_coll: MultiSphereCollision
) -> None:
    q0 = teaarm_robot.get_default_config()
    robot_model = cast(RobotLike, teaarm_robot.robot)
    dist = jax.jit(multi_sphere_coll.compute_self_collision_distance)(robot_model, q0)
    assert bool(jnp.all(jnp.isfinite(dist)))


def test_grad_compatible(
    teaarm_robot: TeaArmRobot, multi_sphere_coll: MultiSphereCollision
) -> None:
    q0 = teaarm_robot.get_default_config()
    robot_model = cast(RobotLike, teaarm_robot.robot)

    def objective(q: jax.Array) -> jax.Array:
        return multi_sphere_coll.compute_self_collision_distance(robot_model, q).sum()

    grad = jax.grad(objective)(q0)
    assert grad.shape == q0.shape
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_default_pose_no_penetration(
    teaarm_robot: TeaArmRobot, multi_sphere_coll: MultiSphereCollision
) -> None:
    q0 = teaarm_robot.get_default_config()
    robot_model = cast(RobotLike, teaarm_robot.robot)
    distances = multi_sphere_coll.compute_self_collision_distance(robot_model, q0)
    assert float(jnp.min(distances)) > -0.005


def test_collision_cost_integration(teaarm_robot: TeaArmRobot) -> None:
    solver = PyrokiSolver(teaarm_robot)
    q0 = teaarm_robot.get_default_config()

    q_next = solver.solve(
        target_L=jaxlie.SE3.identity(),
        target_R=jaxlie.SE3.identity(),
        target_Head=None,
        q_current=q0,
    )
    assert q_next.shape == q0.shape
    assert bool(jnp.all(jnp.isfinite(q_next)))
