# pyright: basic, reportMissingTypeStubs=false
import io

import numpy as onp
import yourdfpy

from teleop_xr.ik.collision import build_multi_sphere_collision


URDF_TEXT = """
<robot name="adjacency_ignore_order_test">
  <link name="torso_link">
    <collision><geometry><box size="0.08 0.08 0.08"/></geometry></collision>
  </link>
  <link name="left_arm_l1">
    <collision><geometry><box size="0.06 0.06 0.06"/></geometry></collision>
  </link>
  <link name="left_arm_l2">
    <collision><geometry><box size="0.05 0.05 0.05"/></geometry></collision>
  </link>
  <link name="frame_left_arm_ee">
    <collision><geometry><box size="0.04 0.04 0.04"/></geometry></collision>
  </link>
  <link name="right_arm_l1">
    <collision><geometry><box size="0.06 0.06 0.06"/></geometry></collision>
  </link>

  <joint name="torso_to_left_l1" type="fixed">
    <parent link="torso_link"/>
    <child link="left_arm_l1"/>
  </joint>
  <joint name="left_l1_to_left_l2" type="fixed">
    <parent link="left_arm_l1"/>
    <child link="left_arm_l2"/>
  </joint>
  <joint name="left_l2_to_left_ee" type="fixed">
    <parent link="left_arm_l2"/>
    <child link="frame_left_arm_ee"/>
  </joint>
  <joint name="torso_to_right_l1" type="fixed">
    <parent link="torso_link"/>
    <child link="right_arm_l1"/>
  </joint>
</robot>
"""


def _pair_key(name_i: str, name_j: str) -> tuple[str, str]:
    return (name_i, name_j) if name_i < name_j else (name_j, name_i)


def _collect_active_link_pairs() -> set[tuple[str, str]]:
    urdf = yourdfpy.URDF.load(io.StringIO(URDF_TEXT))
    coll = build_multi_sphere_collision(urdf, ignore_adj_order=2)

    sphere_link_indices = onp.asarray(
        coll.sphere_link_indices, dtype=onp.int32
    ).tolist()
    pair_i = onp.asarray(coll.pair_i, dtype=onp.int32).tolist()
    pair_j = onp.asarray(coll.pair_j, dtype=onp.int32).tolist()

    active_pairs: set[tuple[str, str]] = set()
    for prim_i, prim_j in zip(pair_i, pair_j):
        link_i = coll.link_names[sphere_link_indices[prim_i]]
        link_j = coll.link_names[sphere_link_indices[prim_j]]
        active_pairs.add(_pair_key(link_i, link_j))
    return active_pairs


def test_ignore_adj_order_two_uses_ancestor_chain_with_ee_torso_exemption() -> None:
    active_pairs = _collect_active_link_pairs()

    assert _pair_key("left_arm_l1", "torso_link") not in active_pairs
    assert _pair_key("left_arm_l2", "torso_link") not in active_pairs
    assert _pair_key("frame_left_arm_ee", "torso_link") in active_pairs
    assert _pair_key("left_arm_l1", "right_arm_l1") in active_pairs
