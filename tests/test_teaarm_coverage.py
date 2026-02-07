import jaxlie
from unittest.mock import patch, MagicMock
from teleop_xr.ik.robots.teaarm import TeaArmRobot

TEAARM_URDF = """
<robot name="teaarm">
  <link name="waist_link"/>
  <link name="frame_left_arm_ee"/>
  <link name="frame_right_arm_ee"/>
  <link name="link1"/><link name="link2"/>
  <joint name="waist_yaw" type="revolute"><parent link="waist_link"/><child link="link1"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="waist_pitch" type="revolute"><parent link="link1"/><child link="link2"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="left_j1" type="revolute"><parent link="link2"/><child link="frame_left_arm_ee"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="right_j1" type="revolute"><parent link="link2"/><child link="frame_right_arm_ee"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
</robot>
"""


def _make_teaarm(tmp_path, urdf_text=None):
    """Helper: create a TeaArmRobot with mocked RAM, optionally override description."""
    urdf_file = tmp_path / "teaarm.urdf"
    urdf_file.write_text(urdf_text or TEAARM_URDF)

    mock_resource = MagicMock()
    mock_resource.path = urdf_file
    mock_resource.root = urdf_file.parent

    with (
        patch("teleop_xr.ik.robots.teaarm.ram.get_resource") as mock_get,
        patch("teleop_xr.ik.robots.teaarm.os.path.exists") as mock_exists,
    ):
        mock_get.return_value = mock_resource
        mock_exists.return_value = True
        robot = TeaArmRobot()
    return robot


def test_teaarm_build_costs_full_coverage(tmp_path):
    robot = _make_teaarm(tmp_path)
    robot.set_description(TEAARM_URDF)
    q = robot.get_default_config()
    target = jaxlie.SE3.identity()

    # Test all combinations to hit all branches

    # 1. Full targets + q_current
    costs = robot.build_costs(target, target, target, q_current=q)
    # manipulability + rest + pose_L + pose_R + pose_Head + limit + self_coll = 7
    assert len(costs) == 7

    # 2. No targets + no q_current
    costs = robot.build_costs(None, None, None, q_current=None)
    # manipulability + limit + self_coll = 3
    assert len(costs) == 3

    # 3. Only L target
    costs = robot.build_costs(target, None, None, q_current=None)
    assert len(costs) == 4

    # 4. Only R target
    costs = robot.build_costs(None, target, None, q_current=None)
    assert len(costs) == 4

    # 5. Only Head target
    costs = robot.build_costs(None, None, target, q_current=None)
    assert len(costs) == 4
