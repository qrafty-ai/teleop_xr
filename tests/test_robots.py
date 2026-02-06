import pytest
import jax.numpy as jnp
import jaxlie
from unittest.mock import patch
from teleop_xr.ik.robot import BaseRobot
from teleop_xr.ik.robots.teaarm import TeaArmRobot
from teleop_xr.ik.robots.h1_2 import UnitreeH1Robot

TEAARM_URDF = """
<robot name="teaarm">
  <link name="waist_link"/>
  <link name="frame_left_arm_ee"/>
  <link name="frame_right_arm_ee"/>
  <link name="link1"/><link name="link2"/><link name="link3"/><link name="link4"/>
  <link name="link5"/><link name="link6"/><link name="link7"/><link name="link8"/>
  <link name="link9"/><link name="link10"/><link name="link11"/><link name="link12"/>
  <link name="link13"/><link name="link14"/>
  <joint name="waist_yaw" type="revolute"><parent link="waist_link"/><child link="link1"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="waist_pitch" type="revolute"><parent link="link1"/><child link="link2"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="left_j1" type="revolute"><parent link="link2"/><child link="link3"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="right_j1" type="revolute"><parent link="link2"/><child link="link4"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="left_j2" type="revolute"><parent link="link3"/><child link="link5"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="right_j2" type="revolute"><parent link="link4"/><child link="link6"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="left_j3" type="revolute"><parent link="link5"/><child link="link7"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="right_j3" type="revolute"><parent link="link6"/><child link="link8"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="left_j4" type="revolute"><parent link="link7"/><child link="link9"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="right_j4" type="revolute"><parent link="link8"/><child link="link10"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="left_j5" type="revolute"><parent link="link9"/><child link="link11"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="right_j5" type="revolute"><parent link="link10"/><child link="link12"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="left_j6" type="revolute"><parent link="link11"/><child link="link13"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="right_j6" type="revolute"><parent link="link12"/><child link="link14"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="left_j7" type="revolute"><parent link="link13"/><child link="frame_left_arm_ee"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
  <joint name="right_j7" type="revolute"><parent link="link14"/><child link="frame_right_arm_ee"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint>
</robot>
"""

H1_URDF = """
<robot name="h1_2">
  <link name="torso_link"/>
  <link name="L_hand_base_link"/>
  <link name="R_hand_base_link"/>
  <joint name="left_hip_yaw_joint" type="revolute"><parent link="torso_link"/><child link="l_leg"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <link name="l_leg"/>
  <joint name="j1" type="revolute"><parent link="torso_link"/><child link="L_hand_base_link"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="j2" type="revolute"><parent link="torso_link"/><child link="R_hand_base_link"/><limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>
"""


class MockBaseRobot(BaseRobot):
    @property
    def actuated_joint_names(self):
        return ["j1"]

    @property
    def joint_var_cls(self):
        return None

    def get_vis_config(self):
        return None

    def forward_kinematics(self, config):
        return {"left": jaxlie.SE3.identity()}

    def get_default_config(self):
        return jnp.zeros(1)

    def build_costs(self, target_L, target_R, target_Head, q_current=None):
        return []


def test_base_robot_properties():
    robot = MockBaseRobot()
    assert robot.orientation.as_matrix().shape == (3, 3)
    assert robot.base_to_ros.as_matrix().shape == (3, 3)
    assert robot.ros_to_base.as_matrix().shape == (3, 3)
    assert robot.supported_frames == {"left", "right", "head"}
    assert robot.default_speed_ratio == 1.0


def test_teaarm_robot(tmp_path):
    robot = TeaArmRobot(urdf_string=TEAARM_URDF)
    assert robot.supported_frames == {"left", "right", "head"}
    assert len(robot.actuated_joint_names) == 16
    assert robot.joint_var_cls is not None
    q = robot.get_default_config()
    fk = robot.forward_kinematics(q)
    assert "left" in fk
    assert hasattr(robot, "robot_coll")
    costs = robot.build_costs(
        jaxlie.SE3.identity(), jaxlie.SE3.identity(), jaxlie.SE3.identity(), q_current=q
    )
    assert len(costs) == 7
    assert robot.get_vis_config() is None

    dummy_urdf = tmp_path / "teaarm.urdf"
    dummy_urdf.write_text(TEAARM_URDF)
    with (
        patch("teleop_xr.ik.robots.teaarm.ram.get_resource") as mock_get,
        patch("teleop_xr.ik.robots.teaarm.os.path.exists") as mock_exists,
    ):
        mock_get.return_value = dummy_urdf
        mock_exists.return_value = True
        robot_local = TeaArmRobot()
        assert robot_local.urdf_path == str(dummy_urdf)
        assert robot_local.get_vis_config() is not None


def test_teaarm_robot_errors(tmp_path):
    MISSING_L_EE_URDF = """<robot name="m"><link name="waist_link"/><link name="frame_right_arm_ee"/><joint name="w" type="revolute"><parent link="waist_link"/><child link="c"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint><link name="c"/></robot>"""
    with pytest.raises(ValueError, match="Link frame_left_arm_ee not found"):
        TeaArmRobot(urdf_string=MISSING_L_EE_URDF)
    MISSING_R_EE_URDF = """<robot name="m"><link name="waist_link"/><link name="frame_left_arm_ee"/><joint name="w" type="revolute"><parent link="waist_link"/><child link="c"/><limit effort="1" lower="-1" upper="1" velocity="1"/></joint><link name="c"/></robot>"""
    with pytest.raises(ValueError, match="Link frame_right_arm_ee not found"):
        TeaArmRobot(urdf_string=MISSING_R_EE_URDF)

    with (
        patch("teleop_xr.ik.robots.teaarm.ram.get_resource") as mock_get,
        patch("teleop_xr.ik.robots.teaarm.os.path.exists") as mock_exists,
    ):
        mock_get.return_value = tmp_path / "nonexistent.urdf"
        mock_exists.return_value = False
        with pytest.raises(FileNotFoundError, match="TeaArm URDF not found"):
            TeaArmRobot()


def test_h1_robot(tmp_path):
    dummy_urdf = tmp_path / "h1.urdf"
    dummy_urdf.write_text(H1_URDF)
    with patch("teleop_xr.ik.robots.h1_2.ram.get_resource") as mock_get:
        mock_get.return_value = dummy_urdf
        robot = UnitreeH1Robot()
        assert robot.supported_frames == {"left", "right", "head"}
        assert robot.default_speed_ratio == 1.2
        assert robot.joint_var_cls is not None
        q = robot.get_default_config()
        assert len(robot.actuated_joint_names) > 0
        fk = robot.forward_kinematics(q)
        assert "left" in fk
        costs = robot.build_costs(
            jaxlie.SE3.identity(),
            jaxlie.SE3.identity(),
            jaxlie.SE3.identity(),
            q_current=q,
        )
        assert len(costs) > 0
        assert robot.get_vis_config() is not None
        robot.urdf_path = ""
        assert robot.get_vis_config() is None
