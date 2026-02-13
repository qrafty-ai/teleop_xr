from teleop_xr.ik.robots.franka import FrankaRobot


def test_franka_gripper_joint_fixing():
    urdf_str = """<robot name="panda">
      <link name="panda_link0"/><link name="panda_link1"/><link name="panda_link2"/>
      <joint name="panda_finger_joint1" type="revolute"><parent link="panda_link0"/><child link="panda_link1"/></joint>
      <joint name="panda_finger_joint2" type="revolute"><parent link="panda_link1"/><child link="panda_link2"/></joint>
      <link name="panda_hand"/>
      <joint name="panda_hand_joint" type="fixed"><parent link="panda_link2"/><child link="panda_hand"/></joint>
    </robot>"""

    robot = FrankaRobot(urdf_string=urdf_str)
    assert robot is not None
    assert "panda_finger_joint1" not in robot.actuated_joint_names
    assert "panda_finger_joint2" not in robot.actuated_joint_names
