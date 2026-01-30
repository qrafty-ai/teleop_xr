import os
import tempfile
import mujoco
import pytest
from teleop_xr.ik.model_loader import load_model


def test_load_urdf():
    urdf_content = """
    <robot name="test">
        <link name="base_link">
            <inertial>
                <origin xyz="0 0 0" rpy="0 0 0"/>
                <mass value="10.0"/>
                <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
            </inertial>
            <visual>
                <geometry>
                    <sphere radius="0.1"/>
                </geometry>
            </visual>
        </link>
        <link name="link1">
            <inertial>
                <origin xyz="0 0 0" rpy="0 0 0"/>
                <mass value="10.0"/>
                <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
            </inertial>
            <visual>
                <geometry>
                    <sphere radius="0.1"/>
                </geometry>
            </visual>
        </link>
        <joint name="joint1" type="revolute">
            <parent link="base_link"/>
            <child link="link1"/>
            <axis xyz="0 0 1"/>
            <limit lower="-3.14" upper="3.14" effort="10" velocity="1"/>
        </joint>
    </robot>
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        urdf_path = os.path.join(tmpdir, "test.urdf")
        with open(urdf_path, "w") as f:
            f.write(urdf_content)

        model = load_model(urdf_path)
        assert isinstance(model, mujoco.MjModel)
        assert model.nq == 8  # 7 for freejoint + 1 for revolute joint


def test_load_mjcf():
    mjcf_content = """
    <mujoco model="test">
        <worldbody>
            <body name="body" pos="0 0 0">
                <joint name="joint" type="hinge"/>
                <geom size="0.1" type="sphere"/>
            </body>
        </worldbody>
    </mujoco>
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = os.path.join(tmpdir, "test.xml")
        with open(xml_path, "w") as f:
            f.write(mjcf_content)

        model = load_model(xml_path)
        assert isinstance(model, mujoco.MjModel)
        assert model.nq == 1


def test_load_invalid_extension():
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_model("test.txt")
