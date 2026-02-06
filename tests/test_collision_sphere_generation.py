import numpy as np
import trimesh
from unittest.mock import patch
from teleop_xr.ik.collision import (
    generate_collision_spheres,
    validate_sphere_decomposition,
)


def test_generate_from_primitives():
    urdf_str = """
    <robot name="test">
      <link name="link_sphere">
        <collision>
          <geometry><sphere radius="0.1"/></geometry>
        </collision>
      </link>
      <link name="link_box">
        <collision>
          <geometry><box size="0.2 0.2 0.2"/></geometry>
        </collision>
      </link>
      <link name="link_cylinder">
        <collision>
          <geometry><cylinder radius="0.05" length="0.2"/></geometry>
        </collision>
      </link>
    </robot>
    """
    decomp = generate_collision_spheres(urdf_string=urdf_str, n_spheres_per_link=4)
    assert validate_sphere_decomposition(decomp)

    assert "link_sphere" in decomp
    # Primitives like sphere are handled analytically
    assert len(decomp["link_sphere"]["radii"]) == 1
    assert np.allclose(decomp["link_sphere"]["radii"][0], 0.1)

    assert "link_box" in decomp
    # Box is converted to mesh and spherized
    assert len(decomp["link_box"]["radii"]) >= 1

    assert "link_cylinder" in decomp
    # Cylinder is converted to mesh and spherized
    assert len(decomp["link_cylinder"]["radii"]) >= 1


def test_generate_with_padding():
    urdf_str = """
    <robot name="test">
      <link name="link1">
        <collision>
          <geometry><sphere radius="0.1"/></geometry>
        </collision>
      </link>
    </robot>
    """
    padding = 0.02
    decomp = generate_collision_spheres(urdf_string=urdf_str, padding=padding)
    # 0.1 + 0.02 = 0.12
    assert np.allclose(decomp["link1"]["radii"][0], 0.12)


def test_mesh_decomposition_ballpark():
    # Long box to test decomposition spread
    urdf_str = """
    <robot name="test">
      <link name="link1">
        <collision>
          <geometry><box size="1.0 0.1 0.1"/></geometry>
        </collision>
      </link>
    </robot>
    """
    decomp = generate_collision_spheres(urdf_string=urdf_str, n_spheres_per_link=4)
    centers = np.array(decomp["link1"]["centers"])
    assert len(centers) == 4
    # Check if they are spread along X axis (longest axis)
    assert np.max(centers[:, 0]) - np.min(centers[:, 0]) > 0.5


def test_deterministic_sorting():
    urdf_str = """
    <robot name="test">
      <link name="link1">
        <collision>
          <geometry><box size="1.0 1.0 1.0"/></geometry>
        </collision>
      </link>
    </robot>
    """
    decomp1 = generate_collision_spheres(urdf_string=urdf_str, n_spheres_per_link=8)
    decomp2 = generate_collision_spheres(urdf_string=urdf_str, n_spheres_per_link=8)

    assert decomp1 == decomp2


def test_origin_handling():
    urdf_str = """
    <robot name="test">
      <link name="link1">
        <collision>
          <origin xyz="1 0 0"/>
          <geometry><sphere radius="0.1"/></geometry>
        </collision>
      </link>
    </robot>
    """
    decomp = generate_collision_spheres(urdf_string=urdf_str)
    assert np.allclose(decomp["link1"]["centers"][0], [1.0, 0.0, 0.0])


def test_strict_over_approximation():
    # Test that vertices of a cube are covered by spheres
    urdf_str = """
    <robot name="test">
      <link name="link1">
        <collision>
          <geometry><box size="0.2 0.2 0.2"/></geometry>
        </collision>
      </link>
    </robot>
    """
    decomp = generate_collision_spheres(urdf_string=urdf_str, n_spheres_per_link=8)
    centers = np.array(decomp["link1"]["centers"])
    radii = np.array(decomp["link1"]["radii"])

    # Create a cube mesh to get its vertices
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    vertices = mesh.vertices

    for v in vertices:
        # Check if vertex is inside at least one sphere
        dists = np.linalg.norm(centers - v, axis=1)
        assert np.any(dists <= radii + 1e-4)


def test_multiple_collision_elements():
    urdf_str = """
    <robot name="test">
      <link name="link1">
        <collision>
          <origin xyz="0.2 0 0"/>
          <geometry><box size="0.1 0.1 0.1"/></geometry>
        </collision>
        <collision>
          <origin xyz="-0.2 0 0"/>
          <geometry><sphere radius="0.05"/></geometry>
        </collision>
      </link>
    </robot>
    """
    decomp = generate_collision_spheres(urdf_string=urdf_str, n_spheres_per_link=4)
    assert len(decomp["link1"]["centers"]) == 4
    centers = np.array(decomp["link1"]["centers"])
    # Should have centers near both collision elements
    assert np.any(centers[:, 0] > 0.1)
    assert np.any(centers[:, 0] < -0.1)


def test_decomposition_fallback():
    urdf_str = """
    <robot name="test">
      <link name="link1">
        <collision>
          <geometry><box size="0.2 0.2 0.2"/></geometry>
        </collision>
      </link>
    </robot>
    """
    # Mock ballpark.spherize to fail
    with patch("ballpark.spherize", side_effect=RuntimeError("Ballpark failed")):
        decomp = generate_collision_spheres(urdf_string=urdf_str, n_spheres_per_link=4)

    # Should fallback to bounding sphere (1 sphere)
    assert len(decomp["link1"]["centers"]) == 1
    # Bounding sphere of 0.2 cube has radius ~ 0.1732 (sqrt(3)*0.1)
    assert decomp["link1"]["radii"][0] > 0.15


def test_coverage_check_failure_fallback():
    urdf_str = """
    <robot name="test">
      <link name="link1">
        <collision>
          <geometry><box size="1.0 0.1 0.1"/></geometry>
        </collision>
      </link>
    </robot>
    """

    # Mock ballpark to return spheres that don't cover the mesh
    # centers = [[0,0,0]], radii = [[0.01]] for a 1.0 box
    class MockSphere:
        def __init__(self, center, radius):
            self.center = center
            self.radius = radius

    with patch("ballpark.spherize", return_value=[MockSphere([0, 0, 0], 0.01)]):
        decomp = generate_collision_spheres(urdf_string=urdf_str, n_spheres_per_link=1)

    # Should detect failure and fallback to bounding sphere
    assert len(decomp["link1"]["centers"]) == 1
    # Bounding sphere of 1.0x0.1x0.1 box should be much larger than 0.01
    assert decomp["link1"]["radii"][0] > 0.4
