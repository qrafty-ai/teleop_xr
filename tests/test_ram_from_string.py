import hashlib
from teleop_xr import ram
import sys
from unittest.mock import MagicMock
import types


def test_from_string_basic(tmp_path):
    urdf_content = '<robot name="test_robot"><link name="base_link"/></robot>'
    cache_dir = tmp_path / "cache"

    urdf_path, mesh_path = ram.from_string(urdf_content, cache_dir=cache_dir)

    content_hash = hashlib.sha256(urdf_content.encode()).hexdigest()[:12]
    expected_path = cache_dir / "processed" / f"string_{content_hash}.urdf"

    assert urdf_path == expected_path
    assert urdf_path.exists()
    assert urdf_path.read_text() == urdf_content
    assert mesh_path is None


def test_from_string_with_meshes(tmp_path):
    # Create some dummy mesh files
    mesh_dir = tmp_path / "my_robot" / "meshes"
    mesh_dir.mkdir(parents=True)
    mesh_file = mesh_dir / "base.stl"
    mesh_file.write_text("dummy stl")

    urdf_content = f"""<robot name="test_robot">
    <link name="base_link">
        <visual>
            <geometry>
                <mesh filename="{mesh_file}"/>
            </geometry>
        </visual>
    </link>
</robot>"""

    cache_dir = tmp_path / "cache"
    urdf_path, mesh_path = ram.from_string(urdf_content, cache_dir=cache_dir)

    assert urdf_path.exists()
    # For a single mesh, we expect the directory of that mesh.
    assert mesh_path == str(mesh_dir)


def test_from_string_too_generic_prefix(tmp_path):
    # Test case where common path is /usr (too generic)
    urdf_content = """<robot name="test_robot">
    <link name="l1"><visual><geometry><mesh filename="/usr/share/m1.stl"/></geometry></visual></link>
    <link name="l2"><visual><geometry><mesh filename="/usr/lib/m2.stl"/></geometry></visual></link>
</robot>"""

    cache_dir = tmp_path / "cache"
    urdf_path, mesh_path = ram.from_string(urdf_content, cache_dir=cache_dir)
    assert mesh_path is None


def test_from_string_package_uri(tmp_path, monkeypatch):
    # Mock a package path
    pkg_path = tmp_path / "my_pkg"
    mesh_dir = pkg_path / "meshes"
    mesh_dir.mkdir(parents=True)
    mesh_file = mesh_dir / "link.stl"
    mesh_file.write_text("dummy")

    urdf_content = """<robot name="test_robot">
    <link name="base_link">
        <visual>
            <geometry>
                <mesh filename="package://my_pkg/meshes/link.stl"/>
            </geometry>
        </visual>
    </link>
</robot>"""

    # Correctly mock ament_index_python.packages.get_package_share_directory
    mock_packages = types.ModuleType("ament_index_python.packages")

    def get_pkg_share(pkg):
        if pkg == "my_pkg":
            return str(pkg_path)
        raise Exception("Package not found")

    mock_packages.get_package_share_directory = get_pkg_share

    monkeypatch.setitem(sys.modules, "ament_index_python", MagicMock())
    monkeypatch.setitem(sys.modules, "ament_index_python.packages", mock_packages)

    cache_dir = tmp_path / "cache"
    urdf_path, mesh_path = ram.from_string(urdf_content, cache_dir=cache_dir)

    content = urdf_path.read_text()
    assert "package://" not in content
    assert str(mesh_file) in content
    assert mesh_path == str(mesh_dir)
