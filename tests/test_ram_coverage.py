import sys
import types
from unittest.mock import MagicMock
import pytest
from teleop_xr import ram


def test_resolve_package_metapackage_subdir(tmp_path):
    # Case where package is an immediate subdirectory
    repo_root = tmp_path / "repo"
    pkg_dir = repo_root / "my_pkg"
    pkg_dir.mkdir(parents=True)
    (pkg_dir / "package.xml").write_text("<name>my_pkg</name>")

    with ram._ram_repo_context(repo_root):
        assert ram._resolve_package("my_pkg") == str(pkg_dir)


def test_resolve_package_ros_env(monkeypatch):
    # Case where package is in ROS 2 env
    mock_packages = types.ModuleType("ament_index_python.packages")
    mock_packages.get_package_share_directory = MagicMock(
        return_value="/opt/ros/share/pkg"
    )

    monkeypatch.setitem(sys.modules, "ament_index_python", MagicMock())
    monkeypatch.setitem(sys.modules, "ament_index_python.packages", mock_packages)

    # Ensure _CURRENT_REPO_ROOT is None
    with ram._ram_repo_context(None):
        assert ram._resolve_package("any_pkg") == "/opt/ros/share/pkg"
        mock_packages.get_package_share_directory.assert_called_once_with("any_pkg")


def test_resolve_package_not_found(monkeypatch):
    # Case where package is NOT found anywhere
    monkeypatch.setitem(sys.modules, "ament_index_python", MagicMock())
    mock_packages = types.ModuleType("ament_index_python.packages")
    # Mock it to raise Exception
    mock_packages.get_package_share_directory = MagicMock(
        side_effect=Exception("Not found")
    )
    monkeypatch.setitem(sys.modules, "ament_index_python.packages", mock_packages)

    with ram._ram_repo_context(None):
        with pytest.raises(ValueError, match="not found"):
            ram._resolve_package("missing_pkg")


def test_replace_dae_conversion_failure(tmp_path, monkeypatch):
    # Case where conversion fails and returns original path
    dae_file = tmp_path / "mesh.dae"
    dae_file.touch()

    # Mock _convert_dae_to_glb to return original path
    monkeypatch.setattr(ram, "_convert_dae_to_glb", lambda p: p)

    urdf = f'<mesh filename="{dae_file}"/>'
    result = ram._replace_dae_with_glb(urdf)
    assert result == urdf  # No change if glb_path == original_path


def test_get_resource_urdf_no_resolve(tmp_path):
    # Case where resolve_packages is False for plain URDF
    urdf_file = tmp_path / "robot.urdf"
    urdf_content = '<robot name="test"/>'
    urdf_file.write_text(urdf_content)

    # Should return original file path if no processing needed
    res = ram.get_resource(
        repo_root=tmp_path, path_inside_repo="robot.urdf", resolve_packages=False
    )
    assert res == urdf_file


def test_from_string_ram_internal_resolve(tmp_path):
    # Case where from_string uses internal RAM resolver
    repo_root = tmp_path / "repo"
    pkg_dir = repo_root / "my_pkg"
    pkg_dir.mkdir(parents=True)

    urdf_content = "package://my_pkg/mesh.stl"

    with ram._ram_repo_context(repo_root):
        urdf_path, mesh_path = ram.from_string(
            urdf_content, cache_dir=tmp_path / "cache"
        )
        assert str(pkg_dir / "mesh.stl") in urdf_path.read_text()


def test_from_string_ros_resolve(tmp_path, monkeypatch):
    # Case where from_string uses ROS resolver
    mock_parent = MagicMock()
    mock_packages = MagicMock()
    mock_packages.get_package_share_directory = MagicMock(return_value="/opt/ros/pkg")
    mock_parent.packages = mock_packages

    monkeypatch.setitem(sys.modules, "ament_index_python", mock_parent)
    monkeypatch.setitem(sys.modules, "ament_index_python.packages", mock_packages)

    urdf_content = "package://my_pkg/mesh.stl"

    with ram._ram_repo_context(None):
        urdf_path, mesh_path = ram.from_string(
            urdf_content, cache_dir=tmp_path / "cache"
        )
        assert "/opt/ros/pkg/mesh.stl" in urdf_path.read_text()


def test_from_string_commonpath_error(tmp_path, monkeypatch):
    # Case where commonpath raises ValueError
    urdf_content = (
        '<mesh filename="/abs/path/1.stl"/><mesh filename="/other/path/2.stl"/>'
    )

    # Mock os.path.commonpath to raise ValueError
    monkeypatch.setattr(ram.os.path, "commonpath", MagicMock(side_effect=ValueError))

    urdf_path, mesh_path = ram.from_string(urdf_content, cache_dir=tmp_path / "cache")
    assert mesh_path is None


def test_from_string_ram_internal_resolve_fail(tmp_path):
    # Case where from_string uses internal RAM resolver but it fails
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    urdf_content = "package://missing/mesh.stl"

    with ram._ram_repo_context(repo_root):
        # This should hit the except ValueError block in from_string
        ram.from_string(urdf_content, cache_dir=tmp_path / "cache")


def test_from_string_no_pkg_found(tmp_path, monkeypatch):
    # Case where pkg is not found in from_string
    monkeypatch.setitem(sys.modules, "ament_index_python", MagicMock())
    mock_packages = types.ModuleType("ament_index_python.packages")
    # Mock it to raise Exception
    mock_packages.get_package_share_directory = MagicMock(
        side_effect=Exception("Not found")
    )
    monkeypatch.setitem(sys.modules, "ament_index_python.packages", mock_packages)

    urdf_content = "package://nonexistent/mesh.stl"

    with ram._ram_repo_context(None):
        urdf_path, mesh_path = ram.from_string(
            urdf_content, cache_dir=tmp_path / "cache"
        )
        assert "package://nonexistent/mesh.stl" in urdf_path.read_text()


def test_get_resource_urdf_no_resolve_packages_but_glb(tmp_path, monkeypatch):
    # Case where resolve_packages is False but convert_dae_to_glb is True for plain URDF
    urdf_file = tmp_path / "robot.urdf"
    urdf_content = '<mesh filename="mesh.dae"/>'
    urdf_file.write_text(urdf_content)

    # Mock _replace_dae_with_glb to see if it's called
    mock_replace = MagicMock(return_value=urdf_content)
    monkeypatch.setattr(ram, "_replace_dae_with_glb", mock_replace)

    ram.get_resource(
        repo_root=tmp_path,
        path_inside_repo="robot.urdf",
        resolve_packages=False,
        convert_dae_to_glb=True,
    )
    mock_replace.assert_called_once()
