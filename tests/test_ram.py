import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

try:
    import git  # noqa: F401
except ImportError:
    pytest.skip("git not installed", allow_module_level=True)

from teleop_xr import ram  # noqa: E402


@pytest.fixture
def mock_cache_dir(tmp_path):
    """Fixture to provide a temporary cache directory for RAM."""
    cache_dir = tmp_path / "ram_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def temp_git_repo():
    """Fixture to create a local git repository with dummy assets."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = Path(tmp_dir) / "test_repo"
        repo_path.mkdir()

        # Create dummy assets
        urdf_path = repo_path / "robot.urdf"
        urdf_path.write_text(
            '<robot name="test_robot"><link name="base_link"/></robot>'
        )

        xacro_path = repo_path / "robot.xacro"
        xacro_path.write_text(
            '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="test_xacro"><link name="base"/></robot>'
        )

        mesh_dir = repo_path / "meshes"
        mesh_dir.mkdir()
        (mesh_dir / "base.stl").write_text("dummy mesh content")

        # Initialize git repo
        with git.Repo.init(repo_path) as repo:
            repo.index.add(["robot.urdf", "robot.xacro", "meshes/base.stl"])
            repo.index.commit("Initial commit")

        yield repo_path


def test_get_repo(temp_git_repo, mock_cache_dir):
    """Test get_repo function."""
    repo_url = temp_git_repo.as_uri()
    repo_dir = ram.get_repo(repo_url, cache_dir=mock_cache_dir)
    assert repo_dir.exists()
    assert (repo_dir / "robot.urdf").exists()


def test_fetch_git_repo(temp_git_repo, mock_cache_dir):
    """Test that RAM can fetch a git repo into cache via get_resource."""
    repo_url = temp_git_repo.as_uri()
    asset_path = ram.get_resource(
        repo_url=repo_url, path_inside_repo="robot.urdf", cache_dir=mock_cache_dir
    )
    assert asset_path.exists()
    assert asset_path.suffix == ".urdf"
    assert asset_path.stem.startswith("robot")
    assert (mock_cache_dir / "repos").exists()


def test_package_path_replacement(temp_git_repo, mock_cache_dir):
    """Test that package:// paths are replaced with relative paths."""
    repo_path = temp_git_repo
    urdf_content = """<robot name="test">
    <link name="base">
        <visual>
            <geometry>
                <mesh filename="package://test_repo/meshes/base.stl"/>
            </geometry>
        </visual>
    </link>
</robot>"""
    (repo_path / "robot_pkg.urdf").write_text(urdf_content)
    with git.Repo(repo_path) as repo:
        repo.index.add(["robot_pkg.urdf"])
        repo.index.commit("Add pkg urdf")

    repo_url = repo_path.as_uri()
    asset_path = ram.get_resource(
        repo_url=repo_url, path_inside_repo="robot_pkg.urdf", cache_dir=mock_cache_dir
    )

    content = asset_path.read_text()
    assert "package://test_repo/" not in content
    # Should be replaced by path relative to repo root
    assert "meshes/base.stl" in content


def test_xacro_conversion(temp_git_repo, mock_cache_dir):
    """Test that xacro files are automatically converted to URDF."""
    repo_url = temp_git_repo.as_uri()
    asset_path = ram.get_resource(
        repo_url=repo_url, path_inside_repo="robot.xacro", cache_dir=mock_cache_dir
    )
    assert asset_path.suffix == ".urdf"
    assert "test_xacro" in asset_path.read_text()


def test_process_xacro(temp_git_repo):
    """Test process_xacro directly."""
    xacro_path = temp_git_repo / "robot.xacro"
    urdf_xml = ram.process_xacro(xacro_path, repo_root=temp_git_repo)
    assert "test_xacro" in urdf_xml
    assert "package://" not in urdf_xml


def test_get_cache_root():
    """Test get_cache_root returns the default path."""
    cache_root = ram.get_cache_root()
    assert cache_root.resolve() == (Path.home() / ".cache" / "ram").resolve()
    assert cache_root.exists()


def test_get_repo_default_cache(temp_git_repo, monkeypatch):
    """Test get_repo uses default cache if not provided."""
    # Mock home directory to avoid polluting real home
    with tempfile.TemporaryDirectory() as tmp_home:
        monkeypatch.setenv("HOME", tmp_home)
        monkeypatch.setenv("USERPROFILE", tmp_home)
        repo_url = temp_git_repo.as_uri()
        repo_dir = ram.get_repo(repo_url)
        assert repo_dir.exists()
        assert repo_dir.resolve().is_relative_to(Path(tmp_home).resolve())


def test_update_existing_repo(temp_git_repo, mock_cache_dir):
    """Test that RAM correctly updates an existing repo in cache."""
    repo_url = temp_git_repo.as_uri()

    # First fetch
    repo_dir = ram.get_repo(repo_url, cache_dir=mock_cache_dir)

    # Modify the source repo
    new_file = temp_git_repo / "new_file.txt"
    new_file.write_text("new content")
    with git.Repo(temp_git_repo) as repo:
        repo.index.add(["new_file.txt"])
        repo.index.commit("Add new file")

    # Second fetch should update
    repo_dir2 = ram.get_repo(repo_url, cache_dir=mock_cache_dir)
    assert repo_dir2 == repo_dir
    assert (repo_dir / "new_file.txt").exists()


def test_update_with_branch(temp_git_repo, mock_cache_dir):
    """Test updating an existing repo with a specific branch."""
    repo_url = temp_git_repo.as_uri()
    with git.Repo(temp_git_repo) as repo:
        # Identify default branch (master or main)
        default_branch = repo.active_branch.name

        # Create a new branch
        repo.create_head("feature")
        repo.git.checkout("feature")
        feature_file = temp_git_repo / "feature.txt"
        feature_file.write_text("feature content")
        repo.index.add(["feature.txt"])
        repo.index.commit("Add feature file")
        repo.git.checkout(default_branch)  # Switch back to default

    # Fetch default branch
    ram.get_repo(repo_url, branch=default_branch, cache_dir=mock_cache_dir)

    # Fetch feature branch
    repo_dir = ram.get_repo(repo_url, branch="feature", cache_dir=mock_cache_dir)
    assert (repo_dir / "feature.txt").exists()

    # Test switching back to default
    repo_dir = ram.get_repo(repo_url, branch=default_branch, cache_dir=mock_cache_dir)
    assert not (repo_dir / "feature.txt").exists()


def test_get_resource_default_cache(temp_git_repo, monkeypatch):
    """Test get_resource uses default cache if not provided."""
    with tempfile.TemporaryDirectory() as tmp_home:
        monkeypatch.setenv("HOME", tmp_home)
        monkeypatch.setenv("USERPROFILE", tmp_home)
        repo_url = temp_git_repo.as_uri()
        asset_path = ram.get_resource(repo_url, "robot.urdf")
        assert asset_path.exists()
        assert asset_path.resolve().is_relative_to(Path(tmp_home).resolve())


def test_asset_not_found(temp_git_repo, mock_cache_dir):
    """Test that RAM raises FileNotFoundError if asset is missing."""
    repo_url = temp_git_repo.as_uri()
    with pytest.raises(FileNotFoundError):
        ram.get_resource(
            repo_url=repo_url, path_inside_repo="missing.urdf", cache_dir=mock_cache_dir
        )


def test_get_asset_alias(temp_git_repo, mock_cache_dir):
    """Test that get_asset is indeed an alias for get_resource."""
    repo_url = temp_git_repo.as_uri()
    asset_path = ram.get_asset(
        repo_url=repo_url, path_inside_repo="robot.urdf", cache_dir=mock_cache_dir
    )
    assert asset_path.exists()


def test_checkout_new_remote_branch(temp_git_repo, mock_cache_dir):
    """Test that RAM fetches if checkout fails (e.g. new remote branch)."""
    repo_url = temp_git_repo.as_uri()

    # 1. Initial fetch of default branch
    ram.get_repo(repo_url, cache_dir=mock_cache_dir)

    # 2. Create a new branch in the SOURCE repo
    with git.Repo(temp_git_repo) as repo:
        new_branch_name = "new-remote-branch"
        repo.create_head(new_branch_name)

    # 3. Fetch the new branch.
    # Local cache repo doesn't know about it yet, so initial checkout fails.
    # It should then fetch and succeed.
    repo_dir = ram.get_repo(repo_url, branch=new_branch_name, cache_dir=mock_cache_dir)
    assert repo_dir.exists()

    # Verify we are on the correct branch
    with git.Repo(repo_dir) as cache_repo:
        assert cache_repo.active_branch.name == new_branch_name


def test_resolve_package_in_repo(temp_git_repo, monkeypatch):
    """Test _resolve_package correctly finds a package in the repo."""
    # Create a dummy package structure
    (temp_git_repo / "my_pkg").mkdir()
    (temp_git_repo / "my_pkg" / "package.xml").write_text("<package>my_pkg</package>")

    # Mock _CURRENT_REPO_ROOT
    with ram._ram_repo_context(temp_git_repo):
        path = ram._resolve_package("my_pkg")
        assert Path(path) == temp_git_repo / "my_pkg"


def test_resolve_package_not_found(temp_git_repo):
    """Test _resolve_package raises ValueError if package is missing."""
    with ram._ram_repo_context(temp_git_repo):
        with pytest.raises(ValueError):
            ram._resolve_package("nonexistent_pkg")


def test_mock_eval_find():
    """Test _mock_eval_find calls _resolve_package correctly."""
    # This just ensures the wiring is correct
    with patch("teleop_xr.ram._resolve_package") as mock_resolve:
        ram._mock_eval_find("foo")
        mock_resolve.assert_called_with("foo")


def test_get_resource_no_resolve_packages(temp_git_repo, mock_cache_dir):
    """Test get_resource with resolve_packages=False."""
    repo_url = temp_git_repo.as_uri()

    # Create a urdf with package://
    (temp_git_repo / "pkg.urdf").write_text(
        '<mesh filename="package://my_pkg/mesh.stl"/>'
    )
    with git.Repo(temp_git_repo) as repo:
        repo.index.add(["pkg.urdf"])
        repo.index.commit("Add pkg urdf")

    # Fetch without resolution
    path = ram.get_resource(
        repo_url, "pkg.urdf", cache_dir=mock_cache_dir, resolve_packages=False
    )

    content = path.read_text()
    assert "package://my_pkg/mesh.stl" in content
    assert "file://" not in content  # Should NOT be resolved to absolute path


def test_resolve_package_at_root(temp_git_repo):
    (temp_git_repo / "my_pkg").mkdir()
    with ram._ram_repo_context(temp_git_repo):
        path = ram._resolve_package("my_pkg")
        assert Path(path) == temp_git_repo / "my_pkg"


def test_resolve_package_metapackage(temp_git_repo):
    (temp_git_repo / "subdir").mkdir()
    (temp_git_repo / "subdir" / "my_pkg").mkdir()

    with ram._ram_repo_context(temp_git_repo):
        path = ram._resolve_package("my_pkg")
        assert Path(path) == temp_git_repo / "subdir" / "my_pkg"

    (temp_git_repo / "direct_pkg").mkdir()
    with ram._ram_repo_context(temp_git_repo):
        path = ram._resolve_package("direct_pkg")
        assert Path(path) == temp_git_repo / "direct_pkg"


def test_package_uri_resolution_full(temp_git_repo):
    (temp_git_repo / "my_pkg").mkdir()
    (temp_git_repo / "my_pkg" / "mesh.stl").write_text("foo")
    content = '<mesh filename="package://my_pkg/mesh.stl"/>'
    with ram._ram_repo_context(temp_git_repo):
        resolved = ram._replace_package_uris(content, temp_git_repo)
        assert (temp_git_repo / "my_pkg" / "mesh.stl").resolve().as_posix() in resolved


def test_process_xacro_no_resolve(temp_git_repo):
    xacro_path = temp_git_repo / "robot.xacro"
    urdf_xml = ram.process_xacro(
        xacro_path, repo_root=temp_git_repo, resolve_packages=False
    )
    assert "test_xacro" in urdf_xml


def test_package_uri_fallback(temp_git_repo):
    content = '<mesh filename="package://unknown_pkg/mesh.stl"/>'
    with ram._ram_repo_context(temp_git_repo):
        resolved = ram._replace_package_uris(content, temp_git_repo)
        assert (temp_git_repo / "mesh.stl").resolve().as_posix() in resolved


def test_get_resource_local_processed_with_package_uri(temp_git_repo, mock_cache_dir):
    (temp_git_repo / "pkg.urdf").write_text(
        '<mesh filename="package://test_repo/mesh.stl"/>'
    )
    path = ram.get_resource(
        path_inside_repo="pkg.urdf", repo_root=temp_git_repo, cache_dir=mock_cache_dir
    )
    assert "processed" in str(path)
    assert "mesh.stl" in path.read_text()


def test_resolve_package_repo_root_is_package(tmp_path):
    """Test _resolve_package when the repo root itself IS the package.

    This covers cases like openarm_description where the repo is cloned
    as 'openarm_description/' and $(find openarm_description) should
    resolve to the repo root.
    """
    # Create a directory whose name matches the package
    repo_root = tmp_path / "openarm_description"
    repo_root.mkdir()
    (repo_root / "config").mkdir()

    with ram._ram_repo_context(repo_root):
        path = ram._resolve_package("openarm_description")
        assert Path(path) == repo_root


def test_convert_dae_to_glb_success(tmp_path):
    dae_path = tmp_path / "model.dae"
    dae_path.write_text("dummy dae")
    glb_path = tmp_path / "model.glb"

    with patch("teleop_xr.ram.trimesh.load") as mock_load:
        mock_scene = MagicMock()
        mock_load.return_value = mock_scene
        mock_scene.export.return_value = b"dummy glb content"

        result = ram._convert_dae_to_glb(dae_path)

        assert result == glb_path
        assert glb_path.exists()
        assert glb_path.read_bytes() == b"dummy glb content"


def test_convert_dae_to_glb_failure(tmp_path):
    dae_path = tmp_path / "model.dae"
    dae_path.write_text("dummy dae")

    with patch(
        "teleop_xr.ram.trimesh.load", side_effect=Exception("conversion failed")
    ):
        result = ram._convert_dae_to_glb(dae_path)
        assert result == dae_path


def test_convert_dae_to_glb_already_exists(tmp_path):
    dae_path = tmp_path / "model.dae"
    glb_path = tmp_path / "model.glb"
    glb_path.write_text("existing glb")

    result = ram._convert_dae_to_glb(dae_path)
    assert result == glb_path
    with patch("teleop_xr.ram.trimesh.load") as mock_load:
        ram._convert_dae_to_glb(dae_path)
        mock_load.assert_not_called()


def test_replace_dae_with_glb(tmp_path):
    dae_path = tmp_path / "mesh.dae"
    dae_path.write_text("dummy")
    urdf_content = f'<mesh filename="{dae_path}"/>'

    with patch("teleop_xr.ram._convert_dae_to_glb") as mock_convert:
        mock_convert.return_value = tmp_path / "mesh.glb"

        result = ram._replace_dae_with_glb(urdf_content)
        assert 'filename="' + (tmp_path / "mesh.glb").as_posix() + '"' in result


def test_resolve_package_with_package_xml(tmp_path):
    repo_root = tmp_path / "some_repo_hash"
    repo_root.mkdir()
    (repo_root / "package.xml").write_text("<package><name>my_package</name></package>")

    with ram._ram_repo_context(repo_root):
        path = ram._resolve_package("my_package")
        assert Path(path) == repo_root


def test_resolve_package_with_invalid_package_xml(tmp_path):
    repo_root = tmp_path / "some_repo_hash"
    repo_root.mkdir()
    (repo_root / "package.xml").write_text("not xml")

    with ram._ram_repo_context(repo_root):
        with pytest.raises(ValueError, match="Package 'my_package' not found"):
            ram._resolve_package("my_package")


def test_get_resource_with_dae_conversion_xacro(tmp_path, mock_cache_dir):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    xacro_path = repo_root / "robot.xacro"
    xacro_path.write_text('<robot name="r"/>')

    with (
        patch("teleop_xr.ram.process_xacro") as mock_process,
        patch("teleop_xr.ram._replace_dae_with_glb") as mock_replace,
    ):
        mock_process.return_value = "processed xacro"
        mock_replace.return_value = "replaced dae"

        result_path = ram.get_resource(
            repo_root=repo_root,
            path_inside_repo="robot.xacro",
            cache_dir=mock_cache_dir,
            convert_dae_to_glb=True,
        )

        assert result_path.read_text() == "replaced dae"
        mock_replace.assert_called_once_with("processed xacro")


def test_get_resource_with_dae_conversion_urdf(tmp_path, mock_cache_dir):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    urdf_path = repo_root / "robot.urdf"
    urdf_path.write_text('<mesh filename="model.dae"/>')

    (repo_root / "model.dae").write_text("dummy")

    with patch("teleop_xr.ram._replace_dae_with_glb") as mock_replace:
        mock_replace.return_value = "replaced urdf"

        result_path = ram.get_resource(
            repo_root=repo_root,
            path_inside_repo="robot.urdf",
            cache_dir=mock_cache_dir,
            convert_dae_to_glb=True,
        )

        assert "processed" in str(result_path)
        assert result_path.read_text() == "replaced urdf"


def test_get_resource_urdf_no_processing_needed(tmp_path, mock_cache_dir):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    urdf_path = repo_root / "robot.urdf"
    urdf_path.write_text('<robot name="r"/>')

    result_path = ram.get_resource(
        repo_root=repo_root,
        path_inside_repo="robot.urdf",
        cache_dir=mock_cache_dir,
        convert_dae_to_glb=True,
        resolve_packages=True,
    )

    assert result_path == urdf_path


def test_resolve_package_with_corrupt_package_xml(tmp_path):
    repo_root = tmp_path / "some_repo_hash"
    repo_root.mkdir()
    (repo_root / "package.xml").write_text("<package>")

    with ram._ram_repo_context(repo_root):
        with patch("teleop_xr.ram.re.search", side_effect=Exception("regex error")):
            with pytest.raises(ValueError):
                ram._resolve_package("my_package")


def test_convert_dae_to_glb_non_bytes(tmp_path):
    dae_path = tmp_path / "model.dae"
    dae_path.write_text("dummy dae")

    with patch("teleop_xr.ram.trimesh.load") as mock_load:
        mock_scene = MagicMock()
        mock_load.return_value = mock_scene
        mock_scene.export.return_value = {"not": "bytes or string"}

        result = ram._convert_dae_to_glb(dae_path)
        assert result == dae_path


def test_replace_dae_with_glb_non_dae():
    urdf_content = '<mesh filename="mesh.stl"/>'
    result = ram._replace_dae_with_glb(urdf_content)
    assert 'filename="mesh.stl"' in result


def test_get_resource_errors():
    with pytest.raises(
        ValueError, match="Either repo_url or repo_root must be provided"
    ):
        ram.get_resource()
    with pytest.raises(
        ValueError, match="Only one of repo_url or repo_root can be provided"
    ):
        ram.get_resource(repo_url="http://foo", repo_root=Path("/foo"))
        with pytest.raises(ValueError, match="path_inside_repo must be relative"):
            # Use a truly absolute path that works on all platforms
            abs_path = os.path.abspath("robot.urdf")
            ram.get_resource(repo_root=Path("/foo"), path_inside_repo=abs_path)
