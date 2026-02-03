import tempfile
from pathlib import Path

import pytest
import git
from teleop_xr import ram


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
        repo = git.Repo.init(repo_path)
        repo.index.add(["robot.urdf", "robot.xacro", "meshes/base.stl"])
        repo.index.commit("Initial commit")

        yield repo_path


def test_get_repo(temp_git_repo, mock_cache_dir):
    """Test get_repo function."""
    repo_url = f"file://{temp_git_repo}"
    repo_dir = ram.get_repo(repo_url, cache_dir=mock_cache_dir)
    assert repo_dir.exists()
    assert (repo_dir / "robot.urdf").exists()


def test_fetch_git_repo(temp_git_repo, mock_cache_dir):
    """Test that RAM can fetch a git repo into cache via get_resource."""
    repo_url = f"file://{temp_git_repo}"
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
    repo = git.Repo(repo_path)
    repo.index.add(["robot_pkg.urdf"])
    repo.index.commit("Add pkg urdf")

    repo_url = f"file://{repo_path}"
    asset_path = ram.get_resource(
        repo_url=repo_url, path_inside_repo="robot_pkg.urdf", cache_dir=mock_cache_dir
    )

    content = asset_path.read_text()
    assert "package://test_repo/" not in content
    # Should be replaced by path relative to repo root
    assert "meshes/base.stl" in content


def test_xacro_conversion(temp_git_repo, mock_cache_dir):
    """Test that xacro files are automatically converted to URDF."""
    repo_url = f"file://{temp_git_repo}"
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
    assert cache_root == Path.home() / ".cache" / "ram"
    assert cache_root.exists()


def test_get_repo_default_cache(temp_git_repo, monkeypatch):
    """Test get_repo uses default cache if not provided."""
    # Mock home directory to avoid polluting real home
    with tempfile.TemporaryDirectory() as tmp_home:
        monkeypatch.setenv("HOME", tmp_home)
        repo_url = f"file://{temp_git_repo}"
        repo_dir = ram.get_repo(repo_url)
        assert repo_dir.exists()
        assert str(repo_dir).startswith(tmp_home)


def test_update_existing_repo(temp_git_repo, mock_cache_dir):
    """Test that RAM correctly updates an existing repo in cache."""
    repo_url = f"file://{temp_git_repo}"

    # First fetch
    repo_dir = ram.get_repo(repo_url, cache_dir=mock_cache_dir)

    # Modify the source repo
    new_file = temp_git_repo / "new_file.txt"
    new_file.write_text("new content")
    repo = git.Repo(temp_git_repo)
    repo.index.add(["new_file.txt"])
    repo.index.commit("Add new file")

    # Second fetch should update
    repo_dir2 = ram.get_repo(repo_url, cache_dir=mock_cache_dir)
    assert repo_dir2 == repo_dir
    assert (repo_dir / "new_file.txt").exists()


def test_update_with_branch(temp_git_repo, mock_cache_dir):
    """Test updating an existing repo with a specific branch."""
    repo_url = f"file://{temp_git_repo}"
    repo = git.Repo(temp_git_repo)

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
        repo_url = f"file://{temp_git_repo}"
        asset_path = ram.get_resource(repo_url, "robot.urdf")
        assert asset_path.exists()
        assert str(asset_path).startswith(tmp_home)


def test_asset_not_found(temp_git_repo, mock_cache_dir):
    """Test that RAM raises FileNotFoundError if asset is missing."""
    repo_url = f"file://{temp_git_repo}"
    with pytest.raises(FileNotFoundError):
        ram.get_resource(
            repo_url=repo_url, path_inside_repo="missing.urdf", cache_dir=mock_cache_dir
        )


def test_get_asset_alias(temp_git_repo, mock_cache_dir):
    """Test that get_asset is indeed an alias for get_resource."""
    repo_url = f"file://{temp_git_repo}"
    asset_path = ram.get_asset(
        repo_url=repo_url, path_inside_repo="robot.urdf", cache_dir=mock_cache_dir
    )
    assert asset_path.exists()


def test_checkout_new_remote_branch(temp_git_repo, mock_cache_dir):
    """Test that RAM fetches if checkout fails (e.g. new remote branch)."""
    repo_url = f"file://{temp_git_repo}"

    # 1. Initial fetch of default branch
    ram.get_repo(repo_url, cache_dir=mock_cache_dir)

    # 2. Create a new branch in the SOURCE repo
    repo = git.Repo(temp_git_repo)
    new_branch_name = "new-remote-branch"
    repo.create_head(new_branch_name)

    # 3. Fetch the new branch.
    # Local cache repo doesn't know about it yet, so initial checkout fails.
    # It should then fetch and succeed.
    repo_dir = ram.get_repo(repo_url, branch=new_branch_name, cache_dir=mock_cache_dir)
    assert repo_dir.exists()

    # Verify we are on the correct branch
    cache_repo = git.Repo(repo_dir)
    assert cache_repo.active_branch.name == new_branch_name
