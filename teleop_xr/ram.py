"""
Robot Asset Manager (RAM) module.
Handles fetching and processing robot descriptions (URDF/Xacro) from git.
"""

import hashlib
import re
from pathlib import Path
from typing import Optional, Dict

import git
import xacro
from filelock import FileLock


def get_cache_root() -> Path:
    """Get the root directory for RAM cache."""
    cache_root = Path.home() / ".cache" / "ram"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root


def _get_repo_dir(repo_url: str, cache_dir: Path) -> Path:
    """Get the directory name for a repo based on its URL hash."""
    url_hash = hashlib.sha256(repo_url.encode()).hexdigest()[:12]
    # Try to extract a human readable name from URL
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    return cache_dir / "repos" / f"{repo_name}_{url_hash}"


def _replace_package_uris(urdf_content: str, repo_root: Path) -> str:
    """
    Replace package:// URI with absolute paths to files in the repository.
    """

    def resolve_uri(match):
        # match.group(1) is the package name
        sub_path = match.group(2)
        # Return absolute path to the file in repo_root
        return str((repo_root / sub_path).absolute())

    return re.sub(r"package://([^/]+)/(.*)", resolve_uri, urdf_content)


def get_repo(
    repo_url: str, branch: Optional[str] = None, cache_dir: Optional[Path] = None
) -> Path:
    """
    Fetch a git repository into the cache and return its local path.
    """
    if cache_dir is None:
        cache_dir = get_cache_root()
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)

    repo_dir = _get_repo_dir(repo_url, cache_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    lock_path = repo_dir.with_suffix(".lock")

    with FileLock(lock_path):
        if not repo_dir.exists():
            # Clone repo
            git.Repo.clone_from(repo_url, repo_dir, branch=branch)
        else:
            # Update repo
            repo = git.Repo(repo_dir)
            if branch:
                try:
                    repo.git.checkout(branch)
                except git.exc.GitCommandError:
                    # If checkout fails, try fetching first
                    repo.remotes.origin.fetch()
                    repo.git.checkout(branch)
            repo.remotes.origin.pull()

    return repo_dir


def process_xacro(
    xacro_path: Path, repo_root: Path, mappings: Optional[Dict[str, str]] = None
) -> str:
    """
    Process a xacro file and return the URDF XML string with package:// URIs resolved.
    """
    doc = xacro.process_file(str(xacro_path), mappings=mappings)
    urdf_xml = doc.toprettyxml(indent="  ")
    return _replace_package_uris(urdf_xml, repo_root)


def get_resource(
    repo_url: str,
    path_inside_repo: str,
    branch: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    xacro_args: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Main entry point for fetching a robot resource.
    """
    if cache_dir is None:
        cache_dir = get_cache_root()

    repo_dir = get_repo(repo_url, branch=branch, cache_dir=cache_dir)
    file_path = repo_dir / path_inside_repo

    if not file_path.exists():
        raise FileNotFoundError(
            f"Asset {path_inside_repo} not found in repo {repo_url}"
        )

    if file_path.suffix == ".xacro" or ".urdf.xacro" in file_path.name:
        # Unique name for output based on args
        arg_str = str(sorted(xacro_args.items())) if xacro_args else ""
        arg_hash = hashlib.sha256(arg_str.encode()).hexdigest()[:6]
        output_urdf_path = file_path.parent / f"{file_path.stem}_{arg_hash}.urdf"

        # Process xacro
        urdf_content = process_xacro(file_path, repo_dir, mappings=xacro_args)
        output_urdf_path.write_text(urdf_content)
    else:
        # For plain URDF, if it has package://, we must process it.
        # Otherwise, we can just return the original file_path.
        content = file_path.read_text()
        if "package://" in content:
            output_urdf_path = file_path.parent / f"{file_path.stem}_processed.urdf"
            content = _replace_package_uris(content, repo_dir)
            output_urdf_path.write_text(content)
        else:
            output_urdf_path = file_path

    return output_urdf_path


# Alias get_asset to get_resource for backward compatibility if any
get_asset = get_resource
