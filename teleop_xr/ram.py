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
    """Replace package:// URI with relative paths to files in the repo."""

    # Find all package://<pkg_name>/ sequences
    # We assume the first part of the path is the package name,
    # and in RAM context, we map it to the repo root.
    def replace_match(match):
        match.group(1)
        relative_path = match.group(2)
        # For RAM, we assume the repo root is the package root for all packages in it
        # or we just strip the package://<pkg_name>/ prefix and use the relative path
        return relative_path

    # Regex to match package://[anything until next /]/

    # Actually, simpler: just replace package://<any>/ with nothing
    # so that the remaining path is relative to the URDF file location if URDF is at root,
    # but URDF might be in a subdir.

    # Better: find absolute path in repo
    def resolve_uri(match):
        match.group(1)
        sub_path = match.group(2)
        # We assume the repo_root is where we should look for the sub_path
        # In many ROS repos, the repo contains multiple packages.
        # For now, we'll just return the sub_path and assume the consumer knows how to handle it
        # relative to the repo root.
        return sub_path

    return re.sub(r"package://([^/]+)/(.*)", resolve_uri, urdf_content)


def get_asset(
    repo_url: str,
    path_inside_repo: str,
    branch: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    xacro_args: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Fetch a robot asset from a git repository and return its path.
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
                repo.git.checkout(branch)
            repo.remotes.origin.pull()

    # Resolve file path
    file_path = repo_dir / path_inside_repo
    if not file_path.exists():
        raise FileNotFoundError(
            f"Asset {path_inside_repo} not found in repo {repo_url}"
        )

    output_urdf_path = file_path

    # Handle Xacro
    if file_path.suffix == ".xacro" or ".urdf.xacro" in file_path.name:
        processed_dir = cache_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        # Unique name for output based on repo URL, path and args
        arg_str = str(sorted(xacro_args.items())) if xacro_args else ""
        content_hash = hashlib.sha256(
            (repo_url + path_inside_repo + arg_str).encode()
        ).hexdigest()[:12]
        output_urdf_path = processed_dir / f"{file_path.stem}_{content_hash}.urdf"

        # Process xacro
        doc = xacro.process_file(str(file_path), mappings=xacro_args)
        urdf_xml = doc.toprettyxml(indent="  ")

        # Replace package:// URIs
        urdf_xml = _replace_package_uris(urdf_xml, repo_dir)

        output_urdf_path.write_text(urdf_xml)
    else:
        # For plain URDF, we might still want to replace package:// URIs
        # but we don't want to modify the original file in the repo.
        # We'll copy it to processed dir.
        processed_dir = cache_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        content_hash = hashlib.sha256(
            (repo_url + path_inside_repo).encode()
        ).hexdigest()[:12]
        output_urdf_path = processed_dir / f"{file_path.stem}_{content_hash}.urdf"

        content = file_path.read_text()
        content = _replace_package_uris(content, repo_dir)
        output_urdf_path.write_text(content)

    return output_urdf_path
