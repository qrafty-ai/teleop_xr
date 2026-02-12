import os
import shutil
import subprocess
import sys
from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        print("Running custom build hook...", file=sys.stderr)

        # Define paths
        root_dir = self.root
        webxr_dir = os.path.join(root_dir, "webxr")
        teleop_xr_dir = os.path.join(root_dir, "teleop_xr")
        pkg_dist_dir = os.path.join(teleop_xr_dir, "dist")

        def ensure_placeholder():
            if not os.path.exists(pkg_dist_dir):
                os.makedirs(pkg_dist_dir, exist_ok=True)
            index_path = os.path.join(pkg_dist_dir, "index.html")
            if not os.path.exists(index_path):
                with open(index_path, "w") as f:
                    f.write("<!-- placeholder -->")
            if "teleop_xr/dist" not in build_data["artifacts"]:
                build_data["artifacts"].append("teleop_xr/dist")

        # Check if we should skip the build
        # We skip if:
        # 1. We are not in a git repo (likely building from sdist in isolated env)
        # 2. AND the artifacts already exist
        is_git_repo = os.path.exists(os.path.join(root_dir, ".git"))
        artifacts_exist = os.path.exists(pkg_dist_dir)

        # Check for skip env var
        if os.environ.get("TELEOP_XR_SKIP_WEBXR_BUILD") in ("1", "true", "True"):
            print(
                "Skipping WebXR build due to TELEOP_XR_SKIP_WEBXR_BUILD",
                file=sys.stderr,
            )
            ensure_placeholder()
            return

        if not is_git_repo and artifacts_exist:
            print(
                "Not in git repo and artifacts exist, skipping WebXR build.",
                file=sys.stderr,
            )
            ensure_placeholder()
            return

        if os.environ.get("SKIP_WEBXR_BUILD"):
            print("Skipping WebXR build via SKIP_WEBXR_BUILD env var", file=sys.stderr)
            return

        # Check if npm is available
        if shutil.which("npm") is None:
            print("Warning: npm not found, skipping WebXR build", file=sys.stderr)
            ensure_placeholder()
            return

        # Run npm install and build
        print(f"Building WebXR in {webxr_dir}...", file=sys.stderr)
        is_windows = os.name == "nt"
        try:
            subprocess.check_call(["npm", "install"], cwd=webxr_dir, shell=is_windows)
            subprocess.check_call(
                ["npm", "run", "build"], cwd=webxr_dir, shell=is_windows
            )
        except subprocess.CalledProcessError as e:
            print(
                f"Error building WebXR (exit code {e.returncode}): {e}", file=sys.stderr
            )
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error during WebXR build: {e}", file=sys.stderr)
            sys.exit(1)

        # Copy artifacts
        dist_dir = os.path.join(webxr_dir, "out")
        if not os.path.exists(dist_dir):
            print(f"Error: {dist_dir} does not exist after build", file=sys.stderr)
            sys.exit(1)

        # Destination: teleop_xr/dist
        if os.path.exists(pkg_dist_dir):
            shutil.rmtree(pkg_dist_dir)

        print(f"Copying {dist_dir} to {pkg_dist_dir}", file=sys.stderr)
        shutil.copytree(dist_dir, pkg_dist_dir)

        # Ensure build_data includes these files as artifacts
        build_data["artifacts"].append("teleop_xr/dist")
