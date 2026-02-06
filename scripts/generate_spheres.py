import io
import json
import os

import numpy as np
import tyro
import yourdfpy
from ballpark import BallparkConfig, Robot, SpherePreset
from loguru import logger

from teleop_xr.ik.loader import load_robot_class


def validate_sphere_decomposition(decomposition: dict) -> bool:
    for link_name, data in decomposition.items():
        if not isinstance(data, dict):
            raise ValueError(f"Invalid data for link {link_name}")
        if "centers" not in data:
            raise ValueError(f"Missing centers for link {link_name}")
        if "radii" not in data:
            raise ValueError(f"Missing radii for link {link_name}")
        centers = data["centers"]
        radii = data["radii"]
        if not isinstance(centers, list) or not isinstance(radii, list):
            raise ValueError(f"Centers and radii must be lists for link {link_name}")
        if len(centers) != len(radii):
            raise ValueError(
                f"Length mismatch between centers and radii for link {link_name}"
            )
    return True


def generate_collision_spheres(
    urdf_path: str | None = None,
    urdf_string: str | None = None,
    target_spheres: int | None = 64,
):
    np.random.seed(42)

    if urdf_string is not None:
        urdf = io.StringIO(urdf_string)
    elif urdf_path is not None:
        urdf = urdf_path
    else:
        raise ValueError("Either urdf_path or urdf_string must be provided.")

    urdf_coll = yourdfpy.URDF.load(urdf, load_collision_meshes=True)

    robot = Robot(urdf_coll)
    result = robot.spherize(target_spheres=target_spheres)
    config = BallparkConfig.from_preset(SpherePreset.BALANCED)
    result = robot.refine(result, config=config)

    decomposition = {}
    for link_name, spheres in result.link_spheres.items():
        decomposition[link_name] = {
            "centers": [s.center.tolist() for s in spheres],
            "radii": [float(s.radius) for s in spheres],
        }

    return decomposition


def main(
    robot_class: str,
    target_spheres: int = 64,
) -> None:
    """
    Generate sphere decomposition for a robot and save it to assets.

    Args:
        robot_class: The robot class specification (e.g., 'h1', 'UnitreeH1Robot' or 'module:Class').
        target_spheres: Optional target number of spheres for decomposition.
    """
    logger.info(f"Loading robot class: {robot_class}")
    try:
        # Try as entry point or module:Class first
        RobotCls = load_robot_class(robot_class)
    except Exception as e:
        # Fallback: maybe it's just the class name in teleop_xr.ik.robots.h1_2
        # But load_robot_class should handle it if it's an entry point.
        # Let's try to be helpful if they passed 'UnitreeH1Robot' but it's not an entry point.
        if ":" not in robot_class:
            logger.warning(
                f"Could not load '{robot_class}' directly. Trying common locations..."
            )
            try:
                # This is a bit hacky but helps with the specific requirement "UnitreeH1Robot"
                from teleop_xr.ik.robots.h1_2 import UnitreeH1Robot

                if robot_class == "UnitreeH1Robot":
                    RobotCls = UnitreeH1Robot
                else:
                    raise e
            except ImportError:
                raise e
        else:
            raise e

    try:
        # Instantiate to get urdf_path and name
        robot = RobotCls()
    except Exception as e:
        logger.error(f"Failed to instantiate robot class: {e}")
        return

    if not hasattr(robot, "urdf_path"):
        logger.error("Robot instance does not have 'urdf_path' attribute.")
        return

    urdf_path = getattr(robot, "urdf_path")
    robot_name = robot.name
    logger.info(f"Generating spheres for {robot_name} using URDF: {urdf_path}")

    try:
        decomposition = generate_collision_spheres(
            urdf_path=urdf_path, target_spheres=target_spheres
        )
        validate_sphere_decomposition(decomposition)
    except Exception as e:
        logger.error(f"Failed to generate collision spheres: {e}")
        return

    # Determine save path
    # Assets are in teleop_xr/ik/robots/assets/{name}/sphere.json
    import teleop_xr.ik.robot as robot_mod

    current_dir = os.path.dirname(os.path.abspath(robot_mod.__file__))
    asset_dir = os.path.join(current_dir, "robots", "assets", robot_name)
    os.makedirs(asset_dir, exist_ok=True)
    asset_path = os.path.join(asset_dir, "sphere.json")

    logger.info(f"Saving decomposition to {asset_path}")
    with open(asset_path, "w") as f:
        json.dump(decomposition, f, indent=2)

    logger.success(f"Successfully generated spheres for {robot_name}")


if __name__ == "__main__":
    tyro.cli(main)
