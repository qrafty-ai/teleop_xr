import importlib
import importlib.metadata
from teleop_xr.ik.robot import BaseRobot
from teleop_xr.ik.robots.h1_2 import UnitreeH1Robot


class RobotLoadError(Exception):
    """Custom exception for robot loading errors."""

    pass


def load_robot_class(robot_spec: str | None = None) -> type[BaseRobot]:
    """
    Load a robot class based on the given specification.

    Precedence:
    1. If robot_spec is None, return UnitreeH1Robot.
    2. If robot_spec matches an entry point name in 'teleop_xr.robots', load that.
    3. If robot_spec contains ':', parse as 'module:ClassName'.
    4. Otherwise, raise RobotLoadError.

    Robot Constructor Contract:
    All robot classes must support the following constructor signature:
    `def __init__(self, urdf_string: str | None = None, **kwargs)`

    Args:
        robot_spec: The robot specification string or None.

    Returns:
        type[BaseRobot]: The loaded robot class.

    Raises:
        RobotLoadError: If the robot class cannot be loaded or is invalid.
    """
    if robot_spec is None:
        return UnitreeH1Robot

    # 2. Entry point discovery
    try:
        # Get entry points for the group
        eps = importlib.metadata.entry_points(group="teleop_xr.robots")
        if robot_spec in eps.names:
            cls = eps[robot_spec].load()
            if not isinstance(cls, type) or not issubclass(cls, BaseRobot):
                raise RobotLoadError(
                    f"Entry point '{robot_spec}' did not return a BaseRobot subclass"
                )
            return cls
    except RobotLoadError:
        raise
    except Exception as e:
        # If entry point exists but fails to load, we should probably report it
        # but if it doesn't exist, we just fall through
        eps = importlib.metadata.entry_points(group="teleop_xr.robots")
        if robot_spec in eps.names:
            raise RobotLoadError(
                f"Failed to load robot class from entry point '{robot_spec}': {e}"
            ) from e

    # 3. Explicit module:ClassName
    if ":" in robot_spec:
        try:
            module_name, class_name = robot_spec.split(":", 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            if not isinstance(cls, type) or not issubclass(cls, BaseRobot):
                raise RobotLoadError(
                    f"Class '{class_name}' in module '{module_name}' is not a subclass of BaseRobot"
                )
            return cls
        except (ImportError, AttributeError) as e:
            raise RobotLoadError(
                f"Failed to load robot class from spec '{robot_spec}': {e}"
            ) from e

    raise RobotLoadError(
        f"Invalid robot specification: '{robot_spec}'. Must be an entry point name or 'module:ClassName' format."
    )


def list_available_robots() -> dict[str, str]:
    """
    List available robots via entry points without importing them.

    Returns:
        dict[str, str]: A mapping of robot names to their class paths.
    """
    try:
        eps = importlib.metadata.entry_points(group="teleop_xr.robots")
        return {ep.name: ep.value for ep in eps}
    except Exception:
        return {}
