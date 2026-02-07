import io
import os
import jax.numpy as jnp
import jaxlie
import yourdfpy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal
from teleop_xr.config import RobotVisConfig

# Cost is a jaxls.CostBase object (or similar depending on implementation)
Cost = Any


@dataclass
class RobotDescription:
    """Describes a robot's URDF source.

    Attributes:
        content: Either a filesystem path to a URDF file or a raw URDF XML string.
        kind: Discriminator indicating how to interpret ``content``.
    """

    content: str
    kind: Literal["path", "urdf_string"]


def _detect_description_kind(content: str) -> Literal["path", "urdf_string"]:
    """Auto-detect whether *content* is a URDF XML string or a file path."""
    stripped = content.lstrip()
    if stripped.startswith("<"):
        return "urdf_string"
    return "path"


class BaseRobot(ABC):
    """
    Abstract base class for robot models used in IK optimization.

    This class defines the interface required by the IK solver and controller
    to compute kinematics and optimization costs.

    Subclasses **must**:
    - Call ``super().__init__()``
    - Set ``self._default_description`` in ``__init__``
    - Implement ``_init_from_description`` – (re)initialises all URDF-dependent state.
    """

    def __init__(self) -> None:
        self._description_override: RobotDescription | None = None
        self._default_description: RobotDescription | None = None

    # ------------------------------------------------------------------
    # Robot description management
    # ------------------------------------------------------------------

    @property
    def description(self) -> RobotDescription:
        """Return the current robot description.

        Checks ``self._description_override`` first and falls back to
        ``self._default_description``.
        """
        if self._description_override is not None:
            return self._description_override
        if self._default_description is None:
            raise ValueError(
                "Robot subclass must set _default_description before accessing description"
            )
        return self._default_description

    @abstractmethod
    def _init_from_description(self, description: RobotDescription) -> None:
        """(Re)initialise all URDF-dependent state from *description*.

        Called once during ``__init__`` and again whenever
        :meth:`set_description` is used to override the description at
        runtime.
        """
        pass  # pragma: no cover

    def set_description(
        self,
        content: str,
        kind: Literal["path", "urdf_string"] | None = None,
    ) -> None:
        """Override the robot description and reinitialise URDF-dependent state.

        Args:
            content: A filesystem path or raw URDF XML string.
            kind: Explicit discriminator.  When ``None`` the kind is
                auto-detected (strings starting with ``<`` are treated as
                URDF XML).
        """
        if kind is None:
            kind = _detect_description_kind(content)
        self._description_override = RobotDescription(content=content, kind=kind)
        self._init_from_description(self._description_override)

    def load_urdf_model(
        self, description: RobotDescription, fallback_mesh_path: str | None = None
    ) -> yourdfpy.URDF:
        """Helper to load a URDF model from a description.

        Updates ``self.urdf_path`` and ``self.mesh_path`` as side effects.

        Args:
            description: The description to load from.
            fallback_mesh_path: Path to use for mesh resolution if description
                is a raw string (kind="urdf_string").

        Returns:
            The loaded yourdfpy.URDF object.
        """
        if description.kind == "path":
            self.urdf_path = description.content
            self.mesh_path = os.path.dirname(description.content)
            return yourdfpy.URDF.load(description.content)
        else:
            self.urdf_path = ""
            self.mesh_path = fallback_mesh_path
            return yourdfpy.URDF.load(io.StringIO(description.content))

    # ------------------------------------------------------------------
    # Orientation helpers
    # ------------------------------------------------------------------

    @property
    def orientation(self) -> jaxlie.SO3:
        """
        Rotation from the robot's base frame to the canonical ROS2 frame (X-forward, Z-up).
        """
        return jaxlie.SO3.identity()

    @property
    def base_to_ros(self) -> jaxlie.SO3:
        """Rotation from the robot base frame to the canonical ROS2 frame."""
        return self.orientation

    @property
    def ros_to_base(self) -> jaxlie.SO3:
        """Rotation from the canonical ROS2 frame to the robot base frame."""
        return self.orientation.inverse()

    @abstractmethod
    def get_vis_config(self) -> RobotVisConfig | None:
        """
        Get the visualization configuration for this robot.

        Returns:
            RobotVisConfig | None: Configuration for rendering the robot, or None if not supported.

        Note:
            Subclasses should use `self.orientation` to populate
            `RobotVisConfig.initial_rotation_euler`.
        """
        pass  # pragma: no cover

    @property
    @abstractmethod
    def actuated_joint_names(self) -> list[str]:
        """
        Get the names of the actuated joints.

        Returns:
            list[str]: A list of joint names.
        """
        pass  # pragma: no cover

    @property
    @abstractmethod
    def joint_var_cls(self) -> Any:
        """
        The jaxls.Var class used for joint configurations.

        Returns:
            Type[jaxls.Var]: The variable class to use for optimization.
        """
        pass  # pragma: no cover

    @property
    def supported_frames(self) -> set[str]:
        """
        Get the set of supported frames for this robot.

        Returns:
            set[str]: A set of frame names (e.g., {"left", "right", "head"}).
        """
        return {"left", "right", "head"}

    @property
    def default_speed_ratio(self) -> float:
        """
        Get the default teleop speed ratio for this robot.

        Returns:
            float: The default speed ratio (e.g., 1.0 for 100%, 1.2 for 120%).
        """
        return 1.0

    @abstractmethod
    def forward_kinematics(self, config: jnp.ndarray) -> dict[str, jaxlie.SE3]:
        """
        Compute the forward kinematics for the given configuration.

        Args:
            config: The robot configuration (e.g., joint angles) as a JAX array.

        Returns:
            dict[str, jaxlie.SE3]: A dictionary mapping link names (e.g., "left", "right", "head")
                                  to their respective SE3 poses.
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_default_config(self) -> jnp.ndarray:
        """
        Get the default configuration for the robot.

        Returns:
            jnp.ndarray: The default configuration as a JAX array.
        """
        pass  # pragma: no cover

    @abstractmethod
    def build_costs(
        self,
        target_L: jaxlie.SE3 | None,
        target_R: jaxlie.SE3 | None,
        target_Head: jaxlie.SE3 | None,
        q_current: jnp.ndarray | None = None,
    ) -> list[Cost]:
        """
        Build a list of costs for the robot-specific formulation.

        Args:
            target_L: Target pose for the left end-effector.
            target_R: Target pose for the right end-effector.
            target_Head: Target pose for the head.
            q_current: Current joint configuration (initial guess).

        Returns:
            list[Cost]: A list of jaxls Cost objects representing the optimization objectives.
        """
        pass  # pragma: no cover
