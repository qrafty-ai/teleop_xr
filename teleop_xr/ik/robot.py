import jax.numpy as jnp
import jaxlie
from abc import ABC, abstractmethod
from typing import Any
from teleop_xr.config import RobotVisConfig

# Cost is a jaxls.CostBase object (or similar depending on implementation)
Cost = Any


class BaseRobot(ABC):
    """
    Abstract base class for robot models used in IK optimization.

    This class defines the interface required by the IK solver and controller
    to compute kinematics and optimization costs.
    """

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
