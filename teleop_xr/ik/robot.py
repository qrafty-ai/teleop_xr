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

    @abstractmethod
    def get_vis_config(self) -> RobotVisConfig | None:
        """
        Get the visualization configuration for this robot.

        Returns:
            RobotVisConfig | None: Configuration for rendering the robot, or None if not supported.
        """
        pass

    @property
    @abstractmethod
    def joint_var_cls(self) -> Any:
        """
        The jaxls.Var class used for joint configurations.

        Returns:
            Type[jaxls.Var]: The variable class to use for optimization.
        """
        pass

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
        pass

    @abstractmethod
    def get_default_config(self) -> jnp.ndarray:
        """
        Get the default configuration for the robot.

        Returns:
            jnp.ndarray: The default configuration as a JAX array.
        """
        pass

    @abstractmethod
    def build_costs(
        self, target_L: jaxlie.SE3, target_R: jaxlie.SE3, target_Head: jaxlie.SE3
    ) -> list[Cost]:
        """
        Build a list of costs for the robot-specific formulation.

        Args:
            target_L: Target pose for the left end-effector.
            target_R: Target pose for the right end-effector.
            target_Head: Target pose for the head.

        Returns:
            list[Cost]: A list of jaxls Cost objects representing the optimization objectives.
        """
        pass
