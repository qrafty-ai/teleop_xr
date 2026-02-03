from abc import ABC, abstractmethod
from typing import Any, Optional
from teleop_xr.config import RobotVisConfig

# Placeholder for Cost type until it's defined in the module
Cost = Any


class BaseRobot(ABC):
    """
    Abstract base class for robot models used in IK optimization.
    """

    @abstractmethod
    def get_vis_config(self) -> Optional[RobotVisConfig]:
        """
        Get the visualization configuration for this robot.
        """
        pass

    @property
    @abstractmethod
    def joint_var_cls(self) -> Any:
        """
        The jaxls.Var class used for joint configurations.
        """
        pass

    @abstractmethod
    def forward_kinematics(self, config: Any) -> Any:
        """
        Compute the forward kinematics for the given configuration.

        Args:
            config: The robot configuration (e.g., joint angles).

        Returns:
            The pose of relevant links (e.g., end-effectors, head).
        """
        pass

    @abstractmethod
    def get_default_config(self) -> Any:
        """
        Get the default configuration for the robot.

        Returns:
            The default configuration.
        """
        pass

    @abstractmethod
    def build_costs(self, target_L: Any, target_R: Any, target_Head: Any) -> list[Cost]:
        """
        Build a list of costs for the robot-specific formulation.

        Args:
            target_L: Target pose for the left end-effector.
            target_R: Target pose for the right end-effector.
            target_Head: Target pose for the head.

        Returns:
            A list of Cost objects.
        """
        pass
