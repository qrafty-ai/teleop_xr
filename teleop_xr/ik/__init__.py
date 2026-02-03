"""
Inverse Kinematics (IK) module for teleoperation.

This module provides tools for mapping XR device poses to robot joint configurations
using optimization-based IK solvers.

Main components:
- BaseRobot: Abstract base class for robot-specific kinematic models and costs.
- PyrokiSolver: Optimization-based IK solver using Pyroki and jaxls.
- IKController: High-level controller for managing teleoperation state and snapshots.

Example:
    ```python
    from teleop_xr.ik import IKController, PyrokiSolver
    from my_robot_package import MyRobotModel

    robot = MyRobotModel()
    solver = PyrokiSolver(robot)
    controller = IKController(robot, solver)

    # In your control loop:
    new_q = controller.step(xr_state, current_q)
    ```
"""

from teleop_xr.ik.robot import BaseRobot
from teleop_xr.ik.solver import PyrokiSolver
from teleop_xr.ik.controller import IKController

__all__ = ["BaseRobot", "PyrokiSolver", "IKController"]
