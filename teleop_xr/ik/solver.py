import jax
import jax.numpy as jnp
import jaxls
import jaxlie
from typing import Callable
from teleop_xr.ik.robot import BaseRobot


class PyrokiSolver:
    """
    Inverse Kinematics (IK) solver using Pyroki and jaxls.

    This solver uses optimization (Least Squares) to find joint configurations
    that satisfy target poses for the robot's end-effectors and head.
    It leverages JAX for high-performance, JIT-compiled solving.
    """

    robot: BaseRobot
    _jit_solve: Callable[[jaxlie.SE3, jaxlie.SE3, jaxlie.SE3, jnp.ndarray], jnp.ndarray]

    def __init__(self, robot: BaseRobot):
        """
        Initialize the solver with a robot model.

        Args:
            robot: The robot model providing kinematic info and costs.
        """
        self.robot = robot

        # JIT compile the solve function
        self._jit_solve = jax.jit(self._solve_internal)

        # Warmup to trigger JIT compilation
        self._warmup()

    def _solve_internal(
        self,
        target_L: jaxlie.SE3,
        target_R: jaxlie.SE3,
        target_Head: jaxlie.SE3,
        q_current: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Internal solve function that will be JIT-compiled.

        Args:
            target_L: Target pose for the left end-effector.
            target_R: Target pose for the right end-effector.
            target_Head: Target pose for the head.
            q_current: Current joint configuration (initial guess).

        Returns:
            jnp.ndarray: Optimized joint configuration.
        """
        # 1. Build costs from the robot
        costs = self.robot.build_costs(target_L, target_R, target_Head)

        # 2. Get the joint variable (assuming single timestep index 0)
        # The robot is expected to have joint_var_cls.
        # We use a single timestep for standard IK.
        var_joints = self.robot.joint_var_cls(jnp.array([0]))

        # 3. Construct initial values
        # q_current is expected to be (num_joints,)
        # Variable values usually expect (timesteps, joint_dims)
        initial_vals = jaxls.VarValues.make(
            [var_joints.with_value(q_current[jnp.newaxis, :])]
        )

        # 4. Construct and solve the LeastSquaresProblem
        # We optimize over joint variables.
        # Note: If the robot defines more variables in costs, they should be added here.
        # For a generic solver, we assume joint variables are the primary optimization targets.
        problem = jaxls.LeastSquaresProblem(costs, [var_joints])

        # solve() returns a VarValues object containing the solution
        solution = problem.analyze().solve(
            initial_vals=initial_vals,
            verbose=False,
            linear_solver="dense_cholesky",
            termination=jaxls.TerminationConfig(max_iterations=15),
        )

        # Return the optimized joint configuration for the first (and only) timestep
        return solution[var_joints][0]

    def solve(
        self,
        target_L: jaxlie.SE3,
        target_R: jaxlie.SE3,
        target_Head: jaxlie.SE3,
        q_current: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Solve the IK problem for the given targets and current configuration.

        Args:
            target_L: Target pose for the left end-effector.
            target_R: Target pose for the right end-effector.
            target_Head: Target pose for the head.
            q_current: Current joint configuration (initial guess).

        Returns:
            jnp.ndarray: Optimized joint configuration.
        """
        return self._jit_solve(target_L, target_R, target_Head, q_current)

    def _warmup(self) -> None:
        """
        Triggers JIT compilation by running a solve with dummy data.
        """
        try:
            # Use default config as initial guess
            q_dummy = self.robot.get_default_config()

            # Use identity poses as dummy targets
            target_dummy = jaxlie.SE3.identity()

            # Run solve to trigger JIT
            self.solve(target_dummy, target_dummy, target_dummy, q_dummy)
        except Exception:
            # Warmup might fail if robot is a mock or not fully implemented yet.
            # We don't want to crash during initialization in that case.
            pass
