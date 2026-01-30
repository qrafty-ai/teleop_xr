import numpy as np
import mujoco
import mink
from typing import Dict
from teleop_xr.ik.model_loader import load_model


class MinkIKSolver:
    def __init__(
        self,
        model_path: str,
        end_effector_frames: Dict[str, str],
        task_weights: Dict[str, float],
        posture_weight: float = 0.01,
    ):
        """
        Initialize the Mink IK solver.

        Args:
            model_path: Path to the URDF or MJCF model.
            end_effector_frames: Mapping from task name to frame name in the model.
            task_weights: Mapping from task name to weight for both position and orientation.
            posture_weight: Weight for the posture regularization task.
        """
        self.model = load_model(model_path)
        self.data = mujoco.MjData(self.model)
        self.configuration = mink.Configuration(self.model)

        self.tasks = []
        self.frame_tasks = {}

        for task_name, frame_name in end_effector_frames.items():
            weight = task_weights.get(task_name, 1.0)

            # Heuristic to determine frame type: prefer site, fallback to body
            frame_type = "body"
            if (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, frame_name)
                != -1
            ):
                frame_type = "site"
            elif (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, frame_name)
                != -1
            ):
                frame_type = "body"
            elif (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, frame_name)
                != -1
            ):
                frame_type = "geom"

            task = mink.FrameTask(
                frame_name=frame_name,
                frame_type=frame_type,
                position_cost=weight,
                orientation_cost=weight,
            )
            self.tasks.append(task)
            self.frame_tasks[task_name] = task

        self.posture_task = mink.PostureTask(self.model, cost=posture_weight)
        # Initialize posture target with the initial configuration
        self.posture_task.set_target(self.configuration.q.copy())
        self.tasks.append(self.posture_task)

        self.dt = 0.01  # 100 Hz

    def solve(
        self, targets: Dict[str, np.ndarray], current_q: np.ndarray
    ) -> np.ndarray:
        """
        Solve IK to find the next joint configuration.

        Args:
            targets: Mapping from task name to target 4x4 SE3 matrix.
            current_q: Current joint configuration.

        Returns:
            The new joint configuration.
        """
        self.configuration.update(current_q)

        for task_name, target_matrix in targets.items():
            if task_name in self.frame_tasks:
                target_se3 = mink.SE3.from_matrix(target_matrix)
                self.frame_tasks[task_name].set_target(target_se3)

        try:
            velocity = mink.solve_ik(
                self.configuration,
                self.tasks,
                self.dt,
                solver="daqp",
            )
            # Integrate velocity to get the next configuration
            new_q = self.configuration.integrate(velocity, self.dt)
            return new_q
        except Exception:
            # Handle failures by returning current_q (hold pose)
            return current_q

    def update_viewer(self, viewer):
        """
        Optional helper to sync visualization.
        """
        if viewer is not None:
            viewer.sync()
