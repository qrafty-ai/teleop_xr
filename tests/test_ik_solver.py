import numpy as np
import mujoco
from teleop_xr.ik.mink_solver import MinkIKSolver
from robot_descriptions import panda_mj_description


def test_mink_ik_solver_init():
    model_path = panda_mj_description.MJCF_PATH
    end_effector_frames = {"ee": "link7"}
    task_weights = {"ee": 1.0}

    solver = MinkIKSolver(
        model_path=model_path,
        end_effector_frames=end_effector_frames,
        task_weights=task_weights,
    )

    assert isinstance(solver.model, mujoco.MjModel)
    assert len(solver.tasks) == 2  # 1 FrameTask + 1 PostureTask
    assert "ee" in solver.frame_tasks


def test_mink_ik_solver_solve():
    model_path = panda_mj_description.MJCF_PATH
    end_effector_frames = {"ee": "link7"}
    task_weights = {"ee": 1.0}

    solver = MinkIKSolver(
        model_path=model_path,
        end_effector_frames=end_effector_frames,
        task_weights=task_weights,
    )

    current_q = np.zeros(solver.model.nq)

    # Define a target: slightly shifted from current position
    # First, get current EE pose
    mujoco.mj_forward(solver.model, solver.data)
    # Get ID of link7
    link7_id = mujoco.mj_name2id(solver.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    current_ee_pos = solver.data.xpos[link7_id].copy()
    current_ee_mat = solver.data.xmat[link7_id].reshape(3, 3).copy()

    target_pos = current_ee_pos + np.array([0.05, 0.05, 0.05])
    target_matrix = np.eye(4)
    target_matrix[:3, :3] = current_ee_mat
    target_matrix[:3, 3] = target_pos

    targets = {"ee": target_matrix}

    new_q = solver.solve(targets, current_q)

    assert new_q.shape == current_q.shape
    assert not np.allclose(new_q, current_q), "Joint configuration should have changed"


def test_mink_ik_solver_invalid_task():
    model_path = panda_mj_description.MJCF_PATH
    end_effector_frames = {"ee": "link7"}

    solver = MinkIKSolver(
        model_path=model_path, end_effector_frames=end_effector_frames, task_weights={}
    )

    current_q = np.zeros(solver.model.nq)
    targets = {"invalid": np.eye(4)}

    new_q = solver.solve(targets, current_q)

    # Should still return a valid configuration (probably same as current_q if posture task doesn't move it much)
    assert new_q.shape == current_q.shape
