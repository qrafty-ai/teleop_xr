import tyro
import mujoco
import mujoco.viewer
from dataclasses import dataclass
from robot_descriptions import g1_mj_description

from teleop_xr import Teleop
from teleop_xr.config import TeleopSettings
from teleop_xr.ik.mink_solver import MinkIKSolver
from teleop_xr.common_cli import CommonCLI


@dataclass
class IKSimCLI(CommonCLI):
    model: str = g1_mj_description.MJCF_PATH
    viewer: bool = False
    rate: int = 100
    head_frame: str = "head_link"
    left_hand_frame: str = "left_palm"
    right_hand_frame: str = "right_palm"


def main():
    cli = tyro.cli(IKSimCLI)

    end_effector_frames = {
        "head": cli.head_frame,
        "left_hand": cli.left_hand_frame,
        "right_hand": cli.right_hand_frame,
    }

    task_weights = {
        "head": 0.5,
        "left_hand": 1.0,
        "right_hand": 1.0,
    }

    solver = MinkIKSolver(
        model_path=cli.model,
        end_effector_frames=end_effector_frames,
        task_weights=task_weights,
    )
    solver.dt = 1.0 / cli.rate

    settings = TeleopSettings(
        host=cli.host,
        port=cli.port,
        input_mode=cli.input_mode,
        multi_eef_mode=True,
    )

    teleop = Teleop(settings=settings)

    viewer = None
    if cli.viewer:
        viewer = mujoco.viewer.launch_passive(solver.model, solver.data)

    def callback(targets, xr_state_dict):
        current_q = solver.data.qpos.copy()
        new_q = solver.solve(targets, current_q)

        solver.data.qpos[:] = new_q

        mujoco.mj_forward(solver.model, solver.data)

        if viewer is not None:
            solver.update_viewer(viewer)

    teleop.subscribe(callback)

    try:
        teleop.run()
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
