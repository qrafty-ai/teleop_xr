import pytest
from unittest.mock import MagicMock, patch, mock_open
import numpy as np
from scripts.configure_sphere_collision import (
    _export_collision_data,
    _run_loop_step,
    RobotSpheresResult,
    Sphere,
)


@pytest.fixture
def mock_components():
    gui = MagicMock()
    # Setup gui mocks
    gui.export_filename = "test.json"
    gui._n_samples_slider.value = 100
    gui._threshold_slider.value = 0.01
    gui._threads_slider.value = 1
    gui._generated_ignore_pairs = None
    gui._user_fully_disabled_links = set()
    gui.is_auto_mode = True
    gui.total_spheres = 10
    gui.manual_allocation = {}
    gui.get_config.return_value = MagicMock()
    gui.joint_config = np.array([0.0])
    gui.show_spheres = True
    gui.opacity = 0.8
    gui.refine_enabled = True

    robot = MagicMock()
    robot.collision_links = ["link1"]
    robot.auto_allocate.return_value = {"link1": 10}
    robot.spherize.return_value = RobotSpheresResult(link_spheres={})
    robot.refine.return_value = RobotSpheresResult(link_spheres={})
    robot.compute_transforms.return_value = np.zeros((1, 7))

    visuals = MagicMock()
    urdf_vis = MagicMock()

    return gui, robot, visuals, urdf_vis


def test_export_collision_data(mock_components):
    gui, robot, _, _ = mock_components

    sphere = Sphere(center=np.array([1, 2, 3]), radius=0.1)
    result = RobotSpheresResult(link_spheres={"link1": [sphere], "link2": []})

    with patch(
        "scripts.configure_sphere_collision.compute_collision_ignore_pairs"
    ) as mock_compute:
        mock_compute.return_value = [["link1", "link2"]]

        with patch("builtins.open", mock_open()):
            with patch("json.dump") as mock_json:
                _export_collision_data(gui, robot, result)

                args, _ = mock_json.call_args
                data = args[0]
                assert "spheres" in data
                assert "collision_ignore_pairs" in data
                assert ["link1", "link2"] in data["collision_ignore_pairs"]
                assert data["spheres"]["link1"]["radii"] == [0.1]


def test_export_collision_data_manual_disabled(mock_components):
    gui, robot, _, _ = mock_components
    gui._user_fully_disabled_links = {"link1"}
    robot.collision_links = ["link1", "link2"]

    result = RobotSpheresResult(link_spheres={})

    with patch(
        "scripts.configure_sphere_collision.compute_collision_ignore_pairs",
        return_value=[],
    ):
        with patch("builtins.open", mock_open()):
            with patch("json.dump") as mock_json:
                _export_collision_data(gui, robot, result)

                args, _ = mock_json.call_args
                data = args[0]
                assert ["link1", "link2"] in data["collision_ignore_pairs"] or [
                    "link2",
                    "link1",
                ] in data["collision_ignore_pairs"]


def test_run_loop_step_spherize_auto(mock_components):
    gui, robot, visuals, urdf_vis = mock_components

    gui.needs_spherize = True
    gui.needs_refine_update = False
    gui.needs_visual_update = False

    result = None

    new_result = _run_loop_step(gui, robot, visuals, urdf_vis, result)

    gui.poll.assert_called_once()
    robot.auto_allocate.assert_called_with(10)
    gui.update_sliders_from_allocation.assert_called()
    robot.spherize.assert_called()
    gui.mark_spherized.assert_called()
    gui.set_needs_refine_update.assert_called()
    assert new_result is not None


def test_run_loop_step_refine(mock_components):
    gui, robot, visuals, urdf_vis = mock_components

    gui.needs_spherize = False
    gui.needs_refine_update = True
    gui.needs_visual_update = False

    sphere = Sphere(center=np.array([1, 2, 3]), radius=0.1)
    result = RobotSpheresResult(link_spheres={"link1": [sphere]})

    _run_loop_step(gui, robot, visuals, urdf_vis, result)

    robot.refine.assert_called()
    visuals.update.assert_called()
    gui.mark_refine_updated.assert_called()
    gui.mark_visuals_updated.assert_called()


def test_run_loop_step_visual_update(mock_components):
    gui, robot, visuals, urdf_vis = mock_components

    gui.needs_spherize = False
    gui.needs_refine_update = False
    gui.needs_visual_update = True

    result = RobotSpheresResult(link_spheres={})

    _run_loop_step(gui, robot, visuals, urdf_vis, result)

    visuals.update.assert_called()
    gui.mark_visuals_updated.assert_called()


def test_run_loop_step_update_cfg_and_transforms(mock_components):
    gui, robot, visuals, urdf_vis = mock_components
    gui.needs_spherize = False
    gui.needs_refine_update = False

    _run_loop_step(gui, robot, visuals, urdf_vis, None)

    urdf_vis.update_cfg.assert_called_with(gui.joint_config)
    robot.compute_transforms.assert_called_with(gui.joint_config)
    visuals.update_transforms.assert_called()
