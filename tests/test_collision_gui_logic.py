import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from scripts.configure_sphere_collision import _SpheresGui


@pytest.fixture
def mock_gui():
    server = MagicMock()
    robot = MagicMock()
    robot.collision_links = ["link1", "link2"]
    robot.joint_limits = (np.array([0.0]), np.array([1.0]))
    robot.links = ["link1", "link2"]
    # Mock similarity groups
    robot._similarity.groups = [["link1", "link2"]]

    with patch("viser.ViserServer", return_value=server):
        gui = _SpheresGui(server, robot, "dummy.json")
        return gui


def test_gui_state_flags(mock_gui):
    assert mock_gui.needs_spherize is True
    mock_gui.mark_spherized()
    assert mock_gui.needs_spherize is False

    assert mock_gui.needs_refine_update is True
    mock_gui.mark_refine_updated()
    assert mock_gui.needs_refine_update is False


def test_get_group_for_link(mock_gui):
    group = mock_gui._get_group_for_link("link1")
    assert group == ["link1", "link2"]

    group_none = mock_gui._get_group_for_link("nonexistent")
    assert group_none is None


def test_gui_properties(mock_gui):
    mock_gui._mode.value = "Auto"
    assert mock_gui.is_auto_mode is True

    mock_gui._total_spheres.value = 10
    assert mock_gui.total_spheres == 10

    mock_gui._opacity.value = 0.5
    assert mock_gui.opacity == 0.5
