import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from scripts.configure_sphere_collision import _SpheresGui, BallparkConfig, SpherePreset


@pytest.fixture
def mock_gui():
    server = MagicMock()

    # Mock gui components to have a .value attribute
    def mock_component(initial_value=None):
        m = MagicMock()
        m.value = initial_value
        return m

    server.gui.add_tab_group.return_value.add_tab.return_value.__enter__.return_value = MagicMock()
    server.gui.add_folder.return_value.__enter__.return_value = MagicMock()
    server.gui.add_checkbox.side_effect = lambda *a, **k: mock_component(
        k.get("initial_value")
    )
    server.gui.add_slider.side_effect = lambda *a, **k: mock_component(
        k.get("initial_value")
    )
    server.gui.add_dropdown.side_effect = lambda *a, **k: mock_component(
        k.get("initial_value")
    )
    server.gui.add_text.side_effect = lambda *a, **k: mock_component(
        k.get("initial_value")
    )
    server.gui.add_number.side_effect = lambda *a, **k: mock_component(
        k.get("initial_value")
    )
    server.gui.add_button.return_value = MagicMock()

    robot = MagicMock()
    robot.collision_links = ["link1", "link2"]
    robot.joint_limits = (np.array([0.0]), np.array([1.0]))
    robot.links = ["link1", "link2"]
    robot._similarity.groups = [["link1", "link2"]]
    robot._link_meshes = {"link1": MagicMock(), "link2": MagicMock()}

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


def test_apply_preset(mock_gui):
    mock_gui._apply_preset("Balanced")
    assert (
        mock_gui._current_config.spherize.padding
        == BallparkConfig.from_preset(SpherePreset.BALANCED).spherize.padding
    )

    mock_gui._apply_preset("Custom")
    assert mock_gui._params_folder is not None


def test_get_config(mock_gui):
    mock_gui._preset.value = "Custom"
    mock_gui._apply_preset("Custom")

    mock_gui._params_sliders["padding"].value = 1.1
    cfg = mock_gui.get_config()
    assert cfg.spherize.padding == 1.1


def test_update_sliders_from_allocation(mock_gui):
    alloc = {"link1": 5, "link2": 10}
    mock_gui.update_sliders_from_allocation(alloc)
    assert mock_gui._link_sliders["link1"].value == 5
    assert mock_gui._link_sliders["link2"].value == 10


def test_manual_disable_logic(mock_gui):
    mock_gui._manual_disable_dropdown.value = "link1"
    # Find the callback for the button
    callback = mock_gui._manual_disable_button.on_click.call_args[0][0]
    callback(None)

    assert "link1" in mock_gui._user_fully_disabled_links

    # Test removal
    mock_gui._remove_disabled_link("link1")
    assert "link1" not in mock_gui._user_fully_disabled_links


def test_poll_detection(mock_gui):
    # Change preset
    mock_gui._preset.value = "Custom"
    mock_gui.poll()
    assert mock_gui._last_preset == "Custom"
    assert mock_gui.needs_spherize is True

    # Change mode
    mock_gui._mode.value = "Manual"
    mock_gui.poll()
    assert mock_gui._last_mode == "Manual"

    # Change joint config
    mock_gui._joint_sliders = [MagicMock(value=0.5)]
    mock_gui.poll()
    assert mock_gui.needs_refine_update is True


def test_update_highlights(mock_gui):
    mock_gui._highlight_link_dropdown.value = "link1"
    mock_gui._generated_ignore_pairs = [["link1", "link2"]]
    mock_gui._update_highlights()
    # Check if meshes were added to scene
    mock_gui._server.scene.add_mesh_simple.assert_called()
    assert mock_gui._ignored_links_text.value == "link2"
