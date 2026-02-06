import pytest
from unittest.mock import MagicMock
import numpy as np
from scripts.configure_sphere_collision import _SphereVisuals, SPHERE_COLORS
from ballpark import RobotSpheresResult, Sphere


@pytest.fixture
def mock_server():
    server = MagicMock()
    server.scene.add_frame.return_value = MagicMock()
    server.scene.add_icosphere.return_value = MagicMock()
    return server


@pytest.fixture
def sphere_visuals(mock_server):
    link_names = ["link1", "link2"]
    return _SphereVisuals(mock_server, link_names)


def test_update_visuals(sphere_visuals, mock_server):
    sphere1 = Sphere(center=np.array([1.0, 2.0, 3.0]), radius=np.array(0.1))
    sphere2 = Sphere(center=np.array([4.0, 5.0, 6.0]), radius=np.array(0.2))

    result = RobotSpheresResult(link_spheres={"link1": [sphere1], "link2": [sphere2]})

    sphere_visuals.update(result, opacity=0.8, visible=True)

    assert len(sphere_visuals._frames) == 2
    assert len(sphere_visuals._handles) == 2
    assert mock_server.scene.add_frame.call_count == 2
    assert mock_server.scene.add_icosphere.call_count == 2

    c = SPHERE_COLORS[0]
    expected_color = (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)

    mock_server.scene.add_icosphere.assert_any_call(
        "/sphere_frames/link1_0/sphere",
        radius=0.1,
        position=(1.0, 2.0, 3.0),
        color=expected_color,
        opacity=0.8,
    )


def test_update_visuals_invisible(sphere_visuals, mock_server):
    result = RobotSpheresResult(
        link_spheres={
            "link1": [Sphere(center=np.array([0.0, 0.0, 0.0]), radius=np.array(0.1))]
        }
    )

    sphere_visuals.update(result, opacity=0.8, visible=False)

    assert len(sphere_visuals._frames) == 0
    assert len(sphere_visuals._handles) == 0
    assert mock_server.scene.add_frame.call_count == 0


def test_update_transforms(sphere_visuals, mock_server):
    sphere1 = Sphere(center=np.array([0.0, 0.0, 0.0]), radius=np.array(0.1))
    result = RobotSpheresResult(link_spheres={"link1": [sphere1]})
    sphere_visuals.update(result, opacity=1.0, visible=True)

    frame_mock = sphere_visuals._frames["link1_0"]

    Ts = np.zeros((2, 7))
    Ts[0] = [1, 0, 0, 0, 10, 20, 30]

    sphere_visuals.update_transforms(Ts)

    assert np.allclose(frame_mock.wxyz, [1, 0, 0, 0])
    assert np.allclose(frame_mock.position, [10, 20, 30])
