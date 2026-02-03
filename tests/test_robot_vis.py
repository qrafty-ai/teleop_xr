import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from teleop_xr.robot_vis import RobotVisModule
from teleop_xr.config import RobotVisConfig


@pytest.fixture
def test_app():
    return FastAPI()


@pytest.fixture
def mock_config(tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text("<robot name='test'></robot>")

    mesh_path = tmp_path / "meshes"
    mesh_path.mkdir()
    (mesh_path / "test.stl").write_bytes(b"stl_content")
    (mesh_path / "test.dae").write_bytes(b"dae_content")
    (mesh_path / "test.obj").write_bytes(b"obj_content")

    return RobotVisConfig(urdf_path=str(urdf_path), mesh_path=str(mesh_path))


@pytest.fixture
def client(test_app, mock_config):
    RobotVisModule(test_app, mock_config)
    return TestClient(test_app)


def test_get_robot_urdf(client):
    response = client.get("/robot_assets/robot.urdf")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/xml"
    assert "<robot name='test'></robot>" in response.text


def test_get_mesh_asset(client):
    response = client.get("/robot_assets/package://test.stl")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.content == b"stl_content"


def test_get_mesh_asset_types(client):
    response = client.get("/robot_assets/package://test.dae")
    assert response.status_code == 200
    assert response.headers["content-type"] == "model/vnd.collada+xml"

    response = client.get("/robot_assets/package://test.obj")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_get_mesh_no_mesh_path(test_app, tmp_path):
    dummy_file = tmp_path / "foo.stl"
    dummy_file.write_bytes(b"foo")

    config = RobotVisConfig(urdf_path=str(tmp_path / "robot.urdf"), mesh_path=None)
    RobotVisModule(test_app, config)
    client = TestClient(test_app)

    path_str = str(dummy_file)
    response = client.get(f"/robot_assets/package://{path_str}")
    assert response.status_code == 200
    assert response.content == b"foo"


def test_get_mesh_asset_no_extension(client, tmp_path):
    dummy_file = tmp_path / "meshes" / "no_ext"
    dummy_file.write_bytes(b"content")
    response = client.get("/robot_assets/package://no_ext")
    assert response.status_code == 200
    assert response.content == b"content"


def test_get_asset_not_found(client):
    response = client.get("/robot_assets/nonexistent.file")
    assert response.status_code == 404


def test_frontend_config(test_app, mock_config):
    module = RobotVisModule(test_app, mock_config)
    config = module.get_frontend_config()
    assert config["urdf_url"] == "/robot_assets/robot.urdf"
    assert config["model_scale"] == 1.0
    assert config["initial_rotation_euler"] == [0.0, 0.0, 0.0]


@pytest.mark.anyio
async def test_broadcast_state(test_app, mock_config):
    module = RobotVisModule(test_app, mock_config)

    class MockConnectionManager:
        def __init__(self):
            self.last_message = None

        async def broadcast(self, message):
            self.last_message = message

    manager = MockConnectionManager()
    joints = {"joint1": 0.5}

    await module.broadcast_state(manager, joints)

    assert manager.last_message is not None
    data = json.loads(manager.last_message)
    assert data["type"] == "robot_state"
    assert data["data"]["joints"] == joints
