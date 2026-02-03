import numpy as np
import pytest

import teleop_xr
from teleop_xr.config import TeleopSettings


def test_resolve_frontend_paths_prefers_webxr_dist(tmp_path):
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    repo_root = package_dir.parent
    webxr_dist = repo_root / "webxr" / "dist"
    webxr_dist.mkdir(parents=True)
    (webxr_dist / "index.html").write_text("ok")

    static_dir, index_path, mount_path, mount_name = teleop_xr._resolve_frontend_paths(
        str(package_dir)
    )
    assert static_dir == str(webxr_dist)
    assert index_path == str(webxr_dist / "index.html")
    assert mount_path == "/"
    assert mount_name == "webxr"


def test_resolve_frontend_paths_falls_back_to_package_dist(tmp_path):
    package_dir = tmp_path / "pkg"
    package_dir.mkdir()
    static_dir, index_path, mount_path, mount_name = teleop_xr._resolve_frontend_paths(
        str(package_dir)
    )

    assert static_dir == str(package_dir / "dist")
    assert index_path == str(package_dir / "dist" / "index.html")
    assert mount_path == "/"
    assert mount_name == "webxr"


def test_get_local_ip_success(monkeypatch: pytest.MonkeyPatch):
    class DummySocket:
        def connect(self, addr):
            return None

        def getsockname(self):
            return ("192.168.0.2", 1234)

        def close(self):
            return None

    monkeypatch.setattr(
        teleop_xr.socket, "socket", lambda *args, **kwargs: DummySocket()
    )
    assert teleop_xr.get_local_ip() == "192.168.0.2"


def test_get_local_ip_error(monkeypatch: pytest.MonkeyPatch):
    class DummySocket:
        def connect(self, addr):
            raise RuntimeError("no network")

    monkeypatch.setattr(
        teleop_xr.socket, "socket", lambda *args, **kwargs: DummySocket()
    )
    assert "Error:" in teleop_xr.get_local_ip()


def test_are_close_translation_and_rotation():
    a = np.eye(4)
    b = np.eye(4)
    assert teleop_xr.are_close(a, b)

    b2 = np.eye(4)
    b2[:3, 3] = [0.1, 0.0, 0.0]
    assert teleop_xr.are_close(a, b2, lin_tol=0.11)
    assert teleop_xr.are_close(a, b2, lin_tol=0.05) is False


def test_slerp_handles_negative_dot():
    q1 = np.array([1.0, 0.0, 0.0, 0.0])
    q2 = np.array([-1.0, 0.0, 0.0, 0.0])
    out = teleop_xr.slerp(q1, q2, 0.5)
    np.testing.assert_allclose(out, q1)


def test_interpolate_transforms_translation_endpoints():
    T1 = np.eye(4)
    T2 = np.eye(4)
    T2[:3, 3] = [2.0, 0.0, 0.0]

    out0 = teleop_xr.interpolate_transforms(T1, T2, 0.0)
    out1 = teleop_xr.interpolate_transforms(T1, T2, 1.0)
    out05 = teleop_xr.interpolate_transforms(T1, T2, 0.5)

    np.testing.assert_allclose(out0, T1)
    np.testing.assert_allclose(out1, T2)
    np.testing.assert_allclose(out05[:3, 3], [1.0, 0.0, 0.0])


@pytest.mark.anyio
async def test_connection_manager_broadcast_removes_failed():
    from unittest.mock import AsyncMock

    ok_ws = AsyncMock()
    ok_ws.send_text = AsyncMock()

    bad_ws = AsyncMock()

    async def raise_send(_msg: str):
        raise RuntimeError("fail")

    bad_ws.send_text = AsyncMock(side_effect=raise_send)

    manager = teleop_xr.ConnectionManager()
    manager.active_connections = [ok_ws, bad_ws]

    await manager.broadcast("hi")

    assert ok_ws in manager.active_connections
    assert bad_ws not in manager.active_connections


def _make_teleop(monkeypatch: pytest.MonkeyPatch, tmp_path):
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    index_path = static_dir / "index.html"
    index_path.write_text("ok")

    monkeypatch.setattr(
        teleop_xr,
        "_resolve_frontend_paths",
        lambda _package_dir: (str(static_dir), str(index_path), "/", "webxr"),
    )

    return teleop_xr.Teleop(TeleopSettings())


def test_teleop_apply_move_false_notifies(monkeypatch: pytest.MonkeyPatch, tmp_path):
    t = _make_teleop(monkeypatch, tmp_path)
    got = []

    def cb(pose, info):
        got.append((pose.copy(), dict(info)))

    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]
    t.set_pose(pose)
    t.subscribe(cb)

    t.apply(
        {"x": 0.0, "y": 0.0, "z": 0.0},
        {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        move=False,
    )
    assert len(got) == 1
    np.testing.assert_allclose(got[0][0], pose)


def test_teleop_handle_xr_state_uses_controller_then_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    t = _make_teleop(monkeypatch, tmp_path)
    calls = []

    def record(position, orientation, move=True, scale=1.0, info=None):
        calls.append((position, orientation, move, scale, info))

    t.apply = record

    message = {
        "fetch_latency_ms": 1.0,
        "devices": [
            {
                "role": "head",
                "pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                },
            },
            {
                "role": "controller",
                "handedness": "right",
                "gripPose": {
                    "position": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                },
            },
        ],
    }
    handle = getattr(t, "_Teleop__handle_xr_state")
    handle(message)
    assert len(calls) == 1
    assert calls[0][0]["x"] == 1.0

    calls.clear()
    message2 = {
        "devices": [
            {
                "role": "head",
                "pose": {
                    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "orientation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
                },
            }
        ]
    }
    handle = getattr(t, "_Teleop__handle_xr_state")
    handle(message2)
    assert len(calls) == 1


@pytest.mark.anyio
async def test_teleop_handle_video_message_routes(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    from unittest.mock import AsyncMock, MagicMock

    t = _make_teleop(monkeypatch, tmp_path)
    ws = MagicMock()

    start_video = AsyncMock()
    monkeypatch.setattr(t, "_start_video_session", start_video)

    await t._handle_video_message(ws, {"type": "video_request"})
    start_video.assert_awaited_once()

    session = MagicMock()
    video_sessions = getattr(t, "_Teleop__video_sessions")
    video_sessions[ws] = session
    route = AsyncMock()
    monkeypatch.setattr(teleop_xr, "route_video_message", route)
    await t._handle_video_message(ws, {"type": "video_answer", "data": {"sdp": "sdp"}})
    route.assert_awaited()


def test_teleop_run_calls_uvicorn(monkeypatch: pytest.MonkeyPatch, tmp_path):
    from unittest.mock import MagicMock

    t = _make_teleop(monkeypatch, tmp_path)
    run_mock = MagicMock()
    monkeypatch.setattr(teleop_xr.uvicorn, "run", run_mock)
    monkeypatch.setattr(teleop_xr, "get_local_ip", lambda: "127.0.0.1")

    t.run()
    assert run_mock.call_count == 1
