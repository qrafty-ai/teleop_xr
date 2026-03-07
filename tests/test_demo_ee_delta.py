import logging
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("uvicorn")
pytest.importorskip("rich")

from teleop_xr.demo import __main__ as demo_main
from teleop_xr.demo.__main__ import (
    TerminalKeyReader,
    run_right_ee_absolute_demo,
    run_right_ee_delta_demo,
)


class DummyController:
    def __init__(self):
        self.mode = "teleop"
        self.mode_history = [self.mode]

    def get_mode(self):
        return self.mode

    def set_mode(self, mode):
        self.mode = str(mode)
        self.mode_history.append(self.mode)

    def submit_ee_delta(self, command, q_current):
        assert command["frame"] == "right"
        return np.array(q_current) + 1.0

    def submit_ee_absolute(self, command, q_current):
        assert command["frame"] == "right"
        assert "target_pose" in command
        return np.array(q_current) + 1.0


class DummyRobot:
    actuated_joint_names = ["j1", "j2"]

    def forward_kinematics(self, _config):
        class _Pose:
            @staticmethod
            def translation():
                return np.array([0.0, 0.0, 0.0])

            class _Rotation:
                wxyz = np.array([1.0, 0.0, 0.0, 0.0])

            @staticmethod
            def rotation():
                return _Pose._Rotation()

        return {"right": _Pose()}


class DummyTeleop:
    async def publish_joint_state(self, _joint_dict):
        return None


class DummyLoop:
    def __init__(self, running: bool):
        self._running = running

    def is_running(self):
        return self._running


def test_run_right_ee_delta_demo_switches_mode_and_restores(monkeypatch):
    monkeypatch.setattr("teleop_xr.demo.__main__.time.sleep", lambda _s: None)

    controller = DummyController()
    robot = DummyRobot()
    teleop = DummyTeleop()
    q0 = np.array([0.0, 0.0])

    out = run_right_ee_delta_demo(
        controller=cast(Any, controller),
        robot=cast(Any, robot),
        teleop=cast(Any, teleop),
        q_current=q0,
        teleop_loop=None,
        logger=logging.getLogger("demo-test"),
    )

    np.testing.assert_allclose(out, [40.0, 40.0])
    assert controller.mode == "teleop"
    assert controller.mode_history[1] == "ee_delta"
    assert controller.mode_history[-1] == "teleop"


def test_run_right_ee_delta_demo_publishes_when_loop_running(monkeypatch):
    monkeypatch.setattr("teleop_xr.demo.__main__.time.sleep", lambda _s: None)

    run_coroutine_threadsafe = Mock()

    def fake_run_coroutine_threadsafe(coro, _loop):
        coro.close()
        run_coroutine_threadsafe()
        return SimpleNamespace()

    monkeypatch.setattr(
        "teleop_xr.demo.__main__.asyncio.run_coroutine_threadsafe",
        fake_run_coroutine_threadsafe,
    )

    controller = DummyController()
    robot = DummyRobot()
    teleop = DummyTeleop()
    q0 = np.array([0.0, 0.0])

    out = run_right_ee_delta_demo(
        controller=cast(Any, controller),
        robot=cast(Any, robot),
        teleop=cast(Any, teleop),
        q_current=q0,
        teleop_loop=cast(Any, DummyLoop(running=True)),
        logger=logging.getLogger("demo-test"),
    )

    np.testing.assert_allclose(out, [40.0, 40.0])
    assert run_coroutine_threadsafe.call_count == 40


def test_run_right_ee_delta_demo_restores_mode_on_error(monkeypatch):
    monkeypatch.setattr("teleop_xr.demo.__main__.time.sleep", lambda _s: None)

    class RaisingController(DummyController):
        def submit_ee_delta(self, command, q_current):
            raise RuntimeError(f"boom: {command['frame']}")

    controller = RaisingController()
    robot = DummyRobot()
    teleop = DummyTeleop()

    with pytest.raises(RuntimeError):
        run_right_ee_delta_demo(
            controller=cast(Any, controller),
            robot=cast(Any, robot),
            teleop=cast(Any, teleop),
            q_current=np.array([0.0, 0.0]),
            teleop_loop=None,
            logger=logging.getLogger("demo-test"),
        )

    assert controller.mode == "teleop"
    assert controller.mode_history[1] == "ee_delta"
    assert controller.mode_history[-1] == "teleop"


def test_run_right_ee_absolute_demo_switches_mode_and_restores(monkeypatch):
    monkeypatch.setattr("teleop_xr.demo.__main__.time.sleep", lambda _s: None)

    controller = DummyController()
    robot = DummyRobot()
    teleop = DummyTeleop()
    q0 = np.array([0.0, 0.0])

    out = run_right_ee_absolute_demo(
        controller=cast(Any, controller),
        robot=cast(Any, robot),
        teleop=cast(Any, teleop),
        q_current=q0,
        teleop_loop=None,
        logger=logging.getLogger("demo-test"),
    )

    np.testing.assert_allclose(out, [40.0, 40.0])
    assert controller.mode == "teleop"
    assert controller.mode_history[1] == "ee_absolute"
    assert controller.mode_history[-1] == "teleop"


def test_run_right_ee_absolute_demo_publishes_when_loop_running(monkeypatch):
    monkeypatch.setattr("teleop_xr.demo.__main__.time.sleep", lambda _s: None)

    run_coroutine_threadsafe = Mock()

    def fake_run_coroutine_threadsafe(coro, _loop):
        coro.close()
        run_coroutine_threadsafe()
        return SimpleNamespace()

    monkeypatch.setattr(
        "teleop_xr.demo.__main__.asyncio.run_coroutine_threadsafe",
        fake_run_coroutine_threadsafe,
    )

    controller = DummyController()
    robot = DummyRobot()
    teleop = DummyTeleop()
    q0 = np.array([0.0, 0.0])

    out = run_right_ee_absolute_demo(
        controller=cast(Any, controller),
        robot=cast(Any, robot),
        teleop=cast(Any, teleop),
        q_current=q0,
        teleop_loop=cast(Any, DummyLoop(running=True)),
        logger=logging.getLogger("demo-test"),
    )

    np.testing.assert_allclose(out, [40.0, 40.0])
    assert run_coroutine_threadsafe.call_count == 40


def test_run_right_ee_absolute_demo_restores_mode_on_error(monkeypatch):
    monkeypatch.setattr("teleop_xr.demo.__main__.time.sleep", lambda _s: None)

    class RaisingController(DummyController):
        def submit_ee_absolute(self, command, q_current):
            raise RuntimeError("boom")

    controller = RaisingController()
    robot = DummyRobot()
    teleop = DummyTeleop()

    with pytest.raises(RuntimeError):
        run_right_ee_absolute_demo(
            controller=cast(Any, controller),
            robot=cast(Any, robot),
            teleop=cast(Any, teleop),
            q_current=np.array([0.0, 0.0]),
            teleop_loop=None,
            logger=logging.getLogger("demo-test"),
        )

    assert controller.mode == "teleop"
    assert controller.mode_history[1] == "ee_absolute"
    assert controller.mode_history[-1] == "teleop"


def test_terminal_key_reader_disabled_does_not_touch_termios():
    reader = TerminalKeyReader(enabled=False)
    with reader as entered:
        assert entered is reader
    assert reader.poll_key() is None


def test_terminal_key_reader_happy_path(monkeypatch):
    stdin = SimpleNamespace(isatty=lambda: True, fileno=lambda: 42)
    monkeypatch.setattr(demo_main.sys, "stdin", stdin)
    monkeypatch.setattr(demo_main.sys, "platform", "linux")

    mock_termios = SimpleNamespace(
        TCSADRAIN=1,
        tcgetattr=Mock(return_value=[1, 2, 3]),
        tcsetattr=Mock(),
    )
    mock_tty = SimpleNamespace(setcbreak=Mock())
    monkeypatch.setattr(demo_main, "_termios", mock_termios)
    monkeypatch.setattr(demo_main, "_tty", mock_tty)

    monkeypatch.setattr(
        demo_main.select, "select", lambda *_args, **_kwargs: ([stdin], [], [])
    )
    monkeypatch.setattr(demo_main.os, "read", lambda _fd, _n: b"d")

    with TerminalKeyReader(enabled=True) as reader:
        assert reader.poll_key() == "d"

    mock_termios.tcgetattr.assert_called_once_with(42)
    mock_tty.setcbreak.assert_called_once_with(42)
    mock_termios.tcsetattr.assert_called_once_with(
        42, mock_termios.TCSADRAIN, [1, 2, 3]
    )


def test_terminal_key_reader_poll_key_empty_paths(monkeypatch):
    stdin = SimpleNamespace(isatty=lambda: True, fileno=lambda: 7)
    monkeypatch.setattr(demo_main.sys, "stdin", stdin)
    monkeypatch.setattr(demo_main.sys, "platform", "linux")

    mock_termios = SimpleNamespace(
        TCSADRAIN=1,
        tcgetattr=Mock(return_value=[1]),
        tcsetattr=Mock(),
    )
    mock_tty = SimpleNamespace(setcbreak=Mock())
    monkeypatch.setattr(demo_main, "_termios", mock_termios)
    monkeypatch.setattr(demo_main, "_tty", mock_tty)

    with TerminalKeyReader(enabled=True) as reader:
        monkeypatch.setattr(
            demo_main.select,
            "select",
            lambda *_args, **_kwargs: ([], [], []),
        )
        assert reader.poll_key() is None

        monkeypatch.setattr(
            demo_main.select, "select", lambda *_args, **_kwargs: ([stdin], [], [])
        )
        monkeypatch.setattr(demo_main.os, "read", lambda _fd, _n: b"")
        assert reader.poll_key() is None


def test_main_ik_tui_keybinding_runs_ee_demo(monkeypatch):
    monkeypatch.setattr(
        demo_main.tyro, "cli", lambda _cls: demo_main.DemoCLI(mode="ik")
    )
    monkeypatch.setattr(demo_main, "ensure_ik_dependencies", lambda: None)
    monkeypatch.setattr(demo_main, "build_camera_views_config", lambda **_kwargs: {})
    monkeypatch.setattr(demo_main.time, "sleep", lambda _s: None)

    class FakeRobot:
        default_speed_ratio = 1.0
        actuated_joint_names = ["j1", "j2"]

        def __init__(self, **_kwargs):
            pass

        def get_default_config(self):
            return np.array([0.0, 0.0])

        def get_vis_config(self):
            return None

    class FakeController:
        def __init__(self, _robot, _solver):
            self._mode = SimpleNamespace(value="teleop")

        def get_mode(self):
            return self._mode

    fake_ik_pkg = ModuleType("teleop_xr.ik")
    fake_ik_loader = ModuleType("teleop_xr.ik.loader")
    fake_ik_solver = ModuleType("teleop_xr.ik.solver")
    fake_ik_controller = ModuleType("teleop_xr.ik.controller")
    setattr(fake_ik_loader, "load_robot_class", lambda _robot_class: FakeRobot)
    setattr(fake_ik_solver, "PyrokiSolver", lambda _robot: object())
    setattr(fake_ik_controller, "IKController", FakeController)

    monkeypatch.setitem(demo_main.sys.modules, "teleop_xr.ik", fake_ik_pkg)
    monkeypatch.setitem(demo_main.sys.modules, "teleop_xr.ik.loader", fake_ik_loader)
    monkeypatch.setitem(demo_main.sys.modules, "teleop_xr.ik.solver", fake_ik_solver)
    monkeypatch.setitem(
        demo_main.sys.modules, "teleop_xr.ik.controller", fake_ik_controller
    )

    class FakeTeleop:
        def __init__(self, settings):
            self.settings = settings
            self.mode_provider = None

        def set_pose(self, _pose):
            return None

        def subscribe(self, _callback):
            return None

        def bind_control_mode_provider(self, provider):
            self.mode_provider = provider

        def run(self):
            return None

        async def publish_joint_state(self, _joint_dict):
            return None

    monkeypatch.setattr(demo_main, "Teleop", FakeTeleop)

    class FakeIKWorker:
        def __init__(self, _controller, _robot, _teleop, _state_container, _logger):
            self.teleop_loop = DummyLoop(running=True)
            self.running = True

        def start(self):
            return None

        def join(self):
            return None

        def set_teleop_loop(self, _loop):
            return None

        def update_state(self, _state):
            return None

    monkeypatch.setattr(demo_main, "IKWorker", FakeIKWorker)

    class FakeThread:
        def __init__(self, target=None, daemon=False):
            self._target = target
            self.daemon = daemon
            self._alive_checks = 0

        def start(self):
            if self._target is not None:
                self._target()

        def is_alive(self):
            name = getattr(self._target, "__name__", "")
            if name == "run":
                if self._alive_checks == 0:
                    self._alive_checks += 1
                    return True
                return False
            return False

        def join(self):
            return None

    monkeypatch.setattr(demo_main.threading, "Thread", FakeThread)

    class DummyLock:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return None

    monkeypatch.setattr(demo_main.threading, "Lock", lambda: DummyLock())

    class FakeKeyReader:
        def __init__(self, enabled):
            self.enabled = enabled
            self._calls = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return None

        def poll_key(self):
            if self._calls == 0:
                self._calls += 1
                return "d"
            return None

    monkeypatch.setattr(demo_main, "TerminalKeyReader", FakeKeyReader)

    class FakeLive:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return None

    monkeypatch.setattr(demo_main, "Live", FakeLive)
    monkeypatch.setattr(
        demo_main, "generate_state_table", lambda *_args, **_kwargs: "state"
    )
    monkeypatch.setattr(
        demo_main, "generate_ik_status_table", lambda *_args, **_kwargs: "status"
    )
    monkeypatch.setattr(
        demo_main, "generate_ik_controls_panel", lambda *_args, **_kwargs: "controls"
    )
    monkeypatch.setattr(
        demo_main, "generate_log_panel", lambda *_args, **_kwargs: "logs"
    )

    calls = []

    def fake_run_right_ee_delta_demo(*_args, **_kwargs):
        calls.append("called")
        return np.array([1.0, 2.0])

    monkeypatch.setattr(
        demo_main, "run_right_ee_delta_demo", fake_run_right_ee_delta_demo
    )

    demo_main.main()
    assert calls == ["called"]


def test_main_ik_tui_keybinding_runs_ee_absolute_demo(monkeypatch):
    monkeypatch.setattr(
        demo_main.tyro, "cli", lambda _cls: demo_main.DemoCLI(mode="ik")
    )
    monkeypatch.setattr(demo_main, "ensure_ik_dependencies", lambda: None)
    monkeypatch.setattr(demo_main, "build_camera_views_config", lambda **_kwargs: {})
    monkeypatch.setattr(demo_main.time, "sleep", lambda _s: None)

    class FakeRobot:
        default_speed_ratio = 1.0
        actuated_joint_names = ["j1", "j2"]

        def __init__(self, **_kwargs):
            pass

        def get_default_config(self):
            return np.array([0.0, 0.0])

        def get_vis_config(self):
            return None

    class FakeController:
        def __init__(self, _robot, _solver):
            self._mode = SimpleNamespace(value="teleop")

        def get_mode(self):
            return self._mode

    fake_ik_pkg = ModuleType("teleop_xr.ik")
    fake_ik_loader = ModuleType("teleop_xr.ik.loader")
    fake_ik_solver = ModuleType("teleop_xr.ik.solver")
    fake_ik_controller = ModuleType("teleop_xr.ik.controller")
    setattr(fake_ik_loader, "load_robot_class", lambda _robot_class: FakeRobot)
    setattr(fake_ik_solver, "PyrokiSolver", lambda _robot: object())
    setattr(fake_ik_controller, "IKController", FakeController)

    monkeypatch.setitem(demo_main.sys.modules, "teleop_xr.ik", fake_ik_pkg)
    monkeypatch.setitem(demo_main.sys.modules, "teleop_xr.ik.loader", fake_ik_loader)
    monkeypatch.setitem(demo_main.sys.modules, "teleop_xr.ik.solver", fake_ik_solver)
    monkeypatch.setitem(
        demo_main.sys.modules, "teleop_xr.ik.controller", fake_ik_controller
    )

    class FakeTeleop:
        def __init__(self, settings):
            self.settings = settings
            self.mode_provider = None

        def set_pose(self, _pose):
            return None

        def subscribe(self, _callback):
            return None

        def bind_control_mode_provider(self, provider):
            self.mode_provider = provider

        def run(self):
            return None

        async def publish_joint_state(self, _joint_dict):
            return None

    monkeypatch.setattr(demo_main, "Teleop", FakeTeleop)

    class FakeIKWorker:
        def __init__(self, _controller, _robot, _teleop, _state_container, _logger):
            self.teleop_loop = DummyLoop(running=True)
            self.running = True

        def start(self):
            return None

        def join(self):
            return None

        def set_teleop_loop(self, _loop):
            return None

        def update_state(self, _state):
            return None

    monkeypatch.setattr(demo_main, "IKWorker", FakeIKWorker)

    class FakeThread:
        def __init__(self, target=None, daemon=False):
            self._target = target
            self.daemon = daemon
            self._alive_checks = 0

        def start(self):
            if self._target is not None:
                self._target()

        def is_alive(self):
            name = getattr(self._target, "__name__", "")
            if name == "run":
                if self._alive_checks == 0:
                    self._alive_checks += 1
                    return True
                return False
            return False

        def join(self):
            return None

    monkeypatch.setattr(demo_main.threading, "Thread", FakeThread)

    class DummyLock:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return None

    monkeypatch.setattr(demo_main.threading, "Lock", lambda: DummyLock())

    class FakeKeyReader:
        def __init__(self, enabled):
            self.enabled = enabled
            self._calls = 0

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return None

        def poll_key(self):
            if self._calls == 0:
                self._calls += 1
                return "a"
            return None

    monkeypatch.setattr(demo_main, "TerminalKeyReader", FakeKeyReader)

    class FakeLive:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return None

    monkeypatch.setattr(demo_main, "Live", FakeLive)
    monkeypatch.setattr(
        demo_main, "generate_state_table", lambda *_args, **_kwargs: "state"
    )
    monkeypatch.setattr(
        demo_main, "generate_ik_status_table", lambda *_args, **_kwargs: "status"
    )
    monkeypatch.setattr(
        demo_main, "generate_ik_controls_panel", lambda *_args, **_kwargs: "controls"
    )
    monkeypatch.setattr(
        demo_main, "generate_log_panel", lambda *_args, **_kwargs: "logs"
    )

    calls = []

    def fake_run_right_ee_absolute_demo(*_args, **_kwargs):
        calls.append("called")
        return np.array([1.0, 2.0])

    monkeypatch.setattr(
        demo_main, "run_right_ee_absolute_demo", fake_run_right_ee_absolute_demo
    )

    demo_main.main()
    assert calls == ["called"]
