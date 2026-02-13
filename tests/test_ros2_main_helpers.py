import sys
from unittest.mock import MagicMock

import pytest

import teleop_xr.ros2.__main__ as ros2_main


def test_ros2_list_available_robots(monkeypatch):
    sentinel = {"franka": "path"}
    monkeypatch.setattr(ros2_main, "_default_list_available_robots", lambda: sentinel)

    assert ros2_main.list_available_robots() is sentinel


def test_ros2_list_robots_or_exit_logs_none(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(ros2_main, "logger", fake_logger)
    monkeypatch.setattr(ros2_main, "_default_list_available_robots", lambda: {})

    ros2_main.list_robots_or_exit()

    fake_logger.info.assert_any_call("Available robots (via entry points):")
    fake_logger.info.assert_any_call("  None")


def test_ros2_list_robots_or_exit_logs_entries(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(ros2_main, "logger", fake_logger)
    monkeypatch.setattr(
        ros2_main, "_default_list_available_robots", lambda: {"franka": "path"}
    )

    ros2_main.list_robots_or_exit()

    fake_logger.info.assert_any_call("  franka: path")


def test_ros2_list_robots_or_exit_handles_import_error(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(ros2_main, "logger", fake_logger)

    def raise_import():
        raise ImportError("broken")

    monkeypatch.setattr(ros2_main, "_default_list_available_robots", raise_import)

    with pytest.raises(SystemExit) as excinfo:
        ros2_main.list_robots_or_exit()

    assert excinfo.value.code == 1
    fake_logger.error.assert_called_once()


def test_ros2_load_robot_class(monkeypatch):
    dummy = MagicMock()
    loader_module = MagicMock()
    loader_module.load_robot_class = MagicMock(return_value=dummy)
    monkeypatch.setitem(sys.modules, "teleop_xr.ik.loader", loader_module)

    assert ros2_main.load_robot_class("custom") is dummy
