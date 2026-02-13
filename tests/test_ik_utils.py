import builtins
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import loguru

import teleop_xr.ik_utils as ik_utils


def test_ensure_ik_dependencies_sets_cpu(monkeypatch):
    fake_jax = SimpleNamespace(config=MagicMock())
    monkeypatch.setitem(sys.modules, "jax", fake_jax)

    ik_utils.ensure_ik_dependencies()

    fake_jax.config.update.assert_called_once_with("jax_platform_name", "cpu")


def test_ensure_ik_dependencies_exits_when_missing(monkeypatch, capsys):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "jax":
            raise ImportError("no jax")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(SystemExit) as excinfo:
        ik_utils.ensure_ik_dependencies()

    assert excinfo.value.code == 1
    stderr = capsys.readouterr().err
    assert "pip install 'teleop-xr[ik]'" in stderr


def test_list_available_robots_delegates(monkeypatch):
    loader = SimpleNamespace(list_available_robots=lambda: {"foo": "bar"})
    monkeypatch.setitem(sys.modules, "teleop_xr.ik.loader", loader)

    assert ik_utils.list_available_robots() == {"foo": "bar"}


def test_list_robots_or_exit_logs_empty(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(loguru, "logger", fake_logger)
    monkeypatch.setattr(ik_utils, "list_available_robots", lambda: {})

    with pytest.raises(SystemExit) as excinfo:
        ik_utils.list_robots_or_exit()

    assert excinfo.value.code == 0
    fake_logger.info.assert_any_call("Available robots (via entry points):")
    fake_logger.info.assert_any_call("  None")


def test_list_robots_or_exit_logs_entries(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(loguru, "logger", fake_logger)
    monkeypatch.setattr(ik_utils, "list_available_robots", lambda: {"franka": "path"})

    with pytest.raises(SystemExit):
        ik_utils.list_robots_or_exit()

    fake_logger.info.assert_any_call("  franka: path")


def test_list_robots_or_exit_handles_import_error(monkeypatch):
    fake_logger = MagicMock()
    monkeypatch.setattr(loguru, "logger", fake_logger)

    def raise_import():
        raise ImportError("missing ik")

    monkeypatch.setattr(ik_utils, "list_available_robots", raise_import)

    with pytest.raises(SystemExit) as excinfo:
        ik_utils.list_robots_or_exit()

    assert excinfo.value.code == 1
    fake_logger.error.assert_called_once()
