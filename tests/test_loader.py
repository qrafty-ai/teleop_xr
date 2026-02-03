import pytest
from teleop_xr.ik.loader import load_robot_class, list_available_robots, RobotLoadError
from teleop_xr.ik.robots.h1_2 import UnitreeH1Robot


def test_load_robot_class_none():
    """Test: load_robot_class(None) returns UnitreeH1Robot"""
    cls = load_robot_class(None)
    assert cls == UnitreeH1Robot


def test_load_robot_class_explicit():
    """Test: load_robot_class('teleop_xr.ik.robots.h1_2:UnitreeH1Robot') returns class"""
    cls = load_robot_class("teleop_xr.ik.robots.h1_2:UnitreeH1Robot")
    assert cls == UnitreeH1Robot


def test_load_robot_class_invalid_module():
    """Test: load_robot_class('nonexistent:Nope') raises RobotLoadError"""
    with pytest.raises(RobotLoadError) as excinfo:
        load_robot_class("nonexistent:Nope")
    assert "Failed to load robot class" in str(excinfo.value)


def test_load_robot_class_not_baserobot():
    """Test: load_robot_class('teleop_xr.ik.loader:RobotLoadError') raises (not a BaseRobot)"""
    with pytest.raises(RobotLoadError) as excinfo:
        load_robot_class("teleop_xr.ik.loader:RobotLoadError")
    assert "is not a subclass of BaseRobot" in str(excinfo.value)


def test_list_available_robots():
    """Test: list_available_robots() returns dict with at least 'h1' entry"""
    # Note: This depends on Task 2 (entry points in pyproject.toml)
    # being active in the environment.
    available = list_available_robots()
    assert isinstance(available, dict)
    assert "h1" in available
    assert available["h1"] == "teleop_xr.ik.robots.h1_2:UnitreeH1Robot"


def test_load_robot_class_entry_point():
    """Test: load_robot_class('h1') returns UnitreeH1Robot via entry point"""
    # Note: This also depends on the entry point being registered.
    cls = load_robot_class("h1")
    assert cls == UnitreeH1Robot
