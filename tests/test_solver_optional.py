import pytest

try:
    import jaxls  # noqa: F401
    import pyroki  # noqa: F401
    import jaxlie  # noqa: F401
    import yourdfpy  # noqa: F401
except ImportError:
    pytest.skip(
        "jaxls, pyroki, jaxlie, or yourdfpy not installed", allow_module_level=True
    )

import jaxlie  # noqa: E402
from teleop_xr.ik.solver import PyrokiSolver  # noqa: E402
from teleop_xr.ik.robots.h1_2 import UnitreeH1Robot  # noqa: E402


def test_solver_optional_targets():
    robot = UnitreeH1Robot()
    solver = PyrokiSolver(robot)

    q_current = robot.get_default_config()
    target = jaxlie.SE3.identity()

    q_all = solver.solve(target, target, target, q_current)
    assert q_all.shape == q_current.shape

    q_left = solver.solve(target, None, None, q_current)
    assert q_left.shape == q_current.shape

    q_right = solver.solve(None, target, None, q_current)
    assert q_right.shape == q_current.shape

    q_head = solver.solve(None, None, target, q_current)
    assert q_head.shape == q_current.shape

    q_none = solver.solve(None, None, None, q_current)
    assert q_none.shape == q_current.shape


if __name__ == "__main__":
    test_solver_optional_targets()
