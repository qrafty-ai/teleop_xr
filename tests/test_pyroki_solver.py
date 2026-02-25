import pytest

try:
    import jaxlie  # noqa: F401
    import jaxls  # noqa: F401
    import pyroki  # noqa: F401
    import yourdfpy  # noqa: F401
except ImportError:
    pytest.skip("IK dependencies not installed", allow_module_level=True)

from typing import Any, cast  # noqa: E402

from teleop_xr.ik.solver import PyrokiSolver  # noqa: E402


def test_pyroki_solver_warmup_handles_robot_errors():
    class DummyRobot:
        def get_default_config(self):
            raise RuntimeError("fail")

        def build_costs(self, target_L, target_R, target_Head):
            raise RuntimeError("not used")

        @property
        def joint_var_cls(self):
            raise RuntimeError("not used")

    solver = PyrokiSolver(cast(Any, DummyRobot()))
    assert solver.robot is not None
    assert solver.warmup_complete is False
    assert solver.warmup_error is not None
    assert solver.warmed_target_patterns == tuple()
