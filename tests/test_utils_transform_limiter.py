import numpy as np
from teleop_xr.utils.transform_limiter import (
    se3_to_twist,
    apply_twist,
    limit_magnitude,
    clamp_twist,
    compute_next_transform,
)


def test_se3_to_twist():
    T1 = np.eye(4)
    T2 = np.eye(4)
    T2[:3, 3] = [1.0, 0.0, 0.0]

    v, w = se3_to_twist(T1, T2)
    np.testing.assert_allclose(v, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(w, [0.0, 0.0, 0.0])


def test_apply_twist():
    T = np.eye(4)
    v = np.array([1.0, 0.0, 0.0])
    w = np.array([0.0, 0.0, 0.0])
    dt = 1.0

    T_new = apply_twist(T, v, w, dt)
    expected = np.eye(4)
    expected[:3, 3] = [1.0, 0.0, 0.0]
    np.testing.assert_allclose(T_new, expected)

    w_nonzero = np.array([0.0, 0.0, np.pi / 2])
    T_rot = apply_twist(T, v, w_nonzero, dt)

    np.testing.assert_allclose(T_rot[:3, 3], [1.0, 0.0, 0.0])

    expected_rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(T_rot[:3, :3], expected_rot, atol=1e-7)


def test_limit_magnitude():
    vec = np.array([3.0, 4.0])
    limited = limit_magnitude(vec, 2.5)
    np.testing.assert_allclose(limited, [1.5, 2.0])

    no_limit = limit_magnitude(vec, 10.0)
    np.testing.assert_allclose(no_limit, vec)


def test_clamp_twist():
    v_prev = np.array([0.0, 0.0, 0.0])
    v_desired = np.array([10.0, 0.0, 0.0])
    max_vel = 5.0
    max_acc = 2.0

    v_new = clamp_twist(v_prev, v_desired, max_vel, max_acc)
    np.testing.assert_allclose(v_new, [2.0, 0.0, 0.0])

    v_prev = np.array([4.0, 0.0, 0.0])
    v_desired = np.array([14.0, 0.0, 0.0])
    max_vel = 5.0
    max_acc = 10.0

    v_new = clamp_twist(v_prev, v_desired, max_vel, max_acc)
    np.testing.assert_allclose(v_new, [5.0, 0.0, 0.0])


def test_compute_next_transform():
    T_tm1 = np.eye(4)
    T_t = np.eye(4)
    T_t[:3, 3] = [1.0, 0.0, 0.0]

    T_tp1_desired = np.eye(4)
    T_tp1_desired[:3, 3] = [10.0, 0.0, 0.0]

    max_lin_vel = 5.0
    max_ang_vel = 1.0
    max_lin_acc = 2.0
    max_ang_acc = 1.0

    T_next = compute_next_transform(
        T_tm1,
        T_t,
        T_tp1_desired,
        max_lin_vel,
        max_ang_vel,
        max_lin_acc,
        max_ang_acc,
        dt=1.0,
    )

    expected_pos = [4.0, 0.0, 0.0]
    np.testing.assert_allclose(T_next[:3, 3], expected_pos)
