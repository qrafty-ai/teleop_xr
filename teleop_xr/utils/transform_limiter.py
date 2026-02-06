import numpy as np
from loguru import logger
from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.quaternions import axangle2quat, quat2axangle


def se3_to_twist(T1, T2):
    """Compute linear and angular velocity from T1 to T2."""
    dT = np.linalg.inv(T1) @ T2
    # Linear velocity
    v = dT[:3, 3]

    # Angular velocity
    R = dT[:3, :3]
    q = mat2quat(R)
    angle, axis = quat2axangle(q)
    w = angle * np.array(axis)

    return v, w


def apply_twist(T, v, w, dt=1.0):
    """Apply twist (v, w) to transformation T over time dt."""
    # Translation update
    t_new = T[:3, 3] + v * dt

    # Rotation update
    angle = np.linalg.norm(w)
    if angle < 1e-8:
        R_update = np.eye(3)
    else:
        axis = w / angle
        q_update = axangle2quat(axis, angle * dt)
        R_update = quat2mat(q_update)

    R_new = R_update @ T[:3, :3]

    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = t_new
    return T_new


def limit_magnitude(vec, max_val):
    norm = np.linalg.norm(vec)
    if norm > max_val:
        return vec * (max_val / norm)
    return vec


def clamp_twist(v_prev, v_desired, max_vel, max_acc):
    # Acceleration
    a = v_desired - v_prev
    a = limit_magnitude(a, max_acc)
    v_new = v_prev + a
    return limit_magnitude(v_new, max_vel)


def compute_next_transform(
    T_tm1,
    T_t,
    T_tp1_desired,
    max_lin_vel,
    max_ang_vel,
    max_lin_acc,
    max_ang_acc,
    dt=1.0,
):
    # Previous and desired velocities
    v_prev, w_prev = se3_to_twist(T_tm1, T_t)
    v_desired, w_desired = se3_to_twist(T_t, T_tp1_desired)

    # Clamp to acceleration and velocity limits
    v_new = clamp_twist(v_prev, v_desired, max_lin_vel * dt, max_lin_acc * dt)
    w_new = clamp_twist(w_prev, w_desired, max_ang_vel * dt, max_ang_acc * dt)

    # Apply twist to current pose
    T_tp1 = apply_twist(T_t, v_new, w_new, dt)

    return T_tp1


if __name__ == "__main__":  # pragma: no cover
    T_tm1 = np.eye(4)
    T_t = np.array(
        [[0.998, -0.05, 0, 0.5], [0.05, 0.998, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]]
    )
    T_tp1_desired = np.array(
        [[0.992, -0.12, 0, 1.2], [0.12, 0.992, 0, 0.3], [0, 0, 1, 0.0], [0, 0, 0, 1]]
    )

    T_tp1_clamped = compute_next_transform(
        T_tm1,
        T_t,
        T_tp1_desired,
        max_lin_vel=1.0,  # m/s
        max_ang_vel=np.radians(45),  # rad/s
        max_lin_acc=0.5,  # m/s²
        max_ang_acc=np.radians(30),  # rad/s²
    )

    logger.info("New transform at t+1 (clamped):\n{}", T_tp1_clamped)
