"""Minimal SE(3) helpers (numpy only) for ov_global."""

import math

import numpy as np


def quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot_to_quat(m: np.ndarray):
    """Return (x, y, z, w)."""
    t = float(np.trace(m))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def yaw_rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def make_T(rot: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def pose_msg_to_T(pose) -> np.ndarray:
    q = pose.orientation
    p = pose.position
    return make_T(quat_to_rot(q.x, q.y, q.z, q.w), [p.x, p.y, p.z])


def yaw_of(T: np.ndarray) -> float:
    return math.atan2(T[1, 0], T[0, 0])


def project_4dof(T: np.ndarray) -> np.ndarray:
    """Keep yaw + translation, drop roll/pitch (both frames gravity aligned)."""
    return make_T(yaw_rot(yaw_of(T)), T[:3, 3])


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def slerp(q0, q1, alpha: float):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + alpha * (q1 - q0)
        return out / np.linalg.norm(out)
    theta0 = math.acos(dot)
    theta = theta0 * alpha
    q2 = q1 - q0 * dot
    q2 /= np.linalg.norm(q2)
    return q0 * math.cos(theta) + q2 * math.sin(theta)


def interpolate_T(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = rot_to_quat(T0[:3, :3])
    q1 = rot_to_quat(T1[:3, :3])
    q = slerp(q0, q1, alpha)
    t = (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3]
    return make_T(quat_to_rot(*q), t)
