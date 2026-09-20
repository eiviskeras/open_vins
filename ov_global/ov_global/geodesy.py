"""WGS-84 geodetic helpers (ENU local tangent plane) for ov_global.

All angles are radians unless a name ends in ``_deg``.
"""

import math

import numpy as np

_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)

# Rotation that maps NED vectors into ENU vectors (and vice versa; it is its own inverse).
R_ENU_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)


def lla_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    s = math.sin(lat)
    n = _A / math.sqrt(1.0 - _E2 * s * s)
    return np.array(
        [
            (n + alt) * math.cos(lat) * math.cos(lon),
            (n + alt) * math.cos(lat) * math.sin(lon),
            (n * (1.0 - _E2) + alt) * s,
        ],
        dtype=np.float64,
    )


def ecef_to_lla(p: np.ndarray) -> np.ndarray:
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    lon = math.atan2(y, x)
    r = math.hypot(x, y)
    lat = math.atan2(z, r * (1.0 - _E2))
    alt = 0.0
    for _ in range(10):
        s = math.sin(lat)
        n = _A / math.sqrt(1.0 - _E2 * s * s)
        alt = r / math.cos(lat) - n
        lat = math.atan2(z, r * (1.0 - _E2 * n / (n + alt)))
    return np.array([lat, lon, alt], dtype=np.float64)


def rotation_ecef_to_enu(lat: float, lon: float) -> np.ndarray:
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ],
        dtype=np.float64,
    )


def lla_to_enu(lla: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
    rot = rotation_ecef_to_enu(float(ref_lla[0]), float(ref_lla[1]))
    return rot @ (lla_to_ecef(*lla) - lla_to_ecef(*ref_lla))


def enu_to_lla(enu: np.ndarray, ref_lla: np.ndarray) -> np.ndarray:
    rot = rotation_ecef_to_enu(float(ref_lla[0]), float(ref_lla[1]))
    return ecef_to_lla(lla_to_ecef(*ref_lla) + rot.T @ enu)


def deg_lla_to_rad(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    return np.array([math.radians(lat_deg), math.radians(lon_deg), alt_m], dtype=np.float64)
