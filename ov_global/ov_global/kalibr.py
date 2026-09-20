"""Read camera intrinsics/extrinsics from an OpenVINS kalibr_imucam_chain.yaml."""

from dataclasses import dataclass

import numpy as np
import yaml


@dataclass
class KalibrCamera:
    cam_id: int
    K: np.ndarray            # 3x3
    D: np.ndarray            # (n,1) radtan [k1 k2 p1 p2]
    T_imu_cam: np.ndarray    # 4x4, camera -> IMU
    resolution: tuple


def _load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        # OpenCV-style "%YAML:1.0" directive is not valid for PyYAML.
        text = '\n'.join(line for line in f.read().splitlines() if not line.lstrip().startswith('%'))
    return yaml.safe_load(text) or {}


def load_camera(path: str, cam_id: int) -> KalibrCamera:
    data = _load_yaml(path)
    key = f'cam{cam_id}'
    if key not in data:
        raise KeyError(f'{key} not found in {path}')
    cam = data[key]
    fx, fy, cx, cy = [float(v) for v in cam['intrinsics']]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([float(v) for v in cam.get('distortion_coeffs', [])], dtype=np.float64).reshape(-1, 1)
    T = np.array(cam['T_imu_cam'], dtype=np.float64).reshape(4, 4)
    res = tuple(int(v) for v in cam.get('resolution', (0, 0)))
    return KalibrCamera(cam_id=cam_id, K=K, D=D, T_imu_cam=T, resolution=res)
