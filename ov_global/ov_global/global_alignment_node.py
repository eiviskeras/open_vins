#!/usr/bin/env python3
"""4-DoF global alignment of OpenVINS with GNSS seed and AprilTag step corrections.

Frames
------
``global``  OpenVINS world frame (gravity aligned, z up, arbitrary yaw/origin).
``imu``     OpenVINS IMU frame (ZED rear-left IMU). Published by OpenVINS as ``global -> imu``.
``map``     Local ENU tangent plane at ``ref_lla_deg``.
``world_ned`` NED frame co-located with ``map`` (same convention as the autoferry novatel relay).
``base_link`` Vessel body frame, SNAME/NED convention (x fwd, y stbd, z down).

State
-----
x = [t_map_global (3), yaw] with covariance P (4x4). The pose of anything in the
map frame is T_map_X = T_map_global(x) * T_global_X. Roll/pitch of the alignment
are zero by construction (both frames are gravity aligned).

Tag sources
-----------
``tag_source: vio_landmarks`` (default, simplest): the four AprilTag corners that
OpenVINS keeps as never-marginalised SLAM landmarks (``/ov_msckf/points_aruco``)
are read back. Their centre and plane normal give the tag pose in ``global``; the
alignment is then a *deterministic anchor* of that estimate onto the surveyed tag.
OpenVINS already does the loop closure internally, so every refinement of the
landmarks moves ``map -> global`` accordingly. No second detector is needed.

``tag_source: pnp``: an external PnP detection (``tag_pnp_node``) is fused as a
chi2-gated EKF update (``correction_mode: kalman``) or a hard reset (``replace``).

Initialisation: GNSS antenna LLA + NED heading at the first VIO message
(``init_mode: gnss``) or from the first accepted tag observation (``init_mode: tag``).
"""

import math
from collections import deque
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64MultiArray
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from ov_global import geodesy
from ov_global.kalibr import load_camera
from ov_global.se3 import (
    inv_T,
    interpolate_T,
    make_T,
    pose_msg_to_T,
    rot_to_quat,
    rpy_to_rot,
    wrap_pi,
    yaw_rot,
)

try:  # optional: publish autoferry NavState when the message package is available
    from autoferry_msgs.msg import NavState  # type: ignore
except Exception:  # pragma: no cover - dependency is optional
    NavState = None

CHI2_99 = {3: 11.345, 4: 13.277}


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _R_enu_body(heading_ned: float) -> np.ndarray:
    """Rotation body(SNAME: x fwd, y stbd, z down) -> ENU for a NED heading (rad, cw from north)."""
    s, c = math.sin(heading_ned), math.cos(heading_ned)
    return np.array([[s, c, 0.0], [c, -s, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)


def _heading_ned_from_R_enu_body(R: np.ndarray) -> float:
    fwd = R[:, 0]
    return math.atan2(fwd[0], fwd[1])  # atan2(east, north)


class GlobalAlignmentNode(Node):
    def __init__(self) -> None:
        super().__init__('global_alignment_node')
        p = self.declare_parameter
        # ---- topics / frames
        p('vio_topic', '/ov_msckf/odomimu')
        p('tag_source', 'vio_landmarks')            # vio_landmarks | pnp
        p('landmarks_topic', '/ov_msckf/points_aruco')
        p('tag_pose_topic', '/ov_global/tag_pose')
        p('map_frame', 'map')
        p('ned_frame', 'world_ned')
        p('global_frame', 'global')
        p('imu_frame', 'imu')
        p('base_frame', 'base_link')
        p('antenna_frame', 'gnss_antenna')
        p('publish_tf', True)
        p('publish_static_tf', True)
        # ---- geometry
        p('imucam_yaml', '')
        p('camera_id', 0)
        p('p_base_cam', [0.0, 0.0, 0.0])           # cam0 optical centre in base_link [m] (frames.yaml entry)
        # camera mount in base_link: rpy [deg] of the *level, forward-looking* camera convention
        # (optical z = body x, optical x = body y, optical y = body z). yaw=180 -> looks aft.
        p('cam_mount_rpy_deg', [0.0, 0.0, 180.0])
        p('p_base_antenna', [0.0, 0.0, 0.0])        # GNSS antenna position in base_link [m]
        # ---- map origin and tag
        p('ref_lla_deg', [float('nan'), float('nan'), 0.0])  # NaN -> use tag centre
        p('tag_center_lla_deg', [0.0, 0.0, 0.0])
        p('tag_center_offset_up_m', 0.0)            # added to the altitude (e.g. bottom-centre survey + half height)
        p('tag_facing_azimuth_deg', float('nan'))   # compass direction the tag face looks towards; NaN -> yaw unobserved
        p('tag_tilt_deg', 0.0)                      # +: top leans away from the viewer
        p('tag_roll_deg', 0.0)                      # rotation about the tag normal
        p('tag_pos_sigma_m', 0.05)
        p('tag_yaw_sigma_deg', 2.0)
        p('tag_size_m', 0.0)                        # edge length; >0 enables the landmark spacing check (2x2 grid)
        p('landmark_grid_cols', 2)                  # fiducial landmark grid: 2x2 for a tag, e.g. 7x5 for a checkerboard
        p('landmark_grid_rows', 2)
        p('landmark_spacing_m', 0.0)                # grid corner spacing; 0 -> tag_size_m
        p('landmark_sigma_m', 0.10)                 # assumed accuracy of OpenVINS fiducial landmarks in global
        p('landmark_square_tolerance', 0.25)        # relative tolerance on spacing / extents / planarity
        # ---- initialisation
        p('init_mode', 'gnss')                      # gnss | tag
        p('init_lla_deg', [0.0, 0.0, 0.0])          # antenna position at VIO start
        p('init_heading_ned_deg', 0.0)              # base_link heading at VIO start
        p('init_pos_sigma_m', 1.0)
        p('init_heading_sigma_deg', 2.0)
        p('gnss_fix_topic', '')                     # optional NavSatFix overriding init_lla_deg
        # ---- correction
        p('correction_mode', 'kalman')              # kalman | replace
        p('chi2_gate_prob', 0.99)
        p('drift_pos_per_m', 0.02)                  # VIO position drift growth [m per m travelled]
        p('drift_yaw_deg_per_m', 0.05)              # VIO yaw drift growth [deg per m travelled]
        p('min_correction_interval_s', 0.0)
        p('max_corrections', 0)                     # pnp source: 0 = unlimited, 1 = anchor once
        p('max_tag_vio_dt_s', 0.05)

        g = lambda name: self.get_parameter(name).value  # noqa: E731
        self.map_frame, self.ned_frame = str(g('map_frame')), str(g('ned_frame'))
        self.global_frame, self.imu_frame = str(g('global_frame')), str(g('imu_frame'))
        self.base_frame, self.antenna_frame = str(g('base_frame')), str(g('antenna_frame'))
        self.publish_tf = bool(g('publish_tf'))

        # geometry
        self.p_base_antenna = np.array([float(v) for v in g('p_base_antenna')])
        self.T_imu_cam = np.eye(4)
        self.T_imu_cams = {}
        yaml_path = str(g('imucam_yaml'))
        if yaml_path:
            for cid in (0, 1):
                try:
                    self.T_imu_cams[cid] = load_camera(yaml_path, cid).T_imu_cam
                except KeyError:
                    pass
            self.T_imu_cam = self.T_imu_cams.get(int(g('camera_id')), np.eye(4))
        else:
            self.get_logger().warning('imucam_yaml not set: assuming camera == IMU frame')
        # base_link -> cam0 (optical) from the surveyed camera position and mount angles,
        # then base_link -> imu through the kalibr extrinsics.
        R_fwd_optical = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # columns: x_c, y_c, z_c in body
        mount = [math.radians(float(v)) for v in g('cam_mount_rpy_deg')]
        R_base_cam = rpy_to_rot(*mount) @ R_fwd_optical
        self.T_base_cam = make_T(R_base_cam, [float(v) for v in g('p_base_cam')])
        self.T_base_imu = self.T_base_cam @ inv_T(self.T_imu_cam)
        self.T_imu_base = inv_T(self.T_base_imu)

        # tag pose in map
        tag_lla = [float(v) for v in g('tag_center_lla_deg')]
        tag_lla[2] += float(g('tag_center_offset_up_m'))
        self.tag_lla = geodesy.deg_lla_to_rad(*tag_lla)
        ref = [float(v) for v in g('ref_lla_deg')]
        self.ref_lla = self.tag_lla.copy() if math.isnan(ref[0]) or math.isnan(ref[1]) else geodesy.deg_lla_to_rad(*ref)
        self.p_map_tag = geodesy.lla_to_enu(self.tag_lla, self.ref_lla)
        az = float(g('tag_facing_azimuth_deg'))
        self.tag_yaw_known = not math.isnan(az)
        self.R_map_tag = self._tag_rotation(math.radians(az if self.tag_yaw_known else 0.0),
                                            math.radians(float(g('tag_tilt_deg'))), math.radians(float(g('tag_roll_deg'))))
        self.T_map_tag = make_T(self.R_map_tag, self.p_map_tag)
        self.tag_pos_var = float(g('tag_pos_sigma_m')) ** 2
        self.tag_yaw_var = math.radians(float(g('tag_yaw_sigma_deg'))) ** 2
        self.tag_size = float(g('tag_size_m'))
        self.grid_cols, self.grid_rows = int(g('landmark_grid_cols')), int(g('landmark_grid_rows'))
        self.grid_n = self.grid_cols * self.grid_rows
        self.grid_spacing = float(g('landmark_spacing_m')) or self.tag_size
        self.landmark_var = float(g('landmark_sigma_m')) ** 2
        self.square_tol = float(g('landmark_square_tolerance'))
        self.tag_source = str(g('tag_source'))
        self.n_G_ref: Optional[np.ndarray] = None   # tag normal sign reference (global frame)

        # init / correction settings
        self.init_mode = str(g('init_mode'))
        self.init_lla = geodesy.deg_lla_to_rad(*[float(v) for v in g('init_lla_deg')])
        self.init_heading = math.radians(float(g('init_heading_ned_deg')))
        self.init_pos_var = float(g('init_pos_sigma_m')) ** 2
        self.init_yaw_var = math.radians(float(g('init_heading_sigma_deg'))) ** 2
        self.correction_mode = str(g('correction_mode'))
        self.gate_prob = float(g('chi2_gate_prob'))
        self.drift_pos = float(g('drift_pos_per_m'))
        self.drift_yaw = math.radians(float(g('drift_yaw_deg_per_m')))
        self.min_dt = float(g('min_correction_interval_s'))
        self.max_corrections = int(g('max_corrections'))
        self.max_sync_dt = float(g('max_tag_vio_dt_s'))

        # state
        self.x = np.zeros(4)                   # [tx, ty, tz, yaw]
        self.P = np.eye(4) * 1e6
        self.initialized = False
        self.dist_since_correction = 0.0
        self.last_correction_t = -math.inf
        self.last_p_G = None
        self.vio_buffer = deque(maxlen=400)    # (t, T_G_I, msg)
        self.gnss_fix: Optional[NavSatFix] = None
        self.n_accepted = 0
        self.n_rejected = 0

        # ROS I/O
        self.tf_br = TransformBroadcaster(self)
        self.static_br = StaticTransformBroadcaster(self)
        self.pub_odom_map = self.create_publisher(Odometry, '~/odom_map', 10)        # ENU, child imu
        self.pub_odom_ned = self.create_publisher(Odometry, '~/odom_ned', 10)        # NED, child base_link
        self.pub_fix_base = self.create_publisher(NavSatFix, '~/fix_base_link', 10)
        self.pub_fix_cam = self.create_publisher(NavSatFix, '~/fix_cam0', 10)
        self.pub_fix_ant = self.create_publisher(NavSatFix, '~/fix_antenna', 10)
        self.pub_residual = self.create_publisher(Float64MultiArray, '~/tag_residual', 10)
        self.pub_align = self.create_publisher(PoseWithCovarianceStamped, '~/map_to_global', 10)
        self.pub_navstate = self.create_publisher(NavState, '~/nav_state', 10) if NavState is not None else None

        self.create_subscription(Odometry, str(g('vio_topic')), self._on_vio, 50)
        if self.tag_source == 'pnp':
            self.create_subscription(PoseWithCovarianceStamped, str(g('tag_pose_topic')), self._on_tag, 10)
        elif self.tag_source == 'vio_landmarks':
            self.create_subscription(PointCloud2, str(g('landmarks_topic')), self._on_landmarks, 10)
        else:
            raise ValueError(f"tag_source must be 'vio_landmarks' or 'pnp', got '{self.tag_source}'")
        fix_topic = str(g('gnss_fix_topic'))
        if fix_topic:
            self.create_subscription(NavSatFix, fix_topic, self._on_fix, 10)

        if bool(g('publish_static_tf')):
            self._publish_static_tf()

        self.get_logger().info(
            f'ov_global alignment: source={self.tag_source} init={self.init_mode} correction={self.correction_mode} '
            f'tag@ENU={np.round(self.p_map_tag, 2).tolist()} yaw_known={self.tag_yaw_known} '
            f'navstate={"on" if self.pub_navstate else "off"}'
        )

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _tag_rotation(azimuth: float, tilt: float, roll: float) -> np.ndarray:
        """R_map_tag for a tag (X right, Y up, Z out of face) facing compass azimuth (cw from north)."""
        n = np.array([math.sin(azimuth), math.cos(azimuth), 0.0])      # face normal, ENU
        up = np.array([0.0, 0.0, 1.0])
        x = np.cross(up, n)
        R = np.column_stack([x, up, n])
        # tilt about tag X (top leans away from viewer for +tilt), roll about tag Z
        Rx = rpy_to_rot(tilt, 0.0, 0.0)
        Rz = yaw_rot(roll)
        return R @ Rx @ Rz

    def _static(self, parent: str, child: str, T: np.ndarray) -> TransformStamped:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = parent
        msg.child_frame_id = child
        t = T[:3, 3]
        qx, qy, qz, qw = rot_to_quat(T[:3, :3])
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, t)
        msg.transform.rotation.x, msg.transform.rotation.y = float(qx), float(qy)
        msg.transform.rotation.z, msg.transform.rotation.w = float(qz), float(qw)
        return msg

    def _publish_static_tf(self) -> None:
        tfs = [
            self._static(self.ned_frame, self.map_frame, make_T(geodesy.R_ENU_NED, [0.0, 0.0, 0.0])),
            self._static(self.imu_frame, self.base_frame, self.T_imu_base),
            self._static(self.base_frame, self.antenna_frame, make_T(np.eye(3), self.p_base_antenna)),
            self._static(self.map_frame, 'apriltag', self.T_map_tag),
        ]
        for cid, T in self.T_imu_cams.items():
            tfs.append(self._static(self.imu_frame, f'cam{cid}', T))
        self.static_br.sendTransform(tfs)

    # -------------------------------------------------------------- helpers
    def T_map_global(self) -> np.ndarray:
        return make_T(yaw_rot(self.x[3]), self.x[:3])

    def _set_alignment(self, T: np.ndarray, P: np.ndarray) -> None:
        self.x[:3] = T[:3, 3]
        self.x[3] = math.atan2(T[1, 0], T[0, 0])
        self.P = P
        self.initialized = True
        self.dist_since_correction = 0.0

    def _lookup_vio(self, t: float) -> Optional[np.ndarray]:
        buf = self.vio_buffer
        if not buf:
            return None
        if t <= buf[0][0]:
            return buf[0][1] if buf[0][0] - t <= self.max_sync_dt else None
        if t >= buf[-1][0]:
            return buf[-1][1] if t - buf[-1][0] <= self.max_sync_dt else None
        for (t0, T0, _), (t1, T1, _) in zip(buf, list(buf)[1:]):
            if t0 <= t <= t1:
                a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
                return interpolate_T(T0, T1, a)
        return None

    # ------------------------------------------------------------ callbacks
    def _on_fix(self, msg: NavSatFix) -> None:
        self.gnss_fix = msg

    def _on_vio(self, msg: Odometry) -> None:
        t = _stamp_to_sec(msg.header.stamp)
        T_G_I = pose_msg_to_T(msg.pose.pose)
        self.vio_buffer.append((t, T_G_I, msg))

        if self.last_p_G is not None:
            self.dist_since_correction += float(np.linalg.norm(T_G_I[:3, 3] - self.last_p_G))
        self.last_p_G = T_G_I[:3, 3].copy()

        if not self.initialized:
            if self.init_mode == 'gnss':
                self._init_from_gnss(T_G_I)
            return

        self._publish_global(msg, T_G_I)

    def _init_from_gnss(self, T_G_I: np.ndarray) -> None:
        lla = self.init_lla
        if self.gnss_fix is not None:
            lla = geodesy.deg_lla_to_rad(self.gnss_fix.latitude, self.gnss_fix.longitude, self.gnss_fix.altitude)
        p_map_ant = geodesy.lla_to_enu(lla, self.ref_lla)
        R_map_base = _R_enu_body(self.init_heading)
        p_map_base = p_map_ant - R_map_base @ self.p_base_antenna
        T_map_base = make_T(R_map_base, p_map_base)
        T_map_I = T_map_base @ self.T_base_imu
        T_map_G = T_map_I @ inv_T(T_G_I)
        yaw = math.atan2(T_map_G[1, 0], T_map_G[0, 0])
        # project to 4-DoF while keeping the IMU position consistent
        R4 = yaw_rot(yaw)
        t4 = T_map_I[:3, 3] - R4 @ T_G_I[:3, 3]
        P = np.diag([self.init_pos_var] * 3 + [self.init_yaw_var])
        self._set_alignment(make_T(R4, t4), P)
        self.get_logger().info(
            f'initialised from GNSS: antenna ENU={np.round(p_map_ant, 2).tolist()} heading={math.degrees(self.init_heading):.1f} deg '
            f'-> map->global t={np.round(t4, 2).tolist()} yaw={math.degrees(yaw):.2f} deg'
        )

    def _on_tag(self, msg: PoseWithCovarianceStamped) -> None:
        t = _stamp_to_sec(msg.header.stamp)
        T_G_I = self._lookup_vio(t)
        if T_G_I is None:
            self.get_logger().warning('tag detection without matching VIO pose (dt too large or VIO not running)', throttle_duration_sec=2.0)
            return
        T_C_tag = pose_msg_to_T(msg.pose.pose)
        cov_C = np.array(msg.pose.covariance).reshape(6, 6)
        T_G_C = T_G_I @ self.T_imu_cam
        T_G_tag = T_G_C @ T_C_tag
        p_G_tag = T_G_tag[:3, 3]

        if not self.initialized:
            if self.init_mode != 'tag':
                return
            if not self.tag_yaw_known:
                self.get_logger().error('init_mode=tag requires tag_facing_azimuth_deg')
                return
            T_map_G = self.T_map_tag @ inv_T(T_G_tag)
            yaw = math.atan2(T_map_G[1, 0], T_map_G[0, 0])
            R4 = yaw_rot(yaw)
            t4 = self.p_map_tag - R4 @ p_G_tag
            P = np.diag([self.tag_pos_var + 1.0] * 3 + [self.tag_yaw_var + math.radians(5.0) ** 2])
            self._set_alignment(make_T(R4, t4), P)
            self.get_logger().info(f'initialised from tag: map->global t={np.round(t4, 2).tolist()} yaw={math.degrees(yaw):.2f} deg')
            return

        if t - self.last_correction_t < self.min_dt:
            return
        if self.max_corrections > 0 and self.n_accepted >= self.max_corrections:
            return

        # ---- predicted measurement and Jacobian (state: [t(3), yaw])
        R_mg = yaw_rot(self.x[3])
        p_pred = R_mg @ p_G_tag + self.x[:3]
        dRdyaw_p = yaw_rot(self.x[3] + math.pi / 2.0) @ np.array([p_G_tag[0], p_G_tag[1], 0.0])
        H = np.zeros((4, 4))
        H[:3, :3] = np.eye(3)
        H[:3, 3] = dRdyaw_p
        r = np.zeros(4)
        r[:3] = self.p_map_tag - p_pred

        # measurement noise: PnP position cov rotated into map + tag survey + VIO drift
        R_map_C = R_mg @ T_G_C[:3, :3]
        Rm = np.zeros((4, 4))
        Rm[:3, :3] = R_map_C @ cov_C[:3, :3] @ R_map_C.T + np.eye(3) * self.tag_pos_var
        rows = 3
        if self.tag_yaw_known:
            n_G = T_G_tag[:3, 2]
            yaw_tag_G = math.atan2(n_G[1], n_G[0])
            yaw_tag_map_known = math.atan2(self.R_map_tag[1, 2], self.R_map_tag[0, 2])
            r[3] = wrap_pi(yaw_tag_map_known - (yaw_tag_G + self.x[3]))
            H[3, 3] = 1.0
            Rm[3, 3] = cov_C[4, 4] + self.tag_yaw_var  # rotation about tag Y (vertical) axis
            rows = 4
        H, r, Rm = H[:rows], r[:rows], Rm[:rows, :rows]

        # ---- prior with VIO drift since the last correction
        d = self.dist_since_correction
        P_prior = self.P + np.diag([(self.drift_pos * d) ** 2] * 3 + [(self.drift_yaw * d) ** 2])
        S = H @ P_prior @ H.T + Rm
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        maha = float(r @ S_inv @ r)
        gate = CHI2_99[rows] if abs(self.gate_prob - 0.99) < 1e-6 else CHI2_99[rows] * (self.gate_prob / 0.99)
        accepted = maha < gate

        out = Float64MultiArray()
        out.data = [float(r[0]), float(r[1]), float(r[2]), float(r[3]) if rows == 4 else float('nan'), maha, 1.0 if accepted else 0.0, d]
        self.pub_residual.publish(out)

        if not accepted:
            self.n_rejected += 1
            self.get_logger().warning(
                f'tag correction rejected: |r|={np.linalg.norm(r[:3]):.2f} m dyaw={math.degrees(r[3]) if rows == 4 else float("nan"):.2f} deg '
                f'chi2={maha:.1f} > {gate:.1f} (accepted {self.n_accepted}, rejected {self.n_rejected})'
            )
            return

        if self.correction_mode == 'replace':
            P_used = np.eye(4) * 1e6
            S = H @ P_used @ H.T + Rm
            S_inv = np.linalg.inv(S)
        else:
            P_used = P_prior
        K = P_used @ H.T @ S_inv
        dx = K @ r
        self.x[:3] += dx[:3]
        self.x[3] = wrap_pi(self.x[3] + dx[3])
        I_KH = np.eye(4) - K @ H
        self.P = I_KH @ P_used @ I_KH.T + K @ Rm @ K.T
        self.dist_since_correction = 0.0
        self.last_correction_t = t
        self.n_accepted += 1
        self.get_logger().info(
            f'tag correction #{self.n_accepted}: step dt={np.round(dx[:3], 3).tolist()} m dyaw={math.degrees(dx[3]):.3f} deg '
            f'(residual {np.linalg.norm(r[:3]):.2f} m, chi2 {maha:.1f}, range {np.linalg.norm(T_C_tag[:3, 3]):.1f} m, travelled {d:.1f} m)'
        )
        self._publish_alignment(msg.header.stamp)

    # ------------------------------------------------ VIO landmark anchor
    def _grid_from_landmarks(self, pts: np.ndarray) -> Optional[tuple]:
        """Return (centre, unit normal, extent) if the points form a planar cols x rows grid of the expected spacing.

        Landmark ids are not available in the point cloud, so the check is order-free: every point
        must have a nearest neighbour at the grid spacing, the two in-plane principal extents must
        match those of a regular grid, and the out-of-plane spread must be small.
        """
        n = self.grid_n
        if pts.shape != (n, 3):
            return None
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        nn = dist.min(axis=1)
        spacing = self.grid_spacing if self.grid_spacing > 0.0 else float(np.median(nn))
        if spacing <= 0.0:
            return None
        tol = self.square_tol * spacing
        if np.any(np.abs(nn - spacing) > tol):
            return None
        centre = pts.mean(axis=0)
        _, s, vt = np.linalg.svd(pts - centre)
        # singular values of a centred regular grid: spacing * sqrt(n * (k^2 - 1) / 12) per axis
        expect = sorted((spacing * math.sqrt(n * (k * k - 1) / 12.0) for k in (self.grid_cols, self.grid_rows)), reverse=True)
        if abs(s[0] - expect[0]) > tol * math.sqrt(n) or abs(s[1] - expect[1]) > tol * math.sqrt(n):
            return None
        if s[2] > tol * math.sqrt(n):  # not planar
            return None
        normal = vt[2] / np.linalg.norm(vt[2])
        extent = spacing * max(self.grid_cols - 1, self.grid_rows - 1)
        return centre, normal, extent

    def _on_landmarks(self, msg: PointCloud2) -> None:
        pts = np.array(
            [(float(p[0]), float(p[1]), float(p[2])) for p in point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)],
            dtype=np.float64,
        ).reshape(-1, 3)
        if pts.shape[0] == 0:
            return
        if pts.shape[0] != self.grid_n:
            self.get_logger().warning(
                f'{pts.shape[0]} fiducial landmarks in state, need exactly {self.grid_n} '
                f'({self.grid_cols}x{self.grid_rows} grid)', throttle_duration_sec=5.0)
            return
        grid = self._grid_from_landmarks(pts)
        if grid is None:
            self.get_logger().warning('fiducial landmarks do not form a valid planar grid yet', throttle_duration_sec=5.0)
            return
        centre_G, n_G, extent = grid

        # resolve the normal sign: towards the camera on first use, then keep continuity
        if self.n_G_ref is None:
            if not self.vio_buffer:
                return
            p_cam_G = (self.vio_buffer[-1][1] @ self.T_imu_cam)[:3, 3]
            if float(np.dot(n_G, p_cam_G - centre_G)) < 0.0:
                n_G = -n_G
            self.n_G_ref = n_G.copy()
        elif float(np.dot(n_G, self.n_G_ref)) < 0.0:
            n_G = -n_G

        # deterministic anchor: surveyed tag <- landmark estimate
        if self.tag_yaw_known:
            yaw_tag_map = math.atan2(self.R_map_tag[1, 2], self.R_map_tag[0, 2])
            yaw = wrap_pi(yaw_tag_map - math.atan2(n_G[1], n_G[0]))
        elif self.initialized:
            yaw = self.x[3]                          # yaw from GNSS seed, position from tag
        else:
            self.get_logger().error('landmark anchor needs tag_facing_azimuth_deg or a GNSS initialisation', throttle_duration_sec=5.0)
            return
        if self.max_corrections > 0 and self.n_accepted >= self.max_corrections:
            return
        R4 = yaw_rot(yaw)
        t4 = self.p_map_tag - R4 @ centre_G
        # normal direction from n landmarks spread over `extent`; centre from their mean
        yaw_var = 2.0 * self.landmark_var / (extent * extent) + self.tag_yaw_var
        P = np.diag([self.landmark_var / self.grid_n + self.tag_pos_var] * 3 + [yaw_var])

        first = not self.initialized
        dt = t4 - self.x[:3]
        dyaw = wrap_pi(yaw - self.x[3])
        self._set_alignment(make_T(R4, t4), P)
        self.last_correction_t = _stamp_to_sec(msg.header.stamp)
        self.n_accepted += 1

        out = Float64MultiArray()
        out.data = [float(dt[0]), float(dt[1]), float(dt[2]), float(dyaw), 0.0, 1.0, self.dist_since_correction]
        self.pub_residual.publish(out)
        if first:
            self.get_logger().info(
                f'anchored from VIO landmarks: tag centre global={np.round(centre_G, 2).tolist()} '
                f'-> map->global t={np.round(t4, 2).tolist()} yaw={math.degrees(yaw):.2f} deg'
            )
        elif np.linalg.norm(dt) > 0.01 or abs(dyaw) > math.radians(0.01):
            self.get_logger().info(
                f'landmark refinement moved alignment by {np.round(dt, 3).tolist()} m, {math.degrees(dyaw):.3f} deg',
                throttle_duration_sec=1.0,
            )
        self._publish_alignment(msg.header.stamp)

    # ------------------------------------------------------------- outputs
    def _publish_alignment(self, stamp) -> None:
        T = self.T_map_global()
        if self.publish_tf:
            tf = self._static(self.map_frame, self.global_frame, T)
            tf.header.stamp = stamp
            self.tf_br.sendTransform(tf)
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.map_frame
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = map(float, T[:3, 3])
        qx, qy, qz, qw = rot_to_quat(T[:3, :3])
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y = float(qx), float(qy)
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = float(qz), float(qw)
        cov = np.zeros((6, 6))
        cov[:3, :3] = self.P[:3, :3]
        cov[5, 5] = self.P[3, 3]
        msg.pose.covariance = cov.reshape(-1).tolist()
        self.pub_align.publish(msg)

    def _fix(self, stamp, frame: str, p_map: np.ndarray, var_xyz: np.ndarray) -> NavSatFix:
        lla = geodesy.enu_to_lla(p_map, self.ref_lla)
        fix = NavSatFix()
        fix.header.stamp = stamp
        fix.header.frame_id = frame
        fix.status.status = 0
        fix.latitude, fix.longitude, fix.altitude = math.degrees(lla[0]), math.degrees(lla[1]), float(lla[2])
        fix.position_covariance = [var_xyz[0], 0.0, 0.0, 0.0, var_xyz[1], 0.0, 0.0, 0.0, var_xyz[2]]
        fix.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        return fix

    def _publish_global(self, vio: Odometry, T_G_I: np.ndarray) -> None:
        stamp = vio.header.stamp
        self._publish_alignment(stamp)
        T_map_G = self.T_map_global()
        T_map_I = T_map_G @ T_G_I
        T_map_base = T_map_I @ self.T_imu_base
        T_map_C = T_map_I @ self.T_imu_cam
        p_map_ant = T_map_base[:3, :3] @ self.p_base_antenna + T_map_base[:3, 3]

        # velocities: OpenVINS twist is expressed in the IMU frame
        v_I = np.array([vio.twist.twist.linear.x, vio.twist.twist.linear.y, vio.twist.twist.linear.z])
        w_I = np.array([vio.twist.twist.angular.x, vio.twist.twist.angular.y, vio.twist.twist.angular.z])
        v_base = self.T_base_imu[:3, :3] @ v_I
        w_base = self.T_base_imu[:3, :3] @ w_I
        v_map = T_map_I[:3, :3] @ v_I

        vio_cov = np.array(vio.pose.covariance).reshape(6, 6)
        var_xyz = np.diag(self.P[:3, :3]) + np.diag(vio_cov[3:6, 3:6]) if np.any(vio_cov) else np.diag(self.P[:3, :3])

        # ENU odometry of the IMU
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.map_frame
        odom.child_frame_id = self.imu_frame
        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = map(float, T_map_I[:3, 3])
        q = rot_to_quat(T_map_I[:3, :3])
        odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = map(float, q)
        odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z = map(float, v_map)
        cov = np.zeros((6, 6))
        cov[:3, :3] = np.diag(var_xyz)
        cov[5, 5] = self.P[3, 3]
        odom.pose.covariance = cov.reshape(-1).tolist()
        self.pub_odom_map.publish(odom)

        # NED odometry of base_link (same layout as the novatel relay)
        R_enu_body = T_map_base[:3, :3]
        heading = _heading_ned_from_R_enu_body(R_enu_body)
        p_ned = geodesy.R_ENU_NED @ T_map_base[:3, 3]
        R_ned_body = geodesy.R_ENU_NED @ R_enu_body
        pitch = math.asin(max(-1.0, min(1.0, -R_ned_body[2, 0])))
        roll = math.atan2(R_ned_body[2, 1], R_ned_body[2, 2])
        odom_ned = Odometry()
        odom_ned.header.stamp = stamp
        odom_ned.header.frame_id = self.ned_frame
        odom_ned.child_frame_id = self.base_frame
        odom_ned.pose.pose.position.x, odom_ned.pose.pose.position.y, odom_ned.pose.pose.position.z = map(float, p_ned)
        qn = rot_to_quat(R_ned_body)
        odom_ned.pose.pose.orientation.x, odom_ned.pose.pose.orientation.y, odom_ned.pose.pose.orientation.z, odom_ned.pose.pose.orientation.w = map(float, qn)
        odom_ned.twist.twist.linear.x, odom_ned.twist.twist.linear.y, odom_ned.twist.twist.linear.z = map(float, v_base)
        odom_ned.twist.twist.angular.x, odom_ned.twist.twist.angular.y, odom_ned.twist.twist.angular.z = map(float, w_base)
        cov_ned = np.zeros((6, 6))
        cov_ned[0, 0], cov_ned[1, 1], cov_ned[2, 2] = var_xyz[1], var_xyz[0], var_xyz[2]
        cov_ned[5, 5] = self.P[3, 3]
        odom_ned.pose.covariance = cov_ned.reshape(-1).tolist()
        self.pub_odom_ned.publish(odom_ned)

        # geodetic fixes
        self.pub_fix_base.publish(self._fix(stamp, self.base_frame, T_map_base[:3, 3], var_xyz))
        self.pub_fix_cam.publish(self._fix(stamp, 'cam0', T_map_C[:3, 3], var_xyz))
        self.pub_fix_ant.publish(self._fix(stamp, self.antenna_frame, p_map_ant, var_xyz))

        if self.pub_navstate is not None:
            lla = geodesy.enu_to_lla(T_map_base[:3, 3], self.ref_lla)
            ns = NavState()
            ns.header.stamp = stamp
            ns.header.frame_id = self.ned_frame
            ns.t_0 = _stamp_to_sec(stamp)
            ns.latitude, ns.longitude = float(lla[0]), float(lla[1])   # radians, as in the novatel relay
            ns.north, ns.east = float(p_ned[0]), float(p_ned[1])
            ns.heading = ns.yaw = float(heading)
            ns.roll, ns.pitch = float(roll), float(pitch)
            ns.surge, ns.sway = float(v_base[0]), float(v_base[1])
            ns.yaw_rate = float(w_base[2])
            ns.cov_mat = [float(var_xyz[1]), float(var_xyz[0]), float(self.P[3, 3])]
            ns.nav_status = 1 if self.n_accepted > 0 else 0
            self.pub_navstate.publish(ns)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GlobalAlignmentNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
