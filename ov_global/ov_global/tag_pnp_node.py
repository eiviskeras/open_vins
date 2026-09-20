#!/usr/bin/env python3
"""Detect one AprilTag and publish its 6-DoF pose in the camera frame (T_C_tag).

Intrinsics come from the OpenVINS kalibr chain YAML so the node works on bags
without ``camera_info``. A ``camera_info`` subscription overrides them if present.

Tag frame (OpenCV IPPE_SQUARE convention): X right, Y up, Z out of the tag face
towards the viewer; origin at the tag centre.
"""

import math
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64

from ov_global.kalibr import load_camera
from ov_global.se3 import rot_to_quat


class TagPnpNode(Node):
    def __init__(self) -> None:
        super().__init__('tag_pnp_node')
        self.declare_parameter('image_topic', '/openvins/zed_rear_left/left/image_raw')
        self.declare_parameter('camera_info_topic', '')
        self.declare_parameter('imucam_yaml', '')
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('camera_frame', 'cam0')
        self.declare_parameter('tag_pose_topic', '/ov_global/tag_pose')
        self.declare_parameter('tag_size_m', 0.162)
        self.declare_parameter('tag_dictionary', 'DICT_APRILTAG_36h11')
        self.declare_parameter('tag_id', 303)
        # normalized [x, y, w, h] region scanned for the tag (same semantics as OpenVINS aruco_detection_roi)
        self.declare_parameter('detection_roi', [0.0, 0.0, 1.0, 1.0])
        self.declare_parameter('corner_sigma_px', 0.5)
        self.declare_parameter('max_reproj_error_px', 2.0)
        self.declare_parameter('max_range_m', 60.0)

        self._camera_frame = str(self.get_parameter('camera_frame').value)
        self._tag_size = float(self.get_parameter('tag_size_m').value)
        self._roi = [float(v) for v in self.get_parameter('detection_roi').value]
        self._sigma_px = float(self.get_parameter('corner_sigma_px').value)
        self._max_reproj = float(self.get_parameter('max_reproj_error_px').value)
        self._max_range = float(self.get_parameter('max_range_m').value)
        self._target_id = int(self.get_parameter('tag_id').value)

        self._K: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        yaml_path = str(self.get_parameter('imucam_yaml').value)
        if yaml_path:
            cam = load_camera(yaml_path, int(self.get_parameter('camera_id').value))
            self._K, self._D = cam.K, cam.D
            self.get_logger().info(f'intrinsics from {yaml_path} cam{cam.cam_id}: fx={cam.K[0,0]:.1f} fy={cam.K[1,1]:.1f}')

        s = self._tag_size / 2.0
        self._obj_pts = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], dtype=np.float64)

        self._detector = self._make_detector(str(self.get_parameter('tag_dictionary').value))
        self._pub = self.create_publisher(PoseWithCovarianceStamped, self.get_parameter('tag_pose_topic').value, 10)
        self._pub_reproj = self.create_publisher(Float64, '~/reprojection_error_px', 10)

        info_topic = str(self.get_parameter('camera_info_topic').value)
        if info_topic:
            self.create_subscription(CameraInfo, info_topic, self._on_camera_info, 10)
        self.create_subscription(Image, self.get_parameter('image_topic').value, self._on_image, 5)
        self.get_logger().info(f'tag_pnp_node: tag {self._target_id}, size {self._tag_size:.3f} m, roi {self._roi}')

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _make_detector(dictionary_name: str):
        dictionary_id = getattr(cv2.aruco, dictionary_name, None)
        if dictionary_id is None:
            raise ValueError(f'unsupported marker dictionary: {dictionary_name}')
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        if hasattr(cv2.aruco, 'DetectorParameters'):
            params = cv2.aruco.DetectorParameters()
        else:
            params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        if hasattr(cv2.aruco, 'ArucoDetector'):
            return cv2.aruco.ArucoDetector(dictionary, params)
        return (dictionary, params)

    def _detect(self, gray: np.ndarray):
        if hasattr(self._detector, 'detectMarkers'):
            return self._detector.detectMarkers(gray)
        dictionary, params = self._detector
        return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self._D = np.array(msg.d, dtype=np.float64).reshape(-1, 1)

    @staticmethod
    def _to_gray(msg: Image) -> np.ndarray:
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        enc = msg.encoding.lower()
        if enc in ('mono8', '8uc1'):
            return buf.reshape(msg.height, msg.step)[:, : msg.width]
        channels = msg.step // msg.width
        img = buf.reshape(msg.height, msg.step)[:, : msg.width * channels].reshape(msg.height, msg.width, channels)
        if channels == 4:
            code = cv2.COLOR_RGBA2GRAY if enc.startswith('rgb') else cv2.COLOR_BGRA2GRAY
        else:
            code = cv2.COLOR_RGB2GRAY if enc.startswith('rgb') else cv2.COLOR_BGR2GRAY
        return cv2.cvtColor(img, code)

    # -------------------------------------------------------------- pipeline
    def _on_image(self, msg: Image) -> None:
        if self._K is None:
            self.get_logger().warning('no intrinsics yet (set imucam_yaml or camera_info_topic)', throttle_duration_sec=5.0)
            return

        gray = self._to_gray(msg)
        h, w = gray.shape[:2]
        x0 = int(max(0, min(w - 1, math.floor(self._roi[0] * w))))
        y0 = int(max(0, min(h - 1, math.floor(self._roi[1] * h))))
        rw = int(max(1, min(w - x0, math.ceil(self._roi[2] * w))))
        rh = int(max(1, min(h - y0, math.ceil(self._roi[3] * h))))
        corners, ids, _ = self._detect(gray[y0 : y0 + rh, x0 : x0 + rw])
        if ids is None:
            return
        idx = [i for i, v in enumerate(ids.flatten().tolist()) if int(v) == self._target_id]
        if not idx:
            return

        img_pts = np.asarray(corners[idx[0]], dtype=np.float64).reshape(4, 2)
        img_pts[:, 0] += x0
        img_pts[:, 1] += y0

        ok, rvec, tvec = cv2.solvePnP(self._obj_pts, img_pts, self._K, self._D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return
        rng = float(np.linalg.norm(tvec))
        if rng > self._max_range or tvec[2] <= 0.0:
            return

        proj, _ = cv2.projectPoints(self._obj_pts, rvec, tvec, self._K, self._D)
        reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - img_pts) ** 2, axis=1))))
        self._pub_reproj.publish(Float64(data=reproj))
        if reproj > self._max_reproj:
            self.get_logger().warning(f'tag {self._target_id}: reprojection {reproj:.2f} px > {self._max_reproj}; dropped')
            return

        # First-order covariance: depth error ~ sigma_px * z^2 / (f * tag_size), lateral ~ sigma_px * z / f,
        # rotation ~ sigma_px * z / (f * tag_size).
        f = 0.5 * (self._K[0, 0] + self._K[1, 1])
        z = float(tvec[2])
        lat_sigma = self._sigma_px * z / f
        depth_sigma = self._sigma_px * z * z / (f * self._tag_size)
        rot_sigma = self._sigma_px * z / (f * self._tag_size)

        R, _ = cv2.Rodrigues(rvec)
        qx, qy, qz, qw = rot_to_quat(R)
        out = PoseWithCovarianceStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self._camera_frame
        out.pose.pose.position.x = float(tvec[0])
        out.pose.pose.position.y = float(tvec[1])
        out.pose.pose.position.z = z
        out.pose.pose.orientation.x = qx
        out.pose.pose.orientation.y = qy
        out.pose.pose.orientation.z = qz
        out.pose.pose.orientation.w = qw
        cov = np.zeros((6, 6))
        cov[0, 0] = cov[1, 1] = lat_sigma**2
        cov[2, 2] = depth_sigma**2
        cov[3, 3] = cov[4, 4] = cov[5, 5] = rot_sigma**2
        out.pose.covariance = cov.reshape(-1).tolist()
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TagPnpNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
