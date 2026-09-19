#!/usr/bin/env python3

"""Independent OpenCV AprilTag detector used to compare against OpenVINS."""

import json
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


class AprilTagOpenCVDebug(Node):
    def __init__(self) -> None:
        super().__init__("apriltag_opencv_debug")
        self.declare_parameter(
            "camera_topics",
            [
                "/openvins/zed_rear_left/left/image_raw",
                "/openvins/zed_rear_left/right/image_raw",
            ],
        )
        self.declare_parameter("equalize_hist", False)
        self.declare_parameter("downsample", False)

        self._bridge = CvBridge()
        self._equalize_hist = bool(self.get_parameter("equalize_hist").value)
        self._downsample = bool(self.get_parameter("downsample").value)
        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11
        )
        detector_parameters = cv2.aruco.DetectorParameters_create()
        self._dictionary = dictionary
        self._detector_parameters = detector_parameters
        self._publisher = self.create_publisher(
            String, "/apriltag_opencv/detections", 100
        )
        self._subscriptions = []
        topics = list(self.get_parameter("camera_topics").value)
        for camera_id, topic in enumerate(topics):
            self._subscriptions.append(
                self.create_subscription(
                    Image,
                    topic,
                    lambda message, cid=camera_id: self._detect(message, cid),
                    10,
                )
            )
        self.get_logger().info(
            "Independent DICT_APRILTAG_36h11 detector subscribed to "
            + ", ".join(topics)
        )

    def _detect(self, message: Image, camera_id: int) -> None:
        started_ns = time.perf_counter_ns()
        image = self._bridge.imgmsg_to_cv2(message, desired_encoding="mono8")
        if self._equalize_hist:
            image = cv2.equalizeHist(image)
        if self._downsample:
            image = cv2.pyrDown(image)

        _, ids, rejected = cv2.aruco.detectMarkers(
            image, self._dictionary, parameters=self._detector_parameters
        )
        detected_ids = [] if ids is None else [int(value) for value in ids.flat]
        timestamp_ns = message.header.stamp.sec * 1_000_000_000
        timestamp_ns += message.header.stamp.nanosec
        payload = {
            "source": "opencv_raw",
            "timestamp_ns": timestamp_ns,
            "camera_id": camera_id,
            "count": len(detected_ids),
            "detected_ids": detected_ids,
            "rejected_candidates": len(rejected),
            "processing_ms": (time.perf_counter_ns() - started_ns) / 1e6,
        }
        output = String()
        output.data = json.dumps(payload, separators=(",", ":"))
        self._publisher.publish(output)
        if detected_ids:
            self.get_logger().info(
                f"OpenCV AprilTag detection: camera={camera_id} "
                f"timestamp_ns={timestamp_ns} ids={detected_ids}"
            )


def main() -> None:
    rclpy.init()
    node = AprilTagOpenCVDebug()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()