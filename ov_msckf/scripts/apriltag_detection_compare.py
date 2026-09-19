#!/usr/bin/env python3

"""Timestamp-match OpenVINS and independent OpenCV AprilTag scan results."""

import json
from typing import Dict, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


Key = Tuple[int, int]


class AprilTagDetectionCompare(Node):
    def __init__(self) -> None:
        super().__init__("apriltag_detection_compare")
        self.declare_parameter("timestamp_tolerance_ns", 1000)
        self.declare_parameter("report_interval_sec", 5.0)
        self._tolerance_ns = int(
            self.get_parameter("timestamp_tolerance_ns").value
        )
        self._pending: Dict[str, Dict[Key, dict]] = {
            "openvins": {},
            "opencv_raw": {},
        }
        self._stats = {
            "matched_frames": 0,
            "both_detected": 0,
            "opencv_raw_only": 0,
            "openvins_only": 0,
            "neither_detected": 0,
        }
        self._publisher = self.create_publisher(
            String, "/apriltag_comparison/stats", 10
        )
        self._subscriptions = [
            self.create_subscription(
                String,
                "/ov_msckf/apriltag_detections",
                lambda message: self._receive(message, "openvins"),
                100,
            ),
            self.create_subscription(
                String,
                "/apriltag_opencv/detections",
                lambda message: self._receive(message, "opencv_raw"),
                100,
            ),
        ]
        interval = float(self.get_parameter("report_interval_sec").value)
        self.create_timer(interval, self._report)

    def _receive(self, message: String, source: str) -> None:
        try:
            result = json.loads(message.data)
            key = (int(result["camera_id"]), int(result["timestamp_ns"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().warning(f"Invalid {source} diagnostic: {error}")
            return

        other_source = "opencv_raw" if source == "openvins" else "openvins"
        match_key = self._nearest_key(self._pending[other_source], key)
        if match_key is None:
            self._pending[source][key] = result
            self._trim_pending()
            return

        other = self._pending[other_source].pop(match_key)
        openvins = result if source == "openvins" else other
        opencv_raw = result if source == "opencv_raw" else other
        self._record_match(openvins, opencv_raw)

    def _nearest_key(self, candidates: Dict[Key, dict], key: Key):
        camera_id, timestamp_ns = key
        matches = [
            candidate
            for candidate in candidates
            if candidate[0] == camera_id
            and abs(candidate[1] - timestamp_ns) <= self._tolerance_ns
        ]
        if not matches:
            return None
        return min(matches, key=lambda candidate: abs(candidate[1] - timestamp_ns))

    def _record_match(self, openvins: dict, opencv_raw: dict) -> None:
        openvins_ids = set(openvins.get("detected_ids", []))
        opencv_ids = set(opencv_raw.get("detected_ids", []))
        self._stats["matched_frames"] += 1
        if openvins_ids and opencv_ids:
            category = "both_detected"
        elif opencv_ids:
            category = "opencv_raw_only"
        elif openvins_ids:
            category = "openvins_only"
        else:
            category = "neither_detected"
        self._stats[category] += 1

        if openvins_ids != opencv_ids and (openvins_ids or opencv_ids):
            self.get_logger().info(
                "Detection mismatch: "
                f"camera={openvins['camera_id']} "
                f"timestamp_ns={opencv_raw['timestamp_ns']} "
                f"openvins={sorted(openvins_ids)} "
                f"opencv_raw={sorted(opencv_ids)}"
            )

    def _trim_pending(self) -> None:
        for pending in self._pending.values():
            while len(pending) > 2000:
                pending.pop(next(iter(pending)))

    def _report(self) -> None:
        report = dict(self._stats)
        report["pending_openvins"] = len(self._pending["openvins"])
        report["pending_opencv_raw"] = len(self._pending["opencv_raw"])
        output = String()
        output.data = json.dumps(report, separators=(",", ":"))
        self._publisher.publish(output)
        self.get_logger().info(f"AprilTag comparison: {output.data}")


def main() -> None:
    rclpy.init()
    node = AprilTagDetectionCompare()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()