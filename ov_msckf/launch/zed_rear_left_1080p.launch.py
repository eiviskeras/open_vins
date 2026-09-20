from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import AndSubstitution, EqualsSubstitution, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config_path = str(
        Path(get_package_share_directory("ov_msckf"))
        / "config"
        / "zed_rear_left_1080p"
        / "estimator_config.yaml"
    )

    imu_topic = LaunchConfiguration("imu_topic")
    left_compressed_topic = LaunchConfiguration("left_compressed_topic")
    right_compressed_topic = LaunchConfiguration("right_compressed_topic")
    rviz_enable = LaunchConfiguration("rviz_enable")
    apriltag_debug_enable = LaunchConfiguration("apriltag_debug_enable")
    ov_global_enable = LaunchConfiguration("ov_global_enable")
    ov_global_config = LaunchConfiguration("ov_global_config")
    ov_global_tag_source = LaunchConfiguration("ov_global_tag_source")
    try:
        ov_global_default_config = str(
            Path(get_package_share_directory("ov_global")) / "config" / "zed_rear_left.yaml"
        )
    except PackageNotFoundError:
        # ov_global not built: keep the launch usable with ov_global_enable:=false
        ov_global_default_config = ""
    imucam_yaml = str(
        Path(get_package_share_directory("ov_msckf"))
        / "config"
        / "zed_rear_left_1080p"
        / "kalibr_imucam_chain.yaml"
    )
    rviz_config = str(
        Path(get_package_share_directory("ov_msckf"))
        / "launch"
        / "zed_rear_left_1080p.rviz"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "imu_topic", default_value="/zed_rear_left/imu/data_raw"
            ),
            DeclareLaunchArgument(
                "left_compressed_topic",
                default_value="/zed_rear_left/left/image/compressed",
            ),
            DeclareLaunchArgument(
                "right_compressed_topic",
                default_value="/zed_rear_left/right/image/compressed",
            ),
            DeclareLaunchArgument(
                "rviz_enable",
                default_value="true",
                description="Launch RViz with stereo images and the estimated path",
            ),
            DeclareLaunchArgument(
                "apriltag_debug_enable",
                default_value="false",
                description="Run an independent OpenCV AprilTag detector for comparison",
            ),
            DeclareLaunchArgument(
                "ov_global_enable",
                default_value="true",
                description="Run the ov_global tag pose and global alignment nodes",
            ),
            DeclareLaunchArgument(
                "ov_global_config",
                default_value=ov_global_default_config,
                description="ov_global parameter file (tag survey, lever arms, init, gating)",
            ),
            DeclareLaunchArgument(
                "ov_global_tag_source",
                default_value="vio_landmarks",
                description="vio_landmarks: anchor on OpenVINS aruco landmarks; pnp: run tag_pnp_node",
            ),
            Node(
                package="image_transport",
                executable="republish",
                name="zed_rear_left_decode_left",
                arguments=["compressed", "raw"],
                remappings=[
                    ("in/compressed", left_compressed_topic),
                    ("out", "/openvins/zed_rear_left/left/image_raw"),
                ],
            ),
            Node(
                package="image_transport",
                executable="republish",
                name="zed_rear_left_decode_right",
                arguments=["compressed", "raw"],
                remappings=[
                    ("in/compressed", right_compressed_topic),
                    ("out", "/openvins/zed_rear_left/right/image_raw"),
                ],
            ),
            Node(
                package="ov_msckf",
                executable="run_subscribe_msckf",
                namespace="ov_msckf",
                name="zed_rear_left",
                output="screen",
                parameters=[
                    {"config_path": config_path},
                ],
                remappings=[
                    ("/zed_rear_left/imu/data_raw", imu_topic),
                ],
            ),
            Node(
                package="ov_msckf",
                executable="apriltag_opencv_debug.py",
                name="apriltag_opencv_debug",
                condition=IfCondition(apriltag_debug_enable),
                output="screen",
                parameters=[
                    {
                        "camera_topics": [
                            "/openvins/zed_rear_left/left/image_raw",
                            "/openvins/zed_rear_left/right/image_raw",
                        ]
                    }
                ],
            ),
            Node(
                package="ov_msckf",
                executable="apriltag_detection_compare.py",
                name="apriltag_detection_compare",
                condition=IfCondition(apriltag_debug_enable),
                output="screen",
            ),
            Node(
                package="ov_global",
                executable="tag_pnp_node",
                name="tag_pnp_node",
                output="screen",
                condition=IfCondition(
                    AndSubstitution(ov_global_enable, EqualsSubstitution(ov_global_tag_source, "pnp"))
                ),
                parameters=[ov_global_config, {"imucam_yaml": imucam_yaml}],
            ),
            Node(
                package="ov_global",
                executable="global_alignment_node",
                name="global_alignment_node",
                output="screen",
                condition=IfCondition(ov_global_enable),
                parameters=[
                    ov_global_config,
                    {"imucam_yaml": imucam_yaml, "tag_source": ov_global_tag_source},
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="zed_rear_left_vins_rviz",
                condition=IfCondition(rviz_enable),
                output="screen",
                arguments=["-d", rviz_config],
            ),
        ]
    )