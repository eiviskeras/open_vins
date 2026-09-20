from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EqualsSubstitution, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_config = str(Path(get_package_share_directory('ov_global')) / 'config' / 'zed_rear_left.yaml')
    default_imucam = str(
        Path(get_package_share_directory('ov_msckf')) / 'config' / 'zed_rear_left_1080p' / 'kalibr_imucam_chain.yaml'
    )
    config = LaunchConfiguration('config')
    imucam_yaml = LaunchConfiguration('imucam_yaml')
    tag_source = LaunchConfiguration('tag_source')

    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=default_config,
                              description='ov_global parameter file (tag survey, lever arms, init, gating)'),
        DeclareLaunchArgument('imucam_yaml', default_value=default_imucam,
                              description='OpenVINS kalibr_imucam_chain.yaml providing K/D and T_imu_cam'),
        DeclareLaunchArgument('tag_source', default_value='vio_landmarks',
                              description='vio_landmarks: anchor on OpenVINS aruco landmarks; pnp: run tag_pnp_node'),
        Node(
            package='ov_global',
            executable='tag_pnp_node',
            name='tag_pnp_node',
            output='screen',
            condition=IfCondition(EqualsSubstitution(tag_source, 'pnp')),
            parameters=[config, {'imucam_yaml': imucam_yaml}],
        ),
        Node(
            package='ov_global',
            executable='global_alignment_node',
            name='global_alignment_node',
            output='screen',
            parameters=[config, {'imucam_yaml': imucam_yaml, 'tag_source': tag_source}],
        ),
    ])
