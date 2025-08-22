#!/usr/bin/env python3
"""
SLAM Backend Launch File for ROS 2
Launches the SLAM processing pipeline without visualization components
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description for SLAM backend pipeline."""

    # Declare launch arguments
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Camera image topic name'
    )

    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/camera/camera_info',
        description='Camera info topic name'
    )

    odom_topic_arg = DeclareLaunchArgument(
        'odom_topic',
        default_value='/odom',
        description='Odometry topic name'
    )

    map_frame_arg = DeclareLaunchArgument(
        'map_frame',
        default_value='map',
        description='Map frame ID'
    )

    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='base_link',
        description='Base frame ID'
    )

    # PX4 and UCI integration arguments
    enable_px4_arg = DeclareLaunchArgument(
        'enable_px4',
        default_value='false',
        description='Enable PX4 integration for UAS operations'
    )

    enable_uci_arg = DeclareLaunchArgument(
        'enable_uci',
        default_value='false',
        description='Enable UCI interface for control applications'
    )

    px4_connection_arg = DeclareLaunchArgument(
        'px4_connection',
        default_value='udp://:14540',
        description='PX4 connection string'
    )

    uci_command_port_arg = DeclareLaunchArgument(
        'uci_command_port',
        default_value='5555',
        description='UCI command port'
    )

    uci_telemetry_port_arg = DeclareLaunchArgument(
        'uci_telemetry_port',
        default_value='5556',
        description='UCI telemetry port'
    )

    # SLAM main node
    slam_node = Node(
        package='python_slam',
        executable='slam_node',
        name='slam_node',
        output='screen',
        parameters=[
            {'camera_topic': LaunchConfiguration('camera_topic')},
            {'camera_info_topic': LaunchConfiguration('camera_info_topic')},
            {'odom_topic': LaunchConfiguration('odom_topic')},
            {'map_frame': LaunchConfiguration('map_frame')},
            {'base_frame': LaunchConfiguration('base_frame')},
            # Integration parameters
            {'enable_px4': LaunchConfiguration('enable_px4')},
            {'enable_uci': LaunchConfiguration('enable_uci')},
            {'px4_connection': LaunchConfiguration('px4_connection')},
            {'uci_command_port': LaunchConfiguration('uci_command_port')},
            {'uci_telemetry_port': LaunchConfiguration('uci_telemetry_port')},
            # SLAM parameters
            {'max_features': 1000},
            {'keyframe_distance': 1.0},
            {'enable_vio': True},
            {'enable_loop_closure': True},
            {'enable_mapping': True},
            {'processing_frequency': 30.0},
            {'state_publish_frequency': 50.0},
            {'map_publish_frequency': 1.0},
        ],
        remappings=[
            ('/camera/image_raw', LaunchConfiguration('camera_topic')),
            ('/camera/camera_info', LaunchConfiguration('camera_info_topic')),
            ('/odom', LaunchConfiguration('odom_topic')),
        ]
    )

    # Feature extraction node
    feature_extraction_node = Node(
        package='python_slam',
        executable='feature_extraction_node',
        name='feature_extraction_node',
        output='screen',
        parameters=[
            {'max_features': 1000},
            {'quality_level': 0.01},
            {'min_distance': 10},
        ]
    )

    # Pose estimation node
    pose_estimation_node = Node(
        package='python_slam',
        executable='pose_estimation_node',
        name='pose_estimation_node',
        output='screen',
        parameters=[
            {'estimation_method': 'pnp'},
            {'min_inliers': 20},
        ]
    )

    # Mapping node
    mapping_node = Node(
        package='python_slam',
        executable='mapping_node',
        name='mapping_node',
        output='screen',
        parameters=[
            {'map_resolution': 0.05},
            {'map_size_x': 2048},
            {'map_size_y': 2048},
        ]
    )

    # Localization node
    localization_node = Node(
        package='python_slam',
        executable='localization_node',
        name='localization_node',
        output='screen',
        parameters=[
            {'localization_frequency': 30.0},
            {'particle_count': 500},
        ]
    )

    # Loop closure node
    loop_closure_node = Node(
        package='python_slam',
        executable='loop_closure_node',
        name='loop_closure_node',
        output='screen',
        parameters=[
            {'enable_loop_closure': True},
            {'similarity_threshold': 0.7},
        ]
    )

    # Flight integration node (conditional)
    flight_integration_node = Node(
        package='python_slam',
        executable='flight_integration_node',
        name='flight_integration_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_px4')),
        parameters=[
            {'px4_connection': LaunchConfiguration('px4_connection')},
            {'flight_mode': 'position'},
        ]
    )

    # UCI interface node (conditional)
    uci_interface_node = Node(
        package='python_slam',
        executable='uci_interface_node',
        name='uci_interface_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('enable_uci')),
        parameters=[
            {'command_port': LaunchConfiguration('uci_command_port')},
            {'telemetry_port': LaunchConfiguration('uci_telemetry_port')},
        ]
    )

    # Transform broadcaster
    tf_broadcaster_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
    )

    return LaunchDescription([
        # Arguments
        camera_topic_arg,
        camera_info_topic_arg,
        odom_topic_arg,
        map_frame_arg,
        base_frame_arg,
        enable_px4_arg,
        enable_uci_arg,
        px4_connection_arg,
        uci_command_port_arg,
        uci_telemetry_port_arg,

        # Log startup
        LogInfo(msg='Starting SLAM Backend Pipeline'),

        # Core SLAM nodes
        slam_node,
        feature_extraction_node,
        pose_estimation_node,
        mapping_node,
        localization_node,
        loop_closure_node,
        tf_broadcaster_node,

        # Conditional nodes
        flight_integration_node,
        uci_interface_node,
    ])
