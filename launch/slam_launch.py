#!/usr/bin/env python3
"""
SLAM Launch File for ROS 2
Launches the complete SLAM pipeline with all necessary nodes
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description for SLAM pipeline."""

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

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )

    # Get package path for config files
    package_path = FindPackageShare('python_slam')

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
            {'ransac_threshold': 1.0},
            {'confidence': 0.99},
            {'max_iterations': 1000},
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
            {'map_size': 1000},
            {'update_frequency': 10.0},
        ]
    )

    # Localization node
    localization_node = Node(
        package='python_slam',
        executable='localization_node',
        name='localization_node',
        output='screen',
        parameters=[
            {'particle_count': 500},
            {'initial_pose_variance': 1.0},
        ]
    )

    # Loop closure node
    loop_closure_node = Node(
        package='python_slam',
        executable='loop_closure_node',
        name='loop_closure_node',
        output='screen',
        parameters=[
            {'similarity_threshold': 0.7},
            {'min_loop_distance': 10.0},
        ]
    )

    # Flight integration node (for drone-specific functionality)
    flight_integration_node = Node(
        package='python_slam',
        executable='flight_integration_node',
        name='flight_integration_node',
        output='screen',
        parameters=[
            {'altitude_control': True},
            {'max_velocity': 2.0},
            {'safety_distance': 1.0},
        ]
    )

    # RViz node (conditional)
    rviz_config_path = os.path.join(package_path, 'rviz', 'slam_config.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        condition=LaunchConfiguration('use_rviz'),
        output='screen'
    )

    return LaunchDescription([
        # Launch arguments
        camera_topic_arg,
        camera_info_topic_arg,
        odom_topic_arg,
        map_frame_arg,
        base_frame_arg,
        use_rviz_arg,

        # Log info
        LogInfo(msg=['Launching Python SLAM pipeline...']),

        # Nodes
        slam_node,
        feature_extraction_node,
        pose_estimation_node,
        mapping_node,
        localization_node,
        loop_closure_node,
        flight_integration_node,
        rviz_node,
    ])
