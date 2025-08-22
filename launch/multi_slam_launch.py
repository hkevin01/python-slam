#!/usr/bin/env python3
"""
Launch file for Multi-Algorithm SLAM system.

Launches the multi-SLAM node with configurable parameters for different
sensor setups and algorithm choices.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition, UnlessCondition
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """Generate launch description for multi-SLAM system."""

    # Package directory
    pkg_dir = get_package_share_directory('python_slam')
    config_dir = os.path.join(pkg_dir, 'config')

    # Launch arguments
    declare_algorithm = DeclareLaunchArgument(
        'algorithm',
        default_value='auto',
        description='SLAM algorithm to use (auto, orb_slam3, rtabmap, cartographer, openvslam, python_slam)'
    )

    declare_sensor_type = DeclareLaunchArgument(
        'sensor_type',
        default_value='auto',
        description='Sensor type (auto, monocular, stereo, rgbd, visual_inertial, lidar, pointcloud)'
    )

    declare_map_frame = DeclareLaunchArgument(
        'map_frame',
        default_value='map',
        description='Map frame ID'
    )

    declare_odom_frame = DeclareLaunchArgument(
        'odom_frame',
        default_value='odom',
        description='Odometry frame ID'
    )

    declare_base_frame = DeclareLaunchArgument(
        'base_frame',
        default_value='base_link',
        description='Base frame ID'
    )

    declare_camera_frame = DeclareLaunchArgument(
        'camera_frame',
        default_value='camera_link',
        description='Camera frame ID'
    )

    # Topic arguments
    declare_image_topic = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='Monocular camera image topic'
    )

    declare_left_image_topic = DeclareLaunchArgument(
        'left_image_topic',
        default_value='/camera/left/image_raw',
        description='Left stereo camera image topic'
    )

    declare_right_image_topic = DeclareLaunchArgument(
        'right_image_topic',
        default_value='/camera/right/image_raw',
        description='Right stereo camera image topic'
    )

    declare_depth_topic = DeclareLaunchArgument(
        'depth_topic',
        default_value='/camera/depth/image_raw',
        description='Depth image topic for RGB-D'
    )

    declare_pointcloud_topic = DeclareLaunchArgument(
        'pointcloud_topic',
        default_value='/velodyne_points',
        description='Point cloud topic'
    )

    declare_imu_topic = DeclareLaunchArgument(
        'imu_topic',
        default_value='/imu/data',
        description='IMU data topic'
    )

    declare_laser_topic = DeclareLaunchArgument(
        'laser_topic',
        default_value='/scan',
        description='Laser scan topic'
    )

    # SLAM parameters
    declare_max_features = DeclareLaunchArgument(
        'max_features',
        default_value='1000',
        description='Maximum number of features to extract'
    )

    declare_enable_loop_closure = DeclareLaunchArgument(
        'enable_loop_closure',
        default_value='true',
        description='Enable loop closure detection'
    )

    declare_enable_mapping = DeclareLaunchArgument(
        'enable_mapping',
        default_value='true',
        description='Enable mapping'
    )

    declare_publish_tf = DeclareLaunchArgument(
        'publish_tf',
        default_value='true',
        description='Publish TF transforms'
    )

    declare_publish_pose = DeclareLaunchArgument(
        'publish_pose',
        default_value='true',
        description='Publish pose estimates'
    )

    declare_publish_map = DeclareLaunchArgument(
        'publish_map',
        default_value='true',
        description='Publish map data'
    )

    # Performance parameters
    declare_pose_publish_rate = DeclareLaunchArgument(
        'pose_publish_rate',
        default_value='30.0',
        description='Pose publishing rate (Hz)'
    )

    declare_map_publish_rate = DeclareLaunchArgument(
        'map_publish_rate',
        default_value='1.0',
        description='Map publishing rate (Hz)'
    )

    declare_status_publish_rate = DeclareLaunchArgument(
        'status_publish_rate',
        default_value='1.0',
        description='Status publishing rate (Hz)'
    )

    # Camera parameters
    declare_camera_fx = DeclareLaunchArgument(
        'camera_fx',
        default_value='525.0',
        description='Camera focal length X'
    )

    declare_camera_fy = DeclareLaunchArgument(
        'camera_fy',
        default_value='525.0',
        description='Camera focal length Y'
    )

    declare_camera_cx = DeclareLaunchArgument(
        'camera_cx',
        default_value='319.5',
        description='Camera principal point X'
    )

    declare_camera_cy = DeclareLaunchArgument(
        'camera_cy',
        default_value='239.5',
        description='Camera principal point Y'
    )

    declare_camera_baseline = DeclareLaunchArgument(
        'camera_baseline',
        default_value='0.1',
        description='Stereo camera baseline (meters)'
    )

    # Debug/logging
    declare_log_level = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level (debug, info, warn, error)'
    )

    declare_enable_rviz = DeclareLaunchArgument(
        'enable_rviz',
        default_value='true',
        description='Launch RViz for visualization'
    )

    # Multi-SLAM node
    multi_slam_node = Node(
        package='python_slam',
        executable='multi_slam_node',
        name='multi_slam_node',
        output='screen',
        parameters=[{
            # Algorithm selection
            'algorithm': LaunchConfiguration('algorithm'),
            'sensor_type': LaunchConfiguration('sensor_type'),

            # Frame names
            'map_frame': LaunchConfiguration('map_frame'),
            'odom_frame': LaunchConfiguration('odom_frame'),
            'base_frame': LaunchConfiguration('base_frame'),
            'camera_frame': LaunchConfiguration('camera_frame'),

            # Topic names
            'image_topic': LaunchConfiguration('image_topic'),
            'left_image_topic': LaunchConfiguration('left_image_topic'),
            'right_image_topic': LaunchConfiguration('right_image_topic'),
            'depth_topic': LaunchConfiguration('depth_topic'),
            'pointcloud_topic': LaunchConfiguration('pointcloud_topic'),
            'imu_topic': LaunchConfiguration('imu_topic'),
            'laser_topic': LaunchConfiguration('laser_topic'),

            # SLAM parameters
            'max_features': LaunchConfiguration('max_features'),
            'enable_loop_closure': LaunchConfiguration('enable_loop_closure'),
            'enable_mapping': LaunchConfiguration('enable_mapping'),
            'publish_tf': LaunchConfiguration('publish_tf'),
            'publish_pose': LaunchConfiguration('publish_pose'),
            'publish_map': LaunchConfiguration('publish_map'),

            # Performance parameters
            'pose_publish_rate': LaunchConfiguration('pose_publish_rate'),
            'map_publish_rate': LaunchConfiguration('map_publish_rate'),
            'status_publish_rate': LaunchConfiguration('status_publish_rate'),

            # Camera parameters
            'camera_fx': LaunchConfiguration('camera_fx'),
            'camera_fy': LaunchConfiguration('camera_fy'),
            'camera_cx': LaunchConfiguration('camera_cx'),
            'camera_cy': LaunchConfiguration('camera_cy'),
            'camera_baseline': LaunchConfiguration('camera_baseline'),
        }],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
        remappings=[
            # Remap standard topics if needed
            ('/camera/image_raw', LaunchConfiguration('image_topic')),
            ('/camera/left/image_raw', LaunchConfiguration('left_image_topic')),
            ('/camera/right/image_raw', LaunchConfiguration('right_image_topic')),
            ('/camera/depth/image_raw', LaunchConfiguration('depth_topic')),
            ('/velodyne_points', LaunchConfiguration('pointcloud_topic')),
            ('/imu/data', LaunchConfiguration('imu_topic')),
            ('/scan', LaunchConfiguration('laser_topic')),
        ]
    )

    # RViz node for visualization
    rviz_config_file = os.path.join(config_dir, 'multi_slam_rviz.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        condition=IfCondition(LaunchConfiguration('enable_rviz')),
        output='screen'
    )

    # Static transform publishers (examples)
    # These would typically be provided by robot description
    base_to_camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        arguments=[
            '0.1', '0.0', '0.1',  # x, y, z
            '0.0', '0.0', '0.0', '1.0',  # qx, qy, qz, qw
            LaunchConfiguration('base_frame'),
            LaunchConfiguration('camera_frame')
        ]
    )

    return LaunchDescription([
        # Declare all launch arguments
        declare_algorithm,
        declare_sensor_type,
        declare_map_frame,
        declare_odom_frame,
        declare_base_frame,
        declare_camera_frame,
        declare_image_topic,
        declare_left_image_topic,
        declare_right_image_topic,
        declare_depth_topic,
        declare_pointcloud_topic,
        declare_imu_topic,
        declare_laser_topic,
        declare_max_features,
        declare_enable_loop_closure,
        declare_enable_mapping,
        declare_publish_tf,
        declare_publish_pose,
        declare_publish_map,
        declare_pose_publish_rate,
        declare_map_publish_rate,
        declare_status_publish_rate,
        declare_camera_fx,
        declare_camera_fy,
        declare_camera_cx,
        declare_camera_cy,
        declare_camera_baseline,
        declare_log_level,
        declare_enable_rviz,

        # Launch nodes
        multi_slam_node,
        rviz_node,
        base_to_camera_tf,
    ])


def generate_monocular_launch():
    """Generate launch description for monocular camera setup."""
    return LaunchDescription([
        DeclareLaunchArgument('algorithm', default_value='orb_slam3'),
        DeclareLaunchArgument('sensor_type', default_value='monocular'),
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),

        Node(
            package='python_slam',
            executable='multi_slam_node',
            name='mono_slam_node',
            parameters=[{
                'algorithm': LaunchConfiguration('algorithm'),
                'sensor_type': LaunchConfiguration('sensor_type'),
                'image_topic': LaunchConfiguration('image_topic'),
            }]
        )
    ])


def generate_stereo_launch():
    """Generate launch description for stereo camera setup."""
    return LaunchDescription([
        DeclareLaunchArgument('algorithm', default_value='orb_slam3'),
        DeclareLaunchArgument('sensor_type', default_value='stereo'),
        DeclareLaunchArgument('left_image_topic', default_value='/camera/left/image_raw'),
        DeclareLaunchArgument('right_image_topic', default_value='/camera/right/image_raw'),

        Node(
            package='python_slam',
            executable='multi_slam_node',
            name='stereo_slam_node',
            parameters=[{
                'algorithm': LaunchConfiguration('algorithm'),
                'sensor_type': LaunchConfiguration('sensor_type'),
                'left_image_topic': LaunchConfiguration('left_image_topic'),
                'right_image_topic': LaunchConfiguration('right_image_topic'),
            }]
        )
    ])


def generate_rgbd_launch():
    """Generate launch description for RGB-D camera setup."""
    return LaunchDescription([
        DeclareLaunchArgument('algorithm', default_value='rtabmap'),
        DeclareLaunchArgument('sensor_type', default_value='rgbd'),
        DeclareLaunchArgument('image_topic', default_value='/camera/color/image_raw'),
        DeclareLaunchArgument('depth_topic', default_value='/camera/depth/image_raw'),

        Node(
            package='python_slam',
            executable='multi_slam_node',
            name='rgbd_slam_node',
            parameters=[{
                'algorithm': LaunchConfiguration('algorithm'),
                'sensor_type': LaunchConfiguration('sensor_type'),
                'image_topic': LaunchConfiguration('image_topic'),
                'depth_topic': LaunchConfiguration('depth_topic'),
            }]
        )
    ])


def generate_lidar_launch():
    """Generate launch description for LiDAR setup."""
    return LaunchDescription([
        DeclareLaunchArgument('algorithm', default_value='cartographer'),
        DeclareLaunchArgument('sensor_type', default_value='lidar'),
        DeclareLaunchArgument('laser_topic', default_value='/scan'),
        DeclareLaunchArgument('pointcloud_topic', default_value='/velodyne_points'),

        Node(
            package='python_slam',
            executable='multi_slam_node',
            name='lidar_slam_node',
            parameters=[{
                'algorithm': LaunchConfiguration('algorithm'),
                'sensor_type': LaunchConfiguration('sensor_type'),
                'laser_topic': LaunchConfiguration('laser_topic'),
                'pointcloud_topic': LaunchConfiguration('pointcloud_topic'),
            }]
        )
    ])
