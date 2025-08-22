"""
ROS2 Launch Configuration for Defense-Oriented SLAM System
Supports PX4 integration, UCI interface, and autonomous navigation
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                           GroupAction, ExecuteProcess, TimerAction)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_defense_launch_description():
    """Generate launch description for defense SLAM system"""

    # Declare launch arguments
    declared_args = [
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'enable_vio',
            default_value='true',
            description='Enable Visual-Inertial Odometry'
        ),
        DeclareLaunchArgument(
            'enable_px4',
            default_value='false',
            description='Enable PX4 integration'
        ),
        DeclareLaunchArgument(
            'enable_uci',
            default_value='false',
            description='Enable UCI interface'
        ),
        DeclareLaunchArgument(
            'enable_oms',
            default_value='false',
            description='Enable OMS integration'
        ),
        DeclareLaunchArgument(
            'autonomous_navigation',
            default_value='false',
            description='Enable autonomous navigation'
        ),
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/image_raw',
            description='Camera image topic'
        ),
        DeclareLaunchArgument(
            'imu_topic',
            default_value='/imu/data',
            description='IMU data topic'
        ),
        DeclareLaunchArgument(
            'gps_topic',
            default_value='/gps/fix',
            description='GPS fix topic'
        ),
        DeclareLaunchArgument(
            'max_features',
            default_value='1000',
            description='Maximum features to track'
        ),
        DeclareLaunchArgument(
            'keyframe_distance',
            default_value='1.0',
            description='Distance threshold for keyframe creation'
        ),
        DeclareLaunchArgument(
            'classification_level',
            default_value='UNCLASSIFIED',
            description='Security classification level'
        ),
        DeclareLaunchArgument(
            'uci_command_port',
            default_value='5555',
            description='UCI command port'
        ),
        DeclareLaunchArgument(
            'uci_telemetry_port',
            default_value='5556',
            description='UCI telemetry port'
        ),
        DeclareLaunchArgument(
            'px4_connection',
            default_value='udp://:14540',
            description='PX4 connection string'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Logging level'
        )
    ]

    # Get launch configurations
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_vio = LaunchConfiguration('enable_vio')
    enable_px4 = LaunchConfiguration('enable_px4')
    enable_uci = LaunchConfiguration('enable_uci')
    enable_oms = LaunchConfiguration('enable_oms')
    autonomous_navigation = LaunchConfiguration('autonomous_navigation')
    camera_topic = LaunchConfiguration('camera_topic')
    imu_topic = LaunchConfiguration('imu_topic')
    gps_topic = LaunchConfiguration('gps_topic')
    max_features = LaunchConfiguration('max_features')
    keyframe_distance = LaunchConfiguration('keyframe_distance')
    classification_level = LaunchConfiguration('classification_level')
    uci_command_port = LaunchConfiguration('uci_command_port')
    uci_telemetry_port = LaunchConfiguration('uci_telemetry_port')
    px4_connection = LaunchConfiguration('px4_connection')
    log_level = LaunchConfiguration('log_level')

    # Main Enhanced SLAM Node
    enhanced_slam_node = Node(
        package='python_slam',
        executable='slam_node',
        name='enhanced_slam_node',
        namespace='',
        parameters=[{
            'use_sim_time': use_sim_time,
            'camera_topic': camera_topic,
            'imu_topic': imu_topic,
            'gps_topic': gps_topic,
            'max_features': max_features,
            'keyframe_distance': keyframe_distance,
            'enable_vio': enable_vio,
            'enable_loop_closure': True,
            'enable_mapping': True,
            'enable_gps_fusion': False,
            'px4_mode': enable_px4,
            'uci_interface': enable_uci,
            'oms_integration': enable_oms,
            'autonomous_navigation': autonomous_navigation,
            'processing_frequency': 30.0,
            'state_publish_frequency': 50.0,
            'map_publish_frequency': 1.0,
            'classification_level': classification_level
        }],
        output='screen',
        arguments=['--ros-args', '--log-level', log_level],
        emulate_tty=True
    )

    # Enhanced GUI Visualization Node
    enhanced_viz_node = Node(
        package='python_slam',
        executable='enhanced_visualization_node',
        name='enhanced_slam_viz',
        namespace='',
        parameters=[{
            'use_sim_time': use_sim_time,
            'enable_3d_viz': True,
            'enable_feature_viz': True,
            'enable_metrics_viz': True,
            'enable_defense_viz': True
        }],
        output='screen',
        condition=UnlessCondition(use_sim_time)  # Disable in simulation
    )

    # PX4 Bridge Node (conditional)
    px4_bridge_node = Node(
        package='python_slam',
        executable='px4_bridge_node',
        name='px4_bridge',
        namespace='',
        condition=IfCondition(enable_px4),
        parameters=[{
            'use_sim_time': use_sim_time,
            'px4_connection': px4_connection,
            'classification_level': classification_level
        }],
        output='screen',
        arguments=['--ros-args', '--log-level', log_level]
    )

    # UCI Interface Node (conditional)
    uci_interface_node = Node(
        package='python_slam',
        executable='uci_interface_node',
        name='uci_interface',
        namespace='',
        condition=IfCondition(enable_uci),
        parameters=[{
            'use_sim_time': use_sim_time,
            'command_port': uci_command_port,
            'telemetry_port': uci_telemetry_port,
            'classification_level': classification_level,
            'node_id': 'SLAM_UCI_NODE'
        }],
        output='screen',
        arguments=['--ros-args', '--log-level', log_level]
    )

    # Compile launch description
    launch_description = LaunchDescription()

    # Add declared arguments
    for arg in declared_args:
        launch_description.add_action(arg)

    # Add core nodes
    launch_description.add_action(enhanced_slam_node)
    launch_description.add_action(enhanced_viz_node)

    # Add conditional nodes with delays for proper initialization
    launch_description.add_action(
        TimerAction(
            period=2.0,
            actions=[px4_bridge_node]
        )
    )

    launch_description.add_action(
        TimerAction(
            period=3.0,
            actions=[uci_interface_node]
        )
    )

    return launch_description

if __name__ == '__main__':
    # For testing launch file syntax
    generate_defense_launch_description()
