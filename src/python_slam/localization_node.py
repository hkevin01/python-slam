#!/usr/bin/env python3
"""
Localization Node for ROS 2
Provides robot localization within the map
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

from .localization import Localization as LocalizationCore


class LocalizationNode(Node):
    """ROS 2 node for localization."""

    def __init__(self):
        super().__init__('localization_node')

        # Initialize localization core
        self.localizer = LocalizationCore()

        # Declare parameters
        self._declare_parameters()

        # Set up QoS
        self.sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            self.sensor_qos
        )

        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.initial_pose_callback,
            self.sensor_qos
        )

        # Create publishers
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/localization/pose',
            self.sensor_qos
        )

        # Create timer for localization updates
        update_frequency = self.get_parameter('update_frequency').get_parameter_value().double_value
        self.localization_timer = self.create_timer(1.0 / update_frequency, self.localization_update)

        self.get_logger().info('Localization Node initialized')

    def _declare_parameters(self):
        """Declare ROS 2 parameters."""
        self.declare_parameter('particle_count', 500)
        self.declare_parameter('initial_pose_variance', 1.0)
        self.declare_parameter('motion_noise_translation', 0.1)
        self.declare_parameter('motion_noise_rotation', 0.05)
        self.declare_parameter('update_frequency', 30.0)
        self.declare_parameter('map_frame', 'map')

    def map_callback(self, msg: OccupancyGrid):
        """Update map for localization."""
        self.localizer.update_map(msg)

    def initial_pose_callback(self, msg: PoseWithCovarianceStamped):
        """Set initial pose for localization."""
        self.localizer.set_initial_pose(msg.pose.pose)
        self.get_logger().info('Initial pose set for localization')

    def localization_update(self):
        """Update localization and publish pose."""
        try:
            # Update localization
            estimated_pose = self.localizer.update()

            if estimated_pose is not None:
                # Create pose message
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
                pose_msg.pose = estimated_pose

                self.pose_pub.publish(pose_msg)

        except Exception as e:
            self.get_logger().error(f'Error in localization update: {str(e)}')


def main(args=None):
    """Main function."""
    rclpy.init(args=args)

    try:
        node = LocalizationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
