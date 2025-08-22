#!/usr/bin/env python3
"""
Mapping Node for ROS 2
Creates and maintains occupancy grid maps
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from cv_bridge import CvBridge

from .mapping import Mapping as MappingCore


class MappingNode(Node):
    """ROS 2 node for mapping."""

    def __init__(self):
        super().__init__('mapping_node')

        # Initialize components
        self.cv_bridge = CvBridge()
        self.mapper = MappingCore()

        # Declare parameters
        self._declare_parameters()

        # Set up QoS
        self.sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.map_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/slam/pose',
            self.pose_callback,
            self.sensor_qos
        )

        self.depth_sub = self.create_subscription(
            Image,
            self.get_parameter('depth_topic').get_parameter_value().string_value,
            self.depth_callback,
            self.sensor_qos
        )

        # Create publishers
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/map',
            self.map_qos
        )

        self.point_cloud_pub = self.create_publisher(
            PointCloud2,
            '/mapping/point_cloud',
            self.sensor_qos
        )

        self.trajectory_pub = self.create_publisher(
            Path,
            '/mapping/trajectory',
            self.sensor_qos
        )

        # Create timer for map updates
        update_frequency = self.get_parameter('update_frequency').get_parameter_value().double_value
        self.map_timer = self.create_timer(1.0 / update_frequency, self.map_update_callback)

        # State variables
        self.current_pose = None
        self.trajectory = Path()

        self.get_logger().info('Mapping Node initialized')

    def _declare_parameters(self):
        """Declare ROS 2 parameters."""
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('map_resolution', 0.05)
        self.declare_parameter('map_size', 1000)
        self.declare_parameter('update_frequency', 10.0)
        self.declare_parameter('occupancy_threshold', 0.65)
        self.declare_parameter('free_threshold', 0.25)
        self.declare_parameter('map_frame', 'map')

    def pose_callback(self, msg: PoseStamped):
        """Update current pose."""
        self.current_pose = msg

        # Add to trajectory
        self.trajectory.header = msg.header
        self.trajectory.poses.append(msg)

        # Update mapper with new pose
        self.mapper.update(msg)

    def depth_callback(self, msg: Image):
        """Process depth images for mapping."""
        try:
            if self.current_pose is None:
                return

            # Convert depth image
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")

            # Update mapping with depth information
            self.mapper.update_with_depth(depth_image, self.current_pose)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def map_update_callback(self):
        """Publish updated map and related data."""
        try:
            # Publish occupancy grid
            occupancy_grid = self.mapper.get_occupancy_grid()
            if occupancy_grid is not None:
                occupancy_grid.header.stamp = self.get_clock().now().to_msg()
                occupancy_grid.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
                self.map_pub.publish(occupancy_grid)

            # Publish point cloud
            point_cloud = self.mapper.get_point_cloud()
            if point_cloud is not None:
                point_cloud.header.stamp = self.get_clock().now().to_msg()
                point_cloud.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
                self.point_cloud_pub.publish(point_cloud)

            # Publish trajectory
            if len(self.trajectory.poses) > 0:
                self.trajectory.header.stamp = self.get_clock().now().to_msg()
                self.trajectory.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
                self.trajectory_pub.publish(self.trajectory)

        except Exception as e:
            self.get_logger().error(f'Error updating map: {str(e)}')


def main(args=None):
    """Main function."""
    rclpy.init(args=args)

    try:
        node = MappingNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
