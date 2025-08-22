#!/usr/bin/env python3
"""
ROS2 Visualization Bridge Node

This node subscribes to SLAM data from the backend and provides
a clean interface for the PyQt5 visualization GUI to consume.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, Image
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, OccupancyGrid
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import Header
import numpy as np
import threading
import time
from collections import deque
import zmq


class VisualizationBridge(Node):
    """Bridge node that receives SLAM data and forwards it to visualization."""

    def __init__(self):
        super().__init__('visualization_bridge')

        # Initialize ZMQ for communication with PyQt5 GUI
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5555")

        # Set up QoS profiles
        self.setup_qos_profiles()

        # Create subscribers for SLAM data
        self.create_subscribers()

        # Data buffers
        self.latest_pose = None
        self.latest_pointcloud = None
        self.latest_map = None
        self.trajectory = deque(maxlen=1000)

        # Thread safety
        self.data_lock = threading.Lock()

        # Publisher for processed data
        self.timer = self.create_timer(0.1, self.publish_visualization_data)

        self.get_logger().info('Visualization Bridge Node initialized')

    def setup_qos_profiles(self):
        """Set up QoS profiles for different data types."""
        self.sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

    def create_subscribers(self):
        """Create ROS2 subscribers for SLAM data."""
        # Subscribe to pose estimates
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/slam/pose',
            self.pose_callback,
            self.sensor_qos
        )

        # Subscribe to point cloud
        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            '/slam/point_cloud',
            self.pointcloud_callback,
            self.sensor_qos
        )

        # Subscribe to trajectory
        self.trajectory_sub = self.create_subscription(
            Path,
            '/slam/trajectory',
            self.trajectory_callback,
            self.sensor_qos
        )

        # Subscribe to occupancy grid map
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            self.map_qos
        )

        # Subscribe to features visualization
        self.features_sub = self.create_subscription(
            Image,
            '/slam/features_image',
            self.features_callback,
            self.sensor_qos
        )

    def pose_callback(self, msg):
        """Handle pose updates."""
        with self.data_lock:
            self.latest_pose = msg

            # Add to trajectory
            pose_point = [
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ]
            self.trajectory.append(pose_point)

    def pointcloud_callback(self, msg):
        """Handle point cloud updates."""
        with self.data_lock:
            self.latest_pointcloud = msg

    def trajectory_callback(self, msg):
        """Handle trajectory updates."""
        with self.data_lock:
            trajectory_points = []
            for pose in msg.poses:
                trajectory_points.append([
                    pose.pose.position.x,
                    pose.pose.position.y,
                    pose.pose.position.z
                ])
            self.trajectory = deque(trajectory_points, maxlen=1000)

    def map_callback(self, msg):
        """Handle map updates."""
        with self.data_lock:
            self.latest_map = msg

    def features_callback(self, msg):
        """Handle feature visualization updates."""
        # Convert ROS image to format suitable for visualization
        pass

    def publish_visualization_data(self):
        """Publish visualization data via ZMQ."""
        with self.data_lock:
            viz_data = {
                'timestamp': time.time(),
                'pose': self.serialize_pose(self.latest_pose) if self.latest_pose else None,
                'pointcloud': self.serialize_pointcloud(self.latest_pointcloud) if self.latest_pointcloud else None,
                'trajectory': list(self.trajectory) if self.trajectory else [],
                'map': self.serialize_map(self.latest_map) if self.latest_map else None,
            }

            try:
                self.socket.send_json(viz_data, zmq.NOBLOCK)
            except zmq.Again:
                # Socket would block, skip this update
                pass

    def serialize_pose(self, pose_msg):
        """Convert PoseStamped to serializable format."""
        if pose_msg is None:
            return None

        return {
            'position': {
                'x': pose_msg.pose.position.x,
                'y': pose_msg.pose.position.y,
                'z': pose_msg.pose.position.z
            },
            'orientation': {
                'x': pose_msg.pose.orientation.x,
                'y': pose_msg.pose.orientation.y,
                'z': pose_msg.pose.orientation.z,
                'w': pose_msg.pose.orientation.w
            },
            'timestamp': pose_msg.header.stamp.sec + pose_msg.header.stamp.nanosec * 1e-9
        }

    def serialize_pointcloud(self, pc_msg):
        """Convert PointCloud2 to serializable format."""
        if pc_msg is None:
            return None

        # Convert point cloud data to numpy array
        # This is a simplified version - full implementation would parse the PointCloud2 format
        return {
            'points': [],  # Would contain actual point data
            'timestamp': pc_msg.header.stamp.sec + pc_msg.header.stamp.nanosec * 1e-9,
            'frame_id': pc_msg.header.frame_id
        }

    def serialize_map(self, map_msg):
        """Convert OccupancyGrid to serializable format."""
        if map_msg is None:
            return None

        return {
            'width': map_msg.info.width,
            'height': map_msg.info.height,
            'resolution': map_msg.info.resolution,
            'origin': {
                'x': map_msg.info.origin.position.x,
                'y': map_msg.info.origin.position.y,
                'theta': map_msg.info.origin.orientation.z  # Simplified
            },
            'data': list(map_msg.data),
            'timestamp': map_msg.header.stamp.sec + map_msg.header.stamp.nanosec * 1e-9
        }

    def destroy_node(self):
        """Clean up ZMQ resources."""
        self.socket.close()
        self.context.term()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        bridge_node = VisualizationBridge()
        rclpy.spin(bridge_node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'bridge_node' in locals():
            bridge_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
