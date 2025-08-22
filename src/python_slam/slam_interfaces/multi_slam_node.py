#!/usr/bin/env python3
"""
Multi-Algorithm SLAM ROS2 Node

ROS2 node that provides a unified interface for multiple SLAM algorithms
with runtime algorithm switching and comprehensive sensor support.
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import numpy as np
import cv2
import yaml
import time
import threading
from typing import Optional, Dict, Any

# ROS2 message types
from sensor_msgs.msg import Image, PointCloud2, Imu, LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped, TransformStamped, PoseWithCovarianceStamped, Point
from std_msgs.msg import String, Header
from visualization_msgs.msg import Marker, MarkerArray

# ROS2 services
from std_srvs.srv import Empty, Trigger
from nav_msgs.srv import GetMap

# TF2
import tf2_ros
import tf2_geometry_msgs
from tf_transformations import quaternion_from_euler, euler_from_quaternion

# CV Bridge
from cv_bridge import CvBridge

# Custom messages and services (would need to be defined)
try:
    from python_slam_msgs.msg import SLAMStatus, PerformanceMetrics
    from python_slam_msgs.srv import SwitchAlgorithm, SaveMap, LoadMap
    CUSTOM_MSGS_AVAILABLE = True
except ImportError:
    CUSTOM_MSGS_AVAILABLE = False

# SLAM interfaces
from python_slam.slam_interfaces import (
    SLAMFactory, SLAMConfiguration, SensorType, SLAMState,
    get_available_algorithms, get_recommended_algorithm
)


class MultiSLAMNode(Node):
    """
    ROS2 node for multi-algorithm SLAM with runtime switching.

    Features:
    - Support for multiple SLAM algorithms
    - Runtime algorithm switching
    - Automatic sensor type detection
    - Performance monitoring
    - Map saving/loading
    - Relocalization support
    """

    def __init__(self):
        super().__init__('multi_slam_node')

        # Initialize components
        self.cv_bridge = CvBridge()
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # SLAM system
        self.slam_factory = SLAMFactory()
        self.slam_system = None
        self.current_algorithm = None

        # Node configuration
        self.declare_parameters()
        self.load_configuration()

        # Synchronization
        self.slam_lock = threading.RLock()
        self.callback_group = ReentrantCallbackGroup()

        # State tracking
        self.last_pose_time = None
        self.pose_timeout = 1.0  # seconds
        self.frame_count = 0
        self.start_time = time.time()

        # Initialize SLAM system
        self.initialize_slam()

        # Setup ROS2 interfaces
        self.setup_publishers()
        self.setup_subscribers()
        self.setup_services()
        self.setup_timers()

        self.get_logger().info(f"Multi-SLAM node initialized with {self.current_algorithm}")

    def declare_parameters(self):
        """Declare ROS2 parameters."""
        # Algorithm selection
        self.declare_parameter('algorithm', 'auto')
        self.declare_parameter('sensor_type', 'auto')

        # Frame names
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_link')

        # Topic names
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('left_image_topic', '/camera/left/image_raw')
        self.declare_parameter('right_image_topic', '/camera/right/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('pointcloud_topic', '/velodyne_points')
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('laser_topic', '/scan')

        # SLAM parameters
        self.declare_parameter('max_features', 1000)
        self.declare_parameter('enable_loop_closure', True)
        self.declare_parameter('enable_mapping', True)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('publish_pose', True)
        self.declare_parameter('publish_map', True)

        # Performance parameters
        self.declare_parameter('pose_publish_rate', 30.0)
        self.declare_parameter('map_publish_rate', 1.0)
        self.declare_parameter('status_publish_rate', 1.0)

        # Camera parameters
        self.declare_parameter('camera_fx', 525.0)
        self.declare_parameter('camera_fy', 525.0)
        self.declare_parameter('camera_cx', 319.5)
        self.declare_parameter('camera_cy', 239.5)
        self.declare_parameter('camera_baseline', 0.1)

    def load_configuration(self):
        """Load configuration from parameters."""
        # Get algorithm and sensor type
        algorithm = self.get_parameter('algorithm').value
        sensor_type_str = self.get_parameter('sensor_type').value

        # Auto-detect sensor type if needed
        if sensor_type_str == 'auto':
            sensor_type = self.detect_sensor_type()
        else:
            sensor_type = SensorType[sensor_type_str.upper()]

        # Auto-select algorithm if needed
        if algorithm == 'auto':
            algorithm = get_recommended_algorithm(sensor_type)
            if algorithm is None:
                available = get_available_algorithms()
                algorithm = available[0] if available else 'python_slam'

        # Camera parameters
        camera_params = {
            'fx': self.get_parameter('camera_fx').value,
            'fy': self.get_parameter('camera_fy').value,
            'cx': self.get_parameter('camera_cx').value,
            'cy': self.get_parameter('camera_cy').value,
            'baseline': self.get_parameter('camera_baseline').value
        }

        # Create SLAM configuration
        self.slam_config = SLAMConfiguration(
            algorithm_name=algorithm,
            sensor_type=sensor_type,
            max_features=self.get_parameter('max_features').value,
            enable_loop_closure=self.get_parameter('enable_loop_closure').value,
            enable_mapping=self.get_parameter('enable_mapping').value,
            custom_params={'camera': camera_params}
        )

        self.get_logger().info(f"Configuration: {algorithm} with {sensor_type.value} sensor")

    def detect_sensor_type(self):
        """Auto-detect sensor type based on available topics."""
        # This is a simplified detection - real implementation would check topic availability
        # For now, default to monocular
        return SensorType.MONOCULAR

    def initialize_slam(self):
        """Initialize the SLAM system."""
        try:
            with self.slam_lock:
                self.slam_system = self.slam_factory.create_algorithm(self.slam_config)

                if self.slam_system.initialize():
                    self.current_algorithm = self.slam_config.algorithm_name
                    self.get_logger().info(f"SLAM system initialized: {self.current_algorithm}")
                else:
                    self.get_logger().error("Failed to initialize SLAM system")

        except Exception as e:
            self.get_logger().error(f"Error initializing SLAM: {e}")

    def setup_publishers(self):
        """Setup ROS2 publishers."""
        # Pose publishers
        self.pose_pub = self.create_publisher(
            PoseStamped, '/slam/pose', 10, callback_group=self.callback_group
        )
        self.odom_pub = self.create_publisher(
            Odometry, '/slam/odometry', 10, callback_group=self.callback_group
        )

        # Map publishers
        self.map_pub = self.create_publisher(
            OccupancyGrid, '/slam/occupancy_grid', 1, callback_group=self.callback_group
        )
        self.map_points_pub = self.create_publisher(
            MarkerArray, '/slam/map_points', 1, callback_group=self.callback_group
        )

        # Status publishers
        self.status_pub = self.create_publisher(
            String, '/slam/status', 1, callback_group=self.callback_group
        )

        if CUSTOM_MSGS_AVAILABLE:
            self.metrics_pub = self.create_publisher(
                PerformanceMetrics, '/slam/metrics', 1, callback_group=self.callback_group
            )

    def setup_subscribers(self):
        """Setup ROS2 subscribers."""
        # Image subscribers
        if self.slam_config.sensor_type == SensorType.MONOCULAR:
            self.image_sub = self.create_subscription(
                Image, self.get_parameter('image_topic').value,
                self.image_callback, 10, callback_group=self.callback_group
            )
        elif self.slam_config.sensor_type == SensorType.STEREO:
            self.left_image_sub = self.create_subscription(
                Image, self.get_parameter('left_image_topic').value,
                self.left_image_callback, 10, callback_group=self.callback_group
            )
            self.right_image_sub = self.create_subscription(
                Image, self.get_parameter('right_image_topic').value,
                self.right_image_callback, 10, callback_group=self.callback_group
            )
        elif self.slam_config.sensor_type == SensorType.RGBD:
            self.rgb_image_sub = self.create_subscription(
                Image, self.get_parameter('image_topic').value,
                self.rgb_image_callback, 10, callback_group=self.callback_group
            )
            self.depth_image_sub = self.create_subscription(
                Image, self.get_parameter('depth_topic').value,
                self.depth_image_callback, 10, callback_group=self.callback_group
            )

        # Other sensors
        self.imu_sub = self.create_subscription(
            Imu, self.get_parameter('imu_topic').value,
            self.imu_callback, 10, callback_group=self.callback_group
        )

        if self.slam_config.sensor_type in [SensorType.LIDAR, SensorType.POINTCLOUD]:
            self.laser_sub = self.create_subscription(
                LaserScan, self.get_parameter('laser_topic').value,
                self.laser_callback, 10, callback_group=self.callback_group
            )
            self.pointcloud_sub = self.create_subscription(
                PointCloud2, self.get_parameter('pointcloud_topic').value,
                self.pointcloud_callback, 10, callback_group=self.callback_group
            )

        # For stereo and RGB-D, we need message synchronization
        self.left_image = None
        self.right_image = None
        self.rgb_image = None
        self.depth_image = None
        self.last_stereo_time = None
        self.last_rgbd_time = None

    def setup_services(self):
        """Setup ROS2 services."""
        # Standard services
        self.reset_srv = self.create_service(
            Trigger, '/slam/reset', self.reset_callback, callback_group=self.callback_group
        )
        self.get_map_srv = self.create_service(
            GetMap, '/slam/get_map', self.get_map_callback, callback_group=self.callback_group
        )

        # Custom services (if available)
        if CUSTOM_MSGS_AVAILABLE:
            self.switch_algorithm_srv = self.create_service(
                SwitchAlgorithm, '/slam/switch_algorithm',
                self.switch_algorithm_callback, callback_group=self.callback_group
            )
            self.save_map_srv = self.create_service(
                SaveMap, '/slam/save_map',
                self.save_map_callback, callback_group=self.callback_group
            )
            self.load_map_srv = self.create_service(
                LoadMap, '/slam/load_map',
                self.load_map_callback, callback_group=self.callback_group
            )

    def setup_timers(self):
        """Setup periodic timers."""
        # Pose publishing timer
        if self.get_parameter('publish_pose').value:
            pose_rate = self.get_parameter('pose_publish_rate').value
            self.pose_timer = self.create_timer(
                1.0 / pose_rate, self.publish_pose_callback, callback_group=self.callback_group
            )

        # Map publishing timer
        if self.get_parameter('publish_map').value:
            map_rate = self.get_parameter('map_publish_rate').value
            self.map_timer = self.create_timer(
                1.0 / map_rate, self.publish_map_callback, callback_group=self.callback_group
            )

        # Status publishing timer
        status_rate = self.get_parameter('status_publish_rate').value
        self.status_timer = self.create_timer(
            1.0 / status_rate, self.publish_status_callback, callback_group=self.callback_group
        )

    # Sensor callbacks
    def image_callback(self, msg):
        """Process monocular image."""
        if self.slam_system is None:
            return

        try:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            with self.slam_lock:
                success = self.slam_system.process_image(msg, timestamp)

            if success:
                self.frame_count += 1
                self.last_pose_time = time.time()

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def left_image_callback(self, msg):
        """Store left stereo image."""
        self.left_image = msg
        self._process_stereo_if_ready()

    def right_image_callback(self, msg):
        """Store right stereo image."""
        self.right_image = msg
        self._process_stereo_if_ready()

    def _process_stereo_if_ready(self):
        """Process stereo images when both are available."""
        if self.left_image is None or self.right_image is None or self.slam_system is None:
            return

        # Check if timestamps are close enough
        left_time = self.left_image.header.stamp.sec + self.left_image.header.stamp.nanosec * 1e-9
        right_time = self.right_image.header.stamp.sec + self.right_image.header.stamp.nanosec * 1e-9

        if abs(left_time - right_time) < 0.1:  # 100ms tolerance
            try:
                timestamp = (left_time + right_time) / 2

                with self.slam_lock:
                    success = self.slam_system.process_stereo_images(
                        self.left_image, self.right_image, timestamp
                    )

                if success:
                    self.frame_count += 1
                    self.last_pose_time = time.time()

                # Clear images
                self.left_image = None
                self.right_image = None

            except Exception as e:
                self.get_logger().error(f"Error processing stereo images: {e}")

    def rgb_image_callback(self, msg):
        """Store RGB image for RGB-D."""
        self.rgb_image = msg
        self._process_rgbd_if_ready()

    def depth_image_callback(self, msg):
        """Store depth image for RGB-D."""
        self.depth_image = msg
        self._process_rgbd_if_ready()

    def _process_rgbd_if_ready(self):
        """Process RGB-D images when both are available."""
        if self.rgb_image is None or self.depth_image is None or self.slam_system is None:
            return

        # Check if timestamps are close enough
        rgb_time = self.rgb_image.header.stamp.sec + self.rgb_image.header.stamp.nanosec * 1e-9
        depth_time = self.depth_image.header.stamp.sec + self.depth_image.header.stamp.nanosec * 1e-9

        if abs(rgb_time - depth_time) < 0.1:  # 100ms tolerance
            try:
                timestamp = (rgb_time + depth_time) / 2

                with self.slam_lock:
                    # RGB-D processing
                    if hasattr(self.slam_system, 'process_rgbd_images'):
                        success = self.slam_system.process_rgbd_images(
                            self.rgb_image, self.depth_image, timestamp
                        )
                    else:
                        # Fallback to separate processing
                        success = self.slam_system.process_image(self.rgb_image, timestamp)

                if success:
                    self.frame_count += 1
                    self.last_pose_time = time.time()

                # Clear images
                self.rgb_image = None
                self.depth_image = None

            except Exception as e:
                self.get_logger().error(f"Error processing RGB-D images: {e}")

    def imu_callback(self, msg):
        """Process IMU data."""
        if self.slam_system is None:
            return

        try:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            with self.slam_lock:
                self.slam_system.process_imu(msg, timestamp)

        except Exception as e:
            self.get_logger().error(f"Error processing IMU: {e}")

    def laser_callback(self, msg):
        """Process laser scan data."""
        if self.slam_system is None:
            return

        try:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            with self.slam_lock:
                if hasattr(self.slam_system, 'process_laser_scan'):
                    success = self.slam_system.process_laser_scan(msg, timestamp)
                else:
                    success = False

            if success:
                self.frame_count += 1
                self.last_pose_time = time.time()

        except Exception as e:
            self.get_logger().error(f"Error processing laser scan: {e}")

    def pointcloud_callback(self, msg):
        """Process point cloud data."""
        if self.slam_system is None:
            return

        try:
            timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            with self.slam_lock:
                success = self.slam_system.process_pointcloud(msg, timestamp)

            if success:
                self.frame_count += 1
                self.last_pose_time = time.time()

        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")

    # Publishing callbacks
    def publish_pose_callback(self):
        """Publish current pose estimate."""
        if self.slam_system is None:
            return

        try:
            with self.slam_lock:
                slam_pose = self.slam_system.get_pose()

            if slam_pose is not None:
                current_time = self.get_clock().now()

                # Publish PoseStamped
                pose_msg = PoseStamped()
                pose_msg.header.stamp = current_time.to_msg()
                pose_msg.header.frame_id = self.get_parameter('map_frame').value

                pose_msg.pose.position.x = float(slam_pose.position[0])
                pose_msg.pose.position.y = float(slam_pose.position[1])
                pose_msg.pose.position.z = float(slam_pose.position[2])

                pose_msg.pose.orientation.x = float(slam_pose.orientation[0])
                pose_msg.pose.orientation.y = float(slam_pose.orientation[1])
                pose_msg.pose.orientation.z = float(slam_pose.orientation[2])
                pose_msg.pose.orientation.w = float(slam_pose.orientation[3])

                self.pose_pub.publish(pose_msg)

                # Publish Odometry
                odom_msg = Odometry()
                odom_msg.header = pose_msg.header
                odom_msg.child_frame_id = self.get_parameter('base_frame').value
                odom_msg.pose.pose = pose_msg.pose

                self.odom_pub.publish(odom_msg)

                # Publish TF
                if self.get_parameter('publish_tf').value:
                    self.publish_transform(slam_pose, current_time)

        except Exception as e:
            self.get_logger().error(f"Error publishing pose: {e}")

    def publish_transform(self, slam_pose, timestamp):
        """Publish TF transform."""
        try:
            transform = TransformStamped()
            transform.header.stamp = timestamp.to_msg()
            transform.header.frame_id = self.get_parameter('map_frame').value
            transform.child_frame_id = self.get_parameter('odom_frame').value

            transform.transform.translation.x = float(slam_pose.position[0])
            transform.transform.translation.y = float(slam_pose.position[1])
            transform.transform.translation.z = float(slam_pose.position[2])

            transform.transform.rotation.x = float(slam_pose.orientation[0])
            transform.transform.rotation.y = float(slam_pose.orientation[1])
            transform.transform.rotation.z = float(slam_pose.orientation[2])
            transform.transform.rotation.w = float(slam_pose.orientation[3])

            self.tf_broadcaster.sendTransform(transform)

        except Exception as e:
            self.get_logger().error(f"Error publishing transform: {e}")

    def publish_map_callback(self):
        """Publish map data."""
        if self.slam_system is None:
            return

        try:
            # Publish occupancy grid
            with self.slam_lock:
                occupancy_grid = self.slam_system.get_occupancy_grid()

            if occupancy_grid is not None:
                occupancy_grid.header.stamp = self.get_clock().now().to_msg()
                self.map_pub.publish(occupancy_grid)

            # Publish map points as markers
            with self.slam_lock:
                map_points = self.slam_system.get_map()

            if map_points:
                self.publish_map_points(map_points)

        except Exception as e:
            self.get_logger().error(f"Error publishing map: {e}")

    def publish_map_points(self, map_points):
        """Publish map points as visualization markers."""
        try:
            marker_array = MarkerArray()

            # Clear previous markers
            clear_marker = Marker()
            clear_marker.action = Marker.DELETEALL
            marker_array.markers.append(clear_marker)

            # Create points marker
            if len(map_points) > 0:
                points_marker = Marker()
                points_marker.header.stamp = self.get_clock().now().to_msg()
                points_marker.header.frame_id = self.get_parameter('map_frame').value
                points_marker.ns = "map_points"
                points_marker.id = 0
                points_marker.type = Marker.POINTS
                points_marker.action = Marker.ADD

                points_marker.scale.x = 0.02
                points_marker.scale.y = 0.02
                points_marker.color.r = 0.0
                points_marker.color.g = 1.0
                points_marker.color.b = 0.0
                points_marker.color.a = 0.8

                for point in map_points:
                    p = Point()
                    p.x = float(point.position[0])
                    p.y = float(point.position[1])
                    p.z = float(point.position[2])
                    points_marker.points.append(p)

                marker_array.markers.append(points_marker)

            self.map_points_pub.publish(marker_array)

        except Exception as e:
            self.get_logger().error(f"Error publishing map points: {e}")

    def publish_status_callback(self):
        """Publish system status and metrics."""
        if self.slam_system is None:
            return

        try:
            # Basic status
            status_msg = String()
            with self.slam_lock:
                state = self.slam_system.state
                is_tracking = self.slam_system.get_pose() is not None

            status_msg.data = f"Algorithm: {self.current_algorithm}, State: {state.value}, Tracking: {is_tracking}"
            self.status_pub.publish(status_msg)

            # Performance metrics (if custom messages available)
            if CUSTOM_MSGS_AVAILABLE:
                with self.slam_lock:
                    metrics = self.slam_system.get_performance_metrics()

                metrics_msg = PerformanceMetrics()
                metrics_msg.algorithm = metrics.get('algorithm', '')
                metrics_msg.frames_processed = metrics.get('frames_processed', 0)
                metrics_msg.average_fps = metrics.get('average_fps', 0.0)
                metrics_msg.avg_processing_time_ms = metrics.get('avg_processing_time_ms', 0.0)

                self.metrics_pub.publish(metrics_msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing status: {e}")

    # Service callbacks
    def reset_callback(self, request, response):
        """Reset SLAM system."""
        try:
            with self.slam_lock:
                if self.slam_system:
                    success = self.slam_system.reset()
                else:
                    success = False

            response.success = success
            response.message = "SLAM system reset" if success else "Failed to reset SLAM system"

            if success:
                self.frame_count = 0
                self.start_time = time.time()
                self.get_logger().info("SLAM system reset successfully")

            return response

        except Exception as e:
            self.get_logger().error(f"Error resetting system: {e}")
            response.success = False
            response.message = str(e)
            return response

    def get_map_callback(self, request, response):
        """Get current map."""
        try:
            with self.slam_lock:
                if self.slam_system:
                    occupancy_grid = self.slam_system.get_occupancy_grid()
                else:
                    occupancy_grid = None

            if occupancy_grid is not None:
                response.map = occupancy_grid
                return response
            else:
                self.get_logger().warning("No map available")
                return response

        except Exception as e:
            self.get_logger().error(f"Error getting map: {e}")
            return response

    def switch_algorithm_callback(self, request, response):
        """Switch SLAM algorithm."""
        if not CUSTOM_MSGS_AVAILABLE:
            return response

        try:
            new_algorithm = request.algorithm_name

            # Check if algorithm is available
            available = get_available_algorithms()
            if new_algorithm not in available:
                response.success = False
                response.message = f"Algorithm {new_algorithm} not available"
                return response

            with self.slam_lock:
                # Save current state if needed
                current_pose = None
                if self.slam_system:
                    current_pose = self.slam_system.get_pose()

                # Create new configuration
                new_config = SLAMConfiguration(
                    algorithm_name=new_algorithm,
                    sensor_type=self.slam_config.sensor_type,
                    max_features=self.slam_config.max_features,
                    enable_loop_closure=self.slam_config.enable_loop_closure,
                    enable_mapping=self.slam_config.enable_mapping,
                    custom_params=self.slam_config.custom_params
                )

                # Switch algorithm
                success = self.slam_factory.switch_algorithm(new_config)

                if success:
                    self.slam_system = self.slam_factory.get_current_algorithm()
                    self.current_algorithm = new_algorithm
                    self.slam_config = new_config

                    # Restore pose if possible
                    if current_pose and hasattr(self.slam_system, 'relocalize'):
                        self.slam_system.relocalize(current_pose)

                    response.success = True
                    response.message = f"Switched to {new_algorithm}"
                    self.get_logger().info(f"Switched to algorithm: {new_algorithm}")
                else:
                    response.success = False
                    response.message = f"Failed to switch to {new_algorithm}"

            return response

        except Exception as e:
            self.get_logger().error(f"Error switching algorithm: {e}")
            response.success = False
            response.message = str(e)
            return response

    def save_map_callback(self, request, response):
        """Save current map."""
        if not CUSTOM_MSGS_AVAILABLE:
            return response

        try:
            filepath = request.filepath

            with self.slam_lock:
                if self.slam_system:
                    success = self.slam_system.save_map(filepath)
                else:
                    success = False

            response.success = success
            response.message = f"Map saved to {filepath}" if success else "Failed to save map"

            return response

        except Exception as e:
            self.get_logger().error(f"Error saving map: {e}")
            response.success = False
            response.message = str(e)
            return response

    def load_map_callback(self, request, response):
        """Load map from file."""
        if not CUSTOM_MSGS_AVAILABLE:
            return response

        try:
            filepath = request.filepath

            with self.slam_lock:
                if self.slam_system:
                    success = self.slam_system.load_map(filepath)
                else:
                    success = False

            response.success = success
            response.message = f"Map loaded from {filepath}" if success else "Failed to load map"

            return response

        except Exception as e:
            self.get_logger().error(f"Error loading map: {e}")
            response.success = False
            response.message = str(e)
            return response


def main(args=None):
    """Main function."""
    rclpy.init(args=args)

    try:
        node = MultiSLAMNode()

        # Use multi-threaded executor for parallel processing
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)

        node.get_logger().info("Multi-SLAM node started")
        executor.spin()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
