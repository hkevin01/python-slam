#!/usr/bin/env python3
"""
Main SLAM Node for ROS 2
Integrates all SLAM components into a unified ROS 2 node
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_geometry_msgs
from typing import Optional, Tuple

# Import SLAM modules
from .feature_extraction import FeatureExtraction
from .pose_estimation import PoseEstimation
from .mapping import Mapping
from .localization import Localization
from .loop_closure import LoopClosure
from .flight_integration import FlightIntegration


class SlamNode(Node):
    """Main SLAM node that coordinates all SLAM components."""

    def __init__(self):
        super().__init__('slam_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Initialize TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Declare parameters
        self._declare_parameters()

        # Initialize SLAM components
        self._initialize_slam_components()

        # Set up QoS profiles
        self._setup_qos_profiles()

        # Create subscribers
        self._create_subscribers()

        # Create publishers
        self._create_publishers()

        # Create timers
        self._create_timers()

        # State variables
        self.current_pose: Optional[PoseStamped] = None
        self.previous_frame: Optional[np.ndarray] = None
        self.frame_count: int = 0

        self.get_logger().info('SLAM Node initialized successfully')

    def _declare_parameters(self):
        """Declare ROS 2 parameters."""
        # Camera topics
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')

        # Frame IDs
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_link')

        # SLAM parameters
        self.declare_parameter('loop_closure_enabled', True)
        self.declare_parameter('mapping_enabled', True)
        self.declare_parameter('localization_enabled', True)

        # Publishing parameters
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('publish_point_cloud', True)
        self.declare_parameter('publish_trajectory', True)

        # Update frequencies
        self.declare_parameter('slam_frequency', 30.0)
        self.declare_parameter('map_update_frequency', 10.0)

    def _initialize_slam_components(self):
        """Initialize all SLAM components."""
        self.feature_extraction = FeatureExtraction()
        self.pose_estimation = PoseEstimation()
        self.mapping = Mapping()
        self.localization = Localization()
        self.loop_closure = LoopClosure()
        self.flight_integration = FlightIntegration()

        self.get_logger().info('SLAM components initialized')

    def _setup_qos_profiles(self):
        """Set up QoS profiles for different topics."""
        self.image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

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

    def _create_subscribers(self):
        """Create ROS 2 subscribers."""
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value

        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            self.image_qos
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            self.sensor_qos
        )

        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            self.sensor_qos
        )

        self.get_logger().info('Subscribers created')

    def _create_publishers(self):
        """Create ROS 2 publishers."""
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/slam/pose',
            self.sensor_qos
        )

        self.map_pub = self.create_publisher(
            OccupancyGrid,
            '/map',
            self.map_qos
        )

        self.trajectory_pub = self.create_publisher(
            Path,
            '/slam/trajectory',
            self.sensor_qos
        )

        self.point_cloud_pub = self.create_publisher(
            PointCloud2,
            '/slam/point_cloud',
            self.sensor_qos
        )

        self.features_image_pub = self.create_publisher(
            Image,
            '/slam/features_image',
            self.image_qos
        )

        self.get_logger().info('Publishers created')

    def _create_timers(self):
        """Create ROS 2 timers."""
        slam_frequency = self.get_parameter('slam_frequency').get_parameter_value().double_value
        map_frequency = self.get_parameter('map_update_frequency').get_parameter_value().double_value

        self.slam_timer = self.create_timer(
            1.0 / slam_frequency,
            self.slam_update_callback
        )

        self.map_timer = self.create_timer(
            1.0 / map_frequency,
            self.map_update_callback
        )

        self.get_logger().info('Timers created')

    def image_callback(self, msg: Image):
        """Process incoming camera images."""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

            # Extract features
            features = self.feature_extraction.extract_features(cv_image)

            # Estimate pose if we have a previous frame
            if self.previous_frame is not None:
                pose_change = self.pose_estimation.estimate_pose(
                    self.previous_frame, cv_image
                )

                # Update current pose
                if self.current_pose is not None and pose_change is not None:
                    self.current_pose = self._update_pose(self.current_pose, pose_change)

                    # Publish pose
                    if self.get_parameter('publish_tf').get_parameter_value().bool_value:
                        self._publish_transform()

                    self.pose_pub.publish(self.current_pose)

            # Store current frame for next iteration
            self.previous_frame = cv_image.copy()
            self.frame_count += 1

            # Publish features visualization
            features_image = self._visualize_features(cv_image, features)
            features_msg = self.cv_bridge.cv2_to_imgmsg(features_image, "bgr8")
            features_msg.header = msg.header
            self.features_image_pub.publish(features_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def camera_info_callback(self, msg: CameraInfo):
        """Process camera info messages."""
        # Update camera parameters in pose estimation
        self.pose_estimation.update_camera_info(msg)

    def depth_callback(self, msg: Image):
        """Process depth images."""
        try:
            # Convert depth image
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")

            # Update mapping with depth information
            if self.current_pose is not None:
                self.mapping.update_with_depth(depth_image, self.current_pose)

        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')

    def slam_update_callback(self):
        """Main SLAM update loop."""
        if self.current_pose is None:
            return

        try:
            # Check for loop closures
            if self.get_parameter('loop_closure_enabled').get_parameter_value().bool_value:
                loop_detected = self.loop_closure.detect_loop(self.current_pose)
                if loop_detected:
                    self.get_logger().info('Loop closure detected!')
                    # Perform loop closure optimization
                    self.loop_closure.close_loop()

            # Update localization
            if self.get_parameter('localization_enabled').get_parameter_value().bool_value:
                self.localization.update(self.current_pose)

            # Flight integration updates
            flight_command = self.flight_integration.update(self.current_pose)
            if flight_command is not None:
                # Publish flight commands (implement based on drone interface)
                pass

        except Exception as e:
            self.get_logger().error(f'Error in SLAM update: {str(e)}')

    def map_update_callback(self):
        """Update and publish map."""
        if not self.get_parameter('mapping_enabled').get_parameter_value().bool_value:
            return

        try:
            # Update mapping
            if self.current_pose is not None:
                self.mapping.update(self.current_pose)

            # Get current map
            occupancy_grid = self.mapping.get_occupancy_grid()
            if occupancy_grid is not None:
                self.map_pub.publish(occupancy_grid)

            # Publish point cloud if enabled
            if self.get_parameter('publish_point_cloud').get_parameter_value().bool_value:
                point_cloud = self.mapping.get_point_cloud()
                if point_cloud is not None:
                    self.point_cloud_pub.publish(point_cloud)

            # Publish trajectory if enabled
            if self.get_parameter('publish_trajectory').get_parameter_value().bool_value:
                trajectory = self.mapping.get_trajectory()
                if trajectory is not None:
                    self.trajectory_pub.publish(trajectory)

        except Exception as e:
            self.get_logger().error(f'Error updating map: {str(e)}')

    def _update_pose(self, current_pose: PoseStamped, pose_change: np.ndarray) -> PoseStamped:
        """Update pose with relative change."""
        # Implementation depends on pose representation
        # This is a simplified version
        new_pose = PoseStamped()
        new_pose.header.stamp = self.get_clock().now().to_msg()
        new_pose.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value

        # Apply pose change (implementation details depend on pose representation)
        # For now, just update timestamp
        new_pose.pose = current_pose.pose

        return new_pose

    def _publish_transform(self):
        """Publish TF transform."""
        if self.current_pose is None:
            return

        try:
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = self.get_parameter('map_frame').get_parameter_value().string_value
            transform.child_frame_id = self.get_parameter('base_frame').get_parameter_value().string_value

            # Copy pose to transform
            transform.transform.translation.x = self.current_pose.pose.position.x
            transform.transform.translation.y = self.current_pose.pose.position.y
            transform.transform.translation.z = self.current_pose.pose.position.z
            transform.transform.rotation = self.current_pose.pose.orientation

            self.tf_broadcaster.sendTransform(transform)

        except Exception as e:
            self.get_logger().error(f'Error publishing transform: {str(e)}')

    def _visualize_features(self, image: np.ndarray, features: list) -> np.ndarray:
        """Visualize extracted features on image."""
        vis_image = image.copy()

        for feature in features:
            # Draw feature points (assuming features have x, y coordinates)
            if hasattr(feature, 'pt'):
                center = (int(feature.pt[0]), int(feature.pt[1]))
                cv2.circle(vis_image, center, 3, (0, 255, 0), -1)

        return vis_image


def main(args=None):
    """Main function to run the SLAM node."""
    rclpy.init(args=args)

    try:
        slam_node = SlamNode()
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error running SLAM node: {str(e)}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
