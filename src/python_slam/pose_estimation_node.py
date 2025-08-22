#!/usr/bin/env python3
"""
Pose Estimation Node for ROS 2
Estimates camera pose from visual features
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from cv_bridge import CvBridge
import numpy as np

from .pose_estimation import PoseEstimation as PoseEstimationCore


class PoseEstimationNode(Node):
    """ROS 2 node for pose estimation."""

    def __init__(self):
        super().__init__('pose_estimation_node')

        # Initialize components
        self.cv_bridge = CvBridge()
        self.pose_estimator = PoseEstimationCore()
        self.tf_broadcaster = TransformBroadcaster(self)

        # Declare parameters
        self._declare_parameters()

        # Set up QoS
        self.sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # State variables
        self.previous_image = None
        self.camera_info = None

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter('camera_topic').get_parameter_value().string_value,
            self.image_callback,
            self.sensor_qos
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').get_parameter_value().string_value,
            self.camera_info_callback,
            self.sensor_qos
        )

        # Create publishers
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/pose_estimation/pose',
            self.sensor_qos
        )

        self.get_logger().info('Pose Estimation Node initialized')

    def _declare_parameters(self):
        """Declare ROS 2 parameters."""
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('ransac_threshold', 1.0)
        self.declare_parameter('confidence', 0.99)
        self.declare_parameter('max_iterations', 1000)
        self.declare_parameter('min_inliers', 50)
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_link')

    def camera_info_callback(self, msg: CameraInfo):
        """Update camera information."""
        self.camera_info = msg
        self.pose_estimator.update_camera_info(msg)

    def image_callback(self, msg: Image):
        """Process images for pose estimation."""
        try:
            # Convert to OpenCV format
            current_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

            # Estimate pose if we have previous image
            if self.previous_image is not None and self.camera_info is not None:
                pose_change = self.pose_estimator.estimate_pose(
                    self.previous_image, current_image
                )

                if pose_change is not None:
                    # Create pose message
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = msg.header.stamp
                    pose_msg.header.frame_id = self.get_parameter('camera_frame').get_parameter_value().string_value

                    # Convert pose_change to pose message (implementation depends on format)
                    # This is a simplified version
                    pose_msg.pose.position.x = 0.0
                    pose_msg.pose.position.y = 0.0
                    pose_msg.pose.position.z = 0.0
                    pose_msg.pose.orientation.w = 1.0

                    self.pose_pub.publish(pose_msg)

                    # Publish TF if enabled
                    if self.get_parameter('publish_tf').get_parameter_value().bool_value:
                        self._publish_transform(pose_msg)

            # Store current image for next iteration
            self.previous_image = current_image.copy()

        except Exception as e:
            self.get_logger().error(f'Error in pose estimation: {str(e)}')

    def _publish_transform(self, pose_msg: PoseStamped):
        """Publish TF transform."""
        try:
            transform = TransformStamped()
            transform.header.stamp = pose_msg.header.stamp
            transform.header.frame_id = self.get_parameter('base_frame').get_parameter_value().string_value
            transform.child_frame_id = pose_msg.header.frame_id

            transform.transform.translation.x = pose_msg.pose.position.x
            transform.transform.translation.y = pose_msg.pose.position.y
            transform.transform.translation.z = pose_msg.pose.position.z
            transform.transform.rotation = pose_msg.pose.orientation

            self.tf_broadcaster.sendTransform(transform)

        except Exception as e:
            self.get_logger().error(f'Error publishing transform: {str(e)}')


def main(args=None):
    """Main function."""
    rclpy.init(args=args)

    try:
        node = PoseEstimationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
