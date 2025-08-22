#!/usr/bin/env python3
"""
Feature Extraction Node for ROS 2
Extracts visual features from camera images
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import List, Tuple

# Import custom messages (would need to be defined)
# from python_slam_msgs.msg import FeatureArray, Feature

from .feature_extraction import FeatureExtraction as FeatureExtractionCore


class FeatureExtractionNode(Node):
    """ROS 2 node for feature extraction."""

    def __init__(self):
        super().__init__('feature_extraction_node')

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Initialize feature extraction core
        self.feature_extractor = FeatureExtractionCore()

        # Declare parameters
        self._declare_parameters()

        # Set up QoS profiles
        self.image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create subscribers
        camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            self.image_qos
        )

        # Create publishers
        self.features_pub = self.create_publisher(
            Image,  # Using Image for visualization, would use custom FeatureArray in practice
            '/features/visualization',
            self.image_qos
        )

        self.get_logger().info('Feature Extraction Node initialized')

    def _declare_parameters(self):
        """Declare ROS 2 parameters."""
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('max_features', 1000)
        self.declare_parameter('quality_level', 0.01)
        self.declare_parameter('min_distance', 10)
        self.declare_parameter('corner_threshold', 0.04)
        self.declare_parameter('publish_visualization', True)

    def image_callback(self, msg: Image):
        """Process incoming images and extract features."""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

            # Update parameters
            self._update_parameters()

            # Extract features
            features = self.feature_extractor.extract_features(cv_image)

            # Publish feature visualization
            if self.get_parameter('publish_visualization').get_parameter_value().bool_value:
                self._publish_feature_visualization(cv_image, features, msg.header)

            self.get_logger().debug(f'Extracted {len(features)} features')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def _update_parameters(self):
        """Update feature extraction parameters from ROS parameters."""
        max_features = self.get_parameter('max_features').get_parameter_value().integer_value
        quality_level = self.get_parameter('quality_level').get_parameter_value().double_value
        min_distance = self.get_parameter('min_distance').get_parameter_value().double_value
        corner_threshold = self.get_parameter('corner_threshold').get_parameter_value().double_value

        # Update core feature extractor parameters
        # This would require updating the FeatureExtraction class to accept these parameters
        pass

    def _publish_feature_visualization(self, image: np.ndarray, features: List, header: Header):
        """Publish visualization of extracted features."""
        try:
            # Create visualization image
            vis_image = image.copy()

            # Draw features
            for feature in features:
                if hasattr(feature, 'pt'):
                    center = (int(feature.pt[0]), int(feature.pt[1]))
                    cv2.circle(vis_image, center, 5, (0, 255, 0), 2)
                    # Optionally draw response strength as color intensity
                    if hasattr(feature, 'response'):
                        intensity = min(255, int(feature.response * 255))
                        cv2.circle(vis_image, center, 3, (0, intensity, 0), -1)

            # Add feature count text
            text = f'Features: {len(features)}'
            cv2.putText(vis_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert back to ROS message
            vis_msg = self.cv_bridge.cv2_to_imgmsg(vis_image, "bgr8")
            vis_msg.header = header

            self.features_pub.publish(vis_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing feature visualization: {str(e)}')


def main(args=None):
    """Main function to run the feature extraction node."""
    rclpy.init(args=args)

    try:
        node = FeatureExtractionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error running feature extraction node: {str(e)}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
