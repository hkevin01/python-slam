#!/usr/bin/env python3
"""
Loop Closure Node for ROS 2
Detects and handles loop closures in SLAM
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from .loop_closure import LoopClosure as LoopClosureCore


class LoopClosureNode(Node):
    """ROS 2 node for loop closure detection."""

    def __init__(self):
        super().__init__('loop_closure_node')

        # Initialize components
        self.cv_bridge = CvBridge()
        self.loop_detector = LoopClosureCore()

        # Declare parameters
        self._declare_parameters()

        # Set up QoS
        self.sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
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

        self.image_sub = self.create_subscription(
            Image,
            self.get_parameter('camera_topic').get_parameter_value().string_value,
            self.image_callback,
            self.sensor_qos
        )

        # Create publishers
        self.loop_detected_pub = self.create_publisher(
            Bool,
            '/loop_closure/detected',
            self.sensor_qos
        )

        self.corrected_pose_pub = self.create_publisher(
            PoseStamped,
            '/loop_closure/corrected_pose',
            self.sensor_qos
        )

        # Create timer for loop closure detection
        detection_frequency = self.get_parameter('detection_frequency').get_parameter_value().double_value
        self.detection_timer = self.create_timer(1.0 / detection_frequency, self.loop_detection_update)

        # State variables
        self.current_pose = None
        self.current_image = None

        self.get_logger().info('Loop Closure Node initialized')

    def _declare_parameters(self):
        """Declare ROS 2 parameters."""
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('similarity_threshold', 0.7)
        self.declare_parameter('min_loop_distance', 10.0)
        self.declare_parameter('detection_frequency', 2.0)
        self.declare_parameter('enable_pose_correction', True)

    def pose_callback(self, msg: PoseStamped):
        """Update current pose."""
        self.current_pose = msg

    def image_callback(self, msg: Image):
        """Process images for loop closure detection."""
        try:
            # Convert to OpenCV format
            self.current_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def loop_detection_update(self):
        """Check for loop closures."""
        try:
            if self.current_pose is None or self.current_image is None:
                return

            # Detect loop closure
            loop_detected = self.loop_detector.detect_loop(
                self.current_pose, self.current_image
            )

            # Publish loop detection result
            loop_msg = Bool()
            loop_msg.data = loop_detected
            self.loop_detected_pub.publish(loop_msg)

            if loop_detected:
                self.get_logger().info('Loop closure detected!')

                # Perform loop closure if enabled
                if self.get_parameter('enable_pose_correction').get_parameter_value().bool_value:
                    corrected_pose = self.loop_detector.close_loop()

                    if corrected_pose is not None:
                        # Create corrected pose message
                        corrected_msg = PoseStamped()
                        corrected_msg.header.stamp = self.get_clock().now().to_msg()
                        corrected_msg.header.frame_id = self.current_pose.header.frame_id
                        corrected_msg.pose = corrected_pose

                        self.corrected_pose_pub.publish(corrected_msg)
                        self.get_logger().info('Published corrected pose after loop closure')

        except Exception as e:
            self.get_logger().error(f'Error in loop closure detection: {str(e)}')


def main(args=None):
    """Main function."""
    rclpy.init(args=args)

    try:
        node = LoopClosureNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
