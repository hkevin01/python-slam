#!/usr/bin/env python3
"""
Basic SLAM pipeline using ORB feature extraction and Essential Matrix pose estimation.
Enhanced for ROS 2 integration and professional usage.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Any
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge


class BasicSlamPipeline:
    """
    Basic SLAM pipeline implementation using ORB features and Essential Matrix.
    Designed for aerial drone competitions with real-time processing capabilities.
    """

    def __init__(self, camera_matrix: Optional[np.ndarray] = None):
        """
        Initialize the SLAM pipeline.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix. If None, uses default values.
        """
        # Initialize ORB feature detector
        self.orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)

        # Initialize matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Camera intrinsic matrix
        if camera_matrix is None:
            # Default camera matrix (update with actual calibration)
            self.K = np.array([
                [525.0, 0.0, 319.5],
                [0.0, 525.0, 239.5],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
        else:
            self.K = camera_matrix

        # State variables
        self.previous_frame: Optional[np.ndarray] = None
        self.current_pose: Optional[np.ndarray] = None
        self.trajectory: List[np.ndarray] = []
        self.map_points: List[np.ndarray] = []

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        """
        Process a single frame through the SLAM pipeline.

        Args:
            frame: Input image frame (BGR format)

        Returns:
            Tuple of (pose_change, map_points) or (None, []) if processing fails
        """
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame

            # Extract features
            keypoints, descriptors = self.orb.detectAndCompute(gray_frame, None)

            if descriptors is None or len(keypoints) < 10:
                self.previous_frame = gray_frame
                return None, []

            # If this is the first frame, store it and return
            if self.previous_frame is None:
                self.previous_frame = gray_frame
                return None, []

            # Process with previous frame
            pose_change = self._estimate_pose_change(self.previous_frame, gray_frame)

            # Update trajectory
            if pose_change is not None:
                if self.current_pose is None:
                    self.current_pose = np.eye(4)
                else:
                    self.current_pose = self.current_pose @ pose_change

                self.trajectory.append(self.current_pose.copy())

            # Store current frame for next iteration
            self.previous_frame = gray_frame

            return pose_change, self.map_points

        except Exception as e:
            print(f"Error processing frame: {e}")
            return None, []

    def process_frames(self, img1_path: str, img2_path: str) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Process two image files for pose estimation.
        Legacy method for compatibility.

        Args:
            img1_path: Path to first image
            img2_path: Path to second image

        Returns:
            Tuple of (R, t, matches) where R is rotation, t is translation, matches are feature matches
        """
        # Load images
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

        if img1 is None or img2 is None:
            raise FileNotFoundError("Images not found.")

        # Extract features
        kp1, des1 = self.orb.detectAndCompute(img1, None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)

        if des1 is None or des2 is None:
            raise ValueError("No features detected in one or both images")

        # Match features
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) < 8:
            raise ValueError("Insufficient matches for pose estimation")

        # Extract matched points
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

        # Estimate pose
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)

        return R, t, matches

    def _estimate_pose_change(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate pose change between two frames.

        Args:
            prev_frame: Previous frame (grayscale)
            curr_frame: Current frame (grayscale)

        Returns:
            4x4 transformation matrix or None if estimation fails
        """
        try:
            # Extract features from both frames
            kp1, des1 = self.orb.detectAndCompute(prev_frame, None)
            kp2, des2 = self.orb.detectAndCompute(curr_frame, None)

            if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
                return None

            # Match features
            matches = self.bf.match(des1, des2)

            if len(matches) < 8:
                return None

            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Extract matched points
            pts1 = np.array([kp1[m.queryIdx].pt for m in matches[:50]], dtype=np.float32)
            pts2 = np.array([kp2[m.trainIdx].pt for m in matches[:50]], dtype=np.float32)

            # Estimate Essential Matrix
            E, mask = cv2.findEssentialMat(
                pts1, pts2, self.K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )

            if E is None:
                return None

            # Recover pose
            _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K)

            # Create 4x4 transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()

            return T

        except Exception as e:
            print(f"Error in pose estimation: {e}")
            return None

    def get_trajectory(self) -> List[np.ndarray]:
        """Get the current trajectory."""
        return self.trajectory.copy()

    def get_current_pose(self) -> Optional[np.ndarray]:
        """Get the current pose."""
        return self.current_pose.copy() if self.current_pose is not None else None

    def reset(self):
        """Reset the SLAM pipeline state."""
        self.previous_frame = None
        self.current_pose = None
        self.trajectory.clear()
        self.map_points.clear()


class BasicSlamPipelineNode(Node):
    """ROS 2 node wrapper for BasicSlamPipeline."""

    def __init__(self):
        super().__init__('basic_slam_pipeline_node')

        # Initialize CV bridge and SLAM pipeline
        self.cv_bridge = CvBridge()
        self.slam = BasicSlamPipeline()

        # Create subscriber
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Create publishers
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/slam/pose',
            10
        )

        self.get_logger().info('Basic SLAM Pipeline Node initialized')

    def image_callback(self, msg: Image):
        """Process incoming images."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process through SLAM pipeline
            pose_change, map_points = self.slam.process_frame(cv_image)

            if pose_change is not None:
                # Create and publish pose message
                pose_msg = PoseStamped()
                pose_msg.header.stamp = msg.header.stamp
                pose_msg.header.frame_id = 'map'

                # Extract position and orientation from transformation matrix
                current_pose = self.slam.get_current_pose()
                if current_pose is not None:
                    pose_msg.pose.position.x = current_pose[0, 3]
                    pose_msg.pose.position.y = current_pose[1, 3]
                    pose_msg.pose.position.z = current_pose[2, 3]

                    # Convert rotation matrix to quaternion (simplified)
                    pose_msg.pose.orientation.w = 1.0
                    pose_msg.pose.orientation.x = 0.0
                    pose_msg.pose.orientation.y = 0.0
                    pose_msg.pose.orientation.z = 0.0

                    self.pose_pub.publish(pose_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    """Main function for ROS 2 node."""
    rclpy.init(args=args)

    try:
        node = BasicSlamPipelineNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    # Check if running as ROS 2 node or standalone
    import sys
    if '--ros-args' in sys.argv or 'ros2' in sys.argv[0]:
        main()
    else:
        # Standalone demo
        print("Basic SLAM Pipeline - Standalone Demo")
        print("For ROS 2 integration, use: ros2 run python_slam basic_slam_pipeline")

        # Create default camera matrix
        K = np.array([
            [525.0, 0.0, 319.5],
            [0.0, 525.0, 239.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # Initialize SLAM pipeline
        slam = BasicSlamPipeline(K)
        print("SLAM pipeline initialized successfully!")
        print("Use slam.process_frame(frame) to process video frames")
        print("Use slam.process_frames(img1_path, img2_path) for static images")
