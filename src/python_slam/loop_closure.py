#!/usr/bin/env python3
"""
Loop Closure Module for Python SLAM
Implements loop closure detection and pose graph optimization
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from geometry_msgs.msg import PoseStamped


class LoopClosure:
    """
    Loop closure detection and handling for SLAM systems.
    """

    def __init__(self, similarity_threshold: float = 0.7, min_loop_distance: float = 10.0):
        """
        Initialize loop closure detector.

        Args:
            similarity_threshold: Threshold for image similarity
            min_loop_distance: Minimum distance for loop closure detection
        """
        self.similarity_threshold = similarity_threshold
        self.min_loop_distance = min_loop_distance

        # Storage for keyframes
        self.keyframes: List[Dict] = []
        self.keyframe_poses: List[np.ndarray] = []
        self.keyframe_descriptors: List[np.ndarray] = []

        # Feature detector for loop closure
        self.detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Loop closure results
        self.detected_loops: List[Tuple[int, int, np.ndarray]] = []

    def add_keyframe(self, image: np.ndarray, pose: PoseStamped, frame_id: int):
        """
        Add a new keyframe for loop closure detection.

        Args:
            image: Keyframe image
            pose: Keyframe pose
            frame_id: Frame identifier
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Extract features
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)

            if descriptors is not None:
                # Store keyframe data
                keyframe_data = {
                    'id': frame_id,
                    'image': gray.copy(),
                    'keypoints': keypoints,
                    'descriptors': descriptors,
                    'pose': pose
                }

                self.keyframes.append(keyframe_data)

                # Store pose as matrix
                pose_matrix = self._pose_to_matrix(pose)
                self.keyframe_poses.append(pose_matrix)
                self.keyframe_descriptors.append(descriptors)

        except Exception as e:
            print(f"Error adding keyframe: {e}")

    def detect_loop(self, current_pose: PoseStamped, current_image: Optional[np.ndarray] = None) -> bool:
        """
        Detect if current pose/image forms a loop closure.

        Args:
            current_pose: Current robot pose
            current_image: Current image (optional)

        Returns:
            True if loop closure detected
        """
        try:
            if len(self.keyframes) < 10:  # Need sufficient keyframes
                return False

            current_position = np.array([
                current_pose.pose.position.x,
                current_pose.pose.position.y,
                current_pose.pose.position.z
            ])

            # Check distance-based loop closure
            for i, keyframe_pose in enumerate(self.keyframe_poses[:-5]):  # Skip recent frames
                keyframe_position = keyframe_pose[:3, 3]
                distance = np.linalg.norm(current_position - keyframe_position)

                if distance < self.min_loop_distance:
                    # If image provided, verify with visual similarity
                    if current_image is not None:
                        if self._verify_visual_loop(current_image, i):
                            self._add_loop_closure(len(self.keyframes), i, current_pose)
                            return True
                    else:
                        # Distance-only loop closure
                        self._add_loop_closure(len(self.keyframes), i, current_pose)
                        return True

            return False

        except Exception as e:
            print(f"Error in loop detection: {e}")
            return False

    def _verify_visual_loop(self, current_image: np.ndarray, keyframe_index: int) -> bool:
        """
        Verify loop closure using visual similarity.

        Args:
            current_image: Current image
            keyframe_index: Index of candidate keyframe

        Returns:
            True if visual loop verified
        """
        try:
            # Convert to grayscale if needed
            if len(current_image.shape) == 3:
                current_gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            else:
                current_gray = current_image

            # Extract features from current image
            current_kp, current_desc = self.detector.detectAndCompute(current_gray, None)

            if current_desc is None:
                return False

            # Get keyframe descriptors
            keyframe_desc = self.keyframe_descriptors[keyframe_index]

            # Match features
            matches = self.matcher.match(current_desc, keyframe_desc)

            # Calculate similarity
            similarity = len(matches) / max(len(current_desc), len(keyframe_desc))

            return similarity > self.similarity_threshold

        except Exception as e:
            print(f"Error in visual loop verification: {e}")
            return False

    def _add_loop_closure(self, current_id: int, loop_id: int, current_pose: PoseStamped):
        """
        Add detected loop closure.

        Args:
            current_id: Current keyframe ID
            loop_id: Loop keyframe ID
            current_pose: Current pose
        """
        try:
            # Calculate relative transformation
            current_matrix = self._pose_to_matrix(current_pose)
            loop_matrix = self.keyframe_poses[loop_id]

            relative_transform = np.linalg.inv(loop_matrix) @ current_matrix

            # Store loop closure
            self.detected_loops.append((current_id, loop_id, relative_transform))

        except Exception as e:
            print(f"Error adding loop closure: {e}")

    def close_loop(self) -> Optional[PoseStamped]:
        """
        Perform loop closure optimization.

        Returns:
            Corrected pose or None
        """
        try:
            if len(self.detected_loops) == 0:
                return None

            # Simplified loop closure - just return the last detected loop
            # In practice, implement pose graph optimization

            # Get the most recent loop
            current_id, loop_id, relative_transform = self.detected_loops[-1]

            # Apply correction (simplified)
            if loop_id < len(self.keyframe_poses):
                corrected_matrix = self.keyframe_poses[loop_id] @ relative_transform
                corrected_pose = self._matrix_to_pose(corrected_matrix)
                return corrected_pose

            return None

        except Exception as e:
            print(f"Error in loop closure: {e}")
            return None

    def _pose_to_matrix(self, pose: PoseStamped) -> np.ndarray:
        """Convert ROS pose to 4x4 transformation matrix."""
        matrix = np.eye(4)
        matrix[0, 3] = pose.pose.position.x
        matrix[1, 3] = pose.pose.position.y
        matrix[2, 3] = pose.pose.position.z

        # Simplified: identity rotation
        # In practice, convert quaternion to rotation matrix

        return matrix

    def _matrix_to_pose(self, matrix: np.ndarray) -> PoseStamped:
        """Convert 4x4 transformation matrix to ROS pose."""
        pose = PoseStamped()
        pose.pose.position.x = matrix[0, 3]
        pose.pose.position.y = matrix[1, 3]
        pose.pose.position.z = matrix[2, 3]
        pose.pose.orientation.w = 1.0  # Simplified

        return pose

    def get_loop_closures(self) -> List[Tuple[int, int, np.ndarray]]:
        """Get all detected loop closures."""
        return self.detected_loops.copy()

    def clear_loops(self):
        """Clear all stored loop closures."""
        self.detected_loops.clear()

    def get_keyframe_count(self) -> int:
        """Get number of stored keyframes."""
        return len(self.keyframes)


if __name__ == "__main__":
    print("Loop Closure Module - Demo")

    # Create loop closure detector
    lc = LoopClosure(similarity_threshold=0.7, min_loop_distance=5.0)

    # Create dummy data
    pose = PoseStamped()
    pose.pose.position.x = 1.0
    pose.pose.position.y = 2.0

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Add keyframe
    lc.add_keyframe(image, pose, 0)

    # Detect loop
    loop_detected = lc.detect_loop(pose, image)

    print(f"Loop detected: {loop_detected}")
    print(f"Keyframes stored: {lc.get_keyframe_count()}")
    print("Loop closure module demo complete!")
