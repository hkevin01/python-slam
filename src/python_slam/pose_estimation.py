#!/usr/bin/env python3
"""
Pose Estimation Module for Python SLAM
Implements visual odometry using Essential Matrix and PnP algorithms
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
from sensor_msgs.msg import CameraInfo


class PoseEstimation:
    """
    Pose estimation class for visual odometry in SLAM systems.
    Uses Essential Matrix decomposition and PnP for camera pose estimation.
    """

    def __init__(self, camera_matrix: Optional[np.ndarray] = None,
                 distortion_coeffs: Optional[np.ndarray] = None):
        """
        Initialize pose estimator.

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            distortion_coeffs: Distortion coefficients [k1, k2, p1, p2, k3]
        """
        # Default camera parameters (update with actual calibration)
        if camera_matrix is None:
            self.camera_matrix = np.array([
                [525.0, 0.0, 319.5],
                [0.0, 525.0, 239.5],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
        else:
            self.camera_matrix = camera_matrix

        if distortion_coeffs is None:
            self.distortion_coeffs = np.zeros(5, dtype=np.float32)
        else:
            self.distortion_coeffs = distortion_coeffs

        # RANSAC parameters
        self.ransac_threshold = 1.0
        self.ransac_confidence = 0.99
        self.ransac_max_iters = 1000
        self.min_inliers = 50

        # State variables
        self.previous_pose: Optional[np.ndarray] = None
        self.current_pose: Optional[np.ndarray] = None

    def update_camera_info(self, camera_info: CameraInfo):
        """
        Update camera parameters from ROS CameraInfo message.

        Args:
            camera_info: ROS CameraInfo message
        """
        # Extract camera matrix
        self.camera_matrix = np.array(camera_info.k).reshape(3, 3)

        # Extract distortion coefficients
        self.distortion_coeffs = np.array(camera_info.d)

    def estimate_pose(self, prev_image: np.ndarray, curr_image: np.ndarray,
                     prev_keypoints: Optional[List] = None, curr_keypoints: Optional[List] = None,
                     matches: Optional[List] = None) -> Optional[np.ndarray]:
        """
        Estimate pose change between two images.

        Args:
            prev_image: Previous image frame
            curr_image: Current image frame
            prev_keypoints: Previous image keypoints (optional)
            curr_keypoints: Current image keypoints (optional)
            matches: Feature matches (optional)

        Returns:
            4x4 transformation matrix or None if estimation fails
        """
        try:
            # If keypoints and matches not provided, extract them
            if prev_keypoints is None or curr_keypoints is None or matches is None:
                from .feature_extraction import FeatureExtraction
                fe = FeatureExtraction()
                prev_keypoints, curr_keypoints, matches = fe.extract_and_match(prev_image, curr_image)

            if len(matches) < self.min_inliers:
                return None

            # Extract matched points
            pts1 = np.array([prev_keypoints[m.queryIdx].pt for m in matches], dtype=np.float32)
            pts2 = np.array([curr_keypoints[m.trainIdx].pt for m in matches], dtype=np.float32)

            # Estimate pose using Essential Matrix
            return self._estimate_pose_essential_matrix(pts1, pts2)

        except Exception as e:
            print(f"Error in pose estimation: {e}")
            return None

    def _estimate_pose_essential_matrix(self, pts1: np.ndarray, pts2: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate pose using Essential Matrix decomposition.

        Args:
            pts1: Points from previous image
            pts2: Points from current image

        Returns:
            4x4 transformation matrix
        """
        try:
            if len(pts1) < 8 or len(pts2) < 8:
                return None

            # Undistort points
            pts1_undist = cv2.undistortPoints(pts1, self.camera_matrix, self.distortion_coeffs)
            pts2_undist = cv2.undistortPoints(pts2, self.camera_matrix, self.distortion_coeffs)

            # Estimate Essential Matrix
            E, mask = cv2.findEssentialMat(
                pts1_undist, pts2_undist,
                focal=1.0, pp=(0.0, 0.0),  # Normalized coordinates
                method=cv2.RANSAC,
                prob=self.ransac_confidence,
                threshold=self.ransac_threshold / max(self.camera_matrix[0, 0], self.camera_matrix[1, 1])
            )

            if E is None or mask is None:
                return None

            # Check if we have enough inliers
            num_inliers = np.sum(mask)
            if num_inliers < self.min_inliers:
                return None

            # Recover pose from Essential Matrix
            _, R, t, mask_pose = cv2.recoverPose(
                E, pts1_undist, pts2_undist,
                focal=1.0, pp=(0.0, 0.0)
            )

            # Create 4x4 transformation matrix
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()

            return T

        except Exception as e:
            print(f"Error in Essential Matrix pose estimation: {e}")
            return None

    def estimate_pose_pnp(self, object_points: np.ndarray, image_points: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate pose using PnP (Perspective-n-Point) algorithm.

        Args:
            object_points: 3D points in world coordinates
            image_points: 2D points in image coordinates

        Returns:
            4x4 transformation matrix
        """
        try:
            if len(object_points) < 4 or len(image_points) < 4:
                return None

            # Solve PnP
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points, image_points,
                self.camera_matrix, self.distortion_coeffs,
                reprojectionError=self.ransac_threshold,
                confidence=self.ransac_confidence,
                iterationsCount=self.ransac_max_iters
            )

            if not success or inliers is None or len(inliers) < self.min_inliers:
                return None

            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            # Create 4x4 transformation matrix
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()

            return T

        except Exception as e:
            print(f"Error in PnP pose estimation: {e}")
            return None

    def track_features_lk(self, prev_image: np.ndarray, curr_image: np.ndarray,
                         prev_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track features using Lucas-Kanade optical flow.

        Args:
            prev_image: Previous image (grayscale)
            curr_image: Current image (grayscale)
            prev_points: Points to track from previous image

        Returns:
            Tuple of (tracked_points, status)
        """
        try:
            # LK optical flow parameters
            lk_params = dict(
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )

            # Track features
            next_points, status, error = cv2.calcOpticalFlowPyrLK(
                prev_image, curr_image, prev_points, None, **lk_params
            )

            # Filter good points
            good_prev = prev_points[status == 1]
            good_next = next_points[status == 1]

            return good_next, status

        except Exception as e:
            print(f"Error in LK tracking: {e}")
            return np.array([]), np.array([])

    def triangulate_points(self, pts1: np.ndarray, pts2: np.ndarray,
                          P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from two views.

        Args:
            pts1: Points from first image
            pts2: Points from second image
            P1: Projection matrix for first camera
            P2: Projection matrix for second camera

        Returns:
            3D points in homogeneous coordinates
        """
        try:
            # Triangulate points
            points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

            # Convert from homogeneous to 3D coordinates
            points_3d = points_4d[:3] / points_4d[3]

            return points_3d.T

        except Exception as e:
            print(f"Error in point triangulation: {e}")
            return np.array([])

    def get_projection_matrix(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Get projection matrix from rotation and translation.

        Args:
            R: 3x3 rotation matrix
            t: 3x1 translation vector

        Returns:
            3x4 projection matrix
        """
        # Create extrinsic matrix
        extrinsic = np.hstack([R, t.reshape(-1, 1)])

        # Compute projection matrix
        P = self.camera_matrix @ extrinsic

        return P

    def compute_reprojection_error(self, object_points: np.ndarray, image_points: np.ndarray,
                                  rvec: np.ndarray, tvec: np.ndarray) -> float:
        """
        Compute reprojection error for pose validation.

        Args:
            object_points: 3D points
            image_points: 2D image points
            rvec: Rotation vector
            tvec: Translation vector

        Returns:
            Mean reprojection error
        """
        try:
            # Project 3D points to image
            projected_points, _ = cv2.projectPoints(
                object_points, rvec, tvec,
                self.camera_matrix, self.distortion_coeffs
            )

            # Compute error
            error = np.linalg.norm(image_points - projected_points.reshape(-1, 2), axis=1)
            mean_error = np.mean(error)

            return mean_error

        except Exception as e:
            print(f"Error computing reprojection error: {e}")
            return float('inf')

    def validate_pose(self, T: np.ndarray) -> bool:
        """
        Validate estimated pose for reasonableness.

        Args:
            T: 4x4 transformation matrix

        Returns:
            True if pose is valid
        """
        try:
            # Extract rotation and translation
            R = T[:3, :3]
            t = T[:3, 3]

            # Check if rotation matrix is valid
            if not self._is_rotation_matrix(R):
                return False

            # Check translation magnitude (should be reasonable for drone motion)
            translation_magnitude = np.linalg.norm(t)
            if translation_magnitude > 10.0:  # Adjust threshold based on application
                return False

            return True

        except Exception:
            return False

    def _is_rotation_matrix(self, R: np.ndarray) -> bool:
        """
        Check if matrix is a valid rotation matrix.

        Args:
            R: 3x3 matrix

        Returns:
            True if valid rotation matrix
        """
        # Check if R^T * R = I
        should_be_identity = np.dot(R.T, R)
        I = np.eye(3, dtype=R.dtype)

        # Check determinant is 1
        det = np.linalg.det(R)

        return (np.allclose(should_be_identity, I, atol=1e-4) and
                np.allclose(det, 1.0, atol=1e-4))

    def update_pose(self, relative_pose: np.ndarray):
        """
        Update current pose with relative transformation.

        Args:
            relative_pose: 4x4 relative transformation matrix
        """
        if self.current_pose is None:
            self.current_pose = relative_pose.copy()
        else:
            self.current_pose = self.current_pose @ relative_pose

        self.previous_pose = self.current_pose.copy()

    def get_current_pose(self) -> Optional[np.ndarray]:
        """Get current pose."""
        return self.current_pose.copy() if self.current_pose is not None else None

    def reset_pose(self):
        """Reset pose estimation state."""
        self.previous_pose = None
        self.current_pose = None


if __name__ == "__main__":
    # Demo usage
    print("Pose Estimation Module - Demo")

    # Create pose estimator
    pe = PoseEstimation()

    # Create dummy images for testing
    img1 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (480, 640), dtype=np.uint8)

    # Estimate pose
    pose = pe.estimate_pose(img1, img2)

    if pose is not None:
        print("Pose estimation successful!")
        print(f"Transformation matrix:\n{pose}")

        # Validate pose
        if pe.validate_pose(pose):
            print("Pose validation: PASSED")
        else:
            print("Pose validation: FAILED")
    else:
        print("Pose estimation failed - insufficient features or matches")
