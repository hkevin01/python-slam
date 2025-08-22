#!/usr/bin/env python3
"""
Python SLAM Wrapper Implementation

Wrapper for the custom Python SLAM implementation providing the standard
SLAM interface. This integrates the existing python-slam components.
"""

import os
import time
import numpy as np
from typing import Optional, List, Dict, Any, Union
import cv2

from ..slam_interface import (
    SLAMInterface, SLAMConfiguration, SLAMPose, SLAMMapPoint,
    SLAMTrajectory, SLAMState, SensorType
)
from sensor_msgs.msg import Image, PointCloud2, Imu
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge

# Import existing python-slam components
try:
    from ...feature_extraction import FeatureExtraction
    from ...pose_estimation import PoseEstimation
    from ...mapping import Mapping
    from ...localization import Localization
    from ...loop_closure import LoopClosure
    PYTHON_SLAM_AVAILABLE = True
except ImportError:
    PYTHON_SLAM_AVAILABLE = False


class PythonSLAMWrapper(SLAMInterface):
    """
    Python SLAM algorithm wrapper.

    Integrates the existing python-slam components with the standard
    SLAM interface for monocular, stereo, and visual-inertial SLAM.
    """

    def __init__(self, config: SLAMConfiguration):
        """Initialize Python SLAM wrapper."""
        super().__init__(config)

        if not PYTHON_SLAM_AVAILABLE:
            raise ImportError("Python SLAM components not available")

        self.cv_bridge = CvBridge()

        # Initialize SLAM components
        self.feature_extractor = None
        self.pose_estimator = None
        self.mapper = None
        self.localizer = None
        self.loop_closer = None

        # Frame storage for stereo/temporal processing
        self.previous_frame = None
        self.previous_features = None
        self.previous_timestamp = None

        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self._setup_camera_parameters()

        # Map and trajectory data
        self.trajectory_poses = []
        self.trajectory_timestamps = []
        self.keyframes = []
        self.map_points_3d = []

        # Performance tracking
        self.processing_times = []
        self.feature_counts = []

        # IMU integration
        self.imu_buffer = []
        self.last_imu_timestamp = None

        # Loop closure
        self.keyframe_interval = config.keyframe_distance
        self.last_keyframe_pose = None

    def _setup_camera_parameters(self):
        """Setup camera parameters from configuration."""
        camera_params = self.config.custom_params.get('camera', {})

        # Default camera matrix (should be calibrated for real use)
        fx = camera_params.get('fx', 525.0)
        fy = camera_params.get('fy', 525.0)
        cx = camera_params.get('cx', 319.5)
        cy = camera_params.get('cy', 239.5)

        self.camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Distortion coefficients
        self.dist_coeffs = np.array([
            camera_params.get('k1', 0.0),
            camera_params.get('k2', 0.0),
            camera_params.get('p1', 0.0),
            camera_params.get('p2', 0.0),
            camera_params.get('k3', 0.0)
        ], dtype=np.float32)

    def initialize(self) -> bool:
        """Initialize Python SLAM components."""
        try:
            # Initialize feature extraction
            self.feature_extractor = FeatureExtraction(
                max_features=self.config.max_features,
                quality_level=self.config.quality_level,
                min_distance=self.config.min_distance
            )

            # Initialize pose estimation
            self.pose_estimator = PoseEstimation(
                camera_matrix=self.camera_matrix
            )

            # Initialize mapping
            self.mapper = Mapping(
                camera_matrix=self.camera_matrix,
                resolution=self.config.map_resolution
            )

            # Initialize localization
            self.localizer = Localization()

            # Initialize loop closure if enabled
            if self.config.enable_loop_closure:
                self.loop_closer = LoopClosure()

            self._update_state(SLAMState.INITIALIZING)
            self._log_info("Python SLAM components initialized successfully")
            return True

        except Exception as e:
            self._log_error(f"Failed to initialize Python SLAM: {e}")
            return False

    def process_image(self, image: Union[np.ndarray, Image], timestamp: float) -> bool:
        """Process monocular camera image."""
        if not self.is_initialized():
            self._log_error("SLAM system not initialized")
            return False

        try:
            # Convert ROS image to OpenCV if necessary
            if isinstance(image, Image):
                cv_image = self.cv_bridge.imgmsg_to_cv2(image, "bgr8")
            else:
                cv_image = image.copy()

            start_time = time.time()

            # Convert to grayscale for feature extraction
            if len(cv_image.shape) == 3:
                gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = cv_image

            # Extract features
            keypoints, descriptors = self.feature_extractor.extract_features(gray_image)
            feature_count = len(keypoints) if keypoints else 0

            pose_estimated = False

            # Estimate pose if we have previous frame
            if self.previous_frame is not None and self.previous_features is not None:
                try:
                    # Match features between frames
                    matches = self._match_features(self.previous_features[1], descriptors)

                    if len(matches) > 10:  # Minimum matches required
                        # Extract matched point coordinates
                        prev_pts = np.array([self.previous_features[0][m.queryIdx].pt
                                           for m in matches], dtype=np.float32)
                        curr_pts = np.array([keypoints[m.trainIdx].pt
                                           for m in matches], dtype=np.float32)

                        # Estimate pose change
                        pose_change = self.pose_estimator.estimate_pose_change(
                            prev_pts, curr_pts, self.camera_matrix
                        )

                        if pose_change is not None:
                            # Update current pose
                            if self.current_pose is not None:
                                # Compose with previous pose
                                new_pose = self._compose_poses(self.current_pose, pose_change, timestamp)
                            else:
                                # First pose
                                new_pose = self._matrix_to_slam_pose(pose_change, timestamp)

                            self.current_pose = new_pose
                            pose_estimated = True

                            # Add to trajectory
                            self.trajectory_poses.append(new_pose)
                            self.trajectory_timestamps.append(timestamp)

                            if self.state == SLAMState.INITIALIZING:
                                self._update_state(SLAMState.TRACKING)

                except Exception as e:
                    self._log_warning(f"Pose estimation failed: {e}")

            # Update mapping if enabled and pose was estimated
            if self.config.enable_mapping and pose_estimated and self.current_pose:
                try:
                    # Add points to map
                    if len(keypoints) > 0:
                        points_3d = self._triangulate_points(keypoints, self.current_pose)
                        if len(points_3d) > 0:
                            self.map_points_3d.extend(points_3d)

                        # Update mapper
                        self.mapper.update(self.current_pose, points_3d)

                except Exception as e:
                    self._log_warning(f"Mapping update failed: {e}")

            # Check for keyframe and loop closure
            if self._should_create_keyframe(timestamp):
                self._create_keyframe(cv_image, keypoints, descriptors, timestamp)

            # Store current frame data for next iteration
            self.previous_frame = gray_image.copy()
            self.previous_features = (keypoints, descriptors)
            self.previous_timestamp = timestamp

            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.feature_counts.append(feature_count)
            self._increment_frame_count()

            # Update state
            if not pose_estimated and self.state == SLAMState.TRACKING:
                self._update_state(SLAMState.LOST)

            return True

        except Exception as e:
            self._log_error(f"Error processing image: {e}")
            return False

    def process_stereo_images(self, left_image: Union[np.ndarray, Image],
                            right_image: Union[np.ndarray, Image],
                            timestamp: float) -> bool:
        """Process stereo camera images."""
        if not self.is_initialized():
            self._log_error("SLAM system not initialized")
            return False

        try:
            # Convert ROS images to OpenCV if necessary
            if isinstance(left_image, Image):
                left_cv = self.cv_bridge.imgmsg_to_cv2(left_image, "bgr8")
            else:
                left_cv = left_image.copy()

            if isinstance(right_image, Image):
                right_cv = self.cv_bridge.imgmsg_to_cv2(right_image, "bgr8")
            else:
                right_cv = right_image.copy()

            start_time = time.time()

            # Convert to grayscale
            if len(left_cv.shape) == 3:
                left_gray = cv2.cvtColor(left_cv, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right_cv, cv2.COLOR_BGR2GRAY)
            else:
                left_gray = left_cv
                right_gray = right_cv

            # Extract features from left image
            left_kp, left_desc = self.feature_extractor.extract_features(left_gray)

            # Stereo matching to get depth
            stereo_points = self._stereo_matching(left_gray, right_gray, left_kp)

            # Estimate pose using stereo information
            pose_estimated = False
            if self.previous_frame is not None and len(stereo_points) > 10:
                try:
                    pose_change = self.pose_estimator.estimate_pose_stereo(
                        self.previous_features[0], left_kp, stereo_points, self.camera_matrix
                    )

                    if pose_change is not None:
                        # Update current pose
                        if self.current_pose is not None:
                            new_pose = self._compose_poses(self.current_pose, pose_change, timestamp)
                        else:
                            new_pose = self._matrix_to_slam_pose(pose_change, timestamp)

                        self.current_pose = new_pose
                        pose_estimated = True

                        # Add to trajectory
                        self.trajectory_poses.append(new_pose)
                        self.trajectory_timestamps.append(timestamp)

                        if self.state == SLAMState.INITIALIZING:
                            self._update_state(SLAMState.TRACKING)

                except Exception as e:
                    self._log_warning(f"Stereo pose estimation failed: {e}")

            # Update mapping with stereo points
            if self.config.enable_mapping and pose_estimated and len(stereo_points) > 0:
                try:
                    # Convert stereo points to 3D map points
                    map_points = []
                    for point in stereo_points:
                        if len(point) >= 3:  # Has depth
                            map_point = SLAMMapPoint(
                                position=np.array([point[0], point[1], point[2]]),
                                confidence=0.8
                            )
                            map_points.append(map_point)

                    self.map_points_3d.extend(map_points)

                except Exception as e:
                    self._log_warning(f"Stereo mapping failed: {e}")

            # Store for next iteration
            self.previous_frame = left_gray.copy()
            self.previous_features = (left_kp, left_desc)
            self.previous_timestamp = timestamp

            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.feature_counts.append(len(left_kp) if left_kp else 0)
            self._increment_frame_count()

            return True

        except Exception as e:
            self._log_error(f"Error processing stereo images: {e}")
            return False

    def process_pointcloud(self, pointcloud: Union[np.ndarray, PointCloud2],
                          timestamp: float) -> bool:
        """Python SLAM doesn't directly process point clouds."""
        self._log_warning("Python SLAM doesn't support direct point cloud processing")
        return False

    def process_imu(self, imu_data: Union[np.ndarray, Imu], timestamp: float) -> bool:
        """Process IMU data for visual-inertial SLAM."""
        if self.config.sensor_type != SensorType.VISUAL_INERTIAL:
            return True  # IMU not used in this mode

        try:
            if isinstance(imu_data, Imu):
                # Extract IMU data from ROS message
                acc = np.array([
                    imu_data.linear_acceleration.x,
                    imu_data.linear_acceleration.y,
                    imu_data.linear_acceleration.z
                ])
                gyro = np.array([
                    imu_data.angular_velocity.x,
                    imu_data.angular_velocity.y,
                    imu_data.angular_velocity.z
                ])
            else:
                # Assume numpy array with [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
                acc = imu_data[:3]
                gyro = imu_data[3:6]

            # Store IMU data in buffer
            imu_measurement = {
                'timestamp': timestamp,
                'acceleration': acc,
                'angular_velocity': gyro
            }

            self.imu_buffer.append(imu_measurement)

            # Keep buffer size manageable
            if len(self.imu_buffer) > 1000:
                self.imu_buffer = self.imu_buffer[-500:]

            self.last_imu_timestamp = timestamp
            return True

        except Exception as e:
            self._log_error(f"Error processing IMU data: {e}")
            return False

    def get_pose(self) -> Optional[SLAMPose]:
        """Get current robot pose estimate."""
        return self.current_pose

    def get_map(self) -> List[SLAMMapPoint]:
        """Get current map representation."""
        return self.map_points_3d.copy()

    def get_occupancy_grid(self) -> Optional[OccupancyGrid]:
        """Get 2D occupancy grid representation."""
        if self.mapper is None:
            return None

        try:
            # Get occupancy grid from mapper
            grid = self.mapper.get_occupancy_grid()
            return grid
        except Exception as e:
            self._log_error(f"Error getting occupancy grid: {e}")
            return None

    def get_trajectory(self) -> SLAMTrajectory:
        """Get robot trajectory."""
        keyframe_indices = list(range(0, len(self.trajectory_poses), 10))  # Every 10th frame
        return SLAMTrajectory(
            poses=self.trajectory_poses.copy(),
            timestamps=self.trajectory_timestamps.copy(),
            keyframe_indices=keyframe_indices
        )

    def reset(self) -> bool:
        """Reset Python SLAM system."""
        try:
            self.current_pose = None
            self.trajectory_poses = []
            self.trajectory_timestamps = []
            self.keyframes = []
            self.map_points_3d = []
            self.processing_times = []
            self.feature_counts = []
            self.imu_buffer = []

            self.previous_frame = None
            self.previous_features = None
            self.previous_timestamp = None
            self.last_imu_timestamp = None
            self.last_keyframe_pose = None

            self._frame_count = 0

            # Reset components
            if self.mapper:
                self.mapper.reset()
            if self.localizer:
                self.localizer.reset()
            if self.loop_closer:
                self.loop_closer.reset()

            self._update_state(SLAMState.INITIALIZING)
            self._log_info("Python SLAM system reset")
            return True

        except Exception as e:
            self._log_error(f"Error resetting system: {e}")
            return False

    def save_map(self, filepath: str) -> bool:
        """Save current map to file."""
        try:
            # Save trajectory and map points
            data = {
                'trajectory_poses': [self._slam_pose_to_dict(pose) for pose in self.trajectory_poses],
                'trajectory_timestamps': self.trajectory_timestamps,
                'map_points': [self._map_point_to_dict(point) for point in self.map_points_3d],
                'camera_matrix': self.camera_matrix.tolist(),
                'config': {
                    'algorithm_name': self.config.algorithm_name,
                    'sensor_type': self.config.sensor_type.value,
                    'max_features': self.config.max_features
                }
            }

            import json
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            self._log_info(f"Map saved to {filepath}")
            return True

        except Exception as e:
            self._log_error(f"Error saving map: {e}")
            return False

    def load_map(self, filepath: str) -> bool:
        """Load map from file."""
        try:
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Load trajectory
            self.trajectory_poses = [self._dict_to_slam_pose(pose_dict)
                                   for pose_dict in data.get('trajectory_poses', [])]
            self.trajectory_timestamps = data.get('trajectory_timestamps', [])

            # Load map points
            self.map_points_3d = [self._dict_to_map_point(point_dict)
                                for point_dict in data.get('map_points', [])]

            # Load camera matrix if available
            if 'camera_matrix' in data:
                self.camera_matrix = np.array(data['camera_matrix'])

            self._log_info(f"Map loaded from {filepath}")
            return True

        except Exception as e:
            self._log_error(f"Error loading map: {e}")
            return False

    def relocalize(self, initial_pose: Optional[SLAMPose] = None) -> bool:
        """Attempt to relocalize after tracking loss."""
        try:
            self._update_state(SLAMState.RELOCALIZATION)

            if initial_pose is not None:
                self.current_pose = initial_pose
                self._log_info("Relocalization with provided initial pose")
                self._update_state(SLAMState.TRACKING)
                return True

            # Try to relocalize using loop closure if available
            if self.loop_closer and len(self.keyframes) > 0:
                # Simplified relocalization
                self.current_pose = self.trajectory_poses[-1] if self.trajectory_poses else None
                if self.current_pose:
                    self._update_state(SLAMState.TRACKING)
                    self._log_info("Relocalization successful")
                    return True

            self._log_warning("Relocalization failed")
            return False

        except Exception as e:
            self._log_error(f"Error during relocalization: {e}")
            return False

    def set_loop_closure_enabled(self, enabled: bool) -> None:
        """Enable/disable loop closure detection."""
        self.config.enable_loop_closure = enabled
        self._log_info(f"Loop closure {'enabled' if enabled else 'disabled'}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get algorithm performance metrics."""
        if not self.processing_times:
            return {'algorithm': 'Python SLAM', 'frames_processed': 0}

        avg_processing_time = np.mean(self.processing_times)
        avg_features = np.mean(self.feature_counts) if self.feature_counts else 0
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0

        return {
            'algorithm': 'Python SLAM',
            'frames_processed': self._frame_count,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'max_processing_time_ms': np.max(self.processing_times) * 1000,
            'min_processing_time_ms': np.min(self.processing_times) * 1000,
            'average_fps': fps,
            'average_features': avg_features,
            'state': self.state.value,
            'map_points': len(self.map_points_3d),
            'trajectory_length': len(self.trajectory_poses),
            'tracking_state': 'tracking' if self.current_pose is not None else 'lost'
        }

    # Helper methods
    def _match_features(self, desc1, desc2):
        """Match features between two frames."""
        if desc1 is None or desc2 is None:
            return []

        try:
            # Use FLANN or BF matcher
            if desc1.dtype == np.uint8:  # ORB descriptors
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            else:  # SIFT/SURF descriptors
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

            matches = matcher.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Filter matches by distance
            good_matches = [m for m in matches if m.distance < 50]
            return good_matches

        except Exception as e:
            self._log_warning(f"Feature matching failed: {e}")
            return []

    def _stereo_matching(self, left_img, right_img, keypoints):
        """Perform stereo matching to get 3D points."""
        try:
            # Simplified stereo matching using OpenCV
            stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=21)
            disparity = stereo.compute(left_img, right_img)

            # Convert keypoints to 3D using disparity
            points_3d = []
            baseline = 0.1  # meters (should be calibrated)

            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
                    d = disparity[y, x]
                    if d > 0:
                        # Convert to 3D coordinates
                        Z = baseline * self.camera_matrix[0, 0] / d
                        X = (x - self.camera_matrix[0, 2]) * Z / self.camera_matrix[0, 0]
                        Y = (y - self.camera_matrix[1, 2]) * Z / self.camera_matrix[1, 1]
                        points_3d.append([X, Y, Z])

            return points_3d

        except Exception as e:
            self._log_warning(f"Stereo matching failed: {e}")
            return []

    def _triangulate_points(self, keypoints, pose):
        """Triangulate 3D points from 2D keypoints."""
        # Simplified triangulation (would need proper stereo/multi-view setup)
        points_3d = []

        try:
            # Assume depth of 1 meter for monocular case
            default_depth = 1.0

            for kp in keypoints:
                x, y = kp.pt
                # Convert to camera coordinates
                X = (x - self.camera_matrix[0, 2]) * default_depth / self.camera_matrix[0, 0]
                Y = (y - self.camera_matrix[1, 2]) * default_depth / self.camera_matrix[1, 1]
                Z = default_depth

                # Transform to world coordinates using current pose
                world_point = self._transform_point_to_world(np.array([X, Y, Z]), pose)

                map_point = SLAMMapPoint(
                    position=world_point,
                    confidence=0.5  # Low confidence for monocular
                )
                points_3d.append(map_point)

        except Exception as e:
            self._log_warning(f"Point triangulation failed: {e}")

        return points_3d

    def _should_create_keyframe(self, timestamp):
        """Check if a new keyframe should be created."""
        if self.last_keyframe_pose is None:
            return True

        # Check distance from last keyframe
        if self.current_pose is not None:
            distance = np.linalg.norm(
                self.current_pose.position - self.last_keyframe_pose.position
            )
            return distance > self.keyframe_interval

        return False

    def _create_keyframe(self, image, keypoints, descriptors, timestamp):
        """Create a new keyframe."""
        if self.current_pose is not None:
            keyframe = {
                'timestamp': timestamp,
                'pose': self.current_pose,
                'image': image.copy(),
                'keypoints': keypoints,
                'descriptors': descriptors
            }

            self.keyframes.append(keyframe)
            self.last_keyframe_pose = self.current_pose

            # Perform loop closure detection
            if self.config.enable_loop_closure and self.loop_closer:
                try:
                    self.loop_closer.detect_loop(keyframe, self.keyframes)
                except Exception as e:
                    self._log_warning(f"Loop closure detection failed: {e}")

    def _compose_poses(self, pose1, pose_change_matrix, timestamp):
        """Compose two poses."""
        # Convert pose1 to matrix
        pose1_matrix = self._slam_pose_to_matrix(pose1)

        # Compose transformations
        new_pose_matrix = np.dot(pose1_matrix, pose_change_matrix)

        # Convert back to SLAMPose
        return self._matrix_to_slam_pose(new_pose_matrix, timestamp)

    def _slam_pose_to_matrix(self, pose):
        """Convert SLAMPose to 4x4 transformation matrix."""
        from scipy.spatial.transform import Rotation

        matrix = np.eye(4)
        matrix[:3, 3] = pose.position

        # Convert quaternion to rotation matrix
        rot = Rotation.from_quat(pose.orientation)
        matrix[:3, :3] = rot.as_matrix()

        return matrix

    def _matrix_to_slam_pose(self, matrix, timestamp):
        """Convert 4x4 transformation matrix to SLAMPose."""
        from scipy.spatial.transform import Rotation

        position = matrix[:3, 3]
        rot = Rotation.from_matrix(matrix[:3, :3])
        orientation = rot.as_quat()  # [x, y, z, w]

        return SLAMPose(
            position=position,
            orientation=orientation,
            timestamp=timestamp,
            frame_id="map"
        )

    def _transform_point_to_world(self, point, pose):
        """Transform point from camera to world coordinates."""
        # Convert pose to matrix
        pose_matrix = self._slam_pose_to_matrix(pose)

        # Add homogeneous coordinate
        point_h = np.append(point, 1.0)

        # Transform to world coordinates
        world_point_h = np.dot(pose_matrix, point_h)

        return world_point_h[:3]

    def _slam_pose_to_dict(self, pose):
        """Convert SLAMPose to dictionary for serialization."""
        return {
            'position': pose.position.tolist(),
            'orientation': pose.orientation.tolist(),
            'timestamp': pose.timestamp,
            'frame_id': pose.frame_id
        }

    def _dict_to_slam_pose(self, pose_dict):
        """Convert dictionary to SLAMPose."""
        return SLAMPose(
            position=np.array(pose_dict['position']),
            orientation=np.array(pose_dict['orientation']),
            timestamp=pose_dict.get('timestamp'),
            frame_id=pose_dict.get('frame_id', 'map')
        )

    def _map_point_to_dict(self, point):
        """Convert SLAMMapPoint to dictionary."""
        return {
            'position': point.position.tolist(),
            'confidence': point.confidence,
            'observations': point.observations
        }

    def _dict_to_map_point(self, point_dict):
        """Convert dictionary to SLAMMapPoint."""
        return SLAMMapPoint(
            position=np.array(point_dict['position']),
            confidence=point_dict.get('confidence', 1.0),
            observations=point_dict.get('observations', 0)
        )
