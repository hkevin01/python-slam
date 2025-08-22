#!/usr/bin/env python3
"""
OpenVSLAM Wrapper Implementation

Wrapper for the OpenVSLAM (formerly OpenVSLAM) visual SLAM algorithm.
Supports monocular, stereo, and RGB-D visual SLAM with BoW-based loop detection.
"""

import os
import time
import numpy as np
from typing import Optional, List, Dict, Any, Union
import cv2
import yaml
import tempfile

from ..slam_interface import (
    SLAMInterface, SLAMConfiguration, SLAMPose, SLAMMapPoint,
    SLAMTrajectory, SLAMState, SensorType
)
from sensor_msgs.msg import Image, PointCloud2, Imu
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge

# OpenVSLAM imports
try:
    import openvslam
    OPENVSLAM_AVAILABLE = True
except ImportError:
    try:
        # Try alternative import name
        import stella_vslam as openvslam
        OPENVSLAM_AVAILABLE = True
    except ImportError:
        OPENVSLAM_AVAILABLE = False


class OpenVSLAMWrapper(SLAMInterface):
    """
    OpenVSLAM algorithm wrapper.

    Provides monocular, stereo, and RGB-D visual SLAM with bag-of-words
    based loop closure detection and pose graph optimization.
    """

    def __init__(self, config: SLAMConfiguration):
        """Initialize OpenVSLAM wrapper."""
        super().__init__(config)

        if not OPENVSLAM_AVAILABLE:
            raise ImportError("OpenVSLAM not available")

        self.cv_bridge = CvBridge()
        self.slam_system = None

        # OpenVSLAM configuration
        self.openvslam_config = {}
        self.config_file_path = None
        self.vocab_file_path = None

        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.baseline = None  # For stereo
        self._setup_camera_parameters()

        # Frame tracking
        self.frame_id = 0
        self.keyframe_database = []

        # Map and trajectory storage
        self.trajectory_poses = []
        self.trajectory_timestamps = []
        self.map_points_3d = []
        self.keyframes = []

        # Performance metrics
        self.processing_times = []
        self.tracking_state_history = []
        self.num_tracked_features = []
        self.loop_closures = 0

        # Working directory
        self.work_dir = tempfile.mkdtemp()

        # Map saving/loading
        self.map_database_path = os.path.join(self.work_dir, "map.msg")

    def _setup_camera_parameters(self):
        """Setup camera parameters from configuration."""
        camera_params = self.config.custom_params.get('camera', {})

        # Camera intrinsics
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

        # Stereo baseline
        self.baseline = camera_params.get('baseline', 0.1)  # meters

    def _create_openvslam_config(self):
        """Create OpenVSLAM configuration file."""
        try:
            # Get image dimensions
            img_width = self.config.custom_params.get('image_width', 640)
            img_height = self.config.custom_params.get('image_height', 480)

            # Base configuration
            config = {
                'Camera': {
                    'name': 'camera',
                    'setup': 'monocular',  # Will be updated based on sensor type
                    'model': 'perspective',
                    'fps': self.config.custom_params.get('fps', 30.0),
                    'cols': img_width,
                    'rows': img_height,
                    'color_order': 'RGB',
                    'fx': float(self.camera_matrix[0, 0]),
                    'fy': float(self.camera_matrix[1, 1]),
                    'cx': float(self.camera_matrix[0, 2]),
                    'cy': float(self.camera_matrix[1, 2]),
                    'k1': float(self.dist_coeffs[0]),
                    'k2': float(self.dist_coeffs[1]),
                    'p1': float(self.dist_coeffs[2]),
                    'p2': float(self.dist_coeffs[3]),
                    'k3': float(self.dist_coeffs[4]),
                },
                'Feature': {
                    'max_num_keypoints': self.config.max_features,
                    'scale_factor': 1.2,
                    'num_levels': 8,
                    'ini_fast_threshold': 20,
                    'min_fast_threshold': 7,
                },
                'Tracking': {
                    'reloc_distance_threshold': 0.2,
                    'reloc_angle_threshold': 0.45,
                    'enable_auto_relocalization': True,
                    'use_robust_matcher': True,
                },
                'Mapping': {
                    'baseline_dist_threshold': 0.02,
                    'redundant_obs_threshold': 3,
                    'enable_temporal_keyframe_only_BA': True,
                },
                'LoopDetector': {
                    'enabled': self.config.enable_loop_closure,
                    'min_continuity': 3,
                    'min_covisibility': 15,
                },
                'PangolinViewer': {
                    'enabled': False,  # Disable viewer for headless operation
                }
            }

            # Sensor-specific configuration
            if self.config.sensor_type == SensorType.STEREO:
                config['Camera']['setup'] = 'stereo'
                config['Camera']['baseline'] = self.baseline
                config['Camera']['depth_threshold'] = 50.0
            elif self.config.sensor_type == SensorType.RGBD:
                config['Camera']['setup'] = 'RGBD'
                config['Camera']['depth_scale'] = 1000.0  # Assuming depth in mm
                config['Camera']['depth_threshold'] = 5.0

            # Custom parameters
            custom_openvslam_params = self.config.custom_params.get('openvslam', {})
            self._deep_update(config, custom_openvslam_params)

            # Save configuration to file
            self.config_file_path = os.path.join(self.work_dir, "openvslam_config.yaml")
            with open(self.config_file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            self.openvslam_config = config
            self._log_info(f"OpenVSLAM config created: {self.config_file_path}")
            return True

        except Exception as e:
            self._log_error(f"Failed to create OpenVSLAM config: {e}")
            return False

    def _setup_vocabulary(self):
        """Setup BoW vocabulary file."""
        try:
            # Check for vocabulary file in config
            vocab_path = self.config.custom_params.get('vocabulary_path')

            if vocab_path and os.path.exists(vocab_path):
                self.vocab_file_path = vocab_path
                self._log_info(f"Using vocabulary file: {vocab_path}")
                return True

            # Try to find default vocabulary file
            possible_paths = [
                "/usr/local/share/openvslam/vocabulary/orb_vocab.dbow2",
                "/opt/openvslam/vocabulary/orb_vocab.dbow2",
                "./vocabulary/orb_vocab.dbow2",
                "../vocabulary/orb_vocab.dbow2"
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    self.vocab_file_path = path
                    self._log_info(f"Found vocabulary file: {path}")
                    return True

            self._log_warning("No vocabulary file found. Loop closure may not work.")
            return False

        except Exception as e:
            self._log_error(f"Error setting up vocabulary: {e}")
            return False

    def initialize(self) -> bool:
        """Initialize OpenVSLAM system."""
        try:
            # Create configuration
            if not self._create_openvslam_config():
                return False

            # Setup vocabulary
            self._setup_vocabulary()

            # Initialize OpenVSLAM
            if self.vocab_file_path:
                self.slam_system = openvslam.System(
                    cfg_file_path=self.config_file_path,
                    vocab_file_path=self.vocab_file_path
                )
            else:
                # Initialize without vocabulary (reduced functionality)
                self.slam_system = openvslam.System(
                    cfg_file_path=self.config_file_path
                )

            # Start SLAM system
            self.slam_system.startup()

            # Disable viewer if available
            if hasattr(self.slam_system, 'disable_viewer'):
                self.slam_system.disable_viewer()

            self.frame_id = 0
            self._update_state(SLAMState.INITIALIZING)
            self._log_info("OpenVSLAM initialized successfully")
            return True

        except Exception as e:
            self._log_error(f"Failed to initialize OpenVSLAM: {e}")
            return False

    def process_image(self, image: Union[np.ndarray, Image], timestamp: float) -> bool:
        """Process monocular camera image."""
        if not self.is_initialized():
            self._log_error("SLAM system not initialized")
            return False

        try:
            # Convert ROS image to OpenCV if necessary
            if isinstance(image, Image):
                cv_image = self.cv_bridge.imgmsg_to_cv2(image, "rgb8")
            else:
                cv_image = image.copy()
                if len(cv_image.shape) == 3 and cv_image.shape[2] == 3:
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            start_time = time.time()

            # Track with OpenVSLAM
            pose = self.slam_system.feed_monocular_frame(cv_image, timestamp)

            success = self._process_pose_result(pose, timestamp)

            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self._increment_frame_count()
            self.frame_id += 1

            return success

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
                left_cv = self.cv_bridge.imgmsg_to_cv2(left_image, "rgb8")
            else:
                left_cv = left_image.copy()
                if len(left_cv.shape) == 3 and left_cv.shape[2] == 3:
                    left_cv = cv2.cvtColor(left_cv, cv2.COLOR_BGR2RGB)

            if isinstance(right_image, Image):
                right_cv = self.cv_bridge.imgmsg_to_cv2(right_image, "rgb8")
            else:
                right_cv = right_image.copy()
                if len(right_cv.shape) == 3 and right_cv.shape[2] == 3:
                    right_cv = cv2.cvtColor(right_cv, cv2.COLOR_BGR2RGB)

            start_time = time.time()

            # Track with OpenVSLAM stereo
            pose = self.slam_system.feed_stereo_frame(left_cv, right_cv, timestamp)

            success = self._process_pose_result(pose, timestamp)

            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self._increment_frame_count()
            self.frame_id += 1

            return success

        except Exception as e:
            self._log_error(f"Error processing stereo images: {e}")
            return False

    def process_rgbd_images(self, rgb_image: Union[np.ndarray, Image],
                           depth_image: Union[np.ndarray, Image],
                           timestamp: float) -> bool:
        """Process RGB-D camera images."""
        if not self.is_initialized():
            self._log_error("SLAM system not initialized")
            return False

        try:
            # Convert ROS images to OpenCV if necessary
            if isinstance(rgb_image, Image):
                rgb_cv = self.cv_bridge.imgmsg_to_cv2(rgb_image, "rgb8")
            else:
                rgb_cv = rgb_image.copy()
                if len(rgb_cv.shape) == 3 and rgb_cv.shape[2] == 3:
                    rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_BGR2RGB)

            if isinstance(depth_image, Image):
                depth_cv = self.cv_bridge.imgmsg_to_cv2(depth_image, "passthrough")
            else:
                depth_cv = depth_image.copy()

            start_time = time.time()

            # Track with OpenVSLAM RGB-D
            pose = self.slam_system.feed_rgbd_frame(rgb_cv, depth_cv, timestamp)

            success = self._process_pose_result(pose, timestamp)

            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self._increment_frame_count()
            self.frame_id += 1

            return success

        except Exception as e:
            self._log_error(f"Error processing RGB-D images: {e}")
            return False

    def process_pointcloud(self, pointcloud: Union[np.ndarray, PointCloud2],
                          timestamp: float) -> bool:
        """OpenVSLAM doesn't directly process point clouds."""
        self._log_warning("OpenVSLAM doesn't support direct point cloud processing")
        return False

    def process_imu(self, imu_data: Union[np.ndarray, Imu], timestamp: float) -> bool:
        """Process IMU data (OpenVSLAM doesn't use IMU directly)."""
        # OpenVSLAM typically doesn't use IMU data directly
        # Could be used for initialization or motion prediction
        return True

    def get_pose(self) -> Optional[SLAMPose]:
        """Get current robot pose estimate."""
        return self.current_pose

    def get_map(self) -> List[SLAMMapPoint]:
        """Get current map representation."""
        if not self.slam_system:
            return []

        try:
            # Get map points from OpenVSLAM
            landmarks = self.slam_system.get_map_landmarks()
            map_points = []

            for landmark in landmarks:
                if landmark is not None:
                    pos = landmark.get_pos_in_world()
                    observations = landmark.get_num_observations()

                    map_point = SLAMMapPoint(
                        position=np.array([pos[0], pos[1], pos[2]]),
                        confidence=min(observations / 10.0, 1.0),
                        observations=observations
                    )
                    map_points.append(map_point)

            return map_points

        except Exception as e:
            self._log_error(f"Error getting map: {e}")
            return []

    def get_occupancy_grid(self) -> Optional[OccupancyGrid]:
        """Get 2D occupancy grid representation."""
        try:
            # OpenVSLAM doesn't directly provide occupancy grids
            # We can create one from the sparse 3D map
            map_points = self.get_map()

            if len(map_points) == 0:
                return None

            # Create occupancy grid from 3D points
            grid = OccupancyGrid()
            grid.header.frame_id = "map"
            grid.header.stamp.sec = int(time.time())

            # Calculate map bounds
            positions = np.array([point.position for point in map_points])
            min_x, max_x = np.min(positions[:, 0]), np.max(positions[:, 0])
            min_y, max_y = np.min(positions[:, 1]), np.max(positions[:, 1])

            # Add padding
            padding = 2.0
            min_x -= padding
            max_x += padding
            min_y -= padding
            max_y += padding

            # Set grid parameters
            resolution = 0.05  # 5cm resolution
            width = int((max_x - min_x) / resolution)
            height = int((max_y - min_y) / resolution)

            grid.info.resolution = resolution
            grid.info.width = width
            grid.info.height = height
            grid.info.origin.position.x = min_x
            grid.info.origin.position.y = min_y
            grid.info.origin.position.z = 0.0

            # Initialize grid as unknown
            grid.data = [-1] * (width * height)

            # Mark occupied cells
            for point in map_points:
                x, y = point.position[:2]
                grid_x = int((x - min_x) / resolution)
                grid_y = int((y - min_y) / resolution)

                if 0 <= grid_x < width and 0 <= grid_y < height:
                    index = grid_y * width + grid_x
                    grid.data[index] = 100  # Occupied

            return grid

        except Exception as e:
            self._log_error(f"Error getting occupancy grid: {e}")
            return None

    def get_trajectory(self) -> SLAMTrajectory:
        """Get robot trajectory."""
        if not self.slam_system:
            return SLAMTrajectory(poses=[], timestamps=[], keyframe_indices=[])

        try:
            # Get keyframes from OpenVSLAM
            keyframes = self.slam_system.get_keyframes()
            keyframe_indices = []

            for i, kf in enumerate(keyframes):
                if kf is not None:
                    keyframe_indices.append(i)

            return SLAMTrajectory(
                poses=self.trajectory_poses.copy(),
                timestamps=self.trajectory_timestamps.copy(),
                keyframe_indices=keyframe_indices
            )

        except Exception as e:
            self._log_error(f"Error getting trajectory: {e}")
            return SLAMTrajectory(poses=[], timestamps=[], keyframe_indices=[])

    def reset(self) -> bool:
        """Reset OpenVSLAM system."""
        try:
            if self.slam_system:
                self.slam_system.reset()

            self.current_pose = None
            self.trajectory_poses = []
            self.trajectory_timestamps = []
            self.map_points_3d = []
            self.keyframes = []
            self.processing_times = []
            self.tracking_state_history = []
            self.num_tracked_features = []
            self.loop_closures = 0
            self.frame_id = 0
            self._frame_count = 0

            self._update_state(SLAMState.INITIALIZING)
            self._log_info("OpenVSLAM system reset")
            return True

        except Exception as e:
            self._log_error(f"Error resetting system: {e}")
            return False

    def save_map(self, filepath: str) -> bool:
        """Save current map to file."""
        try:
            if not self.slam_system:
                return False

            # Save OpenVSLAM map database
            self.slam_system.save_map_database(filepath)

            # Also save our trajectory data
            trajectory_data = {
                'trajectory_poses': [self._slam_pose_to_dict(pose) for pose in self.trajectory_poses],
                'trajectory_timestamps': self.trajectory_timestamps,
                'config': {
                    'algorithm_name': self.config.algorithm_name,
                    'sensor_type': self.config.sensor_type.value,
                    'camera_matrix': self.camera_matrix.tolist(),
                    'dist_coeffs': self.dist_coeffs.tolist()
                }
            }

            import json
            with open(f"{filepath}_trajectory.json", 'w') as f:
                json.dump(trajectory_data, f, indent=2)

            self._log_info(f"Map saved to {filepath}")
            return True

        except Exception as e:
            self._log_error(f"Error saving map: {e}")
            return False

    def load_map(self, filepath: str) -> bool:
        """Load map from file."""
        try:
            if not self.slam_system:
                return False

            # Load OpenVSLAM map database
            self.slam_system.load_map_database(filepath)

            # Load trajectory data if available
            trajectory_file = f"{filepath}_trajectory.json"
            if os.path.exists(trajectory_file):
                import json
                with open(trajectory_file, 'r') as f:
                    data = json.load(f)

                self.trajectory_poses = [self._dict_to_slam_pose(pose_dict)
                                       for pose_dict in data.get('trajectory_poses', [])]
                self.trajectory_timestamps = data.get('trajectory_timestamps', [])

            self._log_info(f"Map loaded from {filepath}")
            return True

        except Exception as e:
            self._log_error(f"Error loading map: {e}")
            return False

    def relocalize(self, initial_pose: Optional[SLAMPose] = None) -> bool:
        """Attempt to relocalize after tracking loss."""
        try:
            self._update_state(SLAMState.RELOCALIZATION)

            if self.slam_system:
                # OpenVSLAM has built-in relocalization
                self.slam_system.request_relocalize()

                # Wait a moment for relocalization to process
                time.sleep(0.1)

                # Check if tracking resumed
                if self.current_pose is not None:
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
        if self.slam_system:
            if enabled:
                self.slam_system.enable_loop_detector()
            else:
                self.slam_system.disable_loop_detector()
        self._log_info(f"Loop closure {'enabled' if enabled else 'disabled'}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get algorithm performance metrics."""
        if not self.processing_times:
            return {'algorithm': 'OpenVSLAM', 'frames_processed': 0}

        avg_processing_time = np.mean(self.processing_times)
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0

        metrics = {
            'algorithm': 'OpenVSLAM',
            'frames_processed': self._frame_count,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'max_processing_time_ms': np.max(self.processing_times) * 1000,
            'min_processing_time_ms': np.min(self.processing_times) * 1000,
            'average_fps': fps,
            'state': self.state.value,
            'loop_closures': self.loop_closures,
            'tracking_state': 'tracking' if self.current_pose is not None else 'lost'
        }

        # Add OpenVSLAM specific metrics
        if self.slam_system:
            try:
                map_points = self.get_map()
                metrics.update({
                    'map_landmarks': len(map_points),
                    'keyframes': len(self.slam_system.get_keyframes()),
                    'current_frame_id': self.frame_id,
                })

                # Add feature tracking info if available
                if self.num_tracked_features:
                    metrics['avg_tracked_features'] = np.mean(self.num_tracked_features)

            except:
                pass

        return metrics

    # Helper methods
    def _process_pose_result(self, pose, timestamp):
        """Process pose result from OpenVSLAM."""
        try:
            if pose is not None:
                # Convert OpenVSLAM pose to SLAMPose
                slam_pose = self._openvslam_pose_to_slam_pose(pose, timestamp)

                if slam_pose is not None:
                    self.current_pose = slam_pose
                    self.trajectory_poses.append(slam_pose)
                    self.trajectory_timestamps.append(timestamp)

                    # Update state based on tracking
                    if self.state in [SLAMState.INITIALIZING, SLAMState.LOST, SLAMState.RELOCALIZATION]:
                        self._update_state(SLAMState.TRACKING)

                    # Check for loop closure
                    if self.slam_system and hasattr(self.slam_system, 'loop_is_detected'):
                        if self.slam_system.loop_is_detected():
                            self.loop_closures += 1
                            self._log_info(f"Loop closure detected! Total: {self.loop_closures}")

                    return True
            else:
                # Tracking lost
                if self.state == SLAMState.TRACKING:
                    self._update_state(SLAMState.LOST)
                return False

        except Exception as e:
            self._log_warning(f"Error processing pose result: {e}")
            return False

    def _openvslam_pose_to_slam_pose(self, openvslam_pose, timestamp):
        """Convert OpenVSLAM pose to SLAMPose."""
        try:
            # Extract pose matrix from OpenVSLAM
            if hasattr(openvslam_pose, 'inverse'):
                # Camera pose (needs to be inverted to get robot pose)
                pose_matrix = openvslam_pose.inverse().matrix()
            elif hasattr(openvslam_pose, 'matrix'):
                pose_matrix = openvslam_pose.matrix()
            else:
                # Assume it's already a matrix
                pose_matrix = np.array(openvslam_pose)

            # Extract position and orientation
            position = pose_matrix[:3, 3]
            rotation_matrix = pose_matrix[:3, :3]

            # Convert rotation matrix to quaternion
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_matrix(rotation_matrix)
            orientation = rot.as_quat()  # [x, y, z, w]

            return SLAMPose(
                position=position,
                orientation=orientation,
                timestamp=timestamp,
                frame_id="map"
            )

        except Exception as e:
            self._log_warning(f"Error converting OpenVSLAM pose: {e}")
            return None

    def _deep_update(self, base_dict, update_dict):
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _slam_pose_to_dict(self, pose):
        """Convert SLAMPose to dictionary."""
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

    def __del__(self):
        """Cleanup OpenVSLAM resources."""
        try:
            if self.slam_system:
                self.slam_system.shutdown()

            # Clean up working directory
            import shutil
            if os.path.exists(self.work_dir):
                shutil.rmtree(self.work_dir)

        except:
            pass
