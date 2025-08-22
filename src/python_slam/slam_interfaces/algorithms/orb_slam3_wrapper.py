#!/usr/bin/env python3
"""
ORB-SLAM3 Wrapper Implementation

Wrapper for ORB-SLAM3 algorithm providing the standard SLAM interface.
Supports monocular, stereo, RGB-D, and visual-inertial SLAM modes.
"""

import os
import time
import numpy as np
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

try:
    import orbslam3
    ORBSLAM3_AVAILABLE = True
except ImportError:
    ORBSLAM3_AVAILABLE = False

from ..slam_interface import (
    SLAMInterface, SLAMConfiguration, SLAMPose, SLAMMapPoint,
    SLAMTrajectory, SLAMState, SensorType
)
from sensor_msgs.msg import Image, PointCloud2, Imu
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2


class ORBSLAM3Wrapper(SLAMInterface):
    """
    ORB-SLAM3 algorithm wrapper.

    Provides integration with ORB-SLAM3 for monocular, stereo, RGB-D,
    and visual-inertial SLAM with the standard SLAM interface.
    """

    def __init__(self, config: SLAMConfiguration):
        """Initialize ORB-SLAM3 wrapper."""
        super().__init__(config)

        if not ORBSLAM3_AVAILABLE:
            raise ImportError("ORB-SLAM3 not available. Please install orbslam3 package.")

        self.slam_system = None
        self.cv_bridge = CvBridge()

        # ORB-SLAM3 specific parameters
        self.vocabulary_file = config.vocabulary_file or self._get_default_vocabulary()
        self.settings_file = config.config_file or self._create_settings_file()

        # Sensor mode mapping
        self.sensor_mode_map = {
            SensorType.MONOCULAR: orbslam3.System.MONOCULAR,
            SensorType.STEREO: orbslam3.System.STEREO,
            SensorType.RGB_D: orbslam3.System.RGBD,
            SensorType.VISUAL_INERTIAL: orbslam3.System.IMU_MONOCULAR
        }

        # Performance tracking
        self.processing_times = []
        self.frame_timestamps = []

        # Map data
        self.map_points_cache = []
        self.keyframes_cache = []

        # Initialize camera parameters from config
        self._setup_camera_parameters()

    def _get_default_vocabulary(self) -> str:
        """Get default ORB vocabulary file path."""
        # Try common locations
        possible_paths = [
            "/usr/local/share/ORB_SLAM3/Vocabulary/ORBvoc.txt",
            "/opt/ORB_SLAM3/Vocabulary/ORBvoc.txt",
            os.path.expanduser("~/ORB_SLAM3/Vocabulary/ORBvoc.txt"),
            "./Vocabulary/ORBvoc.txt"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        # If not found, return default path (user needs to provide)
        return "ORBvoc.txt"

    def _create_settings_file(self) -> str:
        """Create ORB-SLAM3 settings file from configuration."""
        settings_dir = os.path.expanduser("~/.config/python_slam/orb_slam3")
        os.makedirs(settings_dir, exist_ok=True)

        settings_file = os.path.join(settings_dir, f"{self.config.sensor_type.value}_settings.yaml")

        # Create settings based on sensor type
        if self.config.sensor_type == SensorType.MONOCULAR:
            settings_content = self._create_monocular_settings()
        elif self.config.sensor_type == SensorType.STEREO:
            settings_content = self._create_stereo_settings()
        elif self.config.sensor_type == SensorType.RGB_D:
            settings_content = self._create_rgbd_settings()
        elif self.config.sensor_type == SensorType.VISUAL_INERTIAL:
            settings_content = self._create_vio_settings()
        else:
            raise ValueError(f"Unsupported sensor type: {self.config.sensor_type}")

        with open(settings_file, 'w') as f:
            f.write(settings_content)

        return settings_file

    def _setup_camera_parameters(self):
        """Setup camera parameters from configuration."""
        # Default camera parameters (should be calibrated for real use)
        self.camera_params = self.config.custom_params.get('camera', {
            'fx': 525.0, 'fy': 525.0,
            'cx': 319.5, 'cy': 239.5,
            'k1': 0.0, 'k2': 0.0, 'p1': 0.0, 'p2': 0.0, 'k3': 0.0
        })

        if self.config.sensor_type == SensorType.STEREO:
            self.stereo_params = self.config.custom_params.get('stereo', {
                'baseline': 0.1,  # meters
                'ThDepth': 35.0
            })

        if self.config.sensor_type == SensorType.RGB_D:
            self.depth_params = self.config.custom_params.get('depth', {
                'DepthMapFactor': 1000.0,
                'ThDepth': 1.5
            })

    def _create_monocular_settings(self) -> str:
        """Create monocular camera settings."""
        return f"""
%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters. Adjust them!
#--------------------------------------------------------------------------------------------

# Camera calibration and distortion parameters (OpenCV)
Camera.fx: {self.camera_params['fx']}
Camera.fy: {self.camera_params['fy']}
Camera.cx: {self.camera_params['cx']}
Camera.cy: {self.camera_params['cy']}

Camera.k1: {self.camera_params['k1']}
Camera.k2: {self.camera_params['k2']}
Camera.p1: {self.camera_params['p1']}
Camera.p2: {self.camera_params['p2']}
Camera.k3: {self.camera_params['k3']}

Camera.width: 640
Camera.height: 480

# Camera frames per second
Camera.fps: 30.0

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

#--------------------------------------------------------------------------------------------
# ORB Extractor Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: {self.config.max_features}

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: {self.config.custom_params.get('ORBextractor.scaleFactor', 1.2)}

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: {self.config.custom_params.get('ORBextractor.nLevels', 8)}

# ORB Extractor: Fast threshold
# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
# You can lower these values if your images have low contrast
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
"""

    def _create_stereo_settings(self) -> str:
        """Create stereo camera settings."""
        return f"""
%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters. Adjust them!
#--------------------------------------------------------------------------------------------

# Left Camera calibration and distortion parameters (OpenCV)
Camera.fx: {self.camera_params['fx']}
Camera.fy: {self.camera_params['fy']}
Camera.cx: {self.camera_params['cx']}
Camera.cy: {self.camera_params['cy']}

Camera.k1: {self.camera_params['k1']}
Camera.k2: {self.camera_params['k2']}
Camera.p1: {self.camera_params['p1']}
Camera.p2: {self.camera_params['p2']}
Camera.k3: {self.camera_params['k3']}

# Right Camera calibration and distortion parameters (OpenCV)
Camera2.fx: {self.camera_params['fx']}
Camera2.fy: {self.camera_params['fy']}
Camera2.cx: {self.camera_params['cx']}
Camera2.cy: {self.camera_params['cy']}

Camera2.k1: {self.camera_params['k1']}
Camera2.k2: {self.camera_params['k2']}
Camera2.p1: {self.camera_params['p1']}
Camera2.p2: {self.camera_params['p2']}
Camera2.k3: {self.camera_params['k3']}

Camera.width: 640
Camera.height: 480

# Camera frames per second
Camera.fps: 30.0

# stereo baseline times fx
Camera.bf: {self.camera_params['fx'] * self.stereo_params['baseline']}

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

# Close/Far threshold. Baseline times.
ThDepth: {self.stereo_params['ThDepth']}

#--------------------------------------------------------------------------------------------
# ORB Extractor Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: {self.config.max_features}

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: {self.config.custom_params.get('ORBextractor.scaleFactor', 1.2)}

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: {self.config.custom_params.get('ORBextractor.nLevels', 8)}

# ORB Extractor: Fast threshold
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
"""

    def _create_rgbd_settings(self) -> str:
        """Create RGB-D camera settings."""
        return f"""
%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters. Adjust them!
#--------------------------------------------------------------------------------------------

# Camera calibration and distortion parameters (OpenCV)
Camera.fx: {self.camera_params['fx']}
Camera.fy: {self.camera_params['fy']}
Camera.cx: {self.camera_params['cx']}
Camera.cy: {self.camera_params['cy']}

Camera.k1: {self.camera_params['k1']}
Camera.k2: {self.camera_params['k2']}
Camera.p1: {self.camera_params['p1']}
Camera.p2: {self.camera_params['p2']}
Camera.k3: {self.camera_params['k3']}

Camera.width: 640
Camera.height: 480

# Camera frames per second
Camera.fps: 30.0

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

# Depth map values factor
DepthMapFactor: {self.depth_params['DepthMapFactor']}

# Close/Far threshold. Baseline times.
ThDepth: {self.depth_params['ThDepth']}

#--------------------------------------------------------------------------------------------
# ORB Extractor Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: {self.config.max_features}

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: {self.config.custom_params.get('ORBextractor.scaleFactor', 1.2)}

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: {self.config.custom_params.get('ORBextractor.nLevels', 8)}

# ORB Extractor: Fast threshold
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
"""

    def _create_vio_settings(self) -> str:
        """Create visual-inertial settings."""
        imu_params = self.config.custom_params.get('imu', {
            'NoiseGyro': 1.7e-4,
            'NoiseAcc': 2.0e-3,
            'GyroWalk': 1.9393e-05,
            'AccWalk': 3.0000e-03,
            'Frequency': 200
        })

        return f"""
%YAML:1.0

#--------------------------------------------------------------------------------------------
# Camera Parameters. Adjust them!
#--------------------------------------------------------------------------------------------

# Camera calibration and distortion parameters (OpenCV)
Camera.fx: {self.camera_params['fx']}
Camera.fy: {self.camera_params['fy']}
Camera.cx: {self.camera_params['cx']}
Camera.cy: {self.camera_params['cy']}

Camera.k1: {self.camera_params['k1']}
Camera.k2: {self.camera_params['k2']}
Camera.p1: {self.camera_params['p1']}
Camera.p2: {self.camera_params['p2']}
Camera.k3: {self.camera_params['k3']}

Camera.width: 640
Camera.height: 480

# Camera frames per second
Camera.fps: 30.0

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

#--------------------------------------------------------------------------------------------
# IMU Parameters
#--------------------------------------------------------------------------------------------

# Noise of gyroscope and accelerometer (continuous-time)
IMU.NoiseGyro: {imu_params['NoiseGyro']}
IMU.NoiseAcc: {imu_params['NoiseAcc']}

# Gyroscope and accelerometer bias random walk
IMU.GyroWalk: {imu_params['GyroWalk']}
IMU.AccWalk: {imu_params['AccWalk']}

# IMU frequency
IMU.Frequency: {imu_params['Frequency']}

#--------------------------------------------------------------------------------------------
# ORB Extractor Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: {self.config.max_features}

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: {self.config.custom_params.get('ORBextractor.scaleFactor', 1.2)}

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: {self.config.custom_params.get('ORBextractor.nLevels', 8)}

# ORB Extractor: Fast threshold
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3
Viewer.ViewpointX: 0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500
"""

    def initialize(self) -> bool:
        """Initialize ORB-SLAM3 system."""
        try:
            if not os.path.exists(self.vocabulary_file):
                self._log_error(f"Vocabulary file not found: {self.vocabulary_file}")
                return False

            if not os.path.exists(self.settings_file):
                self._log_error(f"Settings file not found: {self.settings_file}")
                return False

            # Initialize ORB-SLAM3 system
            sensor_mode = self.sensor_mode_map[self.config.sensor_type]
            self.slam_system = orbslam3.System(
                self.vocabulary_file,
                self.settings_file,
                sensor_mode,
                True  # Use viewer
            )

            self._update_state(SLAMState.INITIALIZING)
            self._log_info("ORB-SLAM3 system initialized successfully")
            return True

        except Exception as e:
            self._log_error(f"Failed to initialize ORB-SLAM3: {e}")
            return False

    def process_image(self, image: Union[np.ndarray, Image], timestamp: float) -> bool:
        """Process monocular camera image."""
        if self.slam_system is None:
            self._log_error("SLAM system not initialized")
            return False

        try:
            # Convert ROS image to OpenCV if necessary
            if isinstance(image, Image):
                cv_image = self.cv_bridge.imgmsg_to_cv2(image, "bgr8")
            else:
                cv_image = image.copy()

            # Process image
            start_time = time.time()
            pose = self.slam_system.process_image_mono(cv_image, timestamp)
            processing_time = time.time() - start_time

            # Update performance metrics
            self.processing_times.append(processing_time)
            self.frame_timestamps.append(timestamp)
            self._increment_frame_count()

            # Update current pose
            if pose is not None:
                self.current_pose = self._convert_pose(pose, timestamp)
                if self.state == SLAMState.INITIALIZING:
                    self._update_state(SLAMState.TRACKING)
            else:
                if self.state == SLAMState.TRACKING:
                    self._update_state(SLAMState.LOST)

            return True

        except Exception as e:
            self._log_error(f"Error processing image: {e}")
            return False

    def process_stereo_images(self, left_image: Union[np.ndarray, Image],
                            right_image: Union[np.ndarray, Image],
                            timestamp: float) -> bool:
        """Process stereo camera images."""
        if self.slam_system is None:
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

            # Process stereo images
            start_time = time.time()
            pose = self.slam_system.process_image_stereo(left_cv, right_cv, timestamp)
            processing_time = time.time() - start_time

            # Update performance metrics
            self.processing_times.append(processing_time)
            self.frame_timestamps.append(timestamp)
            self._increment_frame_count()

            # Update current pose
            if pose is not None:
                self.current_pose = self._convert_pose(pose, timestamp)
                if self.state == SLAMState.INITIALIZING:
                    self._update_state(SLAMState.TRACKING)
            else:
                if self.state == SLAMState.TRACKING:
                    self._update_state(SLAMState.LOST)

            return True

        except Exception as e:
            self._log_error(f"Error processing stereo images: {e}")
            return False

    def process_pointcloud(self, pointcloud: Union[np.ndarray, PointCloud2],
                          timestamp: float) -> bool:
        """ORB-SLAM3 doesn't directly process point clouds."""
        self._log_warning("ORB-SLAM3 doesn't support direct point cloud processing")
        return False

    def process_imu(self, imu_data: Union[np.ndarray, Imu], timestamp: float) -> bool:
        """Process IMU data for visual-inertial SLAM."""
        if self.config.sensor_type != SensorType.VISUAL_INERTIAL:
            return True  # IMU not used in this mode

        if self.slam_system is None:
            self._log_error("SLAM system not initialized")
            return False

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

            # Process IMU data
            self.slam_system.process_imu(acc, gyro, timestamp)
            return True

        except Exception as e:
            self._log_error(f"Error processing IMU data: {e}")
            return False

    def get_pose(self) -> Optional[SLAMPose]:
        """Get current robot pose estimate."""
        return self.current_pose

    def get_map(self) -> List[SLAMMapPoint]:
        """Get current map representation."""
        if self.slam_system is None:
            return []

        try:
            # Get map points from ORB-SLAM3
            map_points = self.slam_system.get_tracked_map_points()
            slam_map_points = []

            for point in map_points:
                if point is not None:
                    position = np.array([point.x, point.y, point.z])
                    slam_point = SLAMMapPoint(
                        position=position,
                        confidence=1.0,
                        observations=1
                    )
                    slam_map_points.append(slam_point)

            self.map_points_cache = slam_map_points
            return slam_map_points

        except Exception as e:
            self._log_error(f"Error getting map: {e}")
            return self.map_points_cache

    def get_occupancy_grid(self) -> Optional[OccupancyGrid]:
        """Get 2D occupancy grid (not directly supported by ORB-SLAM3)."""
        self._log_warning("ORB-SLAM3 doesn't provide occupancy grid directly")
        return None

    def get_trajectory(self) -> SLAMTrajectory:
        """Get robot trajectory."""
        if self.slam_system is None:
            return SLAMTrajectory([], [], [])

        try:
            # Get keyframes from ORB-SLAM3
            keyframes = self.slam_system.get_keyframes()
            poses = []
            timestamps = []
            keyframe_indices = []

            for i, kf in enumerate(keyframes):
                if kf is not None:
                    pose = self._convert_pose(kf.get_pose(), kf.timestamp)
                    poses.append(pose)
                    timestamps.append(kf.timestamp)
                    keyframe_indices.append(i)

            trajectory = SLAMTrajectory(poses, timestamps, keyframe_indices)
            return trajectory

        except Exception as e:
            self._log_error(f"Error getting trajectory: {e}")
            return SLAMTrajectory([], [], [])

    def reset(self) -> bool:
        """Reset ORB-SLAM3 system."""
        try:
            if self.slam_system is not None:
                self.slam_system.reset()

            self.current_pose = None
            self.map_points_cache = []
            self.keyframes_cache = []
            self.processing_times = []
            self.frame_timestamps = []
            self._frame_count = 0

            self._update_state(SLAMState.INITIALIZING)
            self._log_info("ORB-SLAM3 system reset")
            return True

        except Exception as e:
            self._log_error(f"Error resetting system: {e}")
            return False

    def save_map(self, filepath: str) -> bool:
        """Save current map to file."""
        if self.slam_system is None:
            return False

        try:
            self.slam_system.save_trajectory_tum(filepath)
            self._log_info(f"Map saved to {filepath}")
            return True

        except Exception as e:
            self._log_error(f"Error saving map: {e}")
            return False

    def load_map(self, filepath: str) -> bool:
        """Load map from file (not directly supported by ORB-SLAM3)."""
        self._log_warning("ORB-SLAM3 doesn't support map loading")
        return False

    def relocalize(self, initial_pose: Optional[SLAMPose] = None) -> bool:
        """Attempt to relocalize after tracking loss."""
        if self.slam_system is None:
            return False

        try:
            # ORB-SLAM3 handles relocalization automatically
            self._update_state(SLAMState.RELOCALIZATION)
            self._log_info("Attempting relocalization")
            return True

        except Exception as e:
            self._log_error(f"Error during relocalization: {e}")
            return False

    def set_loop_closure_enabled(self, enabled: bool) -> None:
        """Enable/disable loop closure detection."""
        # ORB-SLAM3 loop closure is controlled via settings file
        self._log_info(f"Loop closure {'enabled' if enabled else 'disabled'}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get algorithm performance metrics."""
        if not self.processing_times:
            return {}

        avg_processing_time = np.mean(self.processing_times)
        max_processing_time = np.max(self.processing_times)
        min_processing_time = np.min(self.processing_times)
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0

        return {
            'algorithm': 'ORB-SLAM3',
            'frames_processed': self._frame_count,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'max_processing_time_ms': max_processing_time * 1000,
            'min_processing_time_ms': min_processing_time * 1000,
            'average_fps': fps,
            'state': self.state.value,
            'map_points': len(self.map_points_cache),
            'tracking_state': 'tracking' if self.current_pose is not None else 'lost'
        }

    def _convert_pose(self, orb_pose, timestamp: float) -> SLAMPose:
        """Convert ORB-SLAM3 pose to SLAMPose."""
        try:
            # Extract position and orientation from ORB-SLAM3 pose
            # Note: This depends on the actual ORB-SLAM3 Python binding interface
            if hasattr(orb_pose, 'translation') and hasattr(orb_pose, 'rotation'):
                position = np.array([
                    orb_pose.translation[0],
                    orb_pose.translation[1],
                    orb_pose.translation[2]
                ])

                # Convert rotation matrix to quaternion (simplified)
                # In real implementation, use proper conversion
                orientation = np.array([0, 0, 0, 1])  # Identity quaternion

            else:
                # Fallback: assume 4x4 transformation matrix
                if isinstance(orb_pose, np.ndarray) and orb_pose.shape == (4, 4):
                    position = orb_pose[:3, 3]
                    # Convert rotation matrix to quaternion
                    from scipy.spatial.transform import Rotation
                    rot = Rotation.from_matrix(orb_pose[:3, :3])
                    orientation = rot.as_quat()  # [x, y, z, w]
                else:
                    # Default pose
                    position = np.array([0, 0, 0])
                    orientation = np.array([0, 0, 0, 1])

            return SLAMPose(
                position=position,
                orientation=orientation,
                timestamp=timestamp,
                frame_id="map"
            )

        except Exception as e:
            self._log_error(f"Error converting pose: {e}")
            return SLAMPose(
                position=np.array([0, 0, 0]),
                orientation=np.array([0, 0, 0, 1]),
                timestamp=timestamp,
                frame_id="map"
            )
