#!/usr/bin/env python3
"""
Abstract SLAM Interface for Multi-Algorithm Framework

This module provides a unified interface for different SLAM algorithms,
enabling hot-swappable algorithm selection and standardized integration
with ROS2 systems.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from geometry_msgs.msg import Pose, Transform
from sensor_msgs.msg import Image, PointCloud2, Imu
from nav_msgs.msg import OccupancyGrid
import cv2


class SensorType(Enum):
    """Supported sensor types for SLAM algorithms."""
    MONOCULAR = "monocular"
    STEREO = "stereo"
    RGB_D = "rgb_d"
    LIDAR_2D = "lidar_2d"
    LIDAR_3D = "lidar_3d"
    VISUAL_INERTIAL = "visual_inertial"


class SLAMState(Enum):
    """SLAM algorithm states."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    LOST = "lost"
    RELOCALIZATION = "relocalization"
    STOPPED = "stopped"


@dataclass
class SLAMConfiguration:
    """Configuration parameters for SLAM algorithms."""
    algorithm_name: str
    sensor_type: SensorType
    max_features: int = 1000
    quality_level: float = 0.01
    min_distance: float = 10.0
    enable_loop_closure: bool = True
    enable_mapping: bool = True
    enable_relocalization: bool = True
    map_resolution: float = 0.05
    keyframe_distance: float = 1.0
    config_file: Optional[str] = None
    vocabulary_file: Optional[str] = None
    custom_params: Dict[str, Any] = None

    def __post_init__(self):
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class SLAMPose:
    """SLAM pose representation with uncertainty."""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # Quaternion [x, y, z, w]
    covariance: Optional[np.ndarray] = None
    timestamp: Optional[float] = None
    frame_id: str = "map"


@dataclass
class SLAMMapPoint:
    """3D map point representation."""
    position: np.ndarray  # [x, y, z]
    descriptor: Optional[np.ndarray] = None
    confidence: float = 1.0
    observations: int = 0


@dataclass
class SLAMTrajectory:
    """Trajectory representation."""
    poses: List[SLAMPose]
    timestamps: List[float]
    keyframe_indices: List[int]


class SLAMInterface(ABC):
    """
    Abstract interface for SLAM algorithms.

    This interface defines the standard methods that all SLAM implementations
    must provide, enabling seamless algorithm switching and unified integration.
    """

    def __init__(self, config: SLAMConfiguration):
        """
        Initialize SLAM algorithm.

        Args:
            config: SLAM configuration parameters
        """
        self.config = config
        self.state = SLAMState.UNINITIALIZED
        self.current_pose = None
        self.map_points = []
        self.trajectory = SLAMTrajectory([], [], [])
        self._frame_count = 0
        self._logger = None

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the SLAM algorithm.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def process_image(self, image: Union[np.ndarray, Image], timestamp: float) -> bool:
        """
        Process camera image for visual SLAM.

        Args:
            image: Input image (numpy array or ROS Image message)
            timestamp: Image timestamp

        Returns:
            bool: True if processing successful, False otherwise
        """
        pass

    @abstractmethod
    def process_stereo_images(self,
                            left_image: Union[np.ndarray, Image],
                            right_image: Union[np.ndarray, Image],
                            timestamp: float) -> bool:
        """
        Process stereo camera images.

        Args:
            left_image: Left camera image
            right_image: Right camera image
            timestamp: Image timestamp

        Returns:
            bool: True if processing successful, False otherwise
        """
        pass

    @abstractmethod
    def process_pointcloud(self, pointcloud: Union[np.ndarray, PointCloud2],
                          timestamp: float) -> bool:
        """
        Process LiDAR point cloud data.

        Args:
            pointcloud: Point cloud data
            timestamp: Point cloud timestamp

        Returns:
            bool: True if processing successful, False otherwise
        """
        pass

    @abstractmethod
    def process_imu(self, imu_data: Union[np.ndarray, Imu], timestamp: float) -> bool:
        """
        Process IMU data for visual-inertial SLAM.

        Args:
            imu_data: IMU measurements (acceleration and angular velocity)
            timestamp: IMU timestamp

        Returns:
            bool: True if processing successful, False otherwise
        """
        pass

    @abstractmethod
    def get_pose(self) -> Optional[SLAMPose]:
        """
        Get current robot pose estimate.

        Returns:
            SLAMPose: Current pose with uncertainty, None if not available
        """
        pass

    @abstractmethod
    def get_map(self) -> List[SLAMMapPoint]:
        """
        Get current map representation.

        Returns:
            List[SLAMMapPoint]: List of 3D map points
        """
        pass

    @abstractmethod
    def get_occupancy_grid(self) -> Optional[OccupancyGrid]:
        """
        Get 2D occupancy grid representation of the map.

        Returns:
            OccupancyGrid: ROS occupancy grid message, None if not available
        """
        pass

    @abstractmethod
    def get_trajectory(self) -> SLAMTrajectory:
        """
        Get robot trajectory.

        Returns:
            SLAMTrajectory: Complete trajectory with keyframes
        """
        pass

    @abstractmethod
    def reset(self) -> bool:
        """
        Reset SLAM algorithm state.

        Returns:
            bool: True if reset successful, False otherwise
        """
        pass

    @abstractmethod
    def save_map(self, filepath: str) -> bool:
        """
        Save current map to file.

        Args:
            filepath: Path to save map file

        Returns:
            bool: True if save successful, False otherwise
        """
        pass

    @abstractmethod
    def load_map(self, filepath: str) -> bool:
        """
        Load map from file.

        Args:
            filepath: Path to map file

        Returns:
            bool: True if load successful, False otherwise
        """
        pass

    @abstractmethod
    def relocalize(self, initial_pose: Optional[SLAMPose] = None) -> bool:
        """
        Attempt to relocalize after tracking loss.

        Args:
            initial_pose: Optional initial pose hint

        Returns:
            bool: True if relocalization successful, False otherwise
        """
        pass

    @abstractmethod
    def set_loop_closure_enabled(self, enabled: bool) -> None:
        """
        Enable/disable loop closure detection.

        Args:
            enabled: True to enable loop closure, False to disable
        """
        pass

    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get algorithm performance metrics.

        Returns:
            Dict[str, Any]: Performance metrics (timing, accuracy, etc.)
        """
        pass

    # Common utility methods (implemented in base class)
    def get_state(self) -> SLAMState:
        """Get current SLAM state."""
        return self.state

    def get_frame_count(self) -> int:
        """Get number of processed frames."""
        return self._frame_count

    def get_config(self) -> SLAMConfiguration:
        """Get SLAM configuration."""
        return self.config

    def is_initialized(self) -> bool:
        """Check if SLAM is initialized."""
        return self.state != SLAMState.UNINITIALIZED

    def is_tracking(self) -> bool:
        """Check if SLAM is currently tracking."""
        return self.state == SLAMState.TRACKING

    def is_lost(self) -> bool:
        """Check if tracking is lost."""
        return self.state == SLAMState.LOST

    def set_logger(self, logger) -> None:
        """Set logger for debugging and monitoring."""
        self._logger = logger

    def _log_info(self, message: str) -> None:
        """Log info message."""
        if self._logger:
            self._logger.info(f"[{self.config.algorithm_name}] {message}")

    def _log_warning(self, message: str) -> None:
        """Log warning message."""
        if self._logger:
            self._logger.warning(f"[{self.config.algorithm_name}] {message}")

    def _log_error(self, message: str) -> None:
        """Log error message."""
        if self._logger:
            self._logger.error(f"[{self.config.algorithm_name}] {message}")

    def _increment_frame_count(self) -> None:
        """Increment frame counter."""
        self._frame_count += 1

    def _update_state(self, new_state: SLAMState) -> None:
        """Update SLAM state."""
        if self.state != new_state:
            self._log_info(f"State changed: {self.state.value} -> {new_state.value}")
            self.state = new_state


class SLAMAlgorithmInfo:
    """Information about available SLAM algorithms."""

    def __init__(self, name: str, supported_sensors: List[SensorType],
                 description: str, performance_rating: int = 0):
        self.name = name
        self.supported_sensors = supported_sensors
        self.description = description
        self.performance_rating = performance_rating  # 1-5 rating
        self.requirements = []
        self.config_template = None

    def supports_sensor(self, sensor_type: SensorType) -> bool:
        """Check if algorithm supports given sensor type."""
        return sensor_type in self.supported_sensors

    def is_available(self) -> bool:
        """Check if algorithm dependencies are available."""
        # This will be implemented by specific algorithm classes
        return True


# Utility functions for SLAM interface
def pose_to_transform_matrix(pose: SLAMPose) -> np.ndarray:
    """Convert SLAMPose to 4x4 transformation matrix."""
    # Convert quaternion to rotation matrix
    q = pose.orientation
    rot_matrix = np.array([
        [1 - 2*(q[1]**2 + q[2]**2), 2*(q[0]*q[1] - q[2]*q[3]), 2*(q[0]*q[2] + q[1]*q[3])],
        [2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[0]**2 + q[2]**2), 2*(q[1]*q[2] - q[0]*q[3])],
        [2*(q[0]*q[2] - q[1]*q[3]), 2*(q[1]*q[2] + q[0]*q[3]), 1 - 2*(q[0]**2 + q[1]**2)]
    ])

    # Create 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = pose.position

    return transform


def transform_matrix_to_pose(transform: np.ndarray, frame_id: str = "map") -> SLAMPose:
    """Convert 4x4 transformation matrix to SLAMPose."""
    position = transform[:3, 3]

    # Convert rotation matrix to quaternion
    rot_matrix = transform[:3, :3]
    trace = np.trace(rot_matrix)

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (rot_matrix[2, 1] - rot_matrix[1, 2]) / S
        y = (rot_matrix[0, 2] - rot_matrix[2, 0]) / S
        z = (rot_matrix[1, 0] - rot_matrix[0, 1]) / S
    else:
        if rot_matrix[0, 0] > rot_matrix[1, 1] and rot_matrix[0, 0] > rot_matrix[2, 2]:
            S = np.sqrt(1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2]) * 2
            w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / S
            x = 0.25 * S
            y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / S
            z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / S
        elif rot_matrix[1, 1] > rot_matrix[2, 2]:
            S = np.sqrt(1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2]) * 2
            w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / S
            x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / S
            y = 0.25 * S
            z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / S
        else:
            S = np.sqrt(1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1]) * 2
            w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / S
            x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / S
            y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / S
            z = 0.25 * S

    orientation = np.array([x, y, z, w])

    return SLAMPose(position=position, orientation=orientation, frame_id=frame_id)
