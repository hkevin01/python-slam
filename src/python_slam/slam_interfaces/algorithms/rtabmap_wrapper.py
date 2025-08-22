#!/usr/bin/env python3
"""
RTAB-Map SLAM Wrapper Implementation

Wrapper for the RTAB-Map Real-Time Appearance-Based Mapping algorithm.
Supports RGB-D, stereo, and visual SLAM with loop closure detection.
"""

import os
import time
import numpy as np
from typing import Optional, List, Dict, Any, Union
import cv2
import tempfile

from ..slam_interface import (
    SLAMInterface, SLAMConfiguration, SLAMPose, SLAMMapPoint,
    SLAMTrajectory, SLAMState, SensorType
)
from sensor_msgs.msg import Image, PointCloud2, Imu
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge

# RTAB-Map imports
try:
    import rtabmap as rtab
    RTABMAP_AVAILABLE = True
except ImportError:
    RTABMAP_AVAILABLE = False

# Point cloud processing
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


class RTABMapWrapper(SLAMInterface):
    """
    RTAB-Map SLAM algorithm wrapper.

    Provides Real-Time Appearance-Based Mapping with loop closure detection
    for RGB-D, stereo, and monocular visual SLAM.
    """

    def __init__(self, config: SLAMConfiguration):
        """Initialize RTAB-Map wrapper."""
        super().__init__(config)

        if not RTABMAP_AVAILABLE:
            raise ImportError("RTAB-Map not available. Install with: pip install rtabmap-python")

        self.cv_bridge = CvBridge()
        self.rtabmap = None

        # RTAB-Map specific parameters
        self.rtab_params = {}
        self.session_id = 0

        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self._setup_camera_parameters()

        # Data storage
        self.trajectory_poses = []
        self.trajectory_timestamps = []
        self.map_points_3d = []
        self.rgb_images = []
        self.depth_images = []

        # Performance metrics
        self.processing_times = []
        self.loop_closures = 0
        self.memory_usage = []

        # Working directory for RTAB-Map database
        self.work_dir = tempfile.mkdtemp()
        self.database_path = os.path.join(self.work_dir, "rtabmap.db")

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

    def _setup_rtabmap_parameters(self):
        """Configure RTAB-Map parameters based on configuration."""
        params = {
            # Memory management
            "Mem/RehearsalSimilarity": "0.30",
            "Mem/RecentWmRatio": "0.2",
            "Mem/STMSize": "30",
            "Mem/BadSignaturesIgnored": "false",

            # RGBD parameters for RGB-D mode
            "RGBD/Enabled": "true" if self.config.sensor_type == SensorType.RGBD else "false",
            "RGBD/ProximityByTime": "false",
            "RGBD/ProximityBySpace": "true",
            "RGBD/AngularUpdate": "0.1",
            "RGBD/LinearUpdate": "0.1",

            # Visual odometry
            "Odom/Strategy": "1",  # 0=GFTT, 1=SURF, 2=SIFT, 3=FAST+BRIEF, 4=FAST+FREAK, 5=GFTT+FREAK, 6=GFTT+BRIEF, 7=GFTT+ORB, 8=BRISK
            "Odom/FeatureType": "6",  # GFTT

            # Loop closure detection
            "Kp/DetectorStrategy": "6",  # GFTT
            "Kp/MaxFeatures": str(self.config.max_features),
            "Kp/RoiRatios": "0.03 0.03 0.04 0.04",

            # Bundle adjustment
            "Vis/EstimationType": "1",  # 0=3D->3D, 1=3D->2D (PnP), 2=2D->2D (Epipolar Geometry)
            "Vis/MinInliers": "15",
            "Vis/RefineIterations": "5",

            # Graph optimization
            "Optimizer/Strategy": "1",  # 0=TORO, 1=g2o, 2=GTSAM
            "Optimizer/Iterations": "20",
            "Optimizer/Slam2D": "false",

            # Database
            "DbSqlite3/InMemory": "false",
            "DatabasePath": self.database_path,
        }

        # Sensor-specific parameters
        if self.config.sensor_type == SensorType.STEREO:
            params.update({
                "RGBD/Enabled": "false",
                "Stereo/Enabled": "true",
                "Stereo/OpticalFlow": "true",
                "Stereo/MaxDisparity": "128.0",
            })
        elif self.config.sensor_type == SensorType.MONOCULAR:
            params.update({
                "RGBD/Enabled": "false",
                "Stereo/Enabled": "false",
                "Vis/EstimationType": "2",  # Epipolar geometry for monocular
            })

        # Custom parameters from configuration
        custom_rtab_params = self.config.custom_params.get('rtabmap', {})
        params.update(custom_rtab_params)

        self.rtab_params = params

    def initialize(self) -> bool:
        """Initialize RTAB-Map system."""
        try:
            self._setup_rtabmap_parameters()

            # Initialize RTAB-Map with parameters
            self.rtabmap = rtab.Rtabmap()

            # Set parameters
            for key, value in self.rtab_params.items():
                self.rtabmap.setParameter(key, value)

            # Initialize with database
            self.rtabmap.init(self.rtab_params, self.database_path)

            self._update_state(SLAMState.INITIALIZING)
            self._log_info("RTAB-Map initialized successfully")
            return True

        except Exception as e:
            self._log_error(f"Failed to initialize RTAB-Map: {e}")
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

            # Process with RTAB-Map
            success = self._process_rtabmap_data(
                rgb_image=cv_image,
                timestamp=timestamp
            )

            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self._increment_frame_count()

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
                left_cv = self.cv_bridge.imgmsg_to_cv2(left_image, "bgr8")
            else:
                left_cv = left_image.copy()

            if isinstance(right_image, Image):
                right_cv = self.cv_bridge.imgmsg_to_cv2(right_image, "bgr8")
            else:
                right_cv = right_image.copy()

            start_time = time.time()

            # Process with RTAB-Map stereo
            success = self._process_rtabmap_data(
                rgb_image=left_cv,
                depth_image=right_cv,  # Right image used as "depth" for stereo
                timestamp=timestamp,
                is_stereo=True
            )

            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self._increment_frame_count()

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
                rgb_cv = self.cv_bridge.imgmsg_to_cv2(rgb_image, "bgr8")
            else:
                rgb_cv = rgb_image.copy()

            if isinstance(depth_image, Image):
                depth_cv = self.cv_bridge.imgmsg_to_cv2(depth_image, "passthrough")
            else:
                depth_cv = depth_image.copy()

            start_time = time.time()

            # Process with RTAB-Map RGB-D
            success = self._process_rtabmap_data(
                rgb_image=rgb_cv,
                depth_image=depth_cv,
                timestamp=timestamp,
                is_stereo=False
            )

            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self._increment_frame_count()

            return success

        except Exception as e:
            self._log_error(f"Error processing RGB-D images: {e}")
            return False

    def process_pointcloud(self, pointcloud: Union[np.ndarray, PointCloud2],
                          timestamp: float) -> bool:
        """Process point cloud data (convert to RGB-D if possible)."""
        if not OPEN3D_AVAILABLE:
            self._log_warning("Open3D not available for point cloud processing")
            return False

        try:
            # Convert PointCloud2 to numpy array if necessary
            if isinstance(pointcloud, PointCloud2):
                # This is a simplified conversion - you'd need proper ROS point cloud tools
                points = self._pointcloud2_to_array(pointcloud)
            else:
                points = pointcloud

            # Project point cloud to RGB-D images
            rgb_image, depth_image = self._project_pointcloud_to_rgbd(points)

            if rgb_image is not None and depth_image is not None:
                return self.process_rgbd_images(rgb_image, depth_image, timestamp)

            return False

        except Exception as e:
            self._log_error(f"Error processing point cloud: {e}")
            return False

    def process_imu(self, imu_data: Union[np.ndarray, Imu], timestamp: float) -> bool:
        """Process IMU data (RTAB-Map can use IMU for odometry)."""
        try:
            # RTAB-Map can incorporate IMU data for better odometry
            # This is a simplified implementation
            return True

        except Exception as e:
            self._log_error(f"Error processing IMU data: {e}")
            return False

    def get_pose(self) -> Optional[SLAMPose]:
        """Get current robot pose estimate."""
        return self.current_pose

    def get_map(self) -> List[SLAMMapPoint]:
        """Get current map representation."""
        if not self.rtabmap:
            return []

        try:
            # Get 3D map from RTAB-Map
            map_data = self.rtabmap.getMap()
            map_points = []

            # Convert RTAB-Map map data to SLAMMapPoint format
            for point in map_data.points:
                map_point = SLAMMapPoint(
                    position=np.array([point.x, point.y, point.z]),
                    confidence=1.0,
                    observations=1
                )
                map_points.append(map_point)

            return map_points

        except Exception as e:
            self._log_error(f"Error getting map: {e}")
            return []

    def get_occupancy_grid(self) -> Optional[OccupancyGrid]:
        """Get 2D occupancy grid representation."""
        if not self.rtabmap:
            return None

        try:
            # Get occupancy grid from RTAB-Map
            grid_map = self.rtabmap.getOccupancyGrid()

            # Convert to ROS OccupancyGrid message
            occupancy_grid = OccupancyGrid()
            occupancy_grid.header.frame_id = "map"
            occupancy_grid.header.stamp.sec = int(time.time())

            # Set grid parameters
            occupancy_grid.info.resolution = grid_map.resolution
            occupancy_grid.info.width = grid_map.width
            occupancy_grid.info.height = grid_map.height

            # Set origin
            occupancy_grid.info.origin.position.x = grid_map.origin_x
            occupancy_grid.info.origin.position.y = grid_map.origin_y
            occupancy_grid.info.origin.position.z = 0.0

            # Convert grid data
            occupancy_grid.data = [int(cell * 100) for cell in grid_map.data]

            return occupancy_grid

        except Exception as e:
            self._log_error(f"Error getting occupancy grid: {e}")
            return None

    def get_trajectory(self) -> SLAMTrajectory:
        """Get robot trajectory."""
        if not self.rtabmap:
            return SLAMTrajectory(poses=[], timestamps=[], keyframe_indices=[])

        try:
            # Get trajectory from RTAB-Map
            trajectory = self.rtabmap.getTrajectory()

            poses = []
            timestamps = []

            for i, pose_data in enumerate(trajectory):
                slam_pose = self._rtabmap_pose_to_slam_pose(pose_data)
                poses.append(slam_pose)
                timestamps.append(slam_pose.timestamp)

            # Keyframes are typically every 10th pose
            keyframe_indices = list(range(0, len(poses), 10))

            return SLAMTrajectory(
                poses=poses,
                timestamps=timestamps,
                keyframe_indices=keyframe_indices
            )

        except Exception as e:
            self._log_error(f"Error getting trajectory: {e}")
            return SLAMTrajectory(poses=[], timestamps=[], keyframe_indices=[])

    def reset(self) -> bool:
        """Reset RTAB-Map system."""
        try:
            if self.rtabmap:
                self.rtabmap.resetMemory()

            self.current_pose = None
            self.trajectory_poses = []
            self.trajectory_timestamps = []
            self.map_points_3d = []
            self.processing_times = []
            self.loop_closures = 0
            self.memory_usage = []
            self._frame_count = 0

            self._update_state(SLAMState.INITIALIZING)
            self._log_info("RTAB-Map system reset")
            return True

        except Exception as e:
            self._log_error(f"Error resetting system: {e}")
            return False

    def save_map(self, filepath: str) -> bool:
        """Save current map to file."""
        try:
            if not self.rtabmap:
                return False

            # Export RTAB-Map database
            self.rtabmap.exportPoses(f"{filepath}_poses.txt")
            self.rtabmap.exportClouds(f"{filepath}_clouds.ply")

            # Copy database file
            import shutil
            shutil.copy2(self.database_path, f"{filepath}.db")

            self._log_info(f"Map saved to {filepath}")
            return True

        except Exception as e:
            self._log_error(f"Error saving map: {e}")
            return False

    def load_map(self, filepath: str) -> bool:
        """Load map from file."""
        try:
            # Copy database file
            import shutil
            if os.path.exists(f"{filepath}.db"):
                shutil.copy2(f"{filepath}.db", self.database_path)

                # Reinitialize with loaded database
                self.rtabmap.close()
                self.rtabmap.init(self.rtab_params, self.database_path)

                self._log_info(f"Map loaded from {filepath}")
                return True

            return False

        except Exception as e:
            self._log_error(f"Error loading map: {e}")
            return False

    def relocalize(self, initial_pose: Optional[SLAMPose] = None) -> bool:
        """Attempt to relocalize after tracking loss."""
        try:
            self._update_state(SLAMState.RELOCALIZATION)

            if self.rtabmap:
                # RTAB-Map has built-in relocalization
                relocalized = self.rtabmap.triggerNewMap()

                if relocalized:
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
        if self.rtabmap:
            self.rtabmap.setParameter("Mem/IncrementalMemory", "true" if enabled else "false")
        self._log_info(f"Loop closure {'enabled' if enabled else 'disabled'}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get algorithm performance metrics."""
        if not self.processing_times:
            return {'algorithm': 'RTAB-Map', 'frames_processed': 0}

        avg_processing_time = np.mean(self.processing_times)
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0

        metrics = {
            'algorithm': 'RTAB-Map',
            'frames_processed': self._frame_count,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'max_processing_time_ms': np.max(self.processing_times) * 1000,
            'min_processing_time_ms': np.min(self.processing_times) * 1000,
            'average_fps': fps,
            'state': self.state.value,
            'loop_closures': self.loop_closures,
            'tracking_state': 'tracking' if self.current_pose is not None else 'lost'
        }

        # Add RTAB-Map specific metrics
        if self.rtabmap:
            try:
                stats = self.rtabmap.getStatistics()
                metrics.update({
                    'total_nodes': stats.get('Nodes', 0),
                    'total_words': stats.get('Words', 0),
                    'database_memory_mb': stats.get('Database memory used', 0) / 1024.0,
                    'working_memory_size': stats.get('Working memory size', 0),
                })
            except:
                pass

        return metrics

    # Helper methods
    def _process_rtabmap_data(self, rgb_image, timestamp, depth_image=None, is_stereo=False):
        """Process data with RTAB-Map."""
        try:
            # Create RTAB-Map sensor data
            if depth_image is not None:
                if is_stereo:
                    # Stereo processing
                    data = rtab.SensorData(
                        rgb_image, depth_image, self.camera_matrix, timestamp
                    )
                else:
                    # RGB-D processing
                    data = rtab.SensorData(
                        rgb_image, depth_image, self.camera_matrix, timestamp
                    )
            else:
                # Monocular processing
                data = rtab.SensorData(rgb_image, self.camera_matrix, timestamp)

            # Process with RTAB-Map
            pose = self.rtabmap.process(data)

            if pose is not None:
                # Convert to SLAMPose
                self.current_pose = self._rtabmap_pose_to_slam_pose(pose, timestamp)

                # Add to trajectory
                self.trajectory_poses.append(self.current_pose)
                self.trajectory_timestamps.append(timestamp)

                # Update state
                if self.state == SLAMState.INITIALIZING:
                    self._update_state(SLAMState.TRACKING)

                # Check for loop closure
                if self.rtabmap.getLoopClosureId() > 0:
                    self.loop_closures += 1
                    self._log_info(f"Loop closure detected! Total: {self.loop_closures}")

                return True
            else:
                if self.state == SLAMState.TRACKING:
                    self._update_state(SLAMState.LOST)
                return False

        except Exception as e:
            self._log_error(f"Error processing RTAB-Map data: {e}")
            return False

    def _rtabmap_pose_to_slam_pose(self, rtab_pose, timestamp=None):
        """Convert RTAB-Map pose to SLAMPose."""
        try:
            # Extract position and orientation from RTAB-Map pose
            position = np.array([rtab_pose.x(), rtab_pose.y(), rtab_pose.z()])

            # Convert rotation matrix to quaternion
            from scipy.spatial.transform import Rotation
            rot_matrix = rtab_pose.rotationMatrix()
            rot = Rotation.from_matrix(rot_matrix)
            orientation = rot.as_quat()  # [x, y, z, w]

            return SLAMPose(
                position=position,
                orientation=orientation,
                timestamp=timestamp or time.time(),
                frame_id="map"
            )

        except Exception as e:
            self._log_warning(f"Error converting RTAB-Map pose: {e}")
            return None

    def _pointcloud2_to_array(self, cloud_msg):
        """Convert PointCloud2 message to numpy array."""
        # This is a simplified implementation
        # In practice, you'd use ros_numpy or similar
        points = []
        # Extract points from PointCloud2 message
        # This would need proper implementation based on point cloud format
        return np.array(points)

    def _project_pointcloud_to_rgbd(self, points):
        """Project 3D point cloud to RGB and depth images."""
        try:
            if not OPEN3D_AVAILABLE:
                return None, None

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])

            if points.shape[1] > 3:
                # Has color information
                pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)

            # Create virtual camera and project
            # This is a simplified projection - would need proper camera setup
            width, height = 640, 480

            # Create depth image
            depth_image = np.zeros((height, width), dtype=np.float32)
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

            # Simple projection using camera matrix
            for point in points:
                if len(point) >= 3:
                    x, y, z = point[:3]
                    if z > 0:
                        u = int(self.camera_matrix[0, 0] * x / z + self.camera_matrix[0, 2])
                        v = int(self.camera_matrix[1, 1] * y / z + self.camera_matrix[1, 2])

                        if 0 <= u < width and 0 <= v < height:
                            depth_image[v, u] = z
                            if len(point) >= 6:
                                rgb_image[v, u] = point[3:6]

            return rgb_image, depth_image

        except Exception as e:
            self._log_warning(f"Point cloud projection failed: {e}")
            return None, None

    def __del__(self):
        """Cleanup RTAB-Map resources."""
        try:
            if self.rtabmap:
                self.rtabmap.close()

            # Clean up working directory
            import shutil
            if os.path.exists(self.work_dir):
                shutil.rmtree(self.work_dir)

        except:
            pass
