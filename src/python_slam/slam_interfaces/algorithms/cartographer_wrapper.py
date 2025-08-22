#!/usr/bin/env python3
"""
Cartographer SLAM Wrapper Implementation

Wrapper for Google's Cartographer algorithm providing real-time SLAM
with 2D and 3D mapping capabilities.
"""

import os
import time
import numpy as np
from typing import Optional, List, Dict, Any, Union
import tempfile
import subprocess

from ..slam_interface import (
    SLAMInterface, SLAMConfiguration, SLAMPose, SLAMMapPoint,
    SLAMTrajectory, SLAMState, SensorType
)
from sensor_msgs.msg import Image, PointCloud2, Imu, LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge

# Cartographer imports (would need cartographer_ros python bindings)
try:
    import cartographer_ros_msgs.msg as carto_msgs
    import cartographer_ros_msgs.srv as carto_srvs
    CARTOGRAPHER_AVAILABLE = True
except ImportError:
    CARTOGRAPHER_AVAILABLE = False

# For point cloud processing
try:
    import sensor_msgs.point_cloud2 as pc2
    POINTCLOUD_UTILS_AVAILABLE = True
except ImportError:
    POINTCLOUD_UTILS_AVAILABLE = False


class CartographerWrapper(SLAMInterface):
    """
    Cartographer SLAM algorithm wrapper.

    Provides Google's Cartographer real-time SLAM with support for
    2D and 3D mapping using laser, RGB-D, and IMU data.
    """

    def __init__(self, config: SLAMConfiguration):
        """Initialize Cartographer wrapper."""
        super().__init__(config)

        if not CARTOGRAPHER_AVAILABLE:
            self._log_warning("Cartographer ROS bindings not available")

        self.cv_bridge = CvBridge()

        # Cartographer configuration
        self.carto_config = {}
        self.config_file_path = None
        self.lua_config_path = None

        # Mapping mode (2D or 3D)
        self.mapping_mode = config.custom_params.get('mapping_mode', '2D')

        # Trajectory and map data
        self.trajectory_poses = []
        self.trajectory_timestamps = []
        self.map_points_3d = []
        self.submap_data = []

        # Performance tracking
        self.processing_times = []
        self.submap_count = 0
        self.constraint_count = 0

        # Sensor data buffers
        self.laser_scans = []
        self.imu_buffer = []
        self.point_clouds = []

        # Working directory
        self.work_dir = tempfile.mkdtemp()

        # Cartographer state
        self.trajectory_id = 0
        self.last_submap_id = -1

    def _create_cartographer_config(self):
        """Create Cartographer Lua configuration file."""
        try:
            # Generate Lua configuration based on sensor type and parameters
            if self.mapping_mode == '2D':
                config_content = self._generate_2d_config()
            else:
                config_content = self._generate_3d_config()

            # Write config file
            config_filename = f"cartographer_config_{self.mapping_mode.lower()}.lua"
            self.lua_config_path = os.path.join(self.work_dir, config_filename)

            with open(self.lua_config_path, 'w') as f:
                f.write(config_content)

            self._log_info(f"Cartographer config created: {self.lua_config_path}")
            return True

        except Exception as e:
            self._log_error(f"Failed to create Cartographer config: {e}")
            return False

    def _generate_2d_config(self):
        """Generate 2D SLAM configuration."""
        config = f'''
include "map_builder.lua"
include "trajectory_builder.lua"

options = {{
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "base_link",
  published_frame = "base_link",
  odom_frame = "odom",
  provide_odom_frame = true,
  publish_frame_projected_to_2d = false,
  use_pose_extrapolator = true,
  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 1,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = {1 if self.config.sensor_type == SensorType.RGBD else 0},
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}}

MAP_BUILDER.use_trajectory_builder_2d = true

TRAJECTORY_BUILDER_2D.submaps.num_range_data = 35
TRAJECTORY_BUILDER_2D.min_range = 0.3
TRAJECTORY_BUILDER_2D.max_range = {self.config.max_range}
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 1.
TRAJECTORY_BUILDER_2D.use_imu_data = {str(self.config.sensor_type == SensorType.VISUAL_INERTIAL).lower()}
TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.linear_search_window = 0.1
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.translation_delta_cost_weight = 10.
TRAJECTORY_BUILDER_2D.real_time_correlative_scan_matcher.rotation_delta_cost_weight = 1e-1

POSE_GRAPH.optimization_problem.huber_scale = 5e2
POSE_GRAPH.optimize_every_n_nodes = 90
POSE_GRAPH.constraint_builder.sampling_ratio = 0.03
POSE_GRAPH.optimization_problem.ceres_solver_options.max_num_iterations = 10
POSE_GRAPH.constraint_builder.min_score = 0.62
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.66

return options
'''
        return config

    def _generate_3d_config(self):
        """Generate 3D SLAM configuration."""
        config = f'''
include "map_builder.lua"
include "trajectory_builder.lua"

options = {{
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "map",
  tracking_frame = "base_link",
  published_frame = "base_link",
  odom_frame = "odom",
  provide_odom_frame = true,
  publish_frame_projected_to_2d = false,
  use_pose_extrapolator = true,
  use_odometry = false,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 0,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 1,
  num_point_clouds = 1,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
}}

MAP_BUILDER.use_trajectory_builder_3d = true

TRAJECTORY_BUILDER_3D.num_accumulated_range_data = 1
TRAJECTORY_BUILDER_3D.min_range = 0.5
TRAJECTORY_BUILDER_3D.max_range = {self.config.max_range}
TRAJECTORY_BUILDER_3D.voxel_filter_size = 0.15
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.max_length = 2.
TRAJECTORY_BUILDER_3D.high_resolution_adaptive_voxel_filter.min_num_points = 150
TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.max_length = 4.
TRAJECTORY_BUILDER_3D.low_resolution_adaptive_voxel_filter.min_num_points = 200
TRAJECTORY_BUILDER_3D.use_online_correlative_scan_matching = false
TRAJECTORY_BUILDER_3D.real_time_correlative_scan_matcher.linear_search_window = 0.15
TRAJECTORY_BUILDER_3D.real_time_correlative_scan_matcher.angular_search_window = math.rad(1.)
TRAJECTORY_BUILDER_3D.ceres_scan_matcher.occupied_space_weight = 1.
TRAJECTORY_BUILDER_3D.ceres_scan_matcher.translation_weight = 10.
TRAJECTORY_BUILDER_3D.ceres_scan_matcher.rotation_weight = 1.
TRAJECTORY_BUILDER_3D.submaps.high_resolution = 0.10
TRAJECTORY_BUILDER_3D.submaps.low_resolution = 0.45
TRAJECTORY_BUILDER_3D.submaps.num_range_data = 130
TRAJECTORY_BUILDER_3D.submaps.range_data_inserter.hit_probability = 0.55
TRAJECTORY_BUILDER_3D.submaps.range_data_inserter.miss_probability = 0.49

POSE_GRAPH.optimization_problem.huber_scale = 5e2
POSE_GRAPH.optimize_every_n_nodes = 320
POSE_GRAPH.constraint_builder.sampling_ratio = 0.03
POSE_GRAPH.optimization_problem.ceres_solver_options.max_num_iterations = 10
POSE_GRAPH.constraint_builder.min_score = 0.62
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.66

return options
'''
        return config

    def initialize(self) -> bool:
        """Initialize Cartographer system."""
        try:
            # Create configuration file
            if not self._create_cartographer_config():
                return False

            # For this implementation, we'll simulate Cartographer initialization
            # In a real implementation, you would start the Cartographer node
            self.trajectory_id = 0

            self._update_state(SLAMState.INITIALIZING)
            self._log_info("Cartographer initialized successfully")
            return True

        except Exception as e:
            self._log_error(f"Failed to initialize Cartographer: {e}")
            return False

    def process_image(self, image: Union[np.ndarray, Image], timestamp: float) -> bool:
        """Process camera image (convert to laser scan or point cloud)."""
        if not self.is_initialized():
            self._log_error("SLAM system not initialized")
            return False

        try:
            # For monocular camera, we need to convert to usable sensor data
            # This is a simplified approach - real implementation would use
            # visual odometry or convert to pseudo-laser scan

            if isinstance(image, Image):
                cv_image = self.cv_bridge.imgmsg_to_cv2(image, "bgr8")
            else:
                cv_image = image.copy()

            start_time = time.time()

            # Convert image to pseudo laser scan (simplified)
            laser_scan = self._image_to_laser_scan(cv_image, timestamp)

            if laser_scan is not None:
                success = self._process_laser_scan(laser_scan, timestamp)
            else:
                success = False

            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self._increment_frame_count()

            return success

        except Exception as e:
            self._log_error(f"Error processing image: {e}")
            return False

    def process_pointcloud(self, pointcloud: Union[np.ndarray, PointCloud2],
                          timestamp: float) -> bool:
        """Process point cloud data."""
        if not self.is_initialized():
            self._log_error("SLAM system not initialized")
            return False

        try:
            start_time = time.time()

            # Process point cloud with Cartographer
            success = self._process_cartographer_pointcloud(pointcloud, timestamp)

            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self._increment_frame_count()

            return success

        except Exception as e:
            self._log_error(f"Error processing point cloud: {e}")
            return False

    def process_laser_scan(self, scan: LaserScan, timestamp: float) -> bool:
        """Process laser scan data."""
        if not self.is_initialized():
            self._log_error("SLAM system not initialized")
            return False

        try:
            start_time = time.time()

            # Process laser scan with Cartographer
            success = self._process_laser_scan(scan, timestamp)

            # Update performance metrics
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self._increment_frame_count()

            return success

        except Exception as e:
            self._log_error(f"Error processing laser scan: {e}")
            return False

    def process_imu(self, imu_data: Union[np.ndarray, Imu], timestamp: float) -> bool:
        """Process IMU data."""
        try:
            if isinstance(imu_data, Imu):
                imu_measurement = {
                    'timestamp': timestamp,
                    'linear_acceleration': [
                        imu_data.linear_acceleration.x,
                        imu_data.linear_acceleration.y,
                        imu_data.linear_acceleration.z
                    ],
                    'angular_velocity': [
                        imu_data.angular_velocity.x,
                        imu_data.angular_velocity.y,
                        imu_data.angular_velocity.z
                    ]
                }
            else:
                imu_measurement = {
                    'timestamp': timestamp,
                    'linear_acceleration': imu_data[:3].tolist(),
                    'angular_velocity': imu_data[3:6].tolist()
                }

            # Store IMU data
            self.imu_buffer.append(imu_measurement)

            # Keep buffer manageable
            if len(self.imu_buffer) > 1000:
                self.imu_buffer = self.imu_buffer[-500:]

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
        try:
            # Simulate getting occupancy grid from Cartographer
            # In real implementation, this would come from Cartographer's map
            if self.mapping_mode == '2D' and len(self.map_points_3d) > 0:
                # Create a simple occupancy grid from map points
                grid = OccupancyGrid()
                grid.header.frame_id = "map"
                grid.header.stamp.sec = int(time.time())

                # Set grid parameters
                grid.info.resolution = 0.05  # 5cm resolution
                grid.info.width = 400
                grid.info.height = 400
                grid.info.origin.position.x = -10.0
                grid.info.origin.position.y = -10.0
                grid.info.origin.position.z = 0.0

                # Initialize grid data
                grid.data = [-1] * (grid.info.width * grid.info.height)

                # Fill grid with map points
                for point in self.map_points_3d:
                    x, y = point.position[:2]
                    grid_x = int((x - grid.info.origin.position.x) / grid.info.resolution)
                    grid_y = int((y - grid.info.origin.position.y) / grid.info.resolution)

                    if 0 <= grid_x < grid.info.width and 0 <= grid_y < grid.info.height:
                        index = grid_y * grid.info.width + grid_x
                        grid.data[index] = 100  # Occupied

                return grid

            return None

        except Exception as e:
            self._log_error(f"Error getting occupancy grid: {e}")
            return None

    def get_trajectory(self) -> SLAMTrajectory:
        """Get robot trajectory."""
        keyframe_indices = list(range(0, len(self.trajectory_poses), 10))
        return SLAMTrajectory(
            poses=self.trajectory_poses.copy(),
            timestamps=self.trajectory_timestamps.copy(),
            keyframe_indices=keyframe_indices
        )

    def reset(self) -> bool:
        """Reset Cartographer system."""
        try:
            self.current_pose = None
            self.trajectory_poses = []
            self.trajectory_timestamps = []
            self.map_points_3d = []
            self.submap_data = []
            self.processing_times = []
            self.submap_count = 0
            self.constraint_count = 0
            self.laser_scans = []
            self.imu_buffer = []
            self.point_clouds = []
            self._frame_count = 0
            self.last_submap_id = -1

            self._update_state(SLAMState.INITIALIZING)
            self._log_info("Cartographer system reset")
            return True

        except Exception as e:
            self._log_error(f"Error resetting system: {e}")
            return False

    def save_map(self, filepath: str) -> bool:
        """Save current map to file."""
        try:
            # Save trajectory and map data
            data = {
                'trajectory_poses': [self._slam_pose_to_dict(pose) for pose in self.trajectory_poses],
                'trajectory_timestamps': self.trajectory_timestamps,
                'map_points': [self._map_point_to_dict(point) for point in self.map_points_3d],
                'submaps': self.submap_data,
                'mapping_mode': self.mapping_mode,
                'config': {
                    'algorithm_name': self.config.algorithm_name,
                    'sensor_type': self.config.sensor_type.value,
                    'mapping_mode': self.mapping_mode
                }
            }

            import json
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)

            # Save Cartographer-specific files
            if self.lua_config_path and os.path.exists(self.lua_config_path):
                import shutil
                shutil.copy2(self.lua_config_path, f"{filepath}.lua")

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

            # Load submaps
            self.submap_data = data.get('submaps', [])

            # Load configuration
            config_data = data.get('config', {})
            self.mapping_mode = config_data.get('mapping_mode', '2D')

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

            # Cartographer has built-in global localization
            # For this simulation, we'll use the last known pose
            if self.trajectory_poses:
                self.current_pose = self.trajectory_poses[-1]
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
            return {'algorithm': 'Cartographer', 'frames_processed': 0}

        avg_processing_time = np.mean(self.processing_times)
        fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0

        return {
            'algorithm': 'Cartographer',
            'frames_processed': self._frame_count,
            'avg_processing_time_ms': avg_processing_time * 1000,
            'max_processing_time_ms': np.max(self.processing_times) * 1000,
            'min_processing_time_ms': np.min(self.processing_times) * 1000,
            'average_fps': fps,
            'state': self.state.value,
            'mapping_mode': self.mapping_mode,
            'submaps': self.submap_count,
            'constraints': self.constraint_count,
            'map_points': len(self.map_points_3d),
            'trajectory_length': len(self.trajectory_poses),
            'tracking_state': 'tracking' if self.current_pose is not None else 'lost'
        }

    # Helper methods
    def _process_laser_scan(self, scan, timestamp):
        """Process laser scan with Cartographer."""
        try:
            # Store scan data
            self.laser_scans.append({
                'timestamp': timestamp,
                'scan': scan
            })

            # Simulate Cartographer processing
            # In real implementation, this would interface with Cartographer
            pose_estimate = self._simulate_cartographer_processing(scan, timestamp)

            if pose_estimate is not None:
                self.current_pose = pose_estimate
                self.trajectory_poses.append(pose_estimate)
                self.trajectory_timestamps.append(timestamp)

                # Update state
                if self.state == SLAMState.INITIALIZING:
                    self._update_state(SLAMState.TRACKING)

                # Generate map points from scan
                if self.config.enable_mapping:
                    self._add_scan_to_map(scan, pose_estimate)

                return True

            return False

        except Exception as e:
            self._log_error(f"Error processing laser scan: {e}")
            return False

    def _process_cartographer_pointcloud(self, pointcloud, timestamp):
        """Process point cloud with Cartographer."""
        try:
            # Convert PointCloud2 to numpy array if necessary
            if isinstance(pointcloud, PointCloud2):
                if not POINTCLOUD_UTILS_AVAILABLE:
                    self._log_warning("Point cloud utilities not available")
                    return False

                points = list(pc2.read_points(pointcloud, field_names=["x", "y", "z"], skip_nans=True))
                points_array = np.array(points)
            else:
                points_array = pointcloud

            # Store point cloud data
            self.point_clouds.append({
                'timestamp': timestamp,
                'points': points_array
            })

            # Simulate Cartographer 3D processing
            pose_estimate = self._simulate_cartographer_3d_processing(points_array, timestamp)

            if pose_estimate is not None:
                self.current_pose = pose_estimate
                self.trajectory_poses.append(pose_estimate)
                self.trajectory_timestamps.append(timestamp)

                # Update state
                if self.state == SLAMState.INITIALIZING:
                    self._update_state(SLAMState.TRACKING)

                # Add points to map
                if self.config.enable_mapping:
                    self._add_pointcloud_to_map(points_array, pose_estimate)

                return True

            return False

        except Exception as e:
            self._log_error(f"Error processing point cloud: {e}")
            return False

    def _simulate_cartographer_processing(self, scan, timestamp):
        """Simulate Cartographer scan matching and pose estimation."""
        try:
            # Very simplified simulation - would be replaced by actual Cartographer
            if hasattr(scan, 'ranges'):
                ranges = np.array(scan.ranges)
            else:
                ranges = scan  # Assume it's already an array

            # Simple motion model for simulation
            if self.current_pose is not None:
                # Add small random motion
                delta_x = np.random.normal(0, 0.02)
                delta_y = np.random.normal(0, 0.02)
                delta_theta = np.random.normal(0, 0.05)

                new_x = self.current_pose.position[0] + delta_x
                new_y = self.current_pose.position[1] + delta_y

                # Convert current orientation to yaw
                from scipy.spatial.transform import Rotation
                current_rot = Rotation.from_quat(self.current_pose.orientation)
                current_yaw = current_rot.as_euler('zyx')[0]
                new_yaw = current_yaw + delta_theta

                # Create new pose
                new_rot = Rotation.from_euler('z', new_yaw)
                new_quat = new_rot.as_quat()

                return SLAMPose(
                    position=np.array([new_x, new_y, 0.0]),
                    orientation=new_quat,
                    timestamp=timestamp,
                    frame_id="map"
                )
            else:
                # Initial pose
                return SLAMPose(
                    position=np.array([0.0, 0.0, 0.0]),
                    orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                    timestamp=timestamp,
                    frame_id="map"
                )

        except Exception as e:
            self._log_warning(f"Simulated processing failed: {e}")
            return None

    def _simulate_cartographer_3d_processing(self, points, timestamp):
        """Simulate Cartographer 3D processing."""
        # Similar to 2D but with 3D motion
        if self.current_pose is not None:
            delta_x = np.random.normal(0, 0.02)
            delta_y = np.random.normal(0, 0.02)
            delta_z = np.random.normal(0, 0.01)

            new_position = self.current_pose.position + np.array([delta_x, delta_y, delta_z])

            return SLAMPose(
                position=new_position,
                orientation=self.current_pose.orientation.copy(),
                timestamp=timestamp,
                frame_id="map"
            )
        else:
            return SLAMPose(
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([0.0, 0.0, 0.0, 1.0]),
                timestamp=timestamp,
                frame_id="map"
            )

    def _add_scan_to_map(self, scan, pose):
        """Add laser scan points to map."""
        try:
            if hasattr(scan, 'ranges') and hasattr(scan, 'angle_min') and hasattr(scan, 'angle_increment'):
                ranges = np.array(scan.ranges)
                angle_min = scan.angle_min
                angle_increment = scan.angle_increment

                # Convert scan to points in map frame
                for i, r in enumerate(ranges):
                    if not np.isnan(r) and not np.isinf(r) and r > 0:
                        angle = angle_min + i * angle_increment

                        # Point in sensor frame
                        x_sensor = r * np.cos(angle)
                        y_sensor = r * np.sin(angle)

                        # Transform to map frame
                        point_map = self._transform_point_to_map(
                            np.array([x_sensor, y_sensor, 0.0]), pose
                        )

                        map_point = SLAMMapPoint(
                            position=point_map,
                            confidence=0.8
                        )
                        self.map_points_3d.append(map_point)

        except Exception as e:
            self._log_warning(f"Error adding scan to map: {e}")

    def _add_pointcloud_to_map(self, points, pose):
        """Add point cloud to map."""
        try:
            # Subsample points for efficiency
            if len(points) > 1000:
                indices = np.random.choice(len(points), 1000, replace=False)
                sampled_points = points[indices]
            else:
                sampled_points = points

            for point in sampled_points:
                if len(point) >= 3:
                    # Transform to map frame
                    point_map = self._transform_point_to_map(point[:3], pose)

                    map_point = SLAMMapPoint(
                        position=point_map,
                        confidence=0.9
                    )
                    self.map_points_3d.append(map_point)

        except Exception as e:
            self._log_warning(f"Error adding point cloud to map: {e}")

    def _transform_point_to_map(self, point, pose):
        """Transform point from sensor to map frame."""
        try:
            # Convert pose to transformation matrix
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_quat(pose.orientation)

            # Apply rotation and translation
            rotated_point = rot.apply(point)
            map_point = rotated_point + pose.position

            return map_point

        except Exception as e:
            self._log_warning(f"Point transformation failed: {e}")
            return point

    def _image_to_laser_scan(self, image, timestamp):
        """Convert image to pseudo laser scan (simplified)."""
        try:
            # This is a very simplified approach
            # Real implementation would use depth estimation or stereo
            height, width = image.shape[:2]
            center_row = height // 2

            # Extract a horizontal line from the center
            line = image[center_row, :]

            # Convert to grayscale if needed
            if len(line.shape) == 3:
                line = np.mean(line, axis=1)

            # Simple obstacle detection based on intensity changes
            ranges = []
            for i in range(len(line)):
                # Simulate range based on intensity
                # This is a placeholder - real implementation needed
                range_val = (255 - line[i]) / 255.0 * 10.0  # Max 10m range
                ranges.append(max(0.1, range_val))  # Min 0.1m

            # Create simplified laser scan structure
            class SimpleLaserScan:
                def __init__(self):
                    self.ranges = ranges
                    self.angle_min = -np.pi/2
                    self.angle_max = np.pi/2
                    self.angle_increment = np.pi / len(ranges)
                    self.time_increment = 0.0
                    self.scan_time = 0.1
                    self.range_min = 0.1
                    self.range_max = 10.0

            return SimpleLaserScan()

        except Exception as e:
            self._log_warning(f"Image to laser scan conversion failed: {e}")
            return None

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

    def __del__(self):
        """Cleanup resources."""
        try:
            # Clean up working directory
            import shutil
            if os.path.exists(self.work_dir):
                shutil.rmtree(self.work_dir)
        except:
            pass
