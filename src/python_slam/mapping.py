#!/usr/bin/env python3
"""
Mapping Module for Python SLAM
Implements 3D mapping and occupancy grid generation
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import PointCloud2


class Mapping:
    """
    Mapping class for creating and maintaining 3D maps and occupancy grids.
    """

    def __init__(self, map_resolution: float = 0.05, map_size: int = 1000):
        """
        Initialize mapping system.

        Args:
            map_resolution: Resolution of occupancy grid in meters per pixel
            map_size: Size of occupancy grid (map_size x map_size)
        """
        self.map_resolution = map_resolution
        self.map_size = map_size

        # Initialize occupancy grid
        self.occupancy_grid = np.zeros((map_size, map_size), dtype=np.float32)
        self.grid_origin = np.array([map_size // 2, map_size // 2])

        # 3D point cloud
        self.point_cloud: List[np.ndarray] = []
        self.point_colors: List[np.ndarray] = []

        # Trajectory
        self.trajectory: List[np.ndarray] = []

        # Map parameters
        self.occupancy_threshold = 0.65
        self.free_threshold = 0.25
        self.update_rate = 0.1

    def update(self, pose: PoseStamped):
        """
        Update map with new pose.

        Args:
            pose: Current robot pose
        """
        try:
            # Extract position
            position = np.array([
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z
            ])

            # Add to trajectory
            self.trajectory.append(position)

            # Update occupancy grid
            self._update_occupancy_grid(position)

        except Exception as e:
            print(f"Error updating map: {e}")

    def update_with_depth(self, depth_image: np.ndarray, pose: PoseStamped):
        """
        Update map with depth information.

        Args:
            depth_image: Depth image
            pose: Camera pose
        """
        try:
            # Extract 3D points from depth image
            points_3d = self._depth_to_points(depth_image)

            if len(points_3d) > 0:
                # Transform points to world coordinates
                world_points = self._transform_points_to_world(points_3d, pose)

                # Add to point cloud
                self.point_cloud.extend(world_points)

                # Update occupancy grid
                self._update_occupancy_with_points(world_points)

        except Exception as e:
            print(f"Error updating map with depth: {e}")

    def _depth_to_points(self, depth_image: np.ndarray, camera_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """Convert depth image to 3D points."""
        try:
            if camera_matrix is None:
                # Default camera matrix
                fx = fy = 525.0
                cx, cy = depth_image.shape[1] // 2, depth_image.shape[0] // 2
            else:
                fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
                cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

            # Create coordinate grids
            h, w = depth_image.shape
            u, v = np.meshgrid(np.arange(w), np.arange(h))

            # Valid depth mask
            valid_mask = (depth_image > 0) & (depth_image < 10.0)

            # Convert to 3D points
            z = depth_image[valid_mask]
            x = (u[valid_mask] - cx) * z / fx
            y = (v[valid_mask] - cy) * z / fy

            points_3d = np.column_stack([x, y, z])
            return points_3d

        except Exception as e:
            print(f"Error converting depth to points: {e}")
            return np.array([])

    def _transform_points_to_world(self, points: np.ndarray, pose: PoseStamped) -> np.ndarray:
        """Transform points from camera to world coordinates."""
        try:
            # Extract pose transformation
            position = np.array([
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z
            ])

            # For simplicity, assume identity rotation
            # In practice, extract quaternion and convert to rotation matrix

            # Transform points
            world_points = points + position
            return world_points

        except Exception as e:
            print(f"Error transforming points: {e}")
            return points

    def _update_occupancy_grid(self, position: np.ndarray):
        """Update occupancy grid with robot position."""
        try:
            # Convert world coordinates to grid coordinates
            grid_x = int(position[0] / self.map_resolution + self.grid_origin[0])
            grid_y = int(position[1] / self.map_resolution + self.grid_origin[1])

            # Check bounds
            if 0 <= grid_x < self.map_size and 0 <= grid_y < self.map_size:
                # Mark as free space
                self.occupancy_grid[grid_y, grid_x] = max(
                    self.occupancy_grid[grid_y, grid_x] - self.update_rate, 0.0
                )

        except Exception as e:
            print(f"Error updating occupancy grid: {e}")

    def _update_occupancy_with_points(self, points: np.ndarray):
        """Update occupancy grid with 3D points."""
        try:
            for point in points:
                # Convert to grid coordinates
                grid_x = int(point[0] / self.map_resolution + self.grid_origin[0])
                grid_y = int(point[1] / self.map_resolution + self.grid_origin[1])

                # Check bounds
                if 0 <= grid_x < self.map_size and 0 <= grid_y < self.map_size:
                    # Mark as occupied
                    self.occupancy_grid[grid_y, grid_x] = min(
                        self.occupancy_grid[grid_y, grid_x] + self.update_rate, 1.0
                    )

        except Exception as e:
            print(f"Error updating occupancy with points: {e}")

    def get_occupancy_grid(self) -> Optional[OccupancyGrid]:
        """Get ROS occupancy grid message."""
        try:
            # Create occupancy grid message
            grid_msg = OccupancyGrid()

            # Set header
            grid_msg.header.frame_id = "map"

            # Set metadata
            grid_msg.info.resolution = self.map_resolution
            grid_msg.info.width = self.map_size
            grid_msg.info.height = self.map_size
            grid_msg.info.origin.position.x = -self.grid_origin[0] * self.map_resolution
            grid_msg.info.origin.position.y = -self.grid_origin[1] * self.map_resolution
            grid_msg.info.origin.orientation.w = 1.0

            # Convert occupancy grid to ROS format
            occupancy_data = self.occupancy_grid.flatten()

            # Convert probabilities to occupancy values (-1: unknown, 0-100: occupied probability)
            ros_data = []
            for prob in occupancy_data:
                if prob < self.free_threshold:
                    ros_data.append(0)  # Free
                elif prob > self.occupancy_threshold:
                    ros_data.append(100)  # Occupied
                else:
                    ros_data.append(-1)  # Unknown

            grid_msg.data = ros_data

            return grid_msg

        except Exception as e:
            print(f"Error creating occupancy grid message: {e}")
            return None

    def get_point_cloud(self) -> Optional[PointCloud2]:
        """Get ROS point cloud message."""
        # Placeholder - would need proper PointCloud2 message creation
        return None

    def get_trajectory(self) -> Optional[Path]:
        """Get ROS path message for trajectory."""
        try:
            if len(self.trajectory) == 0:
                return None

            # Create path message
            path_msg = Path()
            path_msg.header.frame_id = "map"

            # Add poses to path
            for i, position in enumerate(self.trajectory):
                pose_stamped = PoseStamped()
                pose_stamped.header.frame_id = "map"
                pose_stamped.pose.position.x = position[0]
                pose_stamped.pose.position.y = position[1]
                pose_stamped.pose.position.z = position[2]
                pose_stamped.pose.orientation.w = 1.0

                path_msg.poses.append(pose_stamped)

            return path_msg

        except Exception as e:
            print(f"Error creating trajectory path: {e}")
            return None

    def save_map(self, filename: str):
        """Save map to file."""
        try:
            np.save(filename, self.occupancy_grid)
            print(f"Map saved to {filename}")
        except Exception as e:
            print(f"Error saving map: {e}")

    def load_map(self, filename: str):
        """Load map from file."""
        try:
            self.occupancy_grid = np.load(filename)
            print(f"Map loaded from {filename}")
        except Exception as e:
            print(f"Error loading map: {e}")

    def get_map_bounds(self) -> Tuple[float, float, float, float]:
        """Get map bounds in world coordinates."""
        min_x = -self.grid_origin[0] * self.map_resolution
        max_x = (self.map_size - self.grid_origin[0]) * self.map_resolution
        min_y = -self.grid_origin[1] * self.map_resolution
        max_y = (self.map_size - self.grid_origin[1]) * self.map_resolution

        return min_x, max_x, min_y, max_y

    def clear_map(self):
        """Clear the map."""
        self.occupancy_grid.fill(0.0)
        self.point_cloud.clear()
        self.point_colors.clear()
        self.trajectory.clear()


if __name__ == "__main__":
    # Demo usage
    print("Mapping Module - Demo")

    # Create mapper
    mapper = Mapping(map_resolution=0.05, map_size=1000)

    # Create dummy pose
    pose = PoseStamped()
    pose.pose.position.x = 1.0
    pose.pose.position.y = 2.0
    pose.pose.position.z = 0.0

    # Update map
    mapper.update(pose)

    # Get occupancy grid
    grid = mapper.get_occupancy_grid()
    if grid:
        print(f"Created occupancy grid: {grid.info.width}x{grid.info.height}")

    print("Mapping module demo complete!")
