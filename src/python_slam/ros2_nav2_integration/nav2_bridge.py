"""
Nav2 Bridge for Python-SLAM Integration

This module provides the main bridge between Python-SLAM and ROS2 Nav2,
handling data conversion, message publishing, and coordination.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from threading import Lock
import time

logger = logging.getLogger(__name__)

# Mock ROS2 imports for demonstration (in real implementation, use actual ROS2)
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
    from nav_msgs.msg import OccupancyGrid, Path, Odometry
    from sensor_msgs.msg import LaserScan, PointCloud2
    from nav2_msgs.msg import BehaviorTree
    from nav2_msgs.srv import ManageLifecycleNodes
    from tf2_ros import TransformBroadcaster, Buffer, TransformListener
    from tf2_geometry_msgs import do_transform_pose
    import tf2_ros
    ROS2_AVAILABLE = True
except ImportError:
    logger.warning("ROS2 not available - using mock implementations")
    ROS2_AVAILABLE = False
    # Mock classes for development without ROS2
    class Node:
        def __init__(self, name): pass
        def get_logger(self): return logger
        def create_publisher(self, *args, **kwargs): return None
        def create_subscription(self, *args, **kwargs): return None
        def create_service(self, *args, **kwargs): return None
        def create_client(self, *args, **kwargs): return None
    
    class QoSProfile:
        def __init__(self, **kwargs): pass
    
    ReliabilityPolicy = type('ReliabilityPolicy', (), {'RELIABLE': 1, 'BEST_EFFORT': 2})
    HistoryPolicy = type('HistoryPolicy', (), {'KEEP_LAST': 1, 'KEEP_ALL': 2})

@dataclass
class Nav2Status:
    """Status information for Nav2 integration."""
    navigation_active: bool = False
    current_goal: Optional[Dict] = None
    planner_status: str = "idle"
    controller_status: str = "idle"
    behavior_status: str = "idle"
    last_error: Optional[str] = None
    planning_time: float = 0.0
    execution_time: float = 0.0

class Nav2Bridge(Node):
    """Main bridge between Python-SLAM and ROS2 Nav2."""
    
    def __init__(self, slam_system=None):
        if ROS2_AVAILABLE:
            super().__init__('slam_nav2_bridge')
        else:
            self.get_logger = lambda: logger
        
        self.slam_system = slam_system
        self.status = Nav2Status()
        self._lock = Lock()
        
        # ROS2 publishers and subscribers
        self.publishers = {}
        self.subscribers = {}
        self.services = {}
        self.clients = {}
        
        # Transform handling
        self.tf_broadcaster = None
        self.tf_buffer = None
        self.tf_listener = None
        
        # Callbacks
        self.goal_reached_callback: Optional[Callable] = None
        self.planning_failed_callback: Optional[Callable] = None
        
        self._initialize_ros2_interfaces()
    
    def _initialize_ros2_interfaces(self):
        """Initialize ROS2 publishers, subscribers, and services."""
        if not ROS2_AVAILABLE:
            logger.info("ROS2 not available - using mock interfaces")
            return
        
        try:
            # QoS profiles
            reliable_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )
            
            best_effort_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )
            
            # Publishers
            self.publishers['map'] = self.create_publisher(
                OccupancyGrid, '/map', reliable_qos
            )
            
            self.publishers['odom'] = self.create_publisher(
                Odometry, '/odom', best_effort_qos
            )
            
            self.publishers['pose'] = self.create_publisher(
                PoseWithCovarianceStamped, '/amcl_pose', reliable_qos
            )
            
            self.publishers['path'] = self.create_publisher(
                Path, '/plan', reliable_qos
            )
            
            self.publishers['cmd_vel'] = self.create_publisher(
                Twist, '/cmd_vel', best_effort_qos
            )
            
            # Subscribers
            self.subscribers['goal'] = self.create_subscription(
                PoseStamped, '/goal_pose', self._goal_callback, reliable_qos
            )
            
            self.subscribers['initial_pose'] = self.create_subscription(
                PoseWithCovarianceStamped, '/initialpose', self._initial_pose_callback, reliable_qos
            )
            
            self.subscribers['scan'] = self.create_subscription(
                LaserScan, '/scan', self._scan_callback, best_effort_qos
            )
            
            # Transform broadcaster and listener
            self.tf_broadcaster = TransformBroadcaster(self)
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            
            logger.info("ROS2 Nav2 bridge interfaces initialized")
        
        except Exception as e:
            logger.error(f"Failed to initialize ROS2 interfaces: {e}")
    
    def publish_map(self, occupancy_grid: np.ndarray, resolution: float = 0.05, origin: List[float] = [0.0, 0.0, 0.0]):
        """Publish occupancy grid map to Nav2."""
        if not ROS2_AVAILABLE or 'map' not in self.publishers:
            logger.debug("Map publishing not available")
            return
        
        try:
            from builtin_interfaces.msg import Time
            from std_msgs.msg import Header
            from geometry_msgs.msg import Pose, Point, Quaternion
            
            # Create OccupancyGrid message
            map_msg = OccupancyGrid()
            
            # Header
            map_msg.header = Header()
            map_msg.header.stamp = self.get_clock().now().to_msg()
            map_msg.header.frame_id = "map"
            
            # Map metadata
            map_msg.info.resolution = resolution
            map_msg.info.width = occupancy_grid.shape[1]
            map_msg.info.height = occupancy_grid.shape[0]
            
            # Origin pose
            map_msg.info.origin = Pose()
            map_msg.info.origin.position = Point(x=origin[0], y=origin[1], z=origin[2])
            map_msg.info.origin.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            
            # Convert occupancy grid to ROS format
            # SLAM grid: unknown=-1, free=0, occupied=1
            # ROS grid: unknown=-1, free=0, occupied=100
            ros_grid = np.zeros_like(occupancy_grid, dtype=np.int8)
            ros_grid[occupancy_grid == 0] = 0    # free
            ros_grid[occupancy_grid == 1] = 100  # occupied
            ros_grid[occupancy_grid == -1] = -1  # unknown
            
            # Flatten and convert to list (ROS expects row-major order)
            map_msg.data = ros_grid.flatten().tolist()
            
            # Publish
            self.publishers['map'].publish(map_msg)
            logger.debug("Published map to Nav2")
        
        except Exception as e:
            logger.error(f"Failed to publish map: {e}")
    
    def publish_odometry(self, pose: np.ndarray, twist: np.ndarray, covariance: Optional[np.ndarray] = None):
        """Publish odometry data to Nav2."""
        if not ROS2_AVAILABLE or 'odom' not in self.publishers:
            logger.debug("Odometry publishing not available")
            return
        
        try:
            from builtin_interfaces.msg import Time
            from std_msgs.msg import Header
            from geometry_msgs.msg import Point, Quaternion, Vector3, TwistWithCovariance, PoseWithCovariance
            
            # Create Odometry message
            odom_msg = Odometry()
            
            # Header
            odom_msg.header = Header()
            odom_msg.header.stamp = self.get_clock().now().to_msg()
            odom_msg.header.frame_id = "odom"
            odom_msg.child_frame_id = "base_link"
            
            # Pose
            odom_msg.pose = PoseWithCovariance()
            odom_msg.pose.pose.position = Point(x=pose[0], y=pose[1], z=pose[2])
            
            # Convert rotation matrix to quaternion (simplified)
            if len(pose) >= 7:  # Has quaternion
                odom_msg.pose.pose.orientation = Quaternion(x=pose[3], y=pose[4], z=pose[5], w=pose[6])
            else:
                odom_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            
            # Pose covariance
            if covariance is not None:
                odom_msg.pose.covariance = covariance.flatten().tolist()
            else:
                # Default covariance
                default_cov = np.eye(6) * 0.1
                odom_msg.pose.covariance = default_cov.flatten().tolist()
            
            # Twist
            odom_msg.twist = TwistWithCovariance()
            odom_msg.twist.twist.linear = Vector3(x=twist[0], y=twist[1], z=twist[2])
            if len(twist) >= 6:
                odom_msg.twist.twist.angular = Vector3(x=twist[3], y=twist[4], z=twist[5])
            else:
                odom_msg.twist.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)
            
            # Twist covariance (default)
            twist_cov = np.eye(6) * 0.01
            odom_msg.twist.covariance = twist_cov.flatten().tolist()
            
            # Publish
            self.publishers['odom'].publish(odom_msg)
            logger.debug("Published odometry to Nav2")
        
        except Exception as e:
            logger.error(f"Failed to publish odometry: {e}")
    
    def publish_robot_pose(self, pose: np.ndarray, covariance: Optional[np.ndarray] = None):
        """Publish robot pose for localization."""
        if not ROS2_AVAILABLE or 'pose' not in self.publishers:
            logger.debug("Pose publishing not available")
            return
        
        try:
            from builtin_interfaces.msg import Time
            from std_msgs.msg import Header
            from geometry_msgs.msg import Point, Quaternion, PoseWithCovariance
            
            # Create PoseWithCovarianceStamped message
            pose_msg = PoseWithCovarianceStamped()
            
            # Header
            pose_msg.header = Header()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "map"
            
            # Pose
            pose_msg.pose = PoseWithCovariance()
            pose_msg.pose.pose.position = Point(x=pose[0], y=pose[1], z=pose[2])
            
            if len(pose) >= 7:
                pose_msg.pose.pose.orientation = Quaternion(x=pose[3], y=pose[4], z=pose[5], w=pose[6])
            else:
                pose_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            
            # Covariance
            if covariance is not None:
                pose_msg.pose.covariance = covariance.flatten().tolist()
            else:
                default_cov = np.eye(6) * 0.1
                pose_msg.pose.covariance = default_cov.flatten().tolist()
            
            # Publish
            self.publishers['pose'].publish(pose_msg)
            logger.debug("Published robot pose to Nav2")
        
        except Exception as e:
            logger.error(f"Failed to publish robot pose: {e}")
    
    def publish_path(self, waypoints: List[np.ndarray]):
        """Publish planned path to Nav2."""
        if not ROS2_AVAILABLE or 'path' not in self.publishers:
            logger.debug("Path publishing not available")
            return
        
        try:
            from builtin_interfaces.msg import Time
            from std_msgs.msg import Header
            from geometry_msgs.msg import Point, Quaternion
            
            # Create Path message
            path_msg = Path()
            
            # Header
            path_msg.header = Header()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = "map"
            
            # Convert waypoints to PoseStamped messages
            for waypoint in waypoints:
                pose_stamped = PoseStamped()
                pose_stamped.header = path_msg.header
                
                pose_stamped.pose.position = Point(x=waypoint[0], y=waypoint[1], z=waypoint[2] if len(waypoint) > 2 else 0.0)
                
                if len(waypoint) >= 7:
                    pose_stamped.pose.orientation = Quaternion(x=waypoint[3], y=waypoint[4], z=waypoint[5], w=waypoint[6])
                else:
                    pose_stamped.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                
                path_msg.poses.append(pose_stamped)
            
            # Publish
            self.publishers['path'].publish(path_msg)
            logger.debug(f"Published path with {len(waypoints)} waypoints")
        
        except Exception as e:
            logger.error(f"Failed to publish path: {e}")
    
    def _goal_callback(self, msg):
        """Handle navigation goal messages from Nav2."""
        try:
            with self._lock:
                goal_pose = {
                    'x': msg.pose.position.x,
                    'y': msg.pose.position.y,
                    'z': msg.pose.position.z,
                    'qx': msg.pose.orientation.x,
                    'qy': msg.pose.orientation.y,
                    'qz': msg.pose.orientation.z,
                    'qw': msg.pose.orientation.w
                }
                
                self.status.current_goal = goal_pose
                self.status.navigation_active = True
                self.status.planner_status = "planning"
                
                logger.info(f"Received navigation goal: {goal_pose}")
                
                # Trigger SLAM-based path planning
                if self.slam_system:
                    self._plan_path_to_goal(goal_pose)
        
        except Exception as e:
            logger.error(f"Goal callback failed: {e}")
            self.status.last_error = str(e)
    
    def _initial_pose_callback(self, msg):
        """Handle initial pose messages for localization."""
        try:
            initial_pose = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ])
            
            logger.info(f"Received initial pose: {initial_pose}")
            
            # Initialize SLAM with this pose
            if self.slam_system and hasattr(self.slam_system, 'set_initial_pose'):
                self.slam_system.set_initial_pose(initial_pose)
        
        except Exception as e:
            logger.error(f"Initial pose callback failed: {e}")
    
    def _scan_callback(self, msg):
        """Handle laser scan data for SLAM updates."""
        try:
            # Convert LaserScan to numpy array
            ranges = np.array(msg.ranges)
            angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
            
            # Filter out invalid ranges
            valid_mask = (ranges >= msg.range_min) & (ranges <= msg.range_max) & np.isfinite(ranges)
            valid_ranges = ranges[valid_mask]
            valid_angles = angles[valid_mask]
            
            # Convert to Cartesian coordinates
            scan_points = np.column_stack([
                valid_ranges * np.cos(valid_angles),
                valid_ranges * np.sin(valid_angles)
            ])
            
            # Update SLAM with scan data
            if self.slam_system and hasattr(self.slam_system, 'process_scan'):
                self.slam_system.process_scan(scan_points, msg.header.stamp)
        
        except Exception as e:
            logger.error(f"Scan callback failed: {e}")
    
    def _plan_path_to_goal(self, goal_pose: Dict):
        """Plan path using SLAM map data."""
        try:
            start_time = time.time()
            
            # Get current robot pose from SLAM
            if not self.slam_system or not hasattr(self.slam_system, 'get_current_pose'):
                logger.error("SLAM system not available for path planning")
                return
            
            current_pose = self.slam_system.get_current_pose()
            if current_pose is None:
                logger.error("Could not get current pose from SLAM")
                return
            
            # Get occupancy grid from SLAM
            occupancy_grid = self.slam_system.get_occupancy_grid()
            if occupancy_grid is None:
                logger.error("Could not get occupancy grid from SLAM")
                return
            
            # Plan path using A* or other algorithm
            path = self._plan_path_astar(current_pose, goal_pose, occupancy_grid)
            
            if path is not None:
                # Publish path to Nav2
                self.publish_path(path)
                
                # Update status
                with self._lock:
                    self.status.planner_status = "succeeded"
                    self.status.planning_time = time.time() - start_time
                
                logger.info(f"Path planned successfully in {self.status.planning_time:.3f}s")
            else:
                with self._lock:
                    self.status.planner_status = "failed"
                    self.status.last_error = "Path planning failed"
                
                if self.planning_failed_callback:
                    self.planning_failed_callback("No valid path found")
        
        except Exception as e:
            logger.error(f"Path planning failed: {e}")
            with self._lock:
                self.status.planner_status = "failed"
                self.status.last_error = str(e)
    
    def _plan_path_astar(self, start_pose, goal_pose, occupancy_grid):
        """Simple A* path planning implementation."""
        # This is a simplified implementation
        # In practice, would use more sophisticated planners
        
        try:
            start_point = np.array([start_pose[0], start_pose[1]])
            goal_point = np.array([goal_pose['x'], goal_pose['y']])
            
            # For simplicity, create a straight-line path
            num_waypoints = 10
            waypoints = []
            
            for i in range(num_waypoints + 1):
                t = i / num_waypoints
                waypoint = start_point + t * (goal_point - start_point)
                
                # Add z=0 and identity quaternion
                full_waypoint = np.array([waypoint[0], waypoint[1], 0.0, 0.0, 0.0, 0.0, 1.0])
                waypoints.append(full_waypoint)
            
            return waypoints
        
        except Exception as e:
            logger.error(f"A* planning failed: {e}")
            return None
    
    def get_status(self) -> Nav2Status:
        """Get current Nav2 integration status."""
        with self._lock:
            return self.status
    
    def set_goal_reached_callback(self, callback: Callable):
        """Set callback for when navigation goal is reached."""
        self.goal_reached_callback = callback
    
    def set_planning_failed_callback(self, callback: Callable):
        """Set callback for when path planning fails."""
        self.planning_failed_callback = callback
    
    def shutdown(self):
        """Shutdown the Nav2 bridge."""
        try:
            with self._lock:
                self.status.navigation_active = False
                self.status.planner_status = "shutdown"
                self.status.controller_status = "shutdown"
                self.status.behavior_status = "shutdown"
            
            logger.info("Nav2 bridge shutdown completed")
        
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
    
    def __del__(self):
        """Destructor to ensure proper shutdown."""
        self.shutdown()
