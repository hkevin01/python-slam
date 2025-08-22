"""
ROS2 Nav2 Integration for Python-SLAM

This module provides integration with the ROS2 Navigation Stack (Nav2)
for autonomous navigation using SLAM data.
"""

from .nav2_bridge import Nav2Bridge
from .slam_nav2_node import SlamNav2Node
from .nav2_planner import Nav2PathPlanner
from .nav2_controller import Nav2Controller
from .nav2_behavior_tree import Nav2BehaviorTreeManager
from .nav2_costmap import Nav2CostmapManager
from .nav2_lifecycle import Nav2LifecycleManager

__all__ = [
    'Nav2Bridge',
    'SlamNav2Node',
    'Nav2PathPlanner',
    'Nav2Controller',
    'Nav2BehaviorTreeManager',
    'Nav2CostmapManager',
    'Nav2LifecycleManager'
]

# Default Nav2 configuration
DEFAULT_NAV2_CONFIG = {
    "planner_server": {
        "expected_planner_frequency": 20.0,
        "use_sim_time": True,
        "planner_plugins": ["GridBased"],
        "GridBased": {
            "plugin": "nav2_navfn_planner/NavfnPlanner",
            "tolerance": 0.5,
            "use_astar": False,
            "allow_unknown": True
        }
    },
    "controller_server": {
        "controller_frequency": 20.0,
        "min_x_velocity_threshold": 0.001,
        "min_y_velocity_threshold": 0.5,
        "min_theta_velocity_threshold": 0.001,
        "controller_plugins": ["FollowPath"],
        "FollowPath": {
            "plugin": "nav2_regulated_pure_pursuit_controller/RegulatedPurePursuitController",
            "desired_linear_vel": 0.5,
            "lookahead_dist": 0.6,
            "min_lookahead_dist": 0.3,
            "max_lookahead_dist": 0.9,
            "transform_tolerance": 0.1
        }
    },
    "behavior_server": {
        "costmap_topic": "local_costmap/costmap_raw",
        "footprint_topic": "local_costmap/published_footprint",
        "cycle_frequency": 10.0,
        "behavior_plugins": ["spin", "backup", "drive_on_heading", "wait"],
        "spin": {
            "plugin": "nav2_behaviors/Spin"
        },
        "backup": {
            "plugin": "nav2_behaviors/BackUp"
        },
        "drive_on_heading": {
            "plugin": "nav2_behaviors/DriveOnHeading"
        },
        "wait": {
            "plugin": "nav2_behaviors/Wait"
        }
    }
}
