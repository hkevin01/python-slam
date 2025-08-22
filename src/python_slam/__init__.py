"""
Python SLAM Package for ROS 2
Advanced SLAM implementation for aerial drone competitions
"""

__version__ = "1.0.0"
__author__ = "Python SLAM Team"
__email__ = "developer@python-slam.org"

"""
Python SLAM Package for ROS 2
Advanced SLAM implementation with optional pySLAM integration
"""

__version__ = "1.0.0"
__author__ = "Python SLAM Team"
__email__ = "developer@python-slam.org"

# Core SLAM components (non-ROS2 dependencies)
from .feature_extraction import FeatureExtraction
from .pose_estimation import PoseEstimation
from .mapping import Mapping
from .localization import Localization
from .loop_closure import LoopClosure

# pySLAM integration (no ROS2 dependencies)
from .pyslam_integration import pySLAMWrapper, pySLAMConfig
from .pyslam_config import (
    create_default_config,
    load_pyslam_config,
    save_pyslam_config,
    get_pyslam_config
)

# Optional ROS2 components (import only if ROS2 is available)
try:
    from .basic_slam_pipeline import BasicSlamPipeline
    from .flight_integration import FlightIntegration
    from .slam_node import SlamNode
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    BasicSlamPipeline = None
    FlightIntegration = None
    SlamNode = None
from .feature_extraction_node import FeatureExtractionNode
from .pose_estimation_node import PoseEstimationNode
from .mapping_node import MappingNode
from .localization_node import LocalizationNode
from .loop_closure_node import LoopClosureNode
from .flight_integration_node import FlightIntegrationNode

__all__ = [
    # Core components
    'BasicSlamPipeline',
    'FeatureExtraction',
    'PoseEstimation',
    'Mapping',
    'Localization',
    'LoopClosure',
    'FlightIntegration',

    # ROS 2 nodes
    'SlamNode',
    'FeatureExtractionNode',
    'PoseEstimationNode',
    'MappingNode',
    'LocalizationNode',
    'LoopClosureNode',
    'FlightIntegrationNode',
]
