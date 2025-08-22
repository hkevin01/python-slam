#!/usr/bin/env python3
"""
SLAM Algorithm Implementations Package

This package contains wrapper implementations for various SLAM algorithms
that conform to the unified SLAMInterface.

Available algorithms:
- ORB-SLAM3: Feature-based visual and visual-inertial SLAM
- RTAB-Map: Real-time appearance-based mapping with RGB-D support
- Cartographer: Google's 2D and 3D real-time SLAM
- OpenVSLAM: Visual SLAM with BoW-based loop closure
- Python SLAM: Custom python-slam algorithm integration
"""

from .orb_slam3_wrapper import ORBSlam3Wrapper
from .rtabmap_wrapper import RTABMapWrapper
from .cartographer_wrapper import CartographerWrapper
from .openvslam_wrapper import OpenVSLAMWrapper
from .python_slam_wrapper import PythonSLAMWrapper

__all__ = [
    'ORBSlam3Wrapper',
    'RTABMapWrapper',
    'CartographerWrapper',
    'OpenVSLAMWrapper',
    'PythonSLAMWrapper'
]

# Algorithm availability checks
AVAILABLE_ALGORITHMS = {}

try:
    from .orb_slam3_wrapper import ORBSlam3Wrapper
    AVAILABLE_ALGORITHMS['orb_slam3'] = True
except ImportError:
    AVAILABLE_ALGORITHMS['orb_slam3'] = False

try:
    from .rtabmap_wrapper import RTABMapWrapper
    AVAILABLE_ALGORITHMS['rtabmap'] = True
except ImportError:
    AVAILABLE_ALGORITHMS['rtabmap'] = False

try:
    from .cartographer_wrapper import CartographerWrapper
    AVAILABLE_ALGORITHMS['cartographer'] = True
except ImportError:
    AVAILABLE_ALGORITHMS['cartographer'] = False

try:
    from .openvslam_wrapper import OpenVSLAMWrapper
    AVAILABLE_ALGORITHMS['openvslam'] = True
except ImportError:
    AVAILABLE_ALGORITHMS['openvslam'] = False

try:
    from .python_slam_wrapper import PythonSLAMWrapper
    AVAILABLE_ALGORITHMS['python_slam'] = True
except ImportError:
    AVAILABLE_ALGORITHMS['python_slam'] = False


def get_available_algorithms():
    """Get list of available SLAM algorithms."""
    return [name for name, available in AVAILABLE_ALGORITHMS.items() if available]


def is_algorithm_available(algorithm_name):
    """Check if a specific algorithm is available."""
    return AVAILABLE_ALGORITHMS.get(algorithm_name.lower(), False)


def get_algorithm_class(algorithm_name):
    """Get the wrapper class for a specific algorithm."""
    algorithm_classes = {
        'orb_slam3': ORBSlam3Wrapper,
        'rtabmap': RTABMapWrapper,
        'cartographer': CartographerWrapper,
        'openvslam': OpenVSLAMWrapper,
        'python_slam': PythonSLAMWrapper
    }

    algorithm_name = algorithm_name.lower()

    if algorithm_name not in algorithm_classes:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    if not is_algorithm_available(algorithm_name):
        raise ImportError(f"Algorithm {algorithm_name} is not available. Check dependencies.")

    return algorithm_classes[algorithm_name]


def get_algorithm_info():
    """Get information about all algorithms."""
    info = {
        'orb_slam3': {
            'name': 'ORB-SLAM3',
            'description': 'Feature-based visual and visual-inertial SLAM',
            'sensors': ['monocular', 'stereo', 'rgbd', 'visual_inertial'],
            'features': ['loop_closure', 'relocalization', 'mapping'],
            'dependencies': ['orbslam3-python', 'opencv-python', 'numpy']
        },
        'rtabmap': {
            'name': 'RTAB-Map',
            'description': 'Real-time appearance-based mapping',
            'sensors': ['rgbd', 'stereo', 'monocular'],
            'features': ['loop_closure', 'relocalization', 'mapping', 'occupancy_grid'],
            'dependencies': ['rtabmap-python', 'opencv-python', 'numpy']
        },
        'cartographer': {
            'name': 'Cartographer',
            'description': 'Google\'s 2D and 3D real-time SLAM',
            'sensors': ['laser', 'pointcloud', 'imu'],
            'features': ['2d_mapping', '3d_mapping', 'loop_closure', 'occupancy_grid'],
            'dependencies': ['cartographer-ros', 'google-cartographer']
        },
        'openvslam': {
            'name': 'OpenVSLAM',
            'description': 'Visual SLAM with BoW-based loop closure',
            'sensors': ['monocular', 'stereo', 'rgbd'],
            'features': ['loop_closure', 'relocalization', 'mapping', 'bow_vocabulary'],
            'dependencies': ['openvslam', 'opencv-python', 'numpy', 'yaml']
        },
        'python_slam': {
            'name': 'Python SLAM',
            'description': 'Custom python-slam algorithm integration',
            'sensors': ['monocular', 'stereo', 'visual_inertial'],
            'features': ['feature_extraction', 'pose_estimation', 'mapping', 'loop_closure'],
            'dependencies': ['opencv-python', 'numpy', 'scipy']
        }
    }

    # Add availability status
    for alg_name, alg_info in info.items():
        alg_info['available'] = is_algorithm_available(alg_name)

    return info
