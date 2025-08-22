#!/usr/bin/env python3
"""
SLAM Interfaces Package

Unified interface system for multiple SLAM algorithms with factory pattern
for runtime algorithm selection and switching.

This package provides:
- Abstract SLAM interface for algorithm consistency
- Factory pattern for algorithm creation and management
- Wrapper implementations for popular SLAM algorithms
- Configuration management and sensor type detection
- Performance monitoring and error handling

Usage:
    from python_slam.slam_interfaces import SLAMFactory, SLAMConfiguration, SensorType

    # Create configuration
    config = SLAMConfiguration(
        algorithm_name="orb_slam3",
        sensor_type=SensorType.MONOCULAR,
        max_features=1000
    )

    # Create SLAM system
    factory = SLAMFactory()
    slam_system = factory.create_algorithm(config)

    # Initialize and use
    slam_system.initialize()
    slam_system.process_image(image, timestamp)
    pose = slam_system.get_pose()
"""

from .slam_interface import (
    SLAMInterface,
    SLAMConfiguration,
    SLAMPose,
    SLAMMapPoint,
    SLAMTrajectory,
    SLAMState,
    SensorType
)
from .slam_factory import SLAMFactory
from .algorithms import (
    ORBSlam3Wrapper,
    RTABMapWrapper,
    CartographerWrapper,
    OpenVSLAMWrapper,
    PythonSLAMWrapper,
    get_available_algorithms,
    is_algorithm_available,
    get_algorithm_class,
    get_algorithm_info
)

__version__ = "1.0.0"

__all__ = [
    # Core interfaces
    'SLAMInterface',
    'SLAMConfiguration',
    'SLAMPose',
    'SLAMMapPoint',
    'SLAMTrajectory',
    'SLAMState',
    'SensorType',

    # Factory
    'SLAMFactory',

    # Algorithm wrappers
    'ORBSlam3Wrapper',
    'RTABMapWrapper',
    'CartographerWrapper',
    'OpenVSLAMWrapper',
    'PythonSLAMWrapper',

    # Utility functions
    'get_available_algorithms',
    'is_algorithm_available',
    'get_algorithm_class',
    'get_algorithm_info'
]

# Package-level convenience functions
def create_slam_system(algorithm_name, sensor_type, **kwargs):
    """
    Convenience function to create a SLAM system.

    Args:
        algorithm_name: Name of the SLAM algorithm
        sensor_type: Type of sensor (SensorType enum or string)
        **kwargs: Additional configuration parameters

    Returns:
        Initialized SLAM system

    Example:
        slam = create_slam_system("orb_slam3", "monocular", max_features=1000)
    """
    # Convert string sensor type to enum if needed
    if isinstance(sensor_type, str):
        sensor_type = SensorType[sensor_type.upper()]

    # Create configuration
    config = SLAMConfiguration(
        algorithm_name=algorithm_name,
        sensor_type=sensor_type,
        **kwargs
    )

    # Create and return SLAM system
    factory = SLAMFactory()
    return factory.create_algorithm(config)


def list_algorithms():
    """List all available SLAM algorithms with their information."""
    available = get_available_algorithms()
    info = get_algorithm_info()

    print("Available SLAM Algorithms:")
    print("-" * 50)

    for alg_name in available:
        alg_info = info[alg_name]
        print(f"• {alg_info['name']}: {alg_info['description']}")
        print(f"  Sensors: {', '.join(alg_info['sensors'])}")
        print(f"  Features: {', '.join(alg_info['features'])}")
        print()

    unavailable = [name for name, avail in get_algorithm_info().items()
                   if not avail and name not in available]

    if unavailable:
        print("Unavailable Algorithms (missing dependencies):")
        print("-" * 50)
        for alg_name in unavailable:
            alg_info = info[alg_name]
            print(f"• {alg_info['name']}: {alg_info['description']}")
            print(f"  Dependencies: {', '.join(alg_info['dependencies'])}")
            print()


def check_dependencies():
    """Check and report the status of algorithm dependencies."""
    info = get_algorithm_info()

    print("SLAM Algorithm Dependency Status:")
    print("=" * 50)

    for alg_name, alg_info in info.items():
        status = "✓ Available" if alg_info['available'] else "✗ Missing dependencies"
        print(f"{alg_info['name']}: {status}")

        if not alg_info['available']:
            print(f"  Required: {', '.join(alg_info['dependencies'])}")
        print()


def get_recommended_algorithm(sensor_type, requirements=None):
    """
    Get recommended algorithm based on sensor type and requirements.

    Args:
        sensor_type: Type of sensor (SensorType enum or string)
        requirements: List of required features (optional)

    Returns:
        Recommended algorithm name or None if no match
    """
    if isinstance(sensor_type, str):
        sensor_type = SensorType[sensor_type.upper()]

    available = get_available_algorithms()
    info = get_algorithm_info()

    # Algorithm preferences by sensor type
    preferences = {
        SensorType.MONOCULAR: ['orb_slam3', 'openvslam', 'python_slam'],
        SensorType.STEREO: ['orb_slam3', 'rtabmap', 'openvslam', 'python_slam'],
        SensorType.RGBD: ['rtabmap', 'orb_slam3', 'openvslam'],
        SensorType.VISUAL_INERTIAL: ['orb_slam3', 'python_slam'],
        SensorType.LIDAR: ['cartographer'],
        SensorType.POINTCLOUD: ['cartographer', 'rtabmap']
    }

    sensor_str = sensor_type.value.lower()
    candidates = preferences.get(sensor_type, available)

    # Filter by availability and sensor support
    for alg_name in candidates:
        if alg_name in available:
            alg_info = info[alg_name]
            if sensor_str in alg_info['sensors']:
                # Check requirements if specified
                if requirements:
                    if all(req in alg_info['features'] for req in requirements):
                        return alg_name
                else:
                    return alg_name

    return None
