# Python SLAM - Multi-Algorithm SLAM Framework

A comprehensive ROS2 SLAM framework with support for multiple algorithms, runtime switching, and unified interfaces.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Algorithms](#supported-algorithms)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

Python SLAM provides a unified interface for multiple SLAM algorithms with runtime algorithm switching capabilities. The framework supports various sensor modalities including monocular, stereo, RGB-D cameras, IMU, LiDAR, and point clouds.

### Key Components

- **Abstract SLAM Interface**: Unified API for all algorithms
- **Factory Pattern**: Runtime algorithm creation and switching
- **Algorithm Wrappers**: Integration with popular SLAM libraries
- **ROS2 Integration**: Complete ROS2 node with topic/service interfaces
- **Performance Monitoring**: Real-time metrics and diagnostics
- **Configuration Management**: YAML-based parameter management

## Features

- ✅ **Multi-Algorithm Support**: ORB-SLAM3, RTAB-Map, Cartographer, OpenVSLAM, Custom Python SLAM
- ✅ **Runtime Algorithm Switching**: Hot-swap algorithms without restart
- ✅ **Sensor Flexibility**: Monocular, stereo, RGB-D, visual-inertial, LiDAR
- ✅ **Automatic Sensor Detection**: Auto-configure based on available topics
- ✅ **Loop Closure Detection**: Integrated across all algorithms
- ✅ **Map Persistence**: Save/load maps with algorithm state
- ✅ **Performance Monitoring**: Real-time FPS, processing time, memory usage
- ✅ **TF Integration**: Automatic transform publishing
- ✅ **RViz Visualization**: Comprehensive visualization setup

## Supported Algorithms

| Algorithm | Sensors | Features | Dependencies |
|-----------|---------|----------|--------------|
| **ORB-SLAM3** | Mono, Stereo, RGB-D, VI | Feature-based, Loop Closure | `orbslam3-python` |
| **RTAB-Map** | RGB-D, Stereo, Mono | Appearance-based, Graph SLAM | `rtabmap-python` |
| **Cartographer** | LiDAR, Point Cloud | 2D/3D Grid SLAM | `cartographer-ros` |
| **OpenVSLAM** | Mono, Stereo, RGB-D | Visual SLAM, BoW | `openvslam` |
| **Python SLAM** | Mono, Stereo, VI | Custom Implementation | `opencv-python` |

## Installation

### Prerequisites

- ROS2 Humble or later
- Python 3.8+
- OpenCV 4.0+
- NumPy, SciPy

### Install Dependencies

```bash
# Core dependencies
sudo apt update
sudo apt install python3-opencv python3-numpy python3-scipy

# ROS2 dependencies
sudo apt install ros-humble-cv-bridge ros-humble-tf2-ros ros-humble-visualization-msgs

# Optional algorithm dependencies
pip install orbslam3-python  # For ORB-SLAM3
pip install rtabmap-python   # For RTAB-Map
# For Cartographer, follow official installation guide
pip install openvslam        # For OpenVSLAM
```

### Build from Source

```bash
# Create workspace
mkdir -p ~/slam_ws/src
cd ~/slam_ws/src

# Clone repository
git clone https://github.com/hkevin01/python-slam.git

# Build
cd ~/slam_ws
colcon build --packages-select python_slam

# Source workspace
source install/setup.bash
```

## Quick Start

### 1. Basic Monocular SLAM

```bash
# Launch with auto-algorithm selection
ros2 launch python_slam multi_slam_launch.py sensor_type:=monocular

# Or specify algorithm
ros2 launch python_slam multi_slam_launch.py algorithm:=orb_slam3 sensor_type:=monocular
```

### 2. RGB-D SLAM

```bash
ros2 launch python_slam multi_slam_launch.py algorithm:=rtabmap sensor_type:=rgbd \
  image_topic:=/camera/color/image_raw \
  depth_topic:=/camera/depth/image_raw
```

### 3. Stereo SLAM

```bash
ros2 launch python_slam multi_slam_launch.py algorithm:=orb_slam3 sensor_type:=stereo \
  left_image_topic:=/camera/left/image_raw \
  right_image_topic:=/camera/right/image_raw
```

### 4. LiDAR SLAM

```bash
ros2 launch python_slam multi_slam_launch.py algorithm:=cartographer sensor_type:=lidar \
  laser_topic:=/scan
```

## Usage

### Using the Python API

```python
from python_slam.slam_interfaces import create_slam_system, SensorType

# Create SLAM system
slam = create_slam_system("orb_slam3", SensorType.MONOCULAR, max_features=1000)

# Initialize
slam.initialize()

# Process image
import cv2
image = cv2.imread("frame.jpg")
success = slam.process_image(image, timestamp=1.0)

# Get pose
pose = slam.get_pose()
if pose:
    print(f"Position: {pose.position}")
    print(f"Orientation: {pose.orientation}")

# Get map
map_points = slam.get_map()
print(f"Map has {len(map_points)} points")
```

### Runtime Algorithm Switching

```bash
# Switch to different algorithm
ros2 service call /slam/switch_algorithm python_slam_msgs/srv/SwitchAlgorithm \
  "{algorithm_name: 'rtabmap'}"

# Save current map
ros2 service call /slam/save_map python_slam_msgs/srv/SaveMap \
  "{filepath: '/path/to/map'}"

# Reset SLAM system
ros2 service call /slam/reset std_srvs/srv/Trigger
```

### Monitoring Performance

```bash
# View status
ros2 topic echo /slam/status

# View performance metrics
ros2 topic echo /slam/metrics

# View current pose
ros2 topic echo /slam/pose
```

## Configuration

### Parameters

The system uses ROS2 parameters for configuration:

```yaml
# Algorithm selection
algorithm: "auto"              # auto, orb_slam3, rtabmap, cartographer, openvslam, python_slam
sensor_type: "auto"            # auto, monocular, stereo, rgbd, visual_inertial, lidar

# Frame names
map_frame: "map"
odom_frame: "odom"
base_frame: "base_link"
camera_frame: "camera_link"

# SLAM parameters
max_features: 1000
enable_loop_closure: true
enable_mapping: true
publish_tf: true

# Camera intrinsics
camera_fx: 525.0
camera_fy: 525.0
camera_cx: 319.5
camera_cy: 239.5
camera_baseline: 0.1           # For stereo

# Performance
pose_publish_rate: 30.0
map_publish_rate: 1.0
```

### Algorithm-Specific Configuration

Each algorithm can be configured with custom parameters:

```python
config = SLAMConfiguration(
    algorithm_name="orb_slam3",
    sensor_type=SensorType.MONOCULAR,
    custom_params={
        'camera': {'fx': 525.0, 'fy': 525.0},
        'orb_slam3': {
            'nFeatures': 1000,
            'scaleFactor': 1.2,
            'nLevels': 8
        }
    }
)
```

## API Reference

### Core Classes

#### SLAMInterface

Abstract base class for all SLAM algorithms.

```python
class SLAMInterface:
    def initialize() -> bool
    def process_image(image, timestamp) -> bool
    def process_stereo_images(left, right, timestamp) -> bool
    def process_pointcloud(pointcloud, timestamp) -> bool
    def process_imu(imu_data, timestamp) -> bool
    def get_pose() -> Optional[SLAMPose]
    def get_map() -> List[SLAMMapPoint]
    def reset() -> bool
    def save_map(filepath) -> bool
    def load_map(filepath) -> bool
```

#### SLAMFactory

Factory for creating and managing SLAM algorithms.

```python
factory = SLAMFactory()
slam_system = factory.create_algorithm(config)
factory.switch_algorithm(new_config)
available = factory.get_available_algorithms()
```

#### SLAMConfiguration

Configuration class for SLAM parameters.

```python
config = SLAMConfiguration(
    algorithm_name="orb_slam3",
    sensor_type=SensorType.MONOCULAR,
    max_features=1000,
    enable_loop_closure=True,
    custom_params={}
)
```

### Data Structures

#### SLAMPose

Robot pose representation.

```python
@dataclass
class SLAMPose:
    position: np.ndarray      # [x, y, z]
    orientation: np.ndarray   # [qx, qy, qz, qw]
    timestamp: float
    frame_id: str = "map"
```

#### SLAMMapPoint

3D map point representation.

```python
@dataclass
class SLAMMapPoint:
    position: np.ndarray      # [x, y, z]
    confidence: float         # 0.0 to 1.0
    observations: int = 0
```

## Examples

### 1. Custom Algorithm Implementation

```python
from python_slam.slam_interfaces import SLAMInterface, SLAMConfiguration

class CustomSLAM(SLAMInterface):
    def __init__(self, config: SLAMConfiguration):
        super().__init__(config)
        # Initialize custom algorithm

    def initialize(self) -> bool:
        # Setup algorithm
        return True

    def process_image(self, image, timestamp) -> bool:
        # Process image and update pose
        return True

    # Implement other required methods...

# Register with factory
from python_slam.slam_interfaces import SLAMFactory
factory = SLAMFactory()
factory.register_algorithm("custom", CustomSLAM)
```

### 2. Multi-Sensor Fusion

```python
# Create multi-sensor configuration
config = SLAMConfiguration(
    algorithm_name="orb_slam3",
    sensor_type=SensorType.VISUAL_INERTIAL,
    max_features=1500,
    custom_params={
        'camera': {'fx': 525.0, 'fy': 525.0},
        'imu': {'noise_gyro': 0.01, 'noise_acc': 0.1}
    }
)

slam = create_slam_system(config)
slam.initialize()

# Process camera and IMU data
slam.process_image(camera_image, camera_timestamp)
slam.process_imu(imu_data, imu_timestamp)
```

### 3. Map Management

```python
# Save map with timestamp
import time
map_file = f"map_{int(time.time())}.dat"
slam.save_map(map_file)

# Load previous map
slam.load_map("previous_map.dat")

# Get map statistics
map_points = slam.get_map()
trajectory = slam.get_trajectory()
print(f"Map: {len(map_points)} points, {len(trajectory.poses)} poses")
```

## Troubleshooting

### Common Issues

1. **Algorithm not available**
   ```bash
   # Check available algorithms
   python3 -c "from python_slam.slam_interfaces import list_algorithms; list_algorithms()"
   ```

2. **Camera calibration errors**
   - Verify camera parameters in configuration
   - Use camera calibration tools to get accurate intrinsics

3. **Performance issues**
   - Reduce max_features parameter
   - Disable loop closure for real-time performance
   - Check system resources

4. **TF frame errors**
   - Verify frame names match your robot setup
   - Check TF tree with `ros2 run tf2_tools view_frames`

### Debug Mode

Enable verbose logging:

```bash
ros2 launch python_slam multi_slam_launch.py log_level:=debug
```

### Performance Profiling

```python
# Get detailed metrics
metrics = slam.get_performance_metrics()
print(f"Average FPS: {metrics['average_fps']}")
print(f"Processing time: {metrics['avg_processing_time_ms']} ms")
```

## Contributing

### Development Setup

```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
cd ~/slam_ws
colcon test --packages-select python_slam

# Format code
black src/python_slam/
```

### Adding New Algorithms

1. Create wrapper class inheriting from `SLAMInterface`
2. Implement all required methods
3. Add to `algorithms/__init__.py`
4. Update factory registration
5. Add tests and documentation

### Submitting Changes

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ORB-SLAM3 team for the excellent visual SLAM system
- RTAB-Map developers for appearance-based mapping
- Google Cartographer team for grid-based SLAM
- OpenVSLAM community for open-source visual SLAM
- ROS2 community for the robotics framework

## Support

- GitHub Issues: [Report bugs and request features](https://github.com/hkevin01/python-slam/issues)
- Documentation: [Full API documentation](https://github.com/hkevin01/python-slam/wiki)
- Examples: [Additional examples and tutorials](https://github.com/hkevin01/python-slam/examples)

---

For more information, visit the [project homepage](https://github.com/hkevin01/python-slam).
