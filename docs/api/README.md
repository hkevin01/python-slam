# API Reference

Complete API reference for Python-SLAM components.

## Core Modules

### SLAM Engine
- [Feature Detection and Matching](slam_algorithms.md#feature-detection)
- [Pose Estimation](slam_algorithms.md#pose-estimation)
- [Bundle Adjustment](slam_algorithms.md#bundle-adjustment)
- [Loop Closure](slam_algorithms.md#loop-closure)

### GPU Acceleration
- [GPU Detection](gpu_acceleration.md#gpu-detection)
- [GPU Manager](gpu_acceleration.md#gpu-manager)
- [Accelerated Operations](gpu_acceleration.md#accelerated-operations)
- [Backend Support](gpu_acceleration.md#backend-support)

### GUI Framework
- [Main Window](gui_framework.md#main-window)
- [3D Visualization](gui_framework.md#3d-visualization)
- [Control Panels](gui_framework.md#control-panels)
- [Metrics Dashboard](gui_framework.md#metrics-dashboard)

### Benchmarking System
- [Trajectory Metrics](benchmarking.md#trajectory-metrics)
- [Processing Metrics](benchmarking.md#processing-metrics)
- [Benchmark Runner](benchmarking.md#benchmark-runner)
- [Report Generation](benchmarking.md#report-generation)

## Integration Modules

### ROS2 Integration
- [Nav2 Bridge](../advanced/ros2_integration.md#nav2-bridge)
- [Message Handling](../advanced/ros2_integration.md#message-handling)
- [Navigation Interface](../advanced/ros2_integration.md#navigation-interface)

### Embedded Optimization
- [ARM Optimization](../advanced/embedded_optimization.md#arm-optimization)
- [NEON SIMD](../advanced/embedded_optimization.md#neon-simd)
- [Cache Optimization](../advanced/embedded_optimization.md#cache-optimization)
- [Power Management](../advanced/embedded_optimization.md#power-management)

## Main System API

### PythonSLAMSystem

The main system class that orchestrates all components.

```python
from python_slam_main import PythonSLAMSystem

class PythonSLAMSystem:
    """Main Python-SLAM system controller."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the SLAM system.

        Args:
            config: System configuration dictionary
        """

    def initialize(self) -> bool:
        """
        Initialize all system components.

        Returns:
            bool: True if initialization successful
        """

    def run(self, mode: str = "full") -> None:
        """
        Run the SLAM system.

        Args:
            mode: Run mode ("full", "gui", "headless", "benchmark", "ros2")
        """

    def stop(self) -> None:
        """Stop the SLAM system gracefully."""

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status.

        Returns:
            Dict containing system status information
        """
```

### Configuration API

```python
def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Dict: Configuration dictionary
    """

def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration.

    Returns:
        Dict: Default configuration
    """

def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration.

    Args:
        config: Configuration to validate

    Returns:
        Tuple of (is_valid, error_messages)
    """
```

## Error Handling

### Exception Classes

```python
class PythonSLAMError(Exception):
    """Base exception for Python-SLAM."""
    pass

class ConfigurationError(PythonSLAMError):
    """Configuration-related errors."""
    pass

class GPUError(PythonSLAMError):
    """GPU acceleration errors."""
    pass

class SLAMError(PythonSLAMError):
    """SLAM algorithm errors."""
    pass

class BenchmarkError(PythonSLAMError):
    """Benchmarking errors."""
    pass

class GUIError(PythonSLAMError):
    """GUI-related errors."""
    pass
```

## Data Structures

### Common Data Types

```python
from typing import NamedTuple, List, Tuple, Optional
import numpy as np

class Pose3D(NamedTuple):
    """3D pose representation."""
    position: np.ndarray  # [x, y, z]
    orientation: np.ndarray  # [qx, qy, qz, qw] quaternion

class KeyFrame(NamedTuple):
    """SLAM keyframe representation."""
    id: int
    timestamp: float
    pose: Pose3D
    features: np.ndarray
    descriptors: np.ndarray
    image: Optional[np.ndarray] = None

class PointCloudData(NamedTuple):
    """Point cloud data structure."""
    points: np.ndarray  # Nx3 array of points
    colors: Optional[np.ndarray] = None  # Nx3 array of colors
    normals: Optional[np.ndarray] = None  # Nx3 array of normals

class TrajectoryData(NamedTuple):
    """Trajectory data structure."""
    poses: List[Pose3D]
    timestamps: List[float]
    keyframe_indices: Optional[List[int]] = None
```

## Performance Monitoring

### Metrics API

```python
from python_slam.benchmarking.benchmark_metrics import ProcessingMetrics

class ProcessingMetrics:
    """Real-time processing metrics."""

    def record_frame_time(self, frame_time: float) -> None:
        """Record frame processing time."""

    def record_memory_usage(self, memory_mb: float) -> None:
        """Record memory usage."""

    def record_cpu_usage(self, cpu_percent: float) -> None:
        """Record CPU usage."""

    def get_current_fps(self) -> float:
        """Get current FPS."""

    def get_average_fps(self) -> float:
        """Get average FPS."""

    def get_peak_memory_usage(self) -> float:
        """Get peak memory usage."""
```

## Plugin System

### Creating Custom Plugins

Python-SLAM supports plugins for extending functionality:

```python
from python_slam.core.plugin_interface import SLAMPlugin

class CustomSLAMPlugin(SLAMPlugin):
    """Custom SLAM plugin example."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "custom_plugin"
        self.version = "1.0.0"

    def initialize(self) -> bool:
        """Initialize plugin."""
        return True

    def process_frame(self, frame_data: Any) -> Any:
        """Process a single frame."""
        # Custom processing logic
        return processed_data

    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        pass
```

### Plugin Registration

```python
from python_slam.core.plugin_manager import PluginManager

# Register plugin
plugin_manager = PluginManager()
plugin_manager.register_plugin("custom_slam", CustomSLAMPlugin)

# Use plugin
plugin = plugin_manager.create_plugin("custom_slam", config)
```

## Callback System

### Event Callbacks

```python
from python_slam.core.events import EventManager, EventType

class SLAMEventCallback:
    """SLAM event callback interface."""

    def on_frame_processed(self, frame_data: Any) -> None:
        """Called when a frame is processed."""
        pass

    def on_keyframe_added(self, keyframe: KeyFrame) -> None:
        """Called when a keyframe is added."""
        pass

    def on_loop_closure(self, loop_data: Any) -> None:
        """Called when loop closure is detected."""
        pass

# Register callbacks
event_manager = EventManager()
callback = SLAMEventCallback()
event_manager.register_callback(EventType.FRAME_PROCESSED, callback.on_frame_processed)
```

## Configuration Schema

### Configuration Structure

```python
ConfigSchema = {
    "slam": {
        "algorithm": str,  # "basic", "advanced", "custom"
        "feature_detector": str,  # "ORB", "SIFT", "SURF"
        "descriptor_matcher": str,  # "BruteForce", "FLANN"
        "max_features": int,  # Maximum features per frame
        "keyframe_threshold": float,  # Keyframe selection threshold
    },
    "gpu": {
        "enable_acceleration": bool,
        "preferred_backend": str,  # "auto", "cuda", "rocm", "metal", "cpu"
        "memory_limit_mb": int,
        "enable_profiling": bool,
    },
    "gui": {
        "enable_gui": bool,
        "theme": str,  # "dark", "light", "auto"
        "update_rate_hz": int,
        "window_size": List[int],  # [width, height]
    },
    "benchmarking": {
        "enable_metrics": bool,
        "save_trajectory": bool,
        "output_directory": str,
        "evaluation_metrics": List[str],  # ["ATE", "RPE", "processing_time"]
    },
    "ros2": {
        "enable_integration": bool,
        "node_name": str,
        "namespace": str,
        "use_sim_time": bool,
    },
    "embedded": {
        "enable_optimization": bool,
        "target_architecture": str,  # "auto", "arm64", "x86_64"
        "optimization_level": str,  # "conservative", "balanced", "aggressive"
    }
}
```

## Logging and Debugging

### Logging Configuration

```python
import logging
from python_slam.core.logging import configure_logging

# Configure logging
configure_logging(
    level=logging.INFO,
    log_file="python_slam.log",
    enable_console=True,
    enable_file=True
)

# Use logger
logger = logging.getLogger("python_slam")
logger.info("System initialized")
```

### Debug Tools

```python
from python_slam.core.debug import DebugManager

# Enable debug mode
debug_manager = DebugManager()
debug_manager.enable_debug_mode()

# Set debug level
debug_manager.set_debug_level("verbose")  # "none", "basic", "verbose", "all"

# Debug visualization
debug_manager.enable_debug_visualization()
```

## Thread Safety

Python-SLAM is designed to be thread-safe for concurrent operations:

- **GPU operations**: Thread-safe GPU context management
- **Metrics collection**: Lock-free data structures for performance metrics
- **GUI updates**: Proper Qt signal/slot mechanism for thread communication
- **Configuration**: Read-only configuration after initialization

## Memory Management

### Automatic Resource Management

```python
from python_slam.core.resource_manager import ResourceManager

# Automatic cleanup
with ResourceManager() as rm:
    # Resources are automatically cleaned up
    gpu_context = rm.acquire_gpu_context()
    # ... use resources
# Resources released automatically
```

### Manual Resource Control

```python
from python_slam.gpu_acceleration.gpu_manager import GPUManager

gpu_manager = GPUManager()
try:
    gpu_manager.initialize_accelerators()
    # ... use GPU resources
finally:
    gpu_manager.cleanup()  # Manual cleanup
```

This API reference provides a comprehensive overview of the Python-SLAM system. For specific implementation details, refer to the individual module documentation.
