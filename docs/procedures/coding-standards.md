# Coding Standards and Guidelines

## Python-SLAM Real-Time Visual SLAM System

**Document Number**: CS-PYTHON-SLAM-001
**Version**: 1.0
**Date**: October 2, 2025
**Prepared by**: Python-SLAM Development Team
**Approved by**: [Technical Lead]

---

## 1. Introduction

### 1.1 Purpose

This document establishes coding standards, conventions, and best practices for the Python-SLAM project. These standards ensure code consistency, maintainability, readability, and quality across all development team members.

### 1.2 Scope

**Applicable Code**:
- Python source code (`.py` files)
- CUDA/OpenCL kernel code (`.cu`, `.cl` files)
- GLSL shader code (`.vert`, `.frag`, `.geom` files)
- Configuration files (`.json`, `.yaml`, `.toml`)
- Build scripts and automation
- Documentation and comments

### 1.3 Compliance Requirements

**Mandatory Standards** (REQ-NF-M-001):
- All code must pass automated style checks
- Code review required before merge to protected branches
- Documentation required for all public APIs
- Test coverage required for new functionality

---

## 2. Python Coding Standards

### 2.1 Style Guide Compliance

**Primary Standard**: PEP 8 - Python Enhancement Proposal 8

**Key Requirements**:
- Line length: 88 characters (Black formatter default)
- Indentation: 4 spaces (no tabs)
- Encoding: UTF-8
- Import organization: isort with Black compatibility

### 2.2 Naming Conventions

#### 2.2.1 Variable and Function Names

```python
# REQ-NF-M-001: Code Documentation and Maintainability

# Variables: snake_case
frame_count = 0
processing_time_ms = 15.5
camera_intrinsics_matrix = np.eye(3)

# Functions: snake_case with descriptive verbs
def extract_features(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract visual features from input image."""
    pass

def compute_pose_estimate(features: FeatureSet) -> Pose6DOF:
    """Compute 6-DOF pose from feature correspondences."""
    pass

# Private functions: leading underscore
def _validate_input_format(data: Any) -> bool:
    """Internal validation function."""
    pass

# Constants: UPPER_SNAKE_CASE
MAX_FEATURE_COUNT = 1000
DEFAULT_TRACKING_THRESHOLD = 0.75
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.png', '.bmp']
```

#### 2.2.2 Class Names

```python
# Classes: PascalCase with descriptive nouns
class SLAMPipeline:
    """Main SLAM processing pipeline."""

    def __init__(self, config: SLAMConfig) -> None:
        self.config = config
        self._is_initialized = False

    def process_frame(self, frame: StereoFrame) -> SLAMResult:
        """Process single stereo frame through SLAM pipeline."""
        pass

class FeatureExtractor:
    """Visual feature extraction and description."""
    pass

class GPUAcceleratedOperations:
    """GPU-accelerated SLAM operations."""
    pass

# Abstract base classes: suffix with 'Base' or 'ABC'
class SLAMAlgorithmBase(ABC):
    """Abstract base class for SLAM algorithms."""

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data - must be implemented by subclasses."""
        pass
```

#### 2.2.3 Module and Package Names

```python
# Modules: lowercase with underscores
# src/python_slam/feature_extraction.py
# src/python_slam/gpu_acceleration/cuda_operations.py
# src/python_slam/gui/visualization_widgets.py

# Packages: lowercase, short names
# src/python_slam/
# src/python_slam/benchmarking/
# src/python_slam/ros2_integration/
```

### 2.3 Type Annotations

#### 2.3.1 Mandatory Type Hints

```python
from typing import Optional, List, Dict, Tuple, Union, Any
import numpy as np

# REQ-NF-M-001: Comprehensive documentation requirements

def process_stereo_pair(
    left_image: np.ndarray,
    right_image: np.ndarray,
    camera_params: CameraParameters,
    previous_pose: Optional[Pose6DOF] = None
) -> SLAMResult:
    """
    Process stereo image pair through SLAM pipeline.

    Fulfills Requirements:
    - REQ-F-001: Real-time SLAM processing
    - REQ-F-002: Feature extraction and tracking

    Args:
        left_image: Left camera image (H, W, 3) uint8 array
        right_image: Right camera image (H, W, 3) uint8 array
        camera_params: Calibrated stereo camera parameters
        previous_pose: Previous pose estimate for initialization

    Returns:
        Complete SLAM result with pose, map updates, and metadata

    Raises:
        SLAMProcessingError: When processing fails due to insufficient features
        CameraCalibrationError: When camera parameters are invalid

    Example:
        >>> left_img = load_image('left.jpg')
        >>> right_img = load_image('right.jpg')
        >>> params = load_camera_calibration('stereo_calib.json')
        >>> result = process_stereo_pair(left_img, right_img, params)
        >>> print(f"Pose: {result.pose}")
    """
    # Implementation here
    pass

# Class type annotations
class PointCloudRenderer:
    """3D point cloud rendering with OpenGL."""

    def __init__(self) -> None:
        self.vertex_buffer: Optional[int] = None
        self.point_count: int = 0
        self.is_initialized: bool = False

    def update_points(self, points: np.ndarray, colors: np.ndarray) -> bool:
        """Update point cloud data for rendering."""
        pass
```

#### 2.3.2 Complex Type Definitions

```python
from typing import TypeVar, Generic, Protocol
from dataclasses import dataclass

# Type variables for generic classes
T = TypeVar('T')
ImageType = TypeVar('ImageType', bound=np.ndarray)

# Protocol definitions for interface contracts
class SLAMAlgorithm(Protocol):
    """Protocol defining SLAM algorithm interface."""

    def initialize(self, config: Dict[str, Any]) -> bool: ...
    def process_frame(self, data: Any) -> SLAMResult: ...
    def get_current_pose(self) -> Pose6DOF: ...

# Generic classes
class DataBuffer(Generic[T]):
    """Thread-safe circular buffer for data storage."""

    def __init__(self, capacity: int) -> None:
        self._buffer: List[Optional[T]] = [None] * capacity
        self._head: int = 0
        self._tail: int = 0

    def push(self, item: T) -> bool:
        """Add item to buffer."""
        pass

    def pop(self) -> Optional[T]:
        """Remove and return oldest item."""
        pass
```

### 2.4 Error Handling Standards

#### 2.4.1 Exception Hierarchy

```python
# REQ-NF-R-002: Data integrity and error handling
# REQ-NF-R-003: Error recovery capabilities

class PythonSLAMError(Exception):
    """Base exception for all Python-SLAM errors."""

    def __init__(self, message: str, error_code: Optional[str] = None) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = time.time()

class SLAMProcessingError(PythonSLAMError):
    """Raised when SLAM processing fails."""
    pass

class CameraCalibrationError(PythonSLAMError):
    """Raised when camera calibration is invalid."""
    pass

class GPUAccelerationError(PythonSLAMError):
    """Raised when GPU operations fail."""
    pass

class ConfigurationError(PythonSLAMError):
    """Raised when configuration is invalid."""
    pass

# Usage example with proper error handling
def extract_features_safe(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features with comprehensive error handling.

    Fulfills Requirements:
    - REQ-NF-R-002: Data integrity validation
    - REQ-NF-R-003: Graceful error recovery
    """
    try:
        # Validate input data
        if not isinstance(image, np.ndarray):
            raise SLAMProcessingError(
                "Invalid input type: expected numpy array",
                error_code="INVALID_INPUT_TYPE"
            )

        if image.size == 0:
            raise SLAMProcessingError(
                "Empty image provided",
                error_code="EMPTY_IMAGE"
            )

        # Attempt feature extraction
        keypoints, descriptors = _extract_features_impl(image)

        # Validate output
        if len(keypoints) == 0:
            logger.warning("No features detected in image")
            # Return empty results rather than failing
            return np.array([]), np.array([])

        return keypoints, descriptors

    except GPUAccelerationError as e:
        # Fallback to CPU processing
        logger.warning(f"GPU processing failed: {e}, falling back to CPU")
        return _extract_features_cpu(image)

    except Exception as e:
        # Log unexpected errors and re-raise
        logger.error(f"Unexpected error in feature extraction: {e}", exc_info=True)
        raise SLAMProcessingError(
            f"Feature extraction failed: {str(e)}",
            error_code="UNEXPECTED_ERROR"
        ) from e
```

#### 2.4.2 Logging Standards

```python
import logging
from typing import Any, Dict

# REQ-NF-M-001: Comprehensive logging for maintainability

# Module-level logger configuration
logger = logging.getLogger(__name__)

class SLAMLogger:
    """Structured logging for SLAM operations."""

    @staticmethod
    def log_performance(operation: str, duration_ms: float, **kwargs: Any) -> None:
        """Log performance metrics with structured data."""
        logger.info(
            f"PERFORMANCE: {operation} completed in {duration_ms:.2f}ms",
            extra={
                'operation': operation,
                'duration_ms': duration_ms,
                'performance_data': kwargs
            }
        )

    @staticmethod
    def log_slam_result(result: SLAMResult) -> None:
        """Log SLAM processing results."""
        logger.info(
            f"SLAM: pose=({result.pose.translation[0]:.3f}, "
            f"{result.pose.translation[1]:.3f}, {result.pose.translation[2]:.3f}), "
            f"features={result.feature_count}, time={result.processing_time:.2f}ms"
        )

    @staticmethod
    def log_error_context(error: Exception, context: Dict[str, Any]) -> None:
        """Log errors with additional context."""
        logger.error(
            f"ERROR: {type(error).__name__}: {str(error)}",
            extra={'error_context': context},
            exc_info=True
        )

# Usage in SLAM pipeline
def process_frame_with_logging(frame: StereoFrame) -> SLAMResult:
    """Process frame with comprehensive logging."""
    start_time = time.perf_counter()

    try:
        logger.debug(f"Processing frame {frame.timestamp}")

        # Feature extraction with timing
        feature_start = time.perf_counter()
        features = extract_features(frame.left_image)
        feature_time = (time.perf_counter() - feature_start) * 1000

        SLAMLogger.log_performance("feature_extraction", feature_time,
                                 feature_count=len(features))

        # Complete processing
        result = complete_slam_processing(frame, features)

        # Log results
        SLAMLogger.log_slam_result(result)

        total_time = (time.perf_counter() - start_time) * 1000
        SLAMLogger.log_performance("frame_processing", total_time)

        return result

    except Exception as e:
        context = {
            'frame_timestamp': frame.timestamp,
            'image_shape': frame.left_image.shape,
            'processing_time_ms': (time.perf_counter() - start_time) * 1000
        }
        SLAMLogger.log_error_context(e, context)
        raise
```

---

## 3. Documentation Standards

### 3.1 Docstring Format

**Standard**: Google-style docstrings with type information

```python
def compute_fundamental_matrix(
    points1: np.ndarray,
    points2: np.ndarray,
    method: str = 'RANSAC'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute fundamental matrix from point correspondences.

    Fulfills Requirements:
    - REQ-F-002: Feature extraction and tracking
    - REQ-F-004: Loop closure detection

    This function estimates the fundamental matrix that relates corresponding
    points between two views of the same scene. The fundamental matrix
    encodes the epipolar geometry between the cameras.

    Args:
        points1: Points in first image (N, 2) array of [x, y] coordinates
        points2: Corresponding points in second image (N, 2) array
        method: Estimation method ('RANSAC', '8-point', 'LMedS')
            - 'RANSAC': Robust estimation with outlier rejection
            - '8-point': Classical 8-point algorithm
            - 'LMedS': Least Median of Squares estimation

    Returns:
        A tuple containing:
        - fundamental_matrix: 3x3 fundamental matrix
        - inlier_mask: Boolean array indicating inlier correspondences

    Raises:
        SLAMProcessingError: When insufficient point correspondences (<8 points)
        ValueError: When invalid method specified

    Example:
        >>> pts1 = np.array([[100, 200], [150, 250], [200, 300]])
        >>> pts2 = np.array([[105, 205], [155, 255], [205, 305]])
        >>> F, mask = compute_fundamental_matrix(pts1, pts2)
        >>> print(f"Fundamental matrix shape: {F.shape}")
        Fundamental matrix shape: (3, 3)

    Note:
        Points should be normalized to image coordinates for numerical stability.
        The fundamental matrix satisfies: pts2^T * F * pts1 = 0 for all
        corresponding point pairs.

    References:
        Hartley, R. and Zisserman, A. (2003). Multiple View Geometry in Computer Vision.
        Cambridge University Press.
    """
    # Implementation here
    pass
```

### 3.2 Code Comments

```python
# REQ-F-001: Real-time SLAM processing requirements
class SLAMPipeline:
    """
    Main SLAM processing pipeline with real-time constraints.

    This class implements the core SLAM algorithm that processes stereo
    camera input to simultaneously build a map and track camera pose.
    """

    def __init__(self, config: SLAMConfig) -> None:
        # REQ-F-010: Configuration management
        self.config = config

        # Initialize feature extraction with configured detector
        # REQ-F-002: Support for multiple feature detectors (ORB, SIFT, SURF)
        if config.feature_detector == 'ORB':
            self.feature_extractor = ORBExtractor(config.orb_params)
        elif config.feature_detector == 'SIFT':
            self.feature_extractor = SIFTExtractor(config.sift_params)
        else:
            raise ConfigurationError(f"Unsupported detector: {config.feature_detector}")

        # REQ-F-007: GPU acceleration initialization
        self.gpu_manager = GPUManager()
        if config.enable_gpu and self.gpu_manager.is_available():
            logger.info(f"GPU acceleration enabled: {self.gpu_manager.get_device_info()}")
        else:
            logger.info("Using CPU-only processing")

        # Performance monitoring for REQ-NF-P-001
        self.performance_monitor = PerformanceMonitor()

    def process_frame(self, stereo_frame: StereoFrame) -> SLAMResult:
        """
        Process single stereo frame through complete SLAM pipeline.

        REQ-NF-P-001: Must complete processing in <100ms for real-time operation
        """
        start_time = time.perf_counter()

        # Step 1: Feature extraction from left image
        # REQ-F-002: Extract minimum 500 features for robust tracking
        features = self.feature_extractor.extract(stereo_frame.left_image)
        if len(features.keypoints) < self.config.min_features:
            logger.warning(f"Low feature count: {len(features.keypoints)}")

        # Step 2: Stereo matching for depth estimation
        # REQ-F-003: Generate 3D point cloud from stereo vision
        depth_map = self._compute_stereo_depth(
            stereo_frame.left_image,
            stereo_frame.right_image
        )

        # Step 3: Track features from previous frame
        if self.previous_features is not None:
            matches = self._track_features(self.previous_features, features)
        else:
            matches = []  # First frame, no tracking possible

        # Step 4: Pose estimation using PnP if we have 3D-2D correspondences
        if len(matches) >= self.config.min_tracking_features:
            pose = self._estimate_pose_pnp(matches, depth_map)
        else:
            # REQ-NF-R-003: Graceful handling of tracking loss
            logger.warning("Insufficient features for pose estimation")
            pose = self._predict_pose_from_motion_model()

        # Step 5: Update map with new observations
        # REQ-F-003: Maintain consistent 3D map
        self._update_map(features, depth_map, pose)

        # Step 6: Loop closure detection for drift correction
        # REQ-F-004: Detect revisited locations
        loop_closure = self._detect_loop_closure(features)
        if loop_closure.detected:
            logger.info(f"Loop closure detected: {loop_closure.confidence}")
            self._perform_global_optimization(loop_closure)

        # Performance monitoring
        processing_time = (time.perf_counter() - start_time) * 1000
        self.performance_monitor.record_frame_time(processing_time)

        # REQ-NF-P-001: Ensure real-time performance constraint
        if processing_time > 100:  # 100ms threshold
            logger.warning(f"Frame processing exceeded real-time threshold: {processing_time:.2f}ms")

        # Store features for next frame tracking
        self.previous_features = features

        return SLAMResult(
            pose=pose,
            new_map_points=self._get_new_map_points(),
            loop_closure_detected=loop_closure.detected,
            processing_time=processing_time,
            feature_count=len(features.keypoints),
            tracking_quality=self._compute_tracking_quality(matches)
        )
```

### 3.3 File Headers

```python
#!/usr/bin/env python3
"""
GPU Acceleration Manager for Python-SLAM

This module provides unified GPU acceleration across multiple vendors (NVIDIA, AMD, Apple)
with automatic fallback to CPU processing when GPU is unavailable.

Fulfills Requirements:
- REQ-F-007: GPU acceleration support for CUDA, ROCm, and Metal
- REQ-NF-P-002: >3x performance improvement with GPU acceleration
- REQ-NF-R-003: Graceful fallback when GPU unavailable

Author: Python-SLAM Development Team
Created: 2025-10-02
License: MIT
"""

import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

# Version information
__version__ = "1.0.0"
__author__ = "Python-SLAM Development Team"
__email__ = "dev@python-slam.org"

# Module-level logger
logger = logging.getLogger(__name__)

# Public API exports
__all__ = [
    'GPUManager',
    'GPUBackend',
    'CUDABackend',
    'ROCmBackend',
    'MetalBackend',
    'GPUAccelerationError'
]
```

---

## 4. Testing Standards

### 4.1 Test Code Organization

```python
"""
Test suite for GPU acceleration functionality.

Test Categories:
- Unit tests: Individual component testing
- Integration tests: Multi-component interaction testing
- Performance tests: Timing and resource usage validation
- Hardware tests: Platform-specific GPU testing

Fulfills Requirements:
- REQ-F-007: GPU acceleration verification
- REQ-NF-P-002: Performance improvement validation
"""

import unittest
import numpy as np
import time
from unittest.mock import Mock, patch
import pytest

# REQ-F-007: GPU Acceleration Support Testing
class TestGPUManager(unittest.TestCase):
    """Test GPU manager functionality and backend selection."""

    def setUp(self) -> None:
        """Set up test environment before each test."""
        self.test_data = np.random.randn(1000, 128).astype(np.float32)
        self.gpu_manager = GPUManager()

    def tearDown(self) -> None:
        """Clean up after each test."""
        if hasattr(self, 'gpu_manager'):
            self.gpu_manager.cleanup()

    def test_gpu_detection(self) -> None:
        """
        Test GPU detection across multiple vendors.

        Verifies REQ-F-007: Support for NVIDIA, AMD, Apple GPUs
        """
        # Test CUDA detection
        with patch('cupy.cuda.is_available') as mock_cuda:
            mock_cuda.return_value = True
            gpus = self.gpu_manager.detect_all_gpus()
            self.assertGreater(len(gpus), 0, "Should detect at least CPU fallback")

        # Test backend initialization
        success = self.gpu_manager.initialize_accelerators()
        self.assertIsInstance(success, bool, "Initialization should return boolean")

    def test_performance_improvement(self) -> None:
        """
        Test GPU acceleration performance improvement.

        Verifies REQ-NF-P-002: >3x speedup requirement
        """
        # Skip if no GPU available
        if not self.gpu_manager.has_gpu():
            self.skipTest("No GPU available for performance testing")

        # Benchmark CPU processing
        start_time = time.perf_counter()
        cpu_result = self._cpu_matrix_multiply(self.test_data, self.test_data.T)
        cpu_time = time.perf_counter() - start_time

        # Benchmark GPU processing
        start_time = time.perf_counter()
        gpu_result = self.gpu_manager.matrix_multiply(self.test_data, self.test_data.T)
        gpu_time = time.perf_counter() - start_time

        # Verify results are equivalent
        np.testing.assert_allclose(cpu_result, gpu_result, rtol=1e-5)

        # Verify performance improvement
        speedup = cpu_time / gpu_time
        self.assertGreater(speedup, 3.0,
                          f"GPU should be >3x faster than CPU, got {speedup:.2f}x")

    @pytest.mark.slow
    def test_extended_gpu_operation(self) -> None:
        """Test GPU stability under extended operation."""
        if not self.gpu_manager.has_gpu():
            self.skipTest("No GPU available")

        # Run operations for extended period
        for i in range(100):
            result = self.gpu_manager.matrix_multiply(self.test_data, self.test_data.T)
            self.assertIsNotNone(result, f"Operation {i} should not fail")

    def _cpu_matrix_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """CPU reference implementation for performance comparison."""
        return np.dot(a, b)

# Performance benchmarking with pytest-benchmark
def test_feature_extraction_performance(benchmark):
    """Benchmark feature extraction performance."""
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    extractor = FeatureExtractor()

    # Benchmark the feature extraction
    result = benchmark(extractor.extract_features, image)

    # Verify minimum feature count (REQ-F-002)
    keypoints, descriptors = result
    assert len(keypoints) >= 500, "Should extract at least 500 features"

# Integration testing with multiple components
class TestSLAMIntegration(unittest.TestCase):
    """Test integration between SLAM components."""

    def test_end_to_end_processing(self) -> None:
        """
        Test complete SLAM pipeline end-to-end.

        Verifies REQ-F-001: Real-time SLAM processing
        """
        # Create test stereo frame
        left_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        right_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        stereo_frame = StereoFrame(left_image, right_image, timestamp=time.time())

        # Initialize SLAM pipeline
        config = SLAMConfig()
        slam_pipeline = SLAMPipeline(config)

        # Process frame
        start_time = time.perf_counter()
        result = slam_pipeline.process_frame(stereo_frame)
        processing_time = (time.perf_counter() - start_time) * 1000

        # Verify real-time performance (REQ-NF-P-001)
        self.assertLess(processing_time, 100,
                       f"Processing should be <100ms, got {processing_time:.2f}ms")

        # Verify result completeness
        self.assertIsInstance(result, SLAMResult)
        self.assertIsInstance(result.pose, Pose6DOF)
        self.assertGreater(result.feature_count, 0)
```

### 4.2 Test Documentation

```python
def test_loop_closure_detection() -> None:
    """
    Test loop closure detection accuracy and performance.

    Test Scenario:
        Simulate robot returning to previously visited location

    Test Data:
        - Synthetic image sequence with known loop closure at frame 100
        - Indoor office environment with distinctive features
        - Controlled lighting and camera motion

    Expected Results:
        - Loop closure detected within 5 frames of actual closure
        - False positive rate <1%
        - Detection confidence >0.8
        - Processing time <100ms per frame

    Requirements Verified:
        - REQ-F-004: Loop closure detection with >95% accuracy
        - REQ-NF-P-001: Real-time processing performance

    Risk Mitigation:
        - Uses synthetic data to ensure reproducible results
        - Tests multiple lighting conditions and motion patterns
        - Validates both positive and negative cases
    """
    pass
```

---

## 5. Performance Standards

### 5.1 Optimization Guidelines

```python
# Performance-critical code optimization examples

# REQ-NF-P-001: Real-time processing performance requirements
def optimized_feature_matching(
    desc1: np.ndarray,
    desc2: np.ndarray
) -> np.ndarray:
    """
    Optimized feature matching with vectorized operations.

    Performance targets:
    - 1000x1000 descriptor matching in <10ms
    - Memory usage <100MB for typical datasets
    """
    # Use vectorized numpy operations instead of loops
    # GOOD: Vectorized distance computation
    distances = np.sqrt(np.sum((desc1[:, np.newaxis] - desc2[np.newaxis, :]) ** 2, axis=2))

    # AVOID: Nested loops (orders of magnitude slower)
    # for i in range(len(desc1)):
    #     for j in range(len(desc2)):
    #         dist = np.linalg.norm(desc1[i] - desc2[j])

    # Apply ratio test efficiently
    sorted_indices = np.argsort(distances, axis=1)
    best_matches = sorted_indices[:, 0]
    second_best = sorted_indices[:, 1]

    # Vectorized ratio test
    ratios = distances[np.arange(len(desc1)), best_matches] / \
             distances[np.arange(len(desc1)), second_best]

    valid_matches = ratios < 0.8
    return best_matches[valid_matches]

# Memory-efficient point cloud processing
def process_large_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    Process large point clouds with memory efficiency.

    Handles point clouds up to 10M points without memory issues.
    """
    # Process in chunks to avoid memory issues
    chunk_size = 100000
    processed_chunks = []

    for i in range(0, len(points), chunk_size):
        chunk = points[i:i + chunk_size]
        # Process chunk with optimized algorithms
        processed_chunk = _process_point_chunk(chunk)
        processed_chunks.append(processed_chunk)

    return np.vstack(processed_chunks)

# GPU memory management
class GPUMemoryManager:
    """Efficient GPU memory management for large datasets."""

    def __init__(self, max_gpu_memory_gb: float = 4.0):
        self.max_memory_bytes = max_gpu_memory_gb * 1024**3
        self.allocated_memory = 0

    def allocate_if_possible(self, size_bytes: int) -> bool:
        """Allocate GPU memory only if within limits."""
        if self.allocated_memory + size_bytes <= self.max_memory_bytes:
            self.allocated_memory += size_bytes
            return True
        return False
```

### 5.2 Profiling and Benchmarking

```python
import cProfile
import pstats
from functools import wraps
import time

def profile_performance(func):
    """Decorator for profiling function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        profiler.disable()

        # Log performance statistics
        logger.info(f"{func.__name__} completed in {(end_time - start_time)*1000:.2f}ms")

        # Save profile data for analysis
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.dump_stats(f'profile_{func.__name__}.prof')

        return result
    return wrapper

# Usage example
@profile_performance
def slam_process_frame(frame: StereoFrame) -> SLAMResult:
    """Process frame with performance profiling."""
    return slam_pipeline.process_frame(frame)
```

---

## 6. Code Review Guidelines

### 6.1 Review Checklist

**Functional Review**:
- [ ] Code implements specified requirements correctly
- [ ] All edge cases and error conditions handled
- [ ] Input validation performed appropriately
- [ ] Output validation ensures data integrity

**Technical Review**:
- [ ] Code follows established style guidelines
- [ ] Performance implications assessed and acceptable
- [ ] Memory usage patterns efficient
- [ ] Thread safety considered where applicable

**Documentation Review**:
- [ ] Public APIs fully documented with examples
- [ ] Complex algorithms explained with comments
- [ ] Requirements traceability maintained
- [ ] Type hints complete and accurate

**Testing Review**:
- [ ] Adequate test coverage for new functionality
- [ ] Tests cover both positive and negative cases
- [ ] Performance tests included for critical paths
- [ ] Integration tests updated as needed

### 6.2 Review Process

1. **Automated Checks**: All style and basic quality checks pass
2. **Self Review**: Developer reviews own code before submission
3. **Peer Review**: At least one peer developer review
4. **Technical Review**: Lead developer review for architectural changes
5. **Documentation Review**: Technical writer review for public APIs

---

## 7. Security Guidelines

### 7.1 Input Validation

```python
# REQ-NF-S-001: Input validation for security

def validate_image_input(image: np.ndarray) -> None:
    """Validate image input for security and safety."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Image must be numpy array")

    if image.size > 50_000_000:  # 50MP limit
        raise ValueError("Image too large - potential DoS attack")

    if image.dtype not in [np.uint8, np.uint16, np.float32]:
        raise ValueError("Unsupported image data type")

def validate_config_file(config_path: str) -> Dict[str, Any]:
    """Safely load and validate configuration files."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Check file size to prevent DoS
    file_size = os.path.getsize(config_path)
    if file_size > 1024 * 1024:  # 1MB limit
        raise ValueError("Configuration file too large")

    # Safely parse JSON/YAML
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")

    return config
```

---

## 8. Tool Configuration

### 8.1 Automated Code Formatting

**.pre-commit-config.yaml**:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.0.0
    hooks:
      - id: black
        language_version: python3.9
        args: [--line-length=88]

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: [--profile=black]

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.950
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

### 8.2 IDE Configuration

**VS Code settings.json**:
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.testing.pytestEnabled": true,
    "editor.formatOnSave": true,
    "editor.rulers": [88],
    "files.trimTrailingWhitespace": true
}
```

---

**Compliance Enforcement**

All code must pass automated checks before merge:
- Black formatting compliance
- Flake8 linting without errors
- MyPy type checking without errors
- Pytest test suite passing
- Code coverage >80% for new code

*These coding standards are enforced through automated tools and are required for all contributions to the Python-SLAM project.*
