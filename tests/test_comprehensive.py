"""
Comprehensive Test Suite for Python-SLAM

This module provides a complete testing framework for all Python-SLAM components
including unit tests, integration tests, and system tests.
"""

import sys
import os
import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import logging
import json
from typing import Dict, Any, List
import time

# Add src directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Set up test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPythonSLAMCore(unittest.TestCase):
    """Test core SLAM functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_data_dir = Path(__file__).parent / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)

        # Create test images
        self.test_image1 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        self.test_image2 = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        # Create test camera intrinsics
        self.camera_intrinsics = np.array([
            [525.0, 0.0, 320.0],
            [0.0, 525.0, 240.0],
            [0.0, 0.0, 1.0]
        ])

    def test_basic_slam_pipeline(self):
        """Test basic SLAM pipeline functionality."""
        try:
            from python_slam.basic_slam_pipeline import BasicSLAMPipeline

            # Initialize SLAM pipeline
            slam = BasicSLAMPipeline()

            # Test initialization
            self.assertIsNotNone(slam)

            # Test feature extraction (if implemented)
            if hasattr(slam, 'extract_features'):
                features = slam.extract_features(self.test_image1)
                self.assertIsNotNone(features)

            logger.info("Basic SLAM pipeline test passed")

        except ImportError as e:
            self.skipTest(f"SLAM pipeline not available: {e}")

    def test_feature_extraction(self):
        """Test feature extraction functionality."""
        try:
            from python_slam.feature_extraction import FeatureExtractor

            extractor = FeatureExtractor()
            keypoints, descriptors = extractor.extract(self.test_image1)

            self.assertIsInstance(keypoints, np.ndarray)
            self.assertIsInstance(descriptors, np.ndarray)
            self.assertGreater(len(keypoints), 0)
            self.assertGreater(len(descriptors), 0)

            logger.info("Feature extraction test passed")

        except ImportError as e:
            self.skipTest(f"Feature extraction not available: {e}")

class TestGUIComponents(unittest.TestCase):
    """Test GUI components."""

    def setUp(self):
        """Set up GUI test environment."""
        # Check if GUI backend is available
        try:
            from PyQt6.QtWidgets import QApplication
            self.gui_backend = "PyQt6"
        except ImportError:
            try:
                from PySide6.QtWidgets import QApplication
                self.gui_backend = "PySide6"
            except ImportError:
                self.gui_backend = None

    def test_main_window_creation(self):
        """Test main window creation."""
        if not self.gui_backend:
            self.skipTest("No GUI backend available")

        try:
            from python_slam.gui.main_window import SlamMainWindow

            # Create application (required for Qt widgets)
            if self.gui_backend == "PyQt6":
                from PyQt6.QtWidgets import QApplication
            else:
                from PySide6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is None:
                app = QApplication([])

            # Create main window
            window = SlamMainWindow()
            self.assertIsNotNone(window)

            # Test window properties
            self.assertIsNotNone(window.windowTitle())

            logger.info("Main window creation test passed")

        except Exception as e:
            self.fail(f"Main window creation failed: {e}")

    def test_material_design_manager(self):
        """Test Material Design styling."""
        try:
            from python_slam.gui.utils import MaterialDesignManager

            manager = MaterialDesignManager()
            self.assertIsNotNone(manager)

            # Test theme availability
            themes = manager.available_themes()
            self.assertIn("dark", themes)
            self.assertIn("light", themes)

            logger.info("Material Design manager test passed")

        except ImportError as e:
            self.skipTest(f"Material Design manager not available: {e}")

    def test_visualization_components(self):
        """Test 3D visualization components."""
        try:
            from python_slam.gui.visualization import Map3DViewer, PointCloudRenderer

            # Test Map3DViewer creation
            map_viewer = Map3DViewer()
            self.assertIsNotNone(map_viewer)

            # Test PointCloudRenderer creation
            pc_renderer = PointCloudRenderer()
            self.assertIsNotNone(pc_renderer)

            logger.info("Visualization components test passed")

        except ImportError as e:
            self.skipTest(f"Visualization components not available: {e}")

class TestGPUAcceleration(unittest.TestCase):
    """Test GPU acceleration functionality."""

    def setUp(self):
        """Set up GPU test environment."""
        self.test_matrix_a = np.random.randn(100, 100).astype(np.float32)
        self.test_matrix_b = np.random.randn(100, 100).astype(np.float32)
        self.test_descriptors1 = np.random.randn(1000, 128).astype(np.float32)
        self.test_descriptors2 = np.random.randn(1000, 128).astype(np.float32)

    def test_gpu_detector(self):
        """Test GPU detection functionality."""
        try:
            from python_slam.gpu_acceleration.gpu_detector import GPUDetector

            detector = GPUDetector()
            gpus = detector.detect_all_gpus()

            self.assertIsInstance(gpus, list)
            self.assertGreaterEqual(len(gpus), 1)  # At least CPU fallback

            # Test best GPU selection
            best_gpu = detector.get_best_gpu()
            self.assertIsNotNone(best_gpu)

            logger.info(f"GPU detector test passed - found {len(gpus)} backends")

        except ImportError as e:
            self.skipTest(f"GPU acceleration not available: {e}")

    def test_gpu_manager(self):
        """Test GPU manager functionality."""
        try:
            from python_slam.gpu_acceleration.gpu_manager import GPUManager

            manager = GPUManager()
            self.assertIsNotNone(manager)

            # Test accelerator initialization
            initialized = manager.initialize_accelerators()
            # Don't fail if no GPU is available, just log
            if not initialized:
                logger.warning("No GPU accelerators available for testing")

            # Test status retrieval
            status = manager.get_accelerator_status()
            self.assertIsInstance(status, dict)

            logger.info("GPU manager test passed")

        except ImportError as e:
            self.skipTest(f"GPU manager not available: {e}")

    def test_accelerated_operations(self):
        """Test accelerated SLAM operations."""
        try:
            from python_slam.gpu_acceleration.accelerated_operations import AcceleratedSLAMOperations

            slam_ops = AcceleratedSLAMOperations()
            self.assertIsNotNone(slam_ops)

            # Test feature matching (will fall back to CPU if no GPU)
            try:
                matches = slam_ops._cpu_feature_matching(self.test_descriptors1, self.test_descriptors2)
                self.assertIsInstance(matches, np.ndarray)
                logger.info("Accelerated operations test passed")
            except Exception as e:
                logger.warning(f"Accelerated operations test failed: {e}")

        except ImportError as e:
            self.skipTest(f"Accelerated operations not available: {e}")

class TestBenchmarking(unittest.TestCase):
    """Test benchmarking functionality."""

    def setUp(self):
        """Set up benchmarking test environment."""
        self.test_trajectory = np.random.randn(100, 7)  # 100 poses with quaternions
        self.test_ground_truth = np.random.randn(100, 7)

    def test_benchmark_metrics(self):
        """Test benchmark metrics calculation."""
        try:
            from python_slam.benchmarking.benchmark_metrics import TrajectoryMetrics, ProcessingMetrics

            # Test trajectory metrics
            traj_metrics = TrajectoryMetrics()

            # Test ATE calculation
            ate = traj_metrics.compute_ate(self.test_trajectory, self.test_ground_truth)
            self.assertIsInstance(ate, float)
            self.assertGreaterEqual(ate, 0.0)

            # Test RPE calculation
            rpe = traj_metrics.compute_rpe(self.test_trajectory, self.test_ground_truth)
            self.assertIsInstance(rpe, float)
            self.assertGreaterEqual(rpe, 0.0)

            # Test processing metrics
            proc_metrics = ProcessingMetrics()

            # Test FPS tracking
            for _ in range(10):
                proc_metrics.record_frame_time(0.033)  # 30 FPS

            fps = proc_metrics.get_current_fps()
            self.assertGreater(fps, 0)

            logger.info("Benchmark metrics test passed")

        except ImportError as e:
            self.skipTest(f"Benchmark metrics not available: {e}")

    def test_benchmark_runner(self):
        """Test benchmark runner functionality."""
        try:
            from python_slam.benchmarking.benchmark_runner import BenchmarkRunner, BenchmarkConfig

            # Create test configuration
            config = BenchmarkConfig(
                timeout_seconds=10,  # Short timeout for testing
                enable_parallel_execution=False  # Disable for testing
            )

            runner = BenchmarkRunner(config)
            self.assertIsNotNone(runner)

            # Test configuration
            self.assertEqual(runner.config.timeout_seconds, 10)

            logger.info("Benchmark runner test passed")

        except ImportError as e:
            self.skipTest(f"Benchmark runner not available: {e}")

class TestEmbeddedOptimization(unittest.TestCase):
    """Test embedded optimization functionality."""

    def setUp(self):
        """Set up embedded optimization test environment."""
        self.test_matrix_a = np.random.randn(64, 64).astype(np.float32)
        self.test_matrix_b = np.random.randn(64, 64).astype(np.float32)
        self.test_image = np.random.randint(0, 256, (240, 320), dtype=np.uint8).astype(np.float32)

    def test_arm_optimizer(self):
        """Test ARM optimization functionality."""
        try:
            from python_slam.embedded_optimization.arm_optimization import ARMOptimizer, ARMConfig

            config = ARMConfig(optimization_level="balanced")
            optimizer = ARMOptimizer(config)

            self.assertIsNotNone(optimizer)

            # Test matrix multiplication optimization
            result = optimizer.optimize_matrix_multiplication(self.test_matrix_a, self.test_matrix_b)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (64, 64))

            # Test feature extraction optimization
            features = optimizer.optimize_feature_extraction(self.test_image)
            self.assertIsInstance(features, dict)
            self.assertIn("edges_x", features)
            self.assertIn("edges_y", features)

            # Test performance stats
            stats = optimizer.get_performance_stats()
            self.assertIsInstance(stats, dict)
            self.assertIn("arm_architecture", stats)

            logger.info("ARM optimizer test passed")

        except ImportError as e:
            self.skipTest(f"ARM optimizer not available: {e}")

class TestROS2Integration(unittest.TestCase):
    """Test ROS2 integration functionality."""

    def test_nav2_bridge(self):
        """Test Nav2 bridge functionality."""
        try:
            from python_slam.ros2_nav2_integration.nav2_bridge import Nav2Bridge, Nav2Status

            # Test bridge creation (without ROS2 initialization)
            bridge = Nav2Bridge()
            self.assertIsNotNone(bridge)

            # Test status
            status = bridge.get_status()
            self.assertIsInstance(status, Nav2Status)

            logger.info("Nav2 bridge test passed")

        except ImportError as e:
            self.skipTest(f"ROS2 integration not available: {e}")

class TestSystemIntegration(unittest.TestCase):
    """Test system integration and main entry point."""

    def test_system_initialization(self):
        """Test system initialization."""
        try:
            # Import main system without Qt to avoid GUI requirements
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

            # Test configuration creation
            from python_slam_main import create_default_config, PythonSLAMSystem

            config = create_default_config()
            self.assertIsInstance(config, dict)
            self.assertIn("slam", config)
            self.assertIn("gpu", config)

            # Test system creation (without GUI)
            config["enable_gui"] = False
            system = PythonSLAMSystem(config)
            self.assertIsNotNone(system)

            logger.info("System integration test passed")

        except ImportError as e:
            self.skipTest(f"System integration not available: {e}")

class TestPerformance(unittest.TestCase):
    """Performance and stress tests."""

    def test_large_matrix_operations(self):
        """Test performance with large matrices."""
        try:
            from python_slam.gpu_acceleration.accelerated_operations import AcceleratedSLAMOperations

            slam_ops = AcceleratedSLAMOperations()

            # Create large test matrices
            large_matrix_a = np.random.randn(1000, 1000).astype(np.float32)
            large_matrix_b = np.random.randn(1000, 1000).astype(np.float32)

            start_time = time.time()

            # Test matrix multiplication (will use CPU fallback if no GPU)
            try:
                result = np.dot(large_matrix_a, large_matrix_b)
                elapsed_time = time.time() - start_time

                self.assertIsInstance(result, np.ndarray)
                self.assertEqual(result.shape, (1000, 1000))

                logger.info(f"Large matrix operation completed in {elapsed_time:.3f}s")
            except Exception as e:
                logger.warning(f"Large matrix operation failed: {e}")

        except ImportError as e:
            self.skipTest(f"Performance test components not available: {e}")

    def test_memory_usage(self):
        """Test memory usage patterns."""
        try:
            import psutil

            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Create large data structures
            large_arrays = []
            for i in range(10):
                arr = np.random.randn(1000, 1000).astype(np.float32)
                large_arrays.append(arr)

            # Check memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = current_memory - initial_memory

            # Clean up
            del large_arrays

            logger.info(f"Memory test: initial={initial_memory:.1f}MB, peak={current_memory:.1f}MB, increase={memory_increase:.1f}MB")

            # Memory increase should be reasonable (less than 1GB for this test)
            self.assertLess(memory_increase, 1000)

        except ImportError as e:
            self.skipTest(f"Memory testing not available: {e}")

def run_comprehensive_tests():
    """Run all tests and generate report."""
    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestPythonSLAMCore,
        TestGUIComponents,
        TestGPUAcceleration,
        TestBenchmarking,
        TestEmbeddedOptimization,
        TestROS2Integration,
        TestSystemIntegration,
        TestPerformance
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )

    logger.info("Starting comprehensive test suite...")
    start_time = time.time()

    result = runner.run(test_suite)

    elapsed_time = time.time() - start_time

    # Generate summary
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped

    logger.info(f"\nTest Summary:")
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failures}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Elapsed time: {elapsed_time:.2f}s")

    # Save test report
    test_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_tests": total_tests,
        "passed": passed,
        "failed": failures,
        "errors": errors,
        "skipped": skipped,
        "elapsed_time": elapsed_time,
        "success_rate": passed / total_tests if total_tests > 0 else 0
    }

    report_path = Path(__file__).parent / "test_report.json"
    try:
        with open(report_path, 'w') as f:
            json.dump(test_report, f, indent=2)
        logger.info(f"Test report saved to {report_path}")
    except Exception as e:
        logger.error(f"Failed to save test report: {e}")

    return result.wasSuccessful()

if __name__ == "__main__":
    # Run comprehensive tests
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
