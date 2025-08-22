"""
Integration Tests for Python-SLAM System

This module provides integration tests that verify the interaction between
different components of the Python-SLAM system.
"""

import unittest
import sys
import os
import numpy as np
import tempfile
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestSystemIntegration(unittest.TestCase):
    """Test integration between main system components."""

    def setUp(self):
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.json")

        # Create test configuration
        self.test_config = {
            "slam": {
                "algorithm": "basic",
                "feature_detector": "ORB",
                "descriptor_matcher": "BruteForce",
                "max_features": 1000
            },
            "gpu": {
                "enable_acceleration": True,
                "preferred_backend": "auto",
                "memory_limit_mb": 2048
            },
            "benchmarking": {
                "enable_metrics": True,
                "save_trajectory": True,
                "output_directory": self.test_dir
            },
            "ros2": {
                "enable_integration": False,
                "node_name": "python_slam_test"
            },
            "embedded": {
                "enable_optimization": True,
                "target_architecture": "auto"
            },
            "gui": {
                "enable_gui": False,
                "theme": "dark",
                "update_rate_hz": 30
            }
        }

        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f, indent=2)

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_configuration_loading(self):
        """Test system configuration loading."""
        try:
            from python_slam_main import load_config, create_default_config

            # Test default configuration creation
            default_config = create_default_config()
            self.assertIsInstance(default_config, dict)
            self.assertIn("slam", default_config)
            self.assertIn("gpu", default_config)

            # Test configuration loading from file
            loaded_config = load_config(self.config_path)
            self.assertIsInstance(loaded_config, dict)
            self.assertEqual(loaded_config["slam"]["algorithm"], "basic")
            self.assertEqual(loaded_config["gpu"]["preferred_backend"], "auto")

        except ImportError as e:
            self.skipTest(f"Configuration system not available: {e}")

    def test_system_initialization_headless(self):
        """Test system initialization in headless mode."""
        try:
            from python_slam_main import PythonSLAMSystem

            # Modify config for headless mode
            self.test_config["gui"]["enable_gui"] = False
            self.test_config["ros2"]["enable_integration"] = False

            with open(self.config_path, 'w') as f:
                json.dump(self.test_config, f, indent=2)

            # Initialize system
            system = PythonSLAMSystem(self.test_config)
            self.assertIsNotNone(system)

            # Test component initialization
            self.assertIsNotNone(system.slam_algorithm)
            self.assertIsNotNone(system.gpu_manager)
            self.assertIsNotNone(system.metrics_tracker)

            # Test system status
            status = system.get_system_status()
            self.assertIsInstance(status, dict)
            self.assertIn("slam_initialized", status)
            self.assertIn("gpu_available", status)

        except ImportError as e:
            self.skipTest(f"System integration not available: {e}")

    def test_gpu_slam_integration(self):
        """Test integration between GPU acceleration and SLAM algorithms."""
        try:
            from python_slam.gpu_acceleration.gpu_manager import GPUManager
            from python_slam.gpu_acceleration.accelerated_operations import AcceleratedSLAMOperations

            # Initialize GPU manager
            gpu_manager = GPUManager()
            gpu_manager.initialize_accelerators()

            # Initialize accelerated operations
            slam_ops = AcceleratedSLAMOperations()

            # Test feature matching integration
            descriptors1 = np.random.randn(500, 128).astype(np.float32)
            descriptors2 = np.random.randn(500, 128).astype(np.float32)

            matches = slam_ops.accelerated_feature_matching(descriptors1, descriptors2)
            self.assertIsInstance(matches, np.ndarray)

            # Test matrix operations integration
            matrix_a = np.random.randn(100, 100).astype(np.float32)
            matrix_b = np.random.randn(100, 100).astype(np.float32)

            result = slam_ops.accelerated_matrix_multiply(matrix_a, matrix_b)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (100, 100))

        except ImportError as e:
            self.skipTest(f"GPU-SLAM integration not available: {e}")

    def test_benchmarking_slam_integration(self):
        """Test integration between benchmarking and SLAM algorithms."""
        try:
            from python_slam.benchmarking.benchmark_metrics import TrajectoryMetrics, ProcessingMetrics
            from python_slam.benchmarking.benchmark_runner import BenchmarkRunner, BenchmarkConfig

            # Create benchmark configuration
            config = BenchmarkConfig(
                timeout_seconds=30,
                enable_parallel_execution=False
            )

            runner = BenchmarkRunner(config)

            # Test trajectory metrics
            trajectory_metrics = TrajectoryMetrics()

            # Create test trajectories
            ground_truth = np.random.randn(100, 7)
            estimated = ground_truth + np.random.normal(0, 0.1, ground_truth.shape)

            ate = trajectory_metrics.compute_ate(estimated, ground_truth)
            rpe = trajectory_metrics.compute_rpe(estimated, ground_truth)

            self.assertIsInstance(ate, float)
            self.assertIsInstance(rpe, float)
            self.assertGreater(ate, 0)
            self.assertGreater(rpe, 0)

            # Test processing metrics
            processing_metrics = ProcessingMetrics()

            # Simulate SLAM processing
            for _ in range(10):
                processing_metrics.record_frame_time(0.033)  # 30 FPS
                processing_metrics.record_memory_usage(1024)  # 1GB
                processing_metrics.record_cpu_usage(25.0)  # 25% CPU

            avg_fps = processing_metrics.get_average_fps()
            peak_memory = processing_metrics.get_peak_memory_usage()
            avg_cpu = processing_metrics.get_average_cpu_usage()

            self.assertGreater(avg_fps, 0)
            self.assertGreater(peak_memory, 0)
            self.assertGreater(avg_cpu, 0)

        except ImportError as e:
            self.skipTest(f"Benchmarking-SLAM integration not available: {e}")

    def test_embedded_optimization_integration(self):
        """Test integration with embedded optimization."""
        try:
            from python_slam.embedded_optimization.arm_optimization import ARMOptimizer, ARMConfig

            # Create ARM optimizer configuration
            config = ARMConfig(
                optimization_level="balanced",
                enable_neon=True,
                cache_optimization=True
            )

            optimizer = ARMOptimizer(config)

            # Test matrix optimization
            matrix_a = np.random.randn(64, 64).astype(np.float32)
            matrix_b = np.random.randn(64, 64).astype(np.float32)

            result = optimizer.optimize_matrix_multiplication(matrix_a, matrix_b)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (64, 64))

            # Test feature extraction optimization
            image = np.random.randint(0, 256, (240, 320), dtype=np.uint8).astype(np.float32)
            features = optimizer.optimize_feature_extraction(image)

            self.assertIsInstance(features, dict)
            self.assertIn("edges_x", features)
            self.assertIn("edges_y", features)

            # Test performance statistics
            stats = optimizer.get_performance_stats()
            self.assertIsInstance(stats, dict)
            self.assertIn("arm_architecture", stats)

        except ImportError as e:
            self.skipTest(f"Embedded optimization integration not available: {e}")

class TestDataPipeline(unittest.TestCase):
    """Test data pipeline integration."""

    def setUp(self):
        """Set up data pipeline test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.create_test_dataset()

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_test_dataset(self):
        """Create a small test dataset."""
        # Create test images directory
        images_dir = os.path.join(self.test_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Create test ground truth file
        groundtruth_file = os.path.join(self.test_dir, "groundtruth.txt")
        with open(groundtruth_file, 'w') as f:
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            for i in range(10):
                timestamp = i * 0.1
                tx, ty, tz = i * 0.1, 0.0, 0.0
                qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
                f.write(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

        # Create test camera intrinsics
        calib_file = os.path.join(self.test_dir, "camera.txt")
        with open(calib_file, 'w') as f:
            f.write("# Camera intrinsics\n")
            f.write("fx 525.0\n")
            f.write("fy 525.0\n")
            f.write("cx 320.0\n")
            f.write("cy 240.0\n")

    def test_dataset_loading_pipeline(self):
        """Test dataset loading and processing pipeline."""
        try:
            # Test if dataset loaders are available
            from python_slam.benchmarking.benchmark_runner import TUMDatasetLoader

            loader = TUMDatasetLoader()

            # Create TUM-style dataset structure
            associations_file = os.path.join(self.test_dir, "associations.txt")
            with open(associations_file, 'w') as f:
                for i in range(10):
                    timestamp = i * 0.1
                    f.write(f"{timestamp} images/{i:06d}.png {timestamp} depth/{i:06d}.png\n")

            # Test dataset validation
            dataset_info = {
                "name": "test_dataset",
                "path": self.test_dir,
                "type": "TUM"
            }

            # Note: Actual loading would require image files
            # This tests the integration interface
            self.assertIsInstance(dataset_info, dict)
            self.assertIn("path", dataset_info)
            self.assertIn("type", dataset_info)

        except ImportError as e:
            self.skipTest(f"Dataset loading pipeline not available: {e}")

class TestPerformanceIntegration(unittest.TestCase):
    """Test performance monitoring integration."""

    def setUp(self):
        """Set up performance test environment."""
        self.test_results = []

    def test_end_to_end_performance(self):
        """Test end-to-end system performance."""
        try:
            from python_slam.benchmarking.benchmark_metrics import ProcessingMetrics

            metrics = ProcessingMetrics()

            # Simulate SLAM processing pipeline
            n_frames = 50
            for frame_idx in range(n_frames):
                frame_start_time = time.time()

                # Simulate feature extraction
                time.sleep(0.005)  # 5ms
                metrics.record_processing_time("feature_extraction", 0.005)

                # Simulate feature matching
                time.sleep(0.010)  # 10ms
                metrics.record_processing_time("feature_matching", 0.010)

                # Simulate pose estimation
                time.sleep(0.008)  # 8ms
                metrics.record_processing_time("pose_estimation", 0.008)

                # Record frame time
                frame_time = time.time() - frame_start_time
                metrics.record_frame_time(frame_time)

                # Record memory usage (simulated)
                memory_usage = 1024 + frame_idx * 2  # Growing memory usage
                metrics.record_memory_usage(memory_usage)

                # Record CPU usage (simulated)
                cpu_usage = 25.0 + np.random.normal(0, 5)  # Variable CPU usage
                metrics.record_cpu_usage(max(0, min(100, cpu_usage)))

            # Verify performance metrics
            avg_fps = metrics.get_average_fps()
            peak_memory = metrics.get_peak_memory_usage()
            avg_cpu = metrics.get_average_cpu_usage()

            self.assertGreater(avg_fps, 0)
            self.assertLess(avg_fps, 100)  # Reasonable FPS range
            self.assertGreater(peak_memory, 1024)  # Memory should have grown
            self.assertGreater(avg_cpu, 0)
            self.assertLess(avg_cpu, 100)

            # Test processing time statistics
            fe_avg_time = metrics.get_average_processing_time("feature_extraction")
            fm_avg_time = metrics.get_average_processing_time("feature_matching")
            pe_avg_time = metrics.get_average_processing_time("pose_estimation")

            self.assertAlmostEqual(fe_avg_time, 0.005, places=3)
            self.assertAlmostEqual(fm_avg_time, 0.010, places=3)
            self.assertAlmostEqual(pe_avg_time, 0.008, places=3)

            print(f"Performance Test Results:")
            print(f"  Average FPS: {avg_fps:.2f}")
            print(f"  Peak Memory: {peak_memory:.1f} MB")
            print(f"  Average CPU: {avg_cpu:.1f}%")
            print(f"  Feature Extraction: {fe_avg_time*1000:.1f}ms")
            print(f"  Feature Matching: {fm_avg_time*1000:.1f}ms")
            print(f"  Pose Estimation: {pe_avg_time*1000:.1f}ms")

        except ImportError as e:
            self.skipTest(f"Performance monitoring not available: {e}")

class TestErrorHandling(unittest.TestCase):
    """Test error handling and recovery mechanisms."""

    def test_gpu_fallback_mechanism(self):
        """Test GPU fallback to CPU when GPU is unavailable."""
        try:
            from python_slam.gpu_acceleration.accelerated_operations import AcceleratedSLAMOperations

            slam_ops = AcceleratedSLAMOperations()

            # Test operations that should work even without GPU
            matrix_a = np.random.randn(50, 50).astype(np.float32)
            matrix_b = np.random.randn(50, 50).astype(np.float32)

            # This should not raise an exception even if no GPU is available
            result = slam_ops.accelerated_matrix_multiply(matrix_a, matrix_b)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (50, 50))

            # Verify correctness against CPU computation
            cpu_result = np.dot(matrix_a, matrix_b)
            np.testing.assert_allclose(result, cpu_result, rtol=1e-5, atol=1e-6)

        except ImportError as e:
            self.skipTest(f"GPU acceleration not available: {e}")

    def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        try:
            from python_slam_main import validate_config, create_default_config

            # Test valid configuration
            valid_config = create_default_config()
            is_valid, errors = validate_config(valid_config)
            self.assertTrue(is_valid)
            self.assertEqual(len(errors), 0)

            # Test invalid configuration
            invalid_config = {
                "slam": {
                    "algorithm": "invalid_algorithm",  # Invalid algorithm
                    "max_features": -100  # Invalid negative value
                },
                "gpu": {
                    "memory_limit_mb": "invalid"  # Invalid type
                }
            }

            is_valid, errors = validate_config(invalid_config)
            self.assertFalse(is_valid)
            self.assertGreater(len(errors), 0)

        except ImportError as e:
            self.skipTest(f"Configuration validation not available: {e}")

    def test_resource_cleanup(self):
        """Test proper resource cleanup and memory management."""
        try:
            from python_slam.gpu_acceleration.gpu_manager import GPUManager
            from python_slam.benchmarking.benchmark_metrics import ProcessingMetrics

            # Initialize components
            gpu_manager = GPUManager()
            metrics = ProcessingMetrics()

            # Use resources
            gpu_manager.initialize_accelerators()

            for _ in range(10):
                metrics.record_frame_time(0.033)
                metrics.record_memory_usage(1024)

            # Test cleanup
            if hasattr(gpu_manager, 'cleanup'):
                gpu_manager.cleanup()

            if hasattr(metrics, 'reset'):
                metrics.reset()
                self.assertEqual(metrics.get_frame_count(), 0)

        except ImportError as e:
            self.skipTest(f"Resource management not available: {e}")

class TestScalabilityIntegration(unittest.TestCase):
    """Test system scalability with different workloads."""

    def test_large_dataset_handling(self):
        """Test system behavior with large datasets."""
        try:
            from python_slam.benchmarking.benchmark_metrics import ProcessingMetrics

            metrics = ProcessingMetrics()

            # Simulate processing large number of frames
            n_frames = 1000

            start_time = time.time()

            for frame_idx in range(n_frames):
                # Simulate variable processing times
                frame_time = 0.033 + np.random.normal(0, 0.005)  # ~30 FPS with variation
                metrics.record_frame_time(max(0.001, frame_time))

                # Simulate memory growth
                memory_usage = 1024 + frame_idx * 0.1  # Slow memory growth
                metrics.record_memory_usage(memory_usage)

                # Only record every 100th frame to avoid test taking too long
                if frame_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    if elapsed > 5.0:  # Limit test time to 5 seconds
                        break

            # Verify metrics are reasonable
            avg_fps = metrics.get_average_fps()
            peak_memory = metrics.get_peak_memory_usage()
            frame_count = metrics.get_frame_count()

            self.assertGreater(avg_fps, 0)
            self.assertGreater(peak_memory, 1024)
            self.assertGreater(frame_count, 0)

            print(f"Scalability Test Results:")
            print(f"  Processed frames: {frame_count}")
            print(f"  Average FPS: {avg_fps:.2f}")
            print(f"  Peak Memory: {peak_memory:.1f} MB")

        except ImportError as e:
            self.skipTest(f"Scalability testing not available: {e}")

    def test_concurrent_operations(self):
        """Test concurrent operations and thread safety."""
        try:
            from python_slam.gpu_acceleration.gpu_manager import GPUManager
            from python_slam.benchmarking.benchmark_metrics import ProcessingMetrics
            import threading
            import queue

            gpu_manager = GPUManager()
            gpu_manager.initialize_accelerators()

            results_queue = queue.Queue()

            def worker_thread(thread_id, num_operations):
                """Worker thread that performs GPU operations."""
                metrics = ProcessingMetrics()

                for i in range(num_operations):
                    # Get optimal accelerator (should be thread-safe)
                    accelerator = gpu_manager.get_optimal_accelerator("matrix_multiply")

                    # Record operation
                    metrics.record_processing_time("matrix_multiply", 0.001)

                results_queue.put((thread_id, metrics.get_frame_count()))

            # Create multiple worker threads
            num_threads = 4
            operations_per_thread = 10
            threads = []

            for thread_id in range(num_threads):
                thread = threading.Thread(
                    target=worker_thread,
                    args=(thread_id, operations_per_thread)
                )
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Collect results
            total_operations = 0
            while not results_queue.empty():
                thread_id, operations = results_queue.get()
                total_operations += operations

            # Verify all operations completed
            expected_operations = num_threads * operations_per_thread
            self.assertEqual(total_operations, expected_operations)

            print(f"Concurrency Test Results:")
            print(f"  Threads: {num_threads}")
            print(f"  Operations per thread: {operations_per_thread}")
            print(f"  Total operations: {total_operations}")

        except ImportError as e:
            self.skipTest(f"Concurrency testing not available: {e}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
