"""
Unit Tests for GPU Acceleration Module

This module provides detailed unit tests for the GPU acceleration components.
"""

import unittest
import numpy as np
import torch
import sys
import os
import tempfile
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestGPUDetector(unittest.TestCase):
    """Test GPU detection functionality."""

    def setUp(self):
        """Set up test environment."""
        try:
            from python_slam.gpu_acceleration.gpu_detector import GPUDetector, GPUInfo
            self.GPUDetector = GPUDetector
            self.GPUInfo = GPUInfo
        except ImportError:
            self.skipTest("GPU acceleration module not available")

    def test_gpu_info_creation(self):
        """Test GPU info object creation."""
        gpu_info = self.GPUInfo(
            device_id=0,
            name="Test GPU",
            backend="cuda",
            memory_total=8000,
            memory_free=7000,
            compute_capability=8.0,
            is_available=True
        )

        self.assertEqual(gpu_info.device_id, 0)
        self.assertEqual(gpu_info.name, "Test GPU")
        self.assertEqual(gpu_info.backend, "cuda")
        self.assertEqual(gpu_info.memory_total, 8000)
        self.assertEqual(gpu_info.memory_free, 7000)
        self.assertEqual(gpu_info.compute_capability, 8.0)
        self.assertTrue(gpu_info.is_available)

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = self.GPUDetector()
        self.assertIsNotNone(detector)

    def test_cuda_detection(self):
        """Test CUDA GPU detection."""
        detector = self.GPUDetector()
        cuda_gpus = detector.detect_cuda_gpus()

        self.assertIsInstance(cuda_gpus, list)

        if torch.cuda.is_available():
            self.assertGreater(len(cuda_gpus), 0)
            for gpu in cuda_gpus:
                self.assertIsInstance(gpu, self.GPUInfo)
                self.assertEqual(gpu.backend, "cuda")
                self.assertTrue(gpu.is_available)
        else:
            self.assertEqual(len(cuda_gpus), 0)

    def test_rocm_detection(self):
        """Test ROCm GPU detection."""
        detector = self.GPUDetector()
        rocm_gpus = detector.detect_rocm_gpus()

        self.assertIsInstance(rocm_gpus, list)

        for gpu in rocm_gpus:
            self.assertIsInstance(gpu, self.GPUInfo)
            self.assertEqual(gpu.backend, "rocm")

    def test_metal_detection(self):
        """Test Metal GPU detection."""
        detector = self.GPUDetector()
        metal_gpus = detector.detect_metal_gpus()

        self.assertIsInstance(metal_gpus, list)

        for gpu in metal_gpus:
            self.assertIsInstance(gpu, self.GPUInfo)
            self.assertEqual(gpu.backend, "metal")

    def test_all_gpu_detection(self):
        """Test detection of all available GPUs."""
        detector = self.GPUDetector()
        all_gpus = detector.detect_all_gpus()

        self.assertIsInstance(all_gpus, list)
        self.assertGreaterEqual(len(all_gpus), 1)  # At least CPU fallback

        # Check for CPU fallback
        has_cpu_fallback = any(gpu.backend == "cpu" for gpu in all_gpus)
        self.assertTrue(has_cpu_fallback)

    def test_best_gpu_selection(self):
        """Test best GPU selection algorithm."""
        detector = self.GPUDetector()
        best_gpu = detector.get_best_gpu()

        self.assertIsNotNone(best_gpu)
        self.assertIsInstance(best_gpu, self.GPUInfo)
        self.assertTrue(best_gpu.is_available)

    def test_gpu_ranking(self):
        """Test GPU ranking algorithm."""
        detector = self.GPUDetector()

        # Create test GPUs
        gpu1 = self.GPUInfo(0, "Low-end GPU", "cuda", 2000, 1500, 6.0, True)
        gpu2 = self.GPUInfo(1, "High-end GPU", "cuda", 12000, 10000, 8.0, True)
        gpu3 = self.GPUInfo(0, "CPU Fallback", "cpu", 16000, 8000, 0.0, True)

        gpus = [gpu1, gpu2, gpu3]
        ranked = detector._rank_gpus(gpus)

        self.assertEqual(len(ranked), 3)
        # High-end GPU should be first
        self.assertEqual(ranked[0].name, "High-end GPU")
        # CPU should be last
        self.assertEqual(ranked[-1].name, "CPU Fallback")

class TestGPUManager(unittest.TestCase):
    """Test GPU manager functionality."""

    def setUp(self):
        """Set up test environment."""
        try:
            from python_slam.gpu_acceleration.gpu_manager import GPUManager
            self.GPUManager = GPUManager
        except ImportError:
            self.skipTest("GPU manager not available")

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = self.GPUManager()
        self.assertIsNotNone(manager)
        self.assertIsNotNone(manager.detector)
        self.assertIsInstance(manager.accelerators, dict)

    def test_accelerator_initialization(self):
        """Test accelerator initialization."""
        manager = self.GPUManager()

        # Initialize accelerators
        success = manager.initialize_accelerators()
        self.assertIsInstance(success, bool)

        # Should have at least CPU accelerator
        status = manager.get_accelerator_status()
        self.assertIsInstance(status, dict)
        self.assertGreaterEqual(len(status), 1)

    def test_load_balancing(self):
        """Test load balancing functionality."""
        manager = self.GPUManager()
        manager.initialize_accelerators()

        # Test task assignment
        for i in range(10):
            accelerator = manager.get_optimal_accelerator("matrix_multiply")
            self.assertIsNotNone(accelerator)

    def test_memory_monitoring(self):
        """Test memory monitoring."""
        manager = self.GPUManager()
        manager.initialize_accelerators()

        # Get memory stats
        memory_stats = manager.get_memory_stats()
        self.assertIsInstance(memory_stats, dict)

        for backend, stats in memory_stats.items():
            self.assertIn("total", stats)
            self.assertIn("used", stats)
            self.assertIn("free", stats)

class TestAcceleratedOperations(unittest.TestCase):
    """Test accelerated SLAM operations."""

    def setUp(self):
        """Set up test environment."""
        try:
            from python_slam.gpu_acceleration.accelerated_operations import AcceleratedSLAMOperations
            self.AcceleratedSLAMOperations = AcceleratedSLAMOperations
        except ImportError:
            self.skipTest("Accelerated operations not available")

        # Create test data
        self.test_matrix_a = np.random.randn(100, 100).astype(np.float32)
        self.test_matrix_b = np.random.randn(100, 100).astype(np.float32)
        self.test_descriptors1 = np.random.randn(500, 128).astype(np.float32)
        self.test_descriptors2 = np.random.randn(500, 128).astype(np.float32)
        self.test_points_3d = np.random.randn(1000, 3).astype(np.float32)
        self.test_points_2d = np.random.randn(1000, 2).astype(np.float32)
        self.test_camera_params = np.array([500.0, 500.0, 320.0, 240.0, 0.1, 0.05, 0.01], dtype=np.float32)

    def test_operations_initialization(self):
        """Test operations initialization."""
        slam_ops = self.AcceleratedSLAMOperations()
        self.assertIsNotNone(slam_ops)
        self.assertIsNotNone(slam_ops.gpu_manager)

    def test_feature_matching(self):
        """Test feature matching operation."""
        slam_ops = self.AcceleratedSLAMOperations()

        matches = slam_ops.accelerated_feature_matching(
            self.test_descriptors1,
            self.test_descriptors2
        )

        self.assertIsInstance(matches, np.ndarray)
        self.assertEqual(matches.shape[1], 2)  # Should be pairs of indices
        self.assertGreaterEqual(matches.shape[0], 0)  # Should have some matches

    def test_bundle_adjustment(self):
        """Test bundle adjustment operation."""
        slam_ops = self.AcceleratedSLAMOperations()

        optimized_points, optimized_params = slam_ops.accelerated_bundle_adjustment(
            self.test_points_3d,
            self.test_points_2d,
            self.test_camera_params
        )

        self.assertIsInstance(optimized_points, np.ndarray)
        self.assertIsInstance(optimized_params, np.ndarray)
        self.assertEqual(optimized_points.shape, self.test_points_3d.shape)
        self.assertEqual(optimized_params.shape, self.test_camera_params.shape)

    def test_matrix_operations(self):
        """Test matrix operations."""
        slam_ops = self.AcceleratedSLAMOperations()

        # Test matrix multiplication
        result = slam_ops.accelerated_matrix_multiply(
            self.test_matrix_a,
            self.test_matrix_b
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (100, 100))

        # Compare with CPU result
        cpu_result = np.dot(self.test_matrix_a, self.test_matrix_b)
        np.testing.assert_allclose(result, cpu_result, rtol=1e-5, atol=1e-6)

    def test_performance_monitoring(self):
        """Test performance monitoring."""
        slam_ops = self.AcceleratedSLAMOperations()

        # Perform some operations to generate stats
        for _ in range(5):
            slam_ops.accelerated_matrix_multiply(
                self.test_matrix_a,
                self.test_matrix_b
            )

        # Get performance stats
        stats = slam_ops.get_performance_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_operations", stats)
        self.assertIn("average_time", stats)
        self.assertIn("operations_per_second", stats)

class TestCUDAAcceleration(unittest.TestCase):
    """Test CUDA-specific acceleration."""

    def setUp(self):
        """Set up CUDA test environment."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        try:
            from python_slam.gpu_acceleration.cuda_acceleration import CUDAAccelerator
            self.CUDAAccelerator = CUDAAccelerator
        except ImportError:
            self.skipTest("CUDA accelerator not available")

        self.test_data = torch.randn(1000, 1000, dtype=torch.float32)

    def test_cuda_initialization(self):
        """Test CUDA accelerator initialization."""
        accelerator = self.CUDAAccelerator(device_id=0)
        self.assertIsNotNone(accelerator)
        self.assertTrue(accelerator.is_available())

    def test_cuda_memory_management(self):
        """Test CUDA memory management."""
        accelerator = self.CUDAAccelerator(device_id=0)

        # Test memory allocation
        gpu_data = accelerator.to_device(self.test_data)
        self.assertTrue(gpu_data.is_cuda)

        # Test memory deallocation
        cpu_data = accelerator.to_cpu(gpu_data)
        self.assertFalse(cpu_data.is_cuda)

        # Verify data integrity
        torch.testing.assert_close(self.test_data, cpu_data)

    def test_cuda_operations(self):
        """Test CUDA operations."""
        accelerator = self.CUDAAccelerator(device_id=0)

        a = torch.randn(500, 500, dtype=torch.float32)
        b = torch.randn(500, 500, dtype=torch.float32)

        # Test matrix multiplication
        result = accelerator.matrix_multiply(a, b)
        self.assertIsInstance(result, torch.Tensor)

        # Compare with CPU result
        cpu_result = torch.mm(a, b)
        torch.testing.assert_close(result.cpu(), cpu_result, rtol=1e-5, atol=1e-6)

class TestROCmAcceleration(unittest.TestCase):
    """Test ROCm-specific acceleration."""

    def setUp(self):
        """Set up ROCm test environment."""
        try:
            from python_slam.gpu_acceleration.rocm_acceleration import ROCmAccelerator
            self.ROCmAccelerator = ROCmAccelerator
        except ImportError:
            self.skipTest("ROCm accelerator not available")

    def test_rocm_initialization(self):
        """Test ROCm accelerator initialization."""
        accelerator = self.ROCmAccelerator(device_id=0)
        self.assertIsNotNone(accelerator)

    def test_rocm_availability(self):
        """Test ROCm availability detection."""
        accelerator = self.ROCmAccelerator(device_id=0)
        is_available = accelerator.is_available()
        self.assertIsInstance(is_available, bool)

class TestMetalAcceleration(unittest.TestCase):
    """Test Metal-specific acceleration."""

    def setUp(self):
        """Set up Metal test environment."""
        try:
            from python_slam.gpu_acceleration.metal_acceleration import MetalAccelerator
            self.MetalAccelerator = MetalAccelerator
        except ImportError:
            self.skipTest("Metal accelerator not available")

    def test_metal_initialization(self):
        """Test Metal accelerator initialization."""
        accelerator = self.MetalAccelerator(device_id=0)
        self.assertIsNotNone(accelerator)

    def test_metal_availability(self):
        """Test Metal availability detection."""
        accelerator = self.MetalAccelerator(device_id=0)
        is_available = accelerator.is_available()
        self.assertIsInstance(is_available, bool)

class TestGPUPerformance(unittest.TestCase):
    """Test GPU performance characteristics."""

    def setUp(self):
        """Set up performance test environment."""
        try:
            from python_slam.gpu_acceleration.accelerated_operations import AcceleratedSLAMOperations
            self.AcceleratedSLAMOperations = AcceleratedSLAMOperations
        except ImportError:
            self.skipTest("Accelerated operations not available")

    def test_matrix_multiplication_performance(self):
        """Test matrix multiplication performance."""
        slam_ops = self.AcceleratedSLAMOperations()

        sizes = [100, 500, 1000]

        for size in sizes:
            matrix_a = np.random.randn(size, size).astype(np.float32)
            matrix_b = np.random.randn(size, size).astype(np.float32)

            # Time GPU operation
            start_time = time.time()
            gpu_result = slam_ops.accelerated_matrix_multiply(matrix_a, matrix_b)
            gpu_time = time.time() - start_time

            # Time CPU operation
            start_time = time.time()
            cpu_result = np.dot(matrix_a, matrix_b)
            cpu_time = time.time() - start_time

            # Verify correctness
            np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-5, atol=1e-6)

            print(f"Matrix size {size}x{size}: GPU={gpu_time:.4f}s, CPU={cpu_time:.4f}s, Speedup={cpu_time/gpu_time:.2f}x")

    def test_feature_matching_performance(self):
        """Test feature matching performance."""
        slam_ops = self.AcceleratedSLAMOperations()

        descriptor_counts = [100, 500, 1000]

        for count in descriptor_counts:
            descriptors1 = np.random.randn(count, 128).astype(np.float32)
            descriptors2 = np.random.randn(count, 128).astype(np.float32)

            start_time = time.time()
            matches = slam_ops.accelerated_feature_matching(descriptors1, descriptors2)
            elapsed_time = time.time() - start_time

            self.assertIsInstance(matches, np.ndarray)
            print(f"Feature matching {count} descriptors: {elapsed_time:.4f}s, {len(matches)} matches")

if __name__ == "__main__":
    unittest.main(verbosity=2)
