"""
Unit Tests for Benchmarking System

This module provides detailed unit tests for the benchmarking components.
"""

import unittest
import numpy as np
import tempfile
import os
import json
import time
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestTrajectoryMetrics(unittest.TestCase):
    """Test trajectory evaluation metrics."""
    
    def setUp(self):
        """Set up test environment."""
        try:
            from python_slam.benchmarking.benchmark_metrics import TrajectoryMetrics
            self.TrajectoryMetrics = TrajectoryMetrics
        except ImportError:
            self.skipTest("Trajectory metrics not available")
        
        # Create test trajectories
        # Ground truth trajectory (perfect circle)
        n_poses = 100
        angles = np.linspace(0, 2*np.pi, n_poses)
        radius = 5.0
        
        self.ground_truth = np.zeros((n_poses, 7))  # [x, y, z, qx, qy, qz, qw]
        self.ground_truth[:, 0] = radius * np.cos(angles)  # x
        self.ground_truth[:, 1] = radius * np.sin(angles)  # y
        self.ground_truth[:, 2] = 0.0  # z
        self.ground_truth[:, 6] = 1.0  # qw (no rotation)
        
        # Estimated trajectory (with some noise)
        self.estimated = self.ground_truth.copy()
        noise_level = 0.1
        self.estimated[:, :3] += np.random.normal(0, noise_level, (n_poses, 3))
        
        # Create a trajectory with systematic error
        self.estimated_systematic = self.ground_truth.copy()
        self.estimated_systematic[:, 0] += 0.5  # Constant offset in x
        self.estimated_systematic[:, 1] += np.linspace(0, 1.0, n_poses)  # Growing offset in y
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = self.TrajectoryMetrics()
        self.assertIsNotNone(metrics)
    
    def test_ate_computation(self):
        """Test Absolute Trajectory Error computation."""
        metrics = self.TrajectoryMetrics()
        
        # Test with identical trajectories
        ate_perfect = metrics.compute_ate(self.ground_truth, self.ground_truth)
        self.assertAlmostEqual(ate_perfect, 0.0, places=10)
        
        # Test with noisy trajectory
        ate_noisy = metrics.compute_ate(self.estimated, self.ground_truth)
        self.assertGreater(ate_noisy, 0.0)
        self.assertLess(ate_noisy, 1.0)  # Should be reasonable given noise level
        
        # Test with systematic error
        ate_systematic = metrics.compute_ate(self.estimated_systematic, self.ground_truth)
        self.assertGreater(ate_systematic, ate_noisy)  # Systematic error should be worse
    
    def test_rpe_computation(self):
        """Test Relative Pose Error computation."""
        metrics = self.TrajectoryMetrics()
        
        # Test with identical trajectories
        rpe_perfect = metrics.compute_rpe(self.ground_truth, self.ground_truth)
        self.assertAlmostEqual(rpe_perfect, 0.0, places=10)
        
        # Test with noisy trajectory
        rpe_noisy = metrics.compute_rpe(self.estimated, self.ground_truth, delta=1)
        self.assertGreater(rpe_noisy, 0.0)
        
        # Test with different delta values
        rpe_delta_5 = metrics.compute_rpe(self.estimated, self.ground_truth, delta=5)
        rpe_delta_10 = metrics.compute_rpe(self.estimated, self.ground_truth, delta=10)
        
        # Larger deltas might have different error characteristics
        self.assertGreater(rpe_delta_5, 0.0)
        self.assertGreater(rpe_delta_10, 0.0)
    
    def test_alignment_computation(self):
        """Test trajectory alignment."""
        metrics = self.TrajectoryMetrics()
        
        # Create a translated and rotated trajectory
        translation = np.array([2.0, 3.0, 1.0])
        rotation_angle = np.pi / 4
        
        translated_traj = self.ground_truth.copy()
        translated_traj[:, :3] += translation
        
        # Test alignment
        if hasattr(metrics, 'align_trajectories'):
            aligned_traj = metrics.align_trajectories(translated_traj, self.ground_truth)
            
            # After alignment, ATE should be much smaller
            ate_before = metrics.compute_ate(translated_traj, self.ground_truth)
            ate_after = metrics.compute_ate(aligned_traj, self.ground_truth)
            
            self.assertLess(ate_after, ate_before)
    
    def test_trajectory_statistics(self):
        """Test trajectory statistics computation."""
        metrics = self.TrajectoryMetrics()
        
        if hasattr(metrics, 'compute_trajectory_statistics'):
            stats = metrics.compute_trajectory_statistics(self.ground_truth)
            
            self.assertIsInstance(stats, dict)
            self.assertIn('total_distance', stats)
            self.assertIn('average_speed', stats)
            self.assertIn('max_speed', stats)
            
            # Check reasonable values
            self.assertGreater(stats['total_distance'], 0)
            self.assertGreaterEqual(stats['average_speed'], 0)
            self.assertGreaterEqual(stats['max_speed'], 0)

class TestProcessingMetrics(unittest.TestCase):
    """Test processing performance metrics."""
    
    def setUp(self):
        """Set up test environment."""
        try:
            from python_slam.benchmarking.benchmark_metrics import ProcessingMetrics
            self.ProcessingMetrics = ProcessingMetrics
        except ImportError:
            self.skipTest("Processing metrics not available")
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = self.ProcessingMetrics()
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.get_frame_count(), 0)
    
    def test_frame_time_recording(self):
        """Test frame time recording."""
        metrics = self.ProcessingMetrics()
        
        # Record some frame times
        frame_times = [0.033, 0.025, 0.040, 0.030, 0.035]  # Various FPS values
        
        for frame_time in frame_times:
            metrics.record_frame_time(frame_time)
        
        self.assertEqual(metrics.get_frame_count(), len(frame_times))
        
        # Test FPS calculation
        current_fps = metrics.get_current_fps()
        self.assertGreater(current_fps, 0)
        self.assertLess(current_fps, 100)  # Reasonable FPS range
        
        average_fps = metrics.get_average_fps()
        self.assertGreater(average_fps, 0)
        self.assertLess(average_fps, 100)
    
    def test_memory_tracking(self):
        """Test memory usage tracking."""
        metrics = self.ProcessingMetrics()
        
        # Record memory usage
        memory_values = [1024, 1056, 1089, 1045, 1078]  # MB
        
        for memory in memory_values:
            metrics.record_memory_usage(memory)
        
        current_memory = metrics.get_current_memory_usage()
        self.assertGreater(current_memory, 0)
        
        peak_memory = metrics.get_peak_memory_usage()
        self.assertGreaterEqual(peak_memory, current_memory)
        self.assertEqual(peak_memory, max(memory_values))
    
    def test_cpu_usage_tracking(self):
        """Test CPU usage tracking."""
        metrics = self.ProcessingMetrics()
        
        # Record CPU usage
        cpu_values = [25.5, 30.2, 28.7, 35.1, 29.8]  # Percentage
        
        for cpu in cpu_values:
            metrics.record_cpu_usage(cpu)
        
        current_cpu = metrics.get_current_cpu_usage()
        self.assertGreaterEqual(current_cpu, 0)
        self.assertLessEqual(current_cpu, 100)
        
        average_cpu = metrics.get_average_cpu_usage()
        self.assertGreaterEqual(average_cpu, 0)
        self.assertLessEqual(average_cpu, 100)
    
    def test_processing_time_tracking(self):
        """Test processing time tracking for different operations."""
        metrics = self.ProcessingMetrics()
        
        # Record processing times for different operations
        operations = {
            'feature_extraction': [0.005, 0.007, 0.006, 0.008],
            'feature_matching': [0.012, 0.015, 0.013, 0.016],
            'pose_estimation': [0.008, 0.010, 0.009, 0.011],
            'bundle_adjustment': [0.025, 0.030, 0.028, 0.032]
        }
        
        for operation, times in operations.items():
            for processing_time in times:
                metrics.record_processing_time(operation, processing_time)
        
        # Test retrieval of processing times
        for operation in operations:
            avg_time = metrics.get_average_processing_time(operation)
            self.assertGreater(avg_time, 0)
            
            total_time = metrics.get_total_processing_time(operation)
            self.assertGreater(total_time, 0)
    
    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        metrics = self.ProcessingMetrics()
        
        # Record some data
        metrics.record_frame_time(0.033)
        metrics.record_memory_usage(1024)
        metrics.record_cpu_usage(25.0)
        
        # Reset metrics
        metrics.reset()
        
        self.assertEqual(metrics.get_frame_count(), 0)
        self.assertEqual(metrics.get_current_memory_usage(), 0)
        self.assertEqual(metrics.get_current_cpu_usage(), 0)

class TestBenchmarkRunner(unittest.TestCase):
    """Test benchmark runner functionality."""
    
    def setUp(self):
        """Set up test environment."""
        try:
            from python_slam.benchmarking.benchmark_runner import BenchmarkRunner, BenchmarkConfig, BenchmarkResult
            self.BenchmarkRunner = BenchmarkRunner
            self.BenchmarkConfig = BenchmarkConfig
            self.BenchmarkResult = BenchmarkResult
        except ImportError:
            self.skipTest("Benchmark runner not available")
        
        # Create temporary test data directory
        self.test_dir = tempfile.mkdtemp()
        self.test_config_path = os.path.join(self.test_dir, "test_config.json")
        
        # Create test configuration
        test_config = {
            "datasets": [
                {
                    "name": "test_dataset",
                    "path": os.path.join(self.test_dir, "test_dataset"),
                    "type": "TUM"
                }
            ],
            "algorithms": [
                {
                    "name": "test_algorithm",
                    "config": {"param1": 1.0, "param2": True}
                }
            ],
            "metrics": ["ATE", "RPE"],
            "timeout_seconds": 30
        }
        
        with open(self.test_config_path, 'w') as f:
            json.dump(test_config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_config_creation(self):
        """Test benchmark configuration creation."""
        config = self.BenchmarkConfig()
        self.assertIsNotNone(config)
        
        # Test default values
        self.assertIsInstance(config.timeout_seconds, (int, float))
        self.assertIsInstance(config.enable_parallel_execution, bool)
        self.assertIsInstance(config.max_workers, int)
    
    def test_config_loading(self):
        """Test configuration loading from file."""
        config = self.BenchmarkConfig.from_file(self.test_config_path)
        self.assertIsNotNone(config)
        
        # Verify loaded configuration
        self.assertIn("datasets", config.__dict__)
        self.assertIn("algorithms", config.__dict__)
        self.assertIn("metrics", config.__dict__)
    
    def test_runner_initialization(self):
        """Test runner initialization."""
        config = self.BenchmarkConfig(timeout_seconds=10)
        runner = self.BenchmarkRunner(config)
        
        self.assertIsNotNone(runner)
        self.assertEqual(runner.config.timeout_seconds, 10)
    
    def test_dataset_validation(self):
        """Test dataset validation."""
        config = self.BenchmarkConfig()
        runner = self.BenchmarkRunner(config)
        
        # Test with valid dataset info
        valid_dataset = {
            "name": "test_dataset",
            "path": self.test_dir,
            "type": "TUM"
        }
        
        is_valid = runner.validate_dataset(valid_dataset)
        self.assertIsInstance(is_valid, bool)
        
        # Test with invalid dataset path
        invalid_dataset = {
            "name": "invalid_dataset",
            "path": "/nonexistent/path",
            "type": "TUM"
        }
        
        is_invalid = runner.validate_dataset(invalid_dataset)
        self.assertFalse(is_invalid)
    
    def test_benchmark_result(self):
        """Test benchmark result structure."""
        result = self.BenchmarkResult(
            dataset_name="test_dataset",
            algorithm_name="test_algorithm",
            metrics={"ATE": 0.15, "RPE": 0.08},
            processing_time=5.2,
            memory_usage=512.0,
            success=True
        )
        
        self.assertEqual(result.dataset_name, "test_dataset")
        self.assertEqual(result.algorithm_name, "test_algorithm")
        self.assertEqual(result.metrics["ATE"], 0.15)
        self.assertEqual(result.metrics["RPE"], 0.08)
        self.assertEqual(result.processing_time, 5.2)
        self.assertEqual(result.memory_usage, 512.0)
        self.assertTrue(result.success)
    
    def test_mock_benchmark_execution(self):
        """Test mock benchmark execution."""
        config = self.BenchmarkConfig(timeout_seconds=5)
        runner = self.BenchmarkRunner(config)
        
        # Create a mock algorithm function
        def mock_algorithm(dataset_path, algorithm_config):
            """Mock algorithm that returns dummy trajectory."""
            time.sleep(0.1)  # Simulate processing time
            
            # Return dummy trajectory
            n_poses = 50
            trajectory = np.random.randn(n_poses, 7)
            return trajectory
        
        # Create mock ground truth
        ground_truth = np.random.randn(50, 7)
        
        # Run mock benchmark
        if hasattr(runner, 'run_single_benchmark'):
            result = runner.run_single_benchmark(
                dataset_path=self.test_dir,
                algorithm_func=mock_algorithm,
                algorithm_config={},
                ground_truth=ground_truth,
                dataset_name="mock_dataset",
                algorithm_name="mock_algorithm"
            )
            
            self.assertIsInstance(result, self.BenchmarkResult)
            self.assertTrue(result.success)
            self.assertIn("ATE", result.metrics)
            self.assertIn("RPE", result.metrics)

class TestDatasetLoaders(unittest.TestCase):
    """Test dataset loading functionality."""
    
    def setUp(self):
        """Set up test environment."""
        try:
            from python_slam.benchmarking.benchmark_runner import TUMDatasetLoader, KITTIDatasetLoader
            self.TUMDatasetLoader = TUMDatasetLoader
            self.KITTIDatasetLoader = KITTIDatasetLoader
        except ImportError:
            self.skipTest("Dataset loaders not available")
        
        # Create temporary test data
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock TUM dataset files
        self.create_mock_tum_dataset()
        
        # Create mock KITTI dataset files
        self.create_mock_kitti_dataset()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_mock_tum_dataset(self):
        """Create mock TUM dataset files."""
        tum_dir = os.path.join(self.test_dir, "tum")
        os.makedirs(tum_dir, exist_ok=True)
        
        # Create mock ground truth file
        groundtruth_file = os.path.join(tum_dir, "groundtruth.txt")
        with open(groundtruth_file, 'w') as f:
            f.write("# ground truth trajectory\n")
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            for i in range(100):
                timestamp = i * 0.033  # 30 FPS
                tx, ty, tz = i * 0.1, 0.0, 0.0  # Simple linear motion
                qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0  # No rotation
                f.write(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
        
        # Create mock RGB images directory
        rgb_dir = os.path.join(tum_dir, "rgb")
        os.makedirs(rgb_dir, exist_ok=True)
        
        # Create mock depth images directory
        depth_dir = os.path.join(tum_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        
        # Create associations file
        associations_file = os.path.join(tum_dir, "associations.txt")
        with open(associations_file, 'w') as f:
            for i in range(100):
                timestamp = i * 0.033
                f.write(f"{timestamp} rgb/{i:06d}.png {timestamp} depth/{i:06d}.png\n")
    
    def create_mock_kitti_dataset(self):
        """Create mock KITTI dataset files."""
        kitti_dir = os.path.join(self.test_dir, "kitti")
        os.makedirs(kitti_dir, exist_ok=True)
        
        # Create mock poses file
        poses_file = os.path.join(kitti_dir, "poses.txt")
        with open(poses_file, 'w') as f:
            for i in range(100):
                # Create identity transformation matrix (flattened)
                pose = "1.0 0.0 0.0 {} 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0".format(i * 0.1)
                f.write(pose + "\n")
        
        # Create mock camera calibration
        calib_file = os.path.join(kitti_dir, "calib.txt")
        with open(calib_file, 'w') as f:
            f.write("P0: 718.856 0.0 607.1928 0.0 0.0 718.856 185.2157 0.0 0.0 0.0 1.0 0.0\n")
            f.write("P1: 718.856 0.0 607.1928 -386.1448 0.0 718.856 185.2157 0.0 0.0 0.0 1.0 0.0\n")
    
    def test_tum_dataset_loader(self):
        """Test TUM dataset loader."""
        loader = self.TUMDatasetLoader()
        tum_path = os.path.join(self.test_dir, "tum")
        
        # Test dataset loading
        dataset = loader.load_dataset(tum_path)
        
        self.assertIsInstance(dataset, dict)
        self.assertIn("ground_truth", dataset)
        self.assertIn("associations", dataset)
        
        # Test ground truth format
        ground_truth = dataset["ground_truth"]
        self.assertIsInstance(ground_truth, np.ndarray)
        self.assertEqual(ground_truth.shape[1], 7)  # [x, y, z, qx, qy, qz, qw]
        self.assertEqual(ground_truth.shape[0], 100)  # 100 poses
    
    def test_kitti_dataset_loader(self):
        """Test KITTI dataset loader."""
        loader = self.KITTIDatasetLoader()
        kitti_path = os.path.join(self.test_dir, "kitti")
        
        # Test dataset loading
        dataset = loader.load_dataset(kitti_path)
        
        self.assertIsInstance(dataset, dict)
        self.assertIn("poses", dataset)
        self.assertIn("calibration", dataset)
        
        # Test poses format
        poses = dataset["poses"]
        self.assertIsInstance(poses, np.ndarray)
        self.assertEqual(poses.shape[0], 100)  # 100 poses
        self.assertEqual(poses.shape[1], 7)  # [x, y, z, qx, qy, qz, qw]

class TestBenchmarkReporting(unittest.TestCase):
    """Test benchmark reporting functionality."""
    
    def setUp(self):
        """Set up test environment."""
        try:
            from python_slam.benchmarking.benchmark_runner import BenchmarkResult, BenchmarkReport
            self.BenchmarkResult = BenchmarkResult
            self.BenchmarkReport = BenchmarkReport
        except ImportError:
            self.skipTest("Benchmark reporting not available")
        
        # Create test results
        self.test_results = [
            self.BenchmarkResult(
                dataset_name="dataset1",
                algorithm_name="algorithm1",
                metrics={"ATE": 0.15, "RPE": 0.08},
                processing_time=5.2,
                memory_usage=512.0,
                success=True
            ),
            self.BenchmarkResult(
                dataset_name="dataset1",
                algorithm_name="algorithm2",
                metrics={"ATE": 0.12, "RPE": 0.06},
                processing_time=7.8,
                memory_usage=768.0,
                success=True
            ),
            self.BenchmarkResult(
                dataset_name="dataset2",
                algorithm_name="algorithm1",
                metrics={"ATE": 0.18, "RPE": 0.09},
                processing_time=4.5,
                memory_usage=480.0,
                success=True
            )
        ]
        
        self.output_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.output_dir, ignore_errors=True)
    
    def test_report_generation(self):
        """Test benchmark report generation."""
        if hasattr(self, 'BenchmarkReport'):
            report = self.BenchmarkReport(self.test_results)
            
            # Test report summary
            summary = report.generate_summary()
            self.assertIsInstance(summary, dict)
            self.assertIn("total_benchmarks", summary)
            self.assertIn("successful_benchmarks", summary)
            self.assertIn("average_metrics", summary)
            
            # Test detailed report
            detailed_report = report.generate_detailed_report()
            self.assertIsInstance(detailed_report, str)
            self.assertGreater(len(detailed_report), 0)
    
    def test_report_export(self):
        """Test report export functionality."""
        if hasattr(self, 'BenchmarkReport'):
            report = self.BenchmarkReport(self.test_results)
            
            # Test JSON export
            json_path = os.path.join(self.output_dir, "benchmark_report.json")
            report.export_json(json_path)
            
            self.assertTrue(os.path.exists(json_path))
            
            # Verify JSON content
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), len(self.test_results))
            
            # Test CSV export
            csv_path = os.path.join(self.output_dir, "benchmark_report.csv")
            report.export_csv(csv_path)
            
            self.assertTrue(os.path.exists(csv_path))

if __name__ == "__main__":
    unittest.main(verbosity=2)
