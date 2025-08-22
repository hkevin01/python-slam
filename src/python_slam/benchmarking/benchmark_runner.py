"""
Benchmark Runner for Python SLAM

Orchestrates benchmark execution, data collection, and result aggregation.
"""

import os
import time
import json
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import concurrent.futures
import logging

import numpy as np

from .benchmark_metrics import (
    TrajectoryMetrics, ProcessingMetrics, MapQualityMetrics,
    MemoryMetrics, MetricResult
)
from .dataset_loader import DatasetLoader
from ..slam_interfaces import SLAMFactory, SLAMConfiguration, SensorType


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    # Test identification
    name: str = "SLAM Benchmark"
    description: str = "Comprehensive SLAM algorithm evaluation"
    version: str = "1.0"

    # Algorithms to test
    algorithms: List[str] = field(default_factory=lambda: ["orb_slam3", "rtabmap", "cartographer"])
    sensor_types: List[SensorType] = field(default_factory=lambda: [SensorType.MONOCULAR])

    # Datasets
    datasets: List[str] = field(default_factory=list)
    dataset_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Metrics to evaluate
    enable_trajectory_metrics: bool = True
    enable_processing_metrics: bool = True
    enable_map_quality_metrics: bool = True
    enable_memory_metrics: bool = True

    # Execution settings
    max_parallel_jobs: int = 1
    timeout_per_test: float = 3600.0  # 1 hour
    repeat_count: int = 3

    # Output settings
    output_directory: str = "benchmark_results"
    save_raw_data: bool = True
    save_trajectories: bool = True
    save_maps: bool = False

    # Advanced settings
    warm_up_frames: int = 10
    max_frames: Optional[int] = None
    skip_frames: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'algorithms': self.algorithms,
            'sensor_types': [st.name for st in self.sensor_types],
            'datasets': self.datasets,
            'dataset_configs': self.dataset_configs,
            'metrics': {
                'trajectory': self.enable_trajectory_metrics,
                'processing': self.enable_processing_metrics,
                'map_quality': self.enable_map_quality_metrics,
                'memory': self.enable_memory_metrics
            },
            'execution': {
                'max_parallel_jobs': self.max_parallel_jobs,
                'timeout_per_test': self.timeout_per_test,
                'repeat_count': self.repeat_count
            },
            'output': {
                'output_directory': self.output_directory,
                'save_raw_data': self.save_raw_data,
                'save_trajectories': self.save_trajectories,
                'save_maps': self.save_maps
            },
            'advanced': {
                'warm_up_frames': self.warm_up_frames,
                'max_frames': self.max_frames,
                'skip_frames': self.skip_frames
            }
        }


@dataclass
class TestResult:
    """Container for individual test results."""
    algorithm: str
    sensor_type: SensorType
    dataset: str
    repetition: int
    metrics: Dict[str, MetricResult]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class BenchmarkRunner:
    """
    Main benchmark execution engine.

    Orchestrates the execution of multiple SLAM algorithms across datasets,
    collects metrics, and generates comprehensive results.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.slam_factory = SLAMFactory()
        self.dataset_loader = DatasetLoader()

        # Setup logging
        self.logger = self._setup_logging()

        # Results storage
        self.results: List[TestResult] = []

        # Progress tracking
        self.total_tests = 0
        self.completed_tests = 0
        self.progress_callback: Optional[Callable[[float], None]] = None

        # Output directory
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_progress_callback(self, callback: Callable[[float], None]):
        """Set callback for progress updates."""
        self.progress_callback = callback

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Execute the complete benchmark suite.

        Returns:
            Dictionary containing aggregated results and statistics
        """
        self.logger.info(f"Starting benchmark: {self.config.name}")
        start_time = time.time()

        # Calculate total number of tests
        self.total_tests = (
            len(self.config.algorithms) *
            len(self.config.sensor_types) *
            len(self.config.datasets) *
            self.config.repeat_count
        )

        self.logger.info(f"Total tests to execute: {self.total_tests}")

        # Generate test matrix
        test_cases = self._generate_test_cases()

        # Execute tests
        if self.config.max_parallel_jobs == 1:
            self._run_sequential(test_cases)
        else:
            self._run_parallel(test_cases)

        # Generate summary
        execution_time = time.time() - start_time
        summary = self._generate_summary(execution_time)

        # Save results
        self._save_results(summary)

        self.logger.info(f"Benchmark completed in {execution_time:.2f} seconds")
        return summary

    def _generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate all test case combinations."""
        test_cases = []

        for algorithm in self.config.algorithms:
            for sensor_type in self.config.sensor_types:
                for dataset in self.config.datasets:
                    for rep in range(self.config.repeat_count):
                        test_cases.append({
                            'algorithm': algorithm,
                            'sensor_type': sensor_type,
                            'dataset': dataset,
                            'repetition': rep + 1
                        })

        return test_cases

    def _run_sequential(self, test_cases: List[Dict[str, Any]]):
        """Run tests sequentially."""
        for test_case in test_cases:
            result = self._execute_single_test(test_case)
            self.results.append(result)
            self._update_progress()

    def _run_parallel(self, test_cases: List[Dict[str, Any]]):
        """Run tests in parallel."""
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.config.max_parallel_jobs
        ) as executor:
            # Submit all jobs
            future_to_test = {
                executor.submit(self._execute_single_test_worker, test_case): test_case
                for test_case in test_cases
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_test):
                test_case = future_to_test[future]
                try:
                    result = future.result(timeout=self.config.timeout_per_test)
                    self.results.append(result)
                except Exception as e:
                    # Create failed result
                    result = TestResult(
                        algorithm=test_case['algorithm'],
                        sensor_type=test_case['sensor_type'],
                        dataset=test_case['dataset'],
                        repetition=test_case['repetition'],
                        metrics={},
                        execution_time=0.0,
                        success=False,
                        error_message=str(e)
                    )
                    self.results.append(result)

                self._update_progress()

    def _execute_single_test(self, test_case: Dict[str, Any]) -> TestResult:
        """Execute a single test case."""
        algorithm = test_case['algorithm']
        sensor_type = test_case['sensor_type']
        dataset = test_case['dataset']
        repetition = test_case['repetition']

        self.logger.info(
            f"Executing test: {algorithm} + {sensor_type.name} + {dataset} (run {repetition})"
        )

        start_time = time.time()

        try:
            # Load dataset
            dataset_data = self.dataset_loader.load_dataset(
                dataset,
                self.config.dataset_configs.get(dataset, {})
            )

            # Create SLAM configuration
            slam_config = SLAMConfiguration(
                algorithm_name=algorithm,
                sensor_type=sensor_type
            )

            # Initialize SLAM system
            slam_system = self.slam_factory.create_algorithm(slam_config)
            slam_system.initialize()

            # Initialize metrics
            metrics_dict = {}

            if self.config.enable_trajectory_metrics:
                traj_metrics = TrajectoryMetrics()
                if dataset_data.ground_truth_trajectory:
                    traj_metrics.set_ground_truth(dataset_data.ground_truth_trajectory)

            if self.config.enable_processing_metrics:
                proc_metrics = ProcessingMetrics()
                proc_metrics.start_timing()

            if self.config.enable_map_quality_metrics:
                map_metrics = MapQualityMetrics()

            if self.config.enable_memory_metrics:
                mem_metrics = MemoryMetrics()
                mem_metrics.start_monitoring()

            # Process dataset
            estimated_trajectory = self._process_dataset(
                slam_system, dataset_data, proc_metrics
            )

            # Collect metrics
            if self.config.enable_trajectory_metrics and dataset_data.ground_truth_trajectory:
                traj_metrics.set_estimated(estimated_trajectory)
                metrics_dict['ate'] = traj_metrics.compute_ate()
                metrics_dict['rpe'] = traj_metrics.compute_rpe()
                metrics_dict['scale_error'] = traj_metrics.compute_scale_error()

            if self.config.enable_processing_metrics:
                metrics_dict['fps'] = proc_metrics.compute_fps()
                metrics_dict['processing_time'] = proc_metrics.compute_processing_time_stats()
                metrics_dict['throughput'] = proc_metrics.compute_throughput()

            if self.config.enable_map_quality_metrics:
                map_points = slam_system.get_map()
                keyframes = []  # Would need to be extracted from SLAM system
                map_metrics.set_map_data(map_points, keyframes)
                metrics_dict['map_density'] = map_metrics.compute_map_density()
                metrics_dict['map_coverage'] = map_metrics.compute_map_coverage()
                metrics_dict['point_confidence'] = map_metrics.compute_point_confidence()

            if self.config.enable_memory_metrics:
                mem_metrics.stop_monitoring()
                metrics_dict['memory_usage'] = mem_metrics.compute_memory_usage()
                metrics_dict['memory_growth'] = mem_metrics.compute_memory_growth()

            execution_time = time.time() - start_time

            # Save intermediate results if requested
            if self.config.save_trajectories:
                self._save_trajectory(estimated_trajectory, test_case)

            if self.config.save_maps:
                self._save_map(slam_system, test_case)

            return TestResult(
                algorithm=algorithm,
                sensor_type=sensor_type,
                dataset=dataset,
                repetition=repetition,
                metrics=metrics_dict,
                execution_time=execution_time,
                success=True
            )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Test failed: {e}")

            return TestResult(
                algorithm=algorithm,
                sensor_type=sensor_type,
                dataset=dataset,
                repetition=repetition,
                metrics={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )

    @staticmethod
    def _execute_single_test_worker(test_case: Dict[str, Any]) -> TestResult:
        """Worker function for parallel execution."""
        # This would need to be a standalone function that can be pickled
        # For now, return a placeholder
        return TestResult(
            algorithm=test_case['algorithm'],
            sensor_type=test_case['sensor_type'],
            dataset=test_case['dataset'],
            repetition=test_case['repetition'],
            metrics={},
            execution_time=0.0,
            success=False,
            error_message="Parallel execution not fully implemented"
        )

    def _process_dataset(self, slam_system, dataset_data, proc_metrics):
        """Process dataset through SLAM system."""
        from ..slam_interfaces import SLAMTrajectory, SLAMPose

        estimated_trajectory = SLAMTrajectory()
        frame_count = 0

        # Warm-up phase
        for _ in range(min(self.config.warm_up_frames, len(dataset_data.images))):
            if frame_count >= len(dataset_data.images):
                break

            image = dataset_data.images[frame_count]
            timestamp = dataset_data.timestamps[frame_count] if dataset_data.timestamps else frame_count

            slam_system.process_image(image, timestamp)
            frame_count += 1

        # Main processing
        while frame_count < len(dataset_data.images):
            if self.config.max_frames and (frame_count - self.config.warm_up_frames) >= self.config.max_frames:
                break

            # Skip frames if configured
            if (frame_count - self.config.warm_up_frames) % (self.config.skip_frames + 1) != 0:
                frame_count += 1
                continue

            image = dataset_data.images[frame_count]
            timestamp = dataset_data.timestamps[frame_count] if dataset_data.timestamps else frame_count

            # Time the processing
            proc_start = time.time()
            success = slam_system.process_image(image, timestamp)
            proc_time = time.time() - proc_start

            proc_metrics.record_frame_processing(proc_time)

            # Get current pose
            if success:
                pose = slam_system.get_pose()
                if pose:
                    estimated_trajectory.poses.append(pose)

            frame_count += 1

        return estimated_trajectory

    def _update_progress(self):
        """Update progress tracking."""
        self.completed_tests += 1
        if self.progress_callback:
            progress = self.completed_tests / self.total_tests
            self.progress_callback(progress)

    def _generate_summary(self, execution_time: float) -> Dict[str, Any]:
        """Generate benchmark summary."""
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]

        summary = {
            'benchmark_info': {
                'name': self.config.name,
                'description': self.config.description,
                'version': self.config.version,
                'timestamp': time.time(),
                'execution_time': execution_time
            },
            'test_summary': {
                'total_tests': len(self.results),
                'successful_tests': len(successful_tests),
                'failed_tests': len(failed_tests),
                'success_rate': len(successful_tests) / len(self.results) if self.results else 0.0
            },
            'algorithm_results': self._aggregate_by_algorithm(),
            'dataset_results': self._aggregate_by_dataset(),
            'metric_statistics': self._compute_metric_statistics(),
            'configuration': self.config.to_dict(),
            'raw_results': [self._result_to_dict(r) for r in self.results] if self.config.save_raw_data else []
        }

        return summary

    def _aggregate_by_algorithm(self) -> Dict[str, Any]:
        """Aggregate results by algorithm."""
        algorithm_results = {}

        for algorithm in self.config.algorithms:
            algo_results = [r for r in self.results if r.algorithm == algorithm and r.success]

            if algo_results:
                algorithm_results[algorithm] = {
                    'test_count': len(algo_results),
                    'success_rate': len(algo_results) / len([r for r in self.results if r.algorithm == algorithm]),
                    'average_metrics': self._average_metrics(algo_results),
                    'execution_time': {
                        'mean': np.mean([r.execution_time for r in algo_results]),
                        'std': np.std([r.execution_time for r in algo_results]),
                        'min': np.min([r.execution_time for r in algo_results]),
                        'max': np.max([r.execution_time for r in algo_results])
                    }
                }

        return algorithm_results

    def _aggregate_by_dataset(self) -> Dict[str, Any]:
        """Aggregate results by dataset."""
        dataset_results = {}

        for dataset in self.config.datasets:
            dataset_test_results = [r for r in self.results if r.dataset == dataset and r.success]

            if dataset_test_results:
                dataset_results[dataset] = {
                    'test_count': len(dataset_test_results),
                    'algorithms_tested': list(set(r.algorithm for r in dataset_test_results)),
                    'average_metrics': self._average_metrics(dataset_test_results)
                }

        return dataset_results

    def _average_metrics(self, results: List[TestResult]) -> Dict[str, Any]:
        """Compute average metrics across results."""
        if not results:
            return {}

        # Collect all metric names
        all_metrics = set()
        for result in results:
            all_metrics.update(result.metrics.keys())

        averaged_metrics = {}
        for metric_name in all_metrics:
            values = []
            for result in results:
                if metric_name in result.metrics:
                    values.append(result.metrics[metric_name].value)

            if values:
                averaged_metrics[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }

        return averaged_metrics

    def _compute_metric_statistics(self) -> Dict[str, Any]:
        """Compute overall metric statistics."""
        successful_results = [r for r in self.results if r.success]

        if not successful_results:
            return {}

        # Collect all metrics
        all_metrics = set()
        for result in successful_results:
            all_metrics.update(result.metrics.keys())

        metric_stats = {}
        for metric_name in all_metrics:
            values = []
            units = None
            description = None

            for result in successful_results:
                if metric_name in result.metrics:
                    metric_result = result.metrics[metric_name]
                    values.append(metric_result.value)
                    if units is None:
                        units = metric_result.unit
                        description = metric_result.description

            if values:
                metric_stats[metric_name] = {
                    'unit': units,
                    'description': description,
                    'statistics': {
                        'count': len(values),
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'median': np.median(values),
                        'p25': np.percentile(values, 25),
                        'p75': np.percentile(values, 75),
                        'p95': np.percentile(values, 95),
                        'p99': np.percentile(values, 99)
                    }
                }

        return metric_stats

    def _result_to_dict(self, result: TestResult) -> Dict[str, Any]:
        """Convert TestResult to dictionary."""
        return {
            'algorithm': result.algorithm,
            'sensor_type': result.sensor_type.name,
            'dataset': result.dataset,
            'repetition': result.repetition,
            'success': result.success,
            'execution_time': result.execution_time,
            'timestamp': result.timestamp,
            'error_message': result.error_message,
            'metrics': {
                name: {
                    'value': metric.value,
                    'unit': metric.unit,
                    'description': metric.description,
                    'metadata': metric.metadata
                }
                for name, metric in result.metrics.items()
            }
        }

    def _save_results(self, summary: Dict[str, Any]):
        """Save benchmark results to files."""
        # Save main summary
        summary_file = self.output_dir / "benchmark_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save configuration
        config_file = self.output_dir / "benchmark_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2, default=str)

        self.logger.info(f"Results saved to {self.output_dir}")

    def _save_trajectory(self, trajectory, test_case: Dict[str, Any]):
        """Save trajectory data."""
        filename = f"trajectory_{test_case['algorithm']}_{test_case['dataset']}_rep{test_case['repetition']}.txt"
        filepath = self.output_dir / "trajectories" / filename
        filepath.parent.mkdir(exist_ok=True)

        with open(filepath, 'w') as f:
            for pose in trajectory.poses:
                f.write(f"{pose.timestamp} {pose.position[0]} {pose.position[1]} {pose.position[2]} "
                       f"{pose.orientation[0]} {pose.orientation[1]} {pose.orientation[2]} {pose.orientation[3]}\n")

    def _save_map(self, slam_system, test_case: Dict[str, Any]):
        """Save map data."""
        filename = f"map_{test_case['algorithm']}_{test_case['dataset']}_rep{test_case['repetition']}.dat"
        filepath = self.output_dir / "maps" / filename
        filepath.parent.mkdir(exist_ok=True)

        slam_system.save_map(str(filepath))

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for benchmark execution."""
        logger = logging.getLogger('benchmark_runner')
        logger.setLevel(logging.INFO)

        # Create file handler
        log_file = self.output_dir / "benchmark.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def create_default_benchmark_config() -> BenchmarkConfig:
    """Create a default benchmark configuration."""
    return BenchmarkConfig(
        name="Python SLAM Default Benchmark",
        description="Standard evaluation of SLAM algorithms",
        algorithms=["orb_slam3", "rtabmap", "python_slam"],
        sensor_types=[SensorType.MONOCULAR, SensorType.STEREO],
        datasets=["sample_dataset"],
        repeat_count=3,
        max_parallel_jobs=1,
        output_directory="benchmark_results"
    )
