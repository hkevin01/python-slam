"""
Standardized Evaluation Metrics for SLAM Benchmarking

Implements standard metrics including ATE, RPE, and custom quality measures.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time
import psutil
import threading
from abc import ABC, abstractmethod

# Import SLAM data structures
from ..slam_interfaces import SLAMPose, SLAMMapPoint, SLAMTrajectory


@dataclass
class MetricResult:
    """Container for metric evaluation results."""
    name: str
    value: float
    unit: str
    description: str
    timestamp: float
    metadata: Dict[str, Any]


class BaseMetric(ABC):
    """Abstract base class for all metrics."""
    
    def __init__(self, name: str, unit: str, description: str):
        self.name = name
        self.unit = unit
        self.description = description
    
    @abstractmethod
    def compute(self, *args, **kwargs) -> MetricResult:
        """Compute the metric value."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset metric state for new evaluation."""
        pass


class TrajectoryMetrics:
    """
    Trajectory evaluation metrics including ATE and RPE.
    
    Based on the TUM RGB-D benchmark evaluation methodology.
    """
    
    def __init__(self):
        self.ground_truth: Optional[SLAMTrajectory] = None
        self.estimated: Optional[SLAMTrajectory] = None
        self.alignment_method = "umeyama"  # or "horn"
    
    def set_ground_truth(self, trajectory: SLAMTrajectory):
        """Set ground truth trajectory."""
        self.ground_truth = trajectory
    
    def set_estimated(self, trajectory: SLAMTrajectory):
        """Set estimated trajectory."""
        self.estimated = trajectory
    
    def compute_ate(self) -> MetricResult:
        """
        Compute Absolute Trajectory Error (ATE).
        
        ATE measures the absolute distances between estimated and ground truth poses
        after optimal alignment.
        """
        if not self.ground_truth or not self.estimated:
            raise ValueError("Both ground truth and estimated trajectories must be set")
        
        # Align trajectories temporally
        aligned_gt, aligned_est = self._align_trajectories()
        
        if len(aligned_gt) == 0:
            return MetricResult(
                name="ATE",
                value=float('inf'),
                unit="meters",
                description="Absolute Trajectory Error",
                timestamp=time.time(),
                metadata={"error": "No aligned poses found"}
            )
        
        # Extract positions
        gt_positions = np.array([pose.position for pose in aligned_gt])
        est_positions = np.array([pose.position for pose in aligned_est])
        
        # Compute optimal alignment transformation
        if self.alignment_method == "umeyama":
            est_aligned = self._umeyama_alignment(est_positions, gt_positions)
        else:
            est_aligned = self._horn_alignment(est_positions, gt_positions)
        
        # Compute absolute errors
        errors = np.linalg.norm(est_aligned - gt_positions, axis=1)
        
        # Statistics
        rmse = np.sqrt(np.mean(errors**2))
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        
        return MetricResult(
            name="ATE",
            value=rmse,
            unit="meters",
            description="Root Mean Square Error of Absolute Trajectory Error",
            timestamp=time.time(),
            metadata={
                "rmse": rmse,
                "mean": mean_error,
                "median": median_error,
                "std": std_error,
                "max": max_error,
                "min": min_error,
                "num_poses": len(errors),
                "alignment_method": self.alignment_method
            }
        )
    
    def compute_rpe(self, delta: float = 1.0) -> MetricResult:
        """
        Compute Relative Pose Error (RPE).
        
        RPE measures the local accuracy of the trajectory over a fixed time interval.
        
        Args:
            delta: Time interval for relative pose computation (seconds)
        """
        if not self.ground_truth or not self.estimated:
            raise ValueError("Both ground truth and estimated trajectories must be set")
        
        # Align trajectories temporally
        aligned_gt, aligned_est = self._align_trajectories()
        
        if len(aligned_gt) < 2:
            return MetricResult(
                name="RPE",
                value=float('inf'),
                unit="meters",
                description="Relative Pose Error",
                timestamp=time.time(),
                metadata={"error": "Insufficient aligned poses"}
            )
        
        translation_errors = []
        rotation_errors = []
        
        # Compute relative poses
        for i in range(len(aligned_gt) - 1):
            # Find poses separated by delta time
            current_time = aligned_gt[i].timestamp
            target_time = current_time + delta
            
            # Find closest pose to target time
            j = i + 1
            while j < len(aligned_gt) and aligned_gt[j].timestamp < target_time:
                j += 1
            
            if j >= len(aligned_gt):
                break
            
            # Compute relative transformations
            gt_rel = self._compute_relative_pose(aligned_gt[i], aligned_gt[j])
            est_rel = self._compute_relative_pose(aligned_est[i], aligned_est[j])
            
            # Compute relative error
            trans_error = np.linalg.norm(gt_rel[:3] - est_rel[:3])
            rot_error = self._rotation_error(gt_rel[3:], est_rel[3:])
            
            translation_errors.append(trans_error)
            rotation_errors.append(rot_error)
        
        if not translation_errors:
            return MetricResult(
                name="RPE",
                value=float('inf'),
                unit="meters",
                description="Relative Pose Error",
                timestamp=time.time(),
                metadata={"error": "No relative poses computed"}
            )
        
        # Statistics
        trans_rmse = np.sqrt(np.mean(np.array(translation_errors)**2))
        rot_rmse = np.sqrt(np.mean(np.array(rotation_errors)**2))
        
        return MetricResult(
            name="RPE",
            value=trans_rmse,
            unit="meters",
            description="Root Mean Square Error of Relative Pose Error (Translation)",
            timestamp=time.time(),
            metadata={
                "translation_rmse": trans_rmse,
                "rotation_rmse": rot_rmse,
                "translation_mean": np.mean(translation_errors),
                "rotation_mean": np.mean(rotation_errors),
                "num_relative_poses": len(translation_errors),
                "delta": delta
            }
        )
    
    def compute_scale_error(self) -> MetricResult:
        """Compute scale drift error."""
        if not self.ground_truth or not self.estimated:
            raise ValueError("Both trajectories must be set")
        
        aligned_gt, aligned_est = self._align_trajectories()
        
        if len(aligned_gt) < 2:
            return MetricResult(
                name="Scale Error",
                value=float('inf'),
                unit="ratio",
                description="Scale drift error",
                timestamp=time.time(),
                metadata={"error": "Insufficient poses"}
            )
        
        # Compute trajectory lengths
        gt_length = self._compute_trajectory_length(aligned_gt)
        est_length = self._compute_trajectory_length(aligned_est)
        
        if gt_length == 0:
            scale_error = float('inf')
        else:
            scale_error = abs(est_length / gt_length - 1.0)
        
        return MetricResult(
            name="Scale Error",
            value=scale_error,
            unit="ratio",
            description="Relative scale error between trajectories",
            timestamp=time.time(),
            metadata={
                "ground_truth_length": gt_length,
                "estimated_length": est_length,
                "scale_ratio": est_length / gt_length if gt_length > 0 else float('inf')
            }
        )
    
    def _align_trajectories(self) -> Tuple[List[SLAMPose], List[SLAMPose]]:
        """Align trajectories temporally using timestamps."""
        if not self.ground_truth or not self.estimated:
            return [], []
        
        gt_poses = self.ground_truth.poses
        est_poses = self.estimated.poses
        
        aligned_gt = []
        aligned_est = []
        
        # Simple nearest neighbor temporal alignment
        max_time_diff = 0.02  # 20ms tolerance
        
        for gt_pose in gt_poses:
            best_match = None
            best_time_diff = float('inf')
            
            for est_pose in est_poses:
                time_diff = abs(gt_pose.timestamp - est_pose.timestamp)
                if time_diff < best_time_diff and time_diff < max_time_diff:
                    best_time_diff = time_diff
                    best_match = est_pose
            
            if best_match:
                aligned_gt.append(gt_pose)
                aligned_est.append(best_match)
        
        return aligned_gt, aligned_est
    
    def _umeyama_alignment(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Umeyama alignment algorithm for point clouds.
        
        Computes optimal similarity transformation (rotation, translation, scale).
        """
        if source.shape != target.shape:
            raise ValueError("Source and target must have same shape")
        
        n, m = source.shape
        
        # Compute centroids
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)
        
        # Center the points
        source_centered = source - source_centroid
        target_centered = target - target_centroid
        
        # Compute cross-covariance matrix
        H = source_centered.T @ target_centered
        
        # SVD
        U, S, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = Vt.T @ U.T
        
        # Ensure proper rotation matrix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Compute scale
        source_var = np.var(source_centered)
        scale = np.trace(np.diag(S)) / source_var if source_var > 0 else 1.0
        
        # Compute translation
        t = target_centroid - scale * R @ source_centroid
        
        # Apply transformation
        return scale * (source @ R.T) + t
    
    def _horn_alignment(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Horn's method for absolute orientation."""
        # Simplified Horn's method (rotation + translation only)
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)
        
        source_centered = source - source_centroid
        target_centered = target - target_centroid
        
        H = source_centered.T @ target_centered
        U, _, Vt = np.linalg.svd(H)
        
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = target_centroid - R @ source_centroid
        
        return (source @ R.T) + t
    
    def _compute_relative_pose(self, pose1: SLAMPose, pose2: SLAMPose) -> np.ndarray:
        """Compute relative pose between two poses."""
        # Simplified relative pose computation
        rel_position = pose2.position - pose1.position
        # For rotation, we'd need proper quaternion math
        rel_orientation = pose2.orientation  # Simplified
        
        return np.concatenate([rel_position, rel_orientation])
    
    def _rotation_error(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Compute rotation error between quaternions."""
        # Simplified rotation error (angular difference)
        # Proper implementation would use quaternion math
        return np.linalg.norm(q1 - q2)
    
    def _compute_trajectory_length(self, poses: List[SLAMPose]) -> float:
        """Compute total trajectory length."""
        if len(poses) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(poses)):
            dist = np.linalg.norm(poses[i].position - poses[i-1].position)
            total_length += dist
        
        return total_length


class ProcessingMetrics:
    """
    Processing performance metrics.
    
    Tracks timing, throughput, and computational efficiency.
    """
    
    def __init__(self):
        self.processing_times = []
        self.frame_timestamps = []
        self.start_time = None
        self.total_frames = 0
    
    def start_timing(self):
        """Start timing session."""
        self.start_time = time.time()
        self.processing_times.clear()
        self.frame_timestamps.clear()
        self.total_frames = 0
    
    def record_frame_processing(self, processing_time: float):
        """Record processing time for a single frame."""
        self.processing_times.append(processing_time)
        self.frame_timestamps.append(time.time())
        self.total_frames += 1
    
    def compute_fps(self) -> MetricResult:
        """Compute frames per second."""
        if len(self.frame_timestamps) < 2:
            fps = 0.0
        else:
            total_time = self.frame_timestamps[-1] - self.frame_timestamps[0]
            fps = (len(self.frame_timestamps) - 1) / total_time
        
        return MetricResult(
            name="FPS",
            value=fps,
            unit="fps",
            description="Frames processed per second",
            timestamp=time.time(),
            metadata={
                "total_frames": self.total_frames,
                "total_time": total_time if len(self.frame_timestamps) >= 2 else 0.0
            }
        )
    
    def compute_processing_time_stats(self) -> MetricResult:
        """Compute processing time statistics."""
        if not self.processing_times:
            return MetricResult(
                name="Processing Time",
                value=0.0,
                unit="ms",
                description="Average processing time per frame",
                timestamp=time.time(),
                metadata={"error": "No processing times recorded"}
            )
        
        times_ms = np.array(self.processing_times) * 1000  # Convert to ms
        
        return MetricResult(
            name="Processing Time",
            value=np.mean(times_ms),
            unit="ms",
            description="Average processing time per frame",
            timestamp=time.time(),
            metadata={
                "mean": np.mean(times_ms),
                "median": np.median(times_ms),
                "std": np.std(times_ms),
                "min": np.min(times_ms),
                "max": np.max(times_ms),
                "p95": np.percentile(times_ms, 95),
                "p99": np.percentile(times_ms, 99)
            }
        )
    
    def compute_throughput(self) -> MetricResult:
        """Compute data throughput."""
        if self.start_time is None or len(self.frame_timestamps) == 0:
            return MetricResult(
                name="Throughput",
                value=0.0,
                unit="frames/min",
                description="Processing throughput",
                timestamp=time.time(),
                metadata={"error": "No timing data available"}
            )
        
        elapsed_time = time.time() - self.start_time
        throughput = (self.total_frames / elapsed_time) * 60  # frames per minute
        
        return MetricResult(
            name="Throughput",
            value=throughput,
            unit="frames/min",
            description="Processing throughput in frames per minute",
            timestamp=time.time(),
            metadata={
                "total_frames": self.total_frames,
                "elapsed_time": elapsed_time
            }
        )


class MapQualityMetrics:
    """
    Map quality assessment metrics.
    
    Evaluates map density, coverage, and consistency.
    """
    
    def __init__(self):
        self.map_points: List[SLAMMapPoint] = []
        self.keyframes: List[SLAMPose] = []
    
    def set_map_data(self, map_points: List[SLAMMapPoint], keyframes: List[SLAMPose]):
        """Set map data for evaluation."""
        self.map_points = map_points
        self.keyframes = keyframes
    
    def compute_map_density(self) -> MetricResult:
        """Compute map point density."""
        if not self.map_points:
            return MetricResult(
                name="Map Density",
                value=0.0,
                unit="points/m³",
                description="Map point density",
                timestamp=time.time(),
                metadata={"error": "No map points available"}
            )
        
        # Compute bounding box volume
        positions = np.array([point.position for point in self.map_points])
        min_bounds = np.min(positions, axis=0)
        max_bounds = np.max(positions, axis=0)
        volume = np.prod(max_bounds - min_bounds)
        
        density = len(self.map_points) / volume if volume > 0 else 0.0
        
        return MetricResult(
            name="Map Density",
            value=density,
            unit="points/m³",
            description="Number of map points per cubic meter",
            timestamp=time.time(),
            metadata={
                "total_points": len(self.map_points),
                "volume": volume,
                "bounds_min": min_bounds.tolist(),
                "bounds_max": max_bounds.tolist()
            }
        )
    
    def compute_map_coverage(self) -> MetricResult:
        """Compute map coverage quality."""
        if not self.map_points or not self.keyframes:
            return MetricResult(
                name="Map Coverage",
                value=0.0,
                unit="ratio",
                description="Map coverage quality",
                timestamp=time.time(),
                metadata={"error": "Insufficient data"}
            )
        
        # Simplified coverage metric based on keyframe distribution
        if len(self.keyframes) < 2:
            coverage = 0.0
        else:
            keyframe_positions = np.array([kf.position for kf in self.keyframes])
            
            # Compute average distance between consecutive keyframes
            distances = []
            for i in range(1, len(keyframe_positions)):
                dist = np.linalg.norm(keyframe_positions[i] - keyframe_positions[i-1])
                distances.append(dist)
            
            avg_keyframe_distance = np.mean(distances) if distances else 0.0
            
            # Coverage is inversely related to keyframe spacing
            coverage = 1.0 / (1.0 + avg_keyframe_distance)
        
        return MetricResult(
            name="Map Coverage",
            value=coverage,
            unit="ratio",
            description="Map coverage quality (0-1)",
            timestamp=time.time(),
            metadata={
                "num_keyframes": len(self.keyframes),
                "avg_keyframe_distance": avg_keyframe_distance
            }
        )
    
    def compute_point_confidence(self) -> MetricResult:
        """Compute average map point confidence."""
        if not self.map_points:
            return MetricResult(
                name="Point Confidence",
                value=0.0,
                unit="ratio",
                description="Average map point confidence",
                timestamp=time.time(),
                metadata={"error": "No map points"}
            )
        
        confidences = [point.confidence for point in self.map_points]
        avg_confidence = np.mean(confidences)
        
        return MetricResult(
            name="Point Confidence",
            value=avg_confidence,
            unit="ratio",
            description="Average confidence of map points",
            timestamp=time.time(),
            metadata={
                "mean": avg_confidence,
                "std": np.std(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
                "num_points": len(self.map_points)
            }
        )


class MemoryMetrics:
    """
    Memory usage monitoring.
    
    Tracks memory consumption and allocation patterns.
    """
    
    def __init__(self):
        self.memory_samples = []
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()
    
    def start_monitoring(self, interval: float = 1.0):
        """Start memory monitoring."""
        self.monitoring = True
        self.memory_samples.clear()
        
        def monitor_memory():
            while self.monitoring:
                try:
                    memory_info = self.process.memory_info()
                    self.memory_samples.append({
                        'timestamp': time.time(),
                        'rss': memory_info.rss,
                        'vms': memory_info.vms,
                        'percent': self.process.memory_percent()
                    })
                    time.sleep(interval)
                except Exception:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def compute_memory_usage(self) -> MetricResult:
        """Compute memory usage statistics."""
        if not self.memory_samples:
            return MetricResult(
                name="Memory Usage",
                value=0.0,
                unit="MB",
                description="Memory usage statistics",
                timestamp=time.time(),
                metadata={"error": "No memory samples"}
            )
        
        rss_values = [sample['rss'] / (1024**2) for sample in self.memory_samples]  # MB
        percent_values = [sample['percent'] for sample in self.memory_samples]
        
        return MetricResult(
            name="Memory Usage",
            value=np.mean(rss_values),
            unit="MB",
            description="Average resident set size",
            timestamp=time.time(),
            metadata={
                "rss_mean": np.mean(rss_values),
                "rss_max": np.max(rss_values),
                "rss_min": np.min(rss_values),
                "percent_mean": np.mean(percent_values),
                "percent_max": np.max(percent_values),
                "num_samples": len(self.memory_samples)
            }
        )
    
    def compute_memory_growth(self) -> MetricResult:
        """Compute memory growth rate."""
        if len(self.memory_samples) < 2:
            return MetricResult(
                name="Memory Growth",
                value=0.0,
                unit="MB/min",
                description="Memory growth rate",
                timestamp=time.time(),
                metadata={"error": "Insufficient samples"}
            )
        
        first_sample = self.memory_samples[0]
        last_sample = self.memory_samples[-1]
        
        time_diff = last_sample['timestamp'] - first_sample['timestamp']  # seconds
        memory_diff = (last_sample['rss'] - first_sample['rss']) / (1024**2)  # MB
        
        growth_rate = (memory_diff / time_diff) * 60 if time_diff > 0 else 0.0  # MB/min
        
        return MetricResult(
            name="Memory Growth",
            value=growth_rate,
            unit="MB/min",
            description="Rate of memory growth",
            timestamp=time.time(),
            metadata={
                "growth_rate": growth_rate,
                "total_growth": memory_diff,
                "time_period": time_diff,
                "initial_memory": first_sample['rss'] / (1024**2),
                "final_memory": last_sample['rss'] / (1024**2)
            }
        )
