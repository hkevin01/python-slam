"""
Comprehensive Benchmarking System for Python SLAM

Standardized evaluation metrics, automated testing, and report generation.
"""

from .benchmark_metrics import (
    TrajectoryMetrics,
    ProcessingMetrics,
    MapQualityMetrics,
    MemoryMetrics
)
from .benchmark_runner import BenchmarkRunner, BenchmarkConfig
from .dataset_loader import DatasetLoader, KITTILoader, TUMLoader, EuRoCLoader
from .report_generator import ReportGenerator, LaTeXReporter, MarkdownReporter
from .simulation import GazeboSimulator, SyntheticDataGenerator

__all__ = [
    'TrajectoryMetrics',
    'ProcessingMetrics',
    'MapQualityMetrics',
    'MemoryMetrics',
    'BenchmarkRunner',
    'BenchmarkConfig',
    'DatasetLoader',
    'KITTILoader',
    'TUMLoader',
    'EuRoCLoader',
    'ReportGenerator',
    'LaTeXReporter',
    'MarkdownReporter',
    'GazeboSimulator',
    'SyntheticDataGenerator'
]

__version__ = "1.0.0"
