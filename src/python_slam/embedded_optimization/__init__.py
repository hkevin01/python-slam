"""
Embedded System Optimization for Python-SLAM

This module provides optimizations for real-time embedded systems
including ARM processors, memory management, and power efficiency.
"""

from .arm_optimization import ARMOptimizer
from .memory_manager import EmbeddedMemoryManager
from .power_manager import PowerManager
from .real_time_scheduler import RealTimeScheduler
from .embedded_slam import EmbeddedSLAMEngine
from .sensor_fusion import EmbeddedSensorFusion
from .edge_computing import EdgeComputingManager

__all__ = [
    'ARMOptimizer',
    'EmbeddedMemoryManager', 
    'PowerManager',
    'RealTimeScheduler',
    'EmbeddedSLAMEngine',
    'EmbeddedSensorFusion',
    'EdgeComputingManager'
]

# Default embedded configuration
DEFAULT_EMBEDDED_CONFIG = {
    "memory_limit_mb": 512,
    "cpu_frequency_mhz": 1000,
    "power_budget_watts": 5.0,
    "real_time_constraints": {
        "max_processing_time_ms": 50,
        "target_fps": 20,
        "priority_levels": 4
    },
    "optimization_level": "balanced",  # "performance", "balanced", "power_save"
    "enable_neon": True,
    "enable_cache_optimization": True,
    "enable_power_management": True
}
