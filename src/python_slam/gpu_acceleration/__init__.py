"""
GPU Acceleration Module for Python-SLAM

This module provides GPU acceleration support for SLAM operations using:
- CUDA (NVIDIA)
- ROCm (AMD)
- Metal (Apple)

The module automatically detects available GPU backends and provides
a unified interface for accelerated SLAM computations.
"""

from .gpu_detector import GPUDetector, GPUBackend
from .cuda_acceleration import CudaAccelerator
from .rocm_acceleration import ROCmAccelerator
from .metal_acceleration import MetalAccelerator
from .gpu_manager import GPUManager
from .accelerated_operations import AcceleratedSLAMOperations

__all__ = [
    'GPUDetector',
    'GPUBackend',
    'CudaAccelerator',
    'ROCmAccelerator', 
    'MetalAccelerator',
    'GPUManager',
    'AcceleratedSLAMOperations'
]

# Convenience function to get the best available GPU accelerator
def get_gpu_accelerator():
    """Get the best available GPU accelerator for the current system."""
    manager = GPUManager()
    return manager.get_best_accelerator()

# Check if any GPU acceleration is available
def is_gpu_available():
    """Check if any GPU acceleration backend is available."""
    detector = GPUDetector()
    return detector.has_any_gpu()
