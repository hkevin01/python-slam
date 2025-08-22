"""
GPU Manager for Python-SLAM

This module provides unified management of GPU acceleration backends,
automatic backend selection, and load balancing across available GPUs.
"""

import logging
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

from .gpu_detector import GPUDetector, GPUBackend, GPUInfo
from .cuda_acceleration import CudaAccelerator, CudaConfig
from .rocm_acceleration import ROCmAccelerator, ROCmConfig
from .metal_acceleration import MetalAccelerator, MetalConfig

logger = logging.getLogger(__name__)

class AcceleratorState(Enum):
    """State of a GPU accelerator."""
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"

@dataclass
class AcceleratorInfo:
    """Information about an active accelerator."""
    backend: GPUBackend
    accelerator: Union[CudaAccelerator, ROCmAccelerator, MetalAccelerator]
    state: AcceleratorState
    load_factor: float = 0.0  # 0.0 = idle, 1.0 = fully loaded
    memory_usage: Dict[str, int] = None

    def __post_init__(self):
        if self.memory_usage is None:
            self.memory_usage = {"total": 0, "used": 0, "free": 0}

class GPUManager:
    """Unified GPU acceleration manager for SLAM operations."""

    def __init__(self, auto_detect: bool = True):
        self.detector = GPUDetector()
        self.accelerators: Dict[GPUBackend, AcceleratorInfo] = {}
        self._preferred_backend: Optional[GPUBackend] = None

        if auto_detect:
            self.initialize_accelerators()

    def initialize_accelerators(self) -> bool:
        """Initialize all available GPU accelerators."""
        logger.info("Initializing GPU accelerators...")

        available_gpus = self.detector.detect_all_gpus()
        success_count = 0

        for gpu_info in available_gpus:
            if gpu_info.backend == GPUBackend.CPU_FALLBACK:
                continue  # Skip CPU fallback for now

            try:
                accelerator = self._create_accelerator(gpu_info.backend)
                if accelerator and accelerator.initialize():
                    self.accelerators[gpu_info.backend] = AcceleratorInfo(
                        backend=gpu_info.backend,
                        accelerator=accelerator,
                        state=AcceleratorState.READY,
                        memory_usage=accelerator.get_memory_info()
                    )
                    success_count += 1
                    logger.info(f"Initialized {gpu_info.backend.value} accelerator")

            except Exception as e:
                logger.error(f"Failed to initialize {gpu_info.backend.value}: {e}")

        if success_count > 0:
            self._set_preferred_backend()
            logger.info(f"Successfully initialized {success_count} GPU accelerators")
            return True
        else:
            logger.warning("No GPU accelerators initialized, falling back to CPU")
            return False

    def _create_accelerator(self, backend: GPUBackend):
        """Create an accelerator instance for the given backend."""
        if backend == GPUBackend.CUDA:
            return CudaAccelerator(CudaConfig())
        elif backend == GPUBackend.ROCM:
            return ROCmAccelerator(ROCmConfig())
        elif backend == GPUBackend.METAL:
            return MetalAccelerator(MetalConfig())
        else:
            logger.warning(f"Unsupported backend: {backend}")
            return None

    def _set_preferred_backend(self):
        """Set the preferred backend based on performance and availability."""
        # Priority order: CUDA > Metal > ROCm > OpenCL
        priority = [GPUBackend.CUDA, GPUBackend.METAL, GPUBackend.ROCM, GPUBackend.OPENCL]

        for backend in priority:
            if backend in self.accelerators:
                self._preferred_backend = backend
                logger.info(f"Set preferred backend to {backend.value}")
                return

    def get_best_accelerator(self) -> Optional[Union[CudaAccelerator, ROCmAccelerator, MetalAccelerator]]:
        """Get the best available accelerator."""
        if not self.accelerators:
            return None

        # Return preferred backend if available and ready
        if (self._preferred_backend and
            self._preferred_backend in self.accelerators and
            self.accelerators[self._preferred_backend].state == AcceleratorState.READY):
            return self.accelerators[self._preferred_backend].accelerator

        # Find the least loaded ready accelerator
        best_accelerator = None
        lowest_load = float('inf')

        for info in self.accelerators.values():
            if info.state == AcceleratorState.READY and info.load_factor < lowest_load:
                lowest_load = info.load_factor
                best_accelerator = info.accelerator

        return best_accelerator

    def get_accelerator(self, backend: GPUBackend) -> Optional[Union[CudaAccelerator, ROCmAccelerator, MetalAccelerator]]:
        """Get a specific accelerator by backend."""
        if backend in self.accelerators:
            info = self.accelerators[backend]
            if info.state == AcceleratorState.READY:
                return info.accelerator
        return None

    def get_available_backends(self) -> List[GPUBackend]:
        """Get list of available and ready backends."""
        return [backend for backend, info in self.accelerators.items()
                if info.state == AcceleratorState.READY]

    def get_accelerator_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all accelerators."""
        status = {}

        for backend, info in self.accelerators.items():
            try:
                # Update memory usage
                info.memory_usage = info.accelerator.get_memory_info()

                status[backend.value] = {
                    "state": info.state.value,
                    "load_factor": info.load_factor,
                    "memory_usage": info.memory_usage,
                    "device_info": (info.accelerator.get_device_info()
                                  if hasattr(info.accelerator, 'get_device_info')
                                  else {})
                }
            except Exception as e:
                status[backend.value] = {
                    "state": AcceleratorState.ERROR.value,
                    "error": str(e)
                }

        return status

    def set_accelerator_state(self, backend: GPUBackend, state: AcceleratorState):
        """Set the state of a specific accelerator."""
        if backend in self.accelerators:
            self.accelerators[backend].state = state
            logger.debug(f"Set {backend.value} state to {state.value}")

    def update_load_factor(self, backend: GPUBackend, load_factor: float):
        """Update the load factor for a specific accelerator."""
        if backend in self.accelerators:
            self.accelerators[backend].load_factor = max(0.0, min(1.0, load_factor))

    def run_benchmark(self) -> Dict[str, Dict[str, float]]:
        """Run performance benchmarks on all available accelerators."""
        logger.info("Running GPU benchmarks...")
        results = {}

        for backend, info in self.accelerators.items():
            if info.state != AcceleratorState.READY:
                continue

            try:
                logger.info(f"Benchmarking {backend.value}...")

                if hasattr(info.accelerator, 'memory_benchmark'):
                    benchmark_results = info.accelerator.memory_benchmark()
                    results[backend.value] = benchmark_results
                    logger.info(f"{backend.value} benchmark completed")
                else:
                    logger.warning(f"{backend.value} does not support benchmarking")

            except Exception as e:
                logger.error(f"Benchmark failed for {backend.value}: {e}")
                results[backend.value] = {"error": str(e)}

        return results

    def feature_matching(self,
                        descriptors1,
                        descriptors2,
                        distance_threshold: float = 0.8,
                        preferred_backend: Optional[GPUBackend] = None):
        """Perform feature matching using the best available accelerator."""
        accelerator = None

        if preferred_backend and preferred_backend in self.accelerators:
            accelerator = self.get_accelerator(preferred_backend)

        if not accelerator:
            accelerator = self.get_best_accelerator()

        if not accelerator:
            raise RuntimeError("No GPU accelerator available for feature matching")

        try:
            return accelerator.feature_matching_gpu(descriptors1, descriptors2, distance_threshold)
        except Exception as e:
            logger.error(f"Feature matching failed: {e}")
            raise

    def matrix_operations(self,
                         matrix_a,
                         matrix_b,
                         operation: str = "multiply",
                         preferred_backend: Optional[GPUBackend] = None):
        """Perform matrix operations using the best available accelerator."""
        accelerator = None

        if preferred_backend and preferred_backend in self.accelerators:
            accelerator = self.get_accelerator(preferred_backend)

        if not accelerator:
            accelerator = self.get_best_accelerator()

        if not accelerator:
            raise RuntimeError("No GPU accelerator available for matrix operations")

        try:
            if hasattr(accelerator, 'matrix_operations_gpu'):
                return accelerator.matrix_operations_gpu(matrix_a, matrix_b, operation)
            else:
                raise NotImplementedError(f"Matrix operations not implemented for this backend")
        except Exception as e:
            logger.error(f"Matrix operations failed: {e}")
            raise

    def bundle_adjustment(self,
                         points_3d,
                         camera_poses,
                         observations,
                         camera_intrinsics,
                         max_iterations: int = 100,
                         preferred_backend: Optional[GPUBackend] = None):
        """Perform bundle adjustment using the best available accelerator."""
        accelerator = None

        if preferred_backend and preferred_backend in self.accelerators:
            accelerator = self.get_accelerator(preferred_backend)

        if not accelerator:
            accelerator = self.get_best_accelerator()

        if not accelerator:
            raise RuntimeError("No GPU accelerator available for bundle adjustment")

        try:
            if hasattr(accelerator, 'bundle_adjustment_gpu'):
                return accelerator.bundle_adjustment_gpu(
                    points_3d, camera_poses, observations, camera_intrinsics, max_iterations
                )
            else:
                raise NotImplementedError(f"Bundle adjustment not implemented for this backend")
        except Exception as e:
            logger.error(f"Bundle adjustment failed: {e}")
            raise

    def get_optimal_backend_for_task(self, task_type: str) -> Optional[GPUBackend]:
        """Get the optimal backend for a specific task type."""
        if not self.accelerators:
            return None

        # Task-specific backend preferences
        preferences = {
            "feature_matching": [GPUBackend.CUDA, GPUBackend.METAL, GPUBackend.ROCM],
            "matrix_operations": [GPUBackend.CUDA, GPUBackend.ROCM, GPUBackend.METAL],
            "bundle_adjustment": [GPUBackend.CUDA, GPUBackend.ROCM, GPUBackend.METAL],
            "image_processing": [GPUBackend.METAL, GPUBackend.CUDA, GPUBackend.ROCM],
            "optimization": [GPUBackend.CUDA, GPUBackend.ROCM, GPUBackend.METAL]
        }

        task_preferences = preferences.get(task_type,
                                         [GPUBackend.CUDA, GPUBackend.METAL, GPUBackend.ROCM])

        for backend in task_preferences:
            if (backend in self.accelerators and
                self.accelerators[backend].state == AcceleratorState.READY):
                return backend

        return None

    def cleanup(self):
        """Clean up all accelerators."""
        logger.info("Cleaning up GPU accelerators...")

        for backend, info in self.accelerators.items():
            try:
                info.accelerator.cleanup()
                logger.debug(f"Cleaned up {backend.value} accelerator")
            except Exception as e:
                logger.error(f"Error cleaning up {backend.value}: {e}")

        self.accelerators.clear()
        logger.info("GPU cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
