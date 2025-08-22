"""
GPU Backend Detection and Management

This module provides automatic detection of available GPU backends
and system capabilities for SLAM acceleration.
"""

import platform
import subprocess
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class GPUBackend(Enum):
    """Supported GPU acceleration backends."""
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"
    OPENCL = "opencl"
    CPU_FALLBACK = "cpu"

@dataclass
class GPUInfo:
    """Information about detected GPU hardware."""
    backend: GPUBackend
    name: str
    memory_mb: int
    compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    available: bool = True

class GPUDetector:
    """Detects available GPU backends and hardware capabilities."""
    
    def __init__(self):
        self.system = platform.system()
        self.machine = platform.machine()
        self._detected_gpus: List[GPUInfo] = []
        self._detection_complete = False
    
    def detect_all_gpus(self) -> List[GPUInfo]:
        """Detect all available GPU backends and hardware."""
        if self._detection_complete:
            return self._detected_gpus
            
        logger.info("Starting GPU detection...")
        
        # Detect CUDA (NVIDIA)
        if self._detect_cuda():
            self._detected_gpus.extend(self._get_cuda_gpus())
        
        # Detect ROCm (AMD)
        if self._detect_rocm():
            self._detected_gpus.extend(self._get_rocm_gpus())
        
        # Detect Metal (Apple)
        if self._detect_metal():
            self._detected_gpus.extend(self._get_metal_gpus())
        
        # Detect OpenCL (Fallback)
        if self._detect_opencl():
            self._detected_gpus.extend(self._get_opencl_gpus())
        
        # Always add CPU fallback
        self._detected_gpus.append(GPUInfo(
            backend=GPUBackend.CPU_FALLBACK,
            name="CPU (Fallback)",
            memory_mb=self._get_system_memory_mb()
        ))
        
        self._detection_complete = True
        logger.info(f"Detected {len(self._detected_gpus)} GPU backends")
        return self._detected_gpus
    
    def get_best_gpu(self) -> GPUInfo:
        """Get the best available GPU for SLAM operations."""
        gpus = self.detect_all_gpus()
        
        # Priority order: CUDA > Metal > ROCm > OpenCL > CPU
        priority = [
            GPUBackend.CUDA,
            GPUBackend.METAL,
            GPUBackend.ROCM,
            GPUBackend.OPENCL,
            GPUBackend.CPU_FALLBACK
        ]
        
        for backend in priority:
            for gpu in gpus:
                if gpu.backend == backend and gpu.available:
                    logger.info(f"Selected GPU: {gpu.name} ({gpu.backend.value})")
                    return gpu
        
        # Fallback to CPU if nothing else works
        return gpus[-1]  # CPU fallback is always last
    
    def has_any_gpu(self) -> bool:
        """Check if any GPU acceleration is available (excluding CPU fallback)."""
        gpus = self.detect_all_gpus()
        return any(gpu.backend != GPUBackend.CPU_FALLBACK and gpu.available for gpu in gpus)
    
    def _detect_cuda(self) -> bool:
        """Detect CUDA availability."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        
        try:
            import cupy
            return True
        except ImportError:
            pass
        
        # Check nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi'], 
                                 capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _detect_rocm(self) -> bool:
        """Detect ROCm availability."""
        # Check for ROCm installation
        try:
            result = subprocess.run(['rocm-smi'], 
                                 capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Check for HIP
        try:
            import torch
            return hasattr(torch.version, 'hip') and torch.version.hip is not None
        except ImportError:
            return False
    
    def _detect_metal(self) -> bool:
        """Detect Metal availability (macOS only)."""
        if self.system != "Darwin":
            return False
        
        try:
            import torch
            return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        except ImportError:
            pass
        
        # Basic macOS check
        try:
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                 capture_output=True, text=True, timeout=10)
            return "Metal" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _detect_opencl(self) -> bool:
        """Detect OpenCL availability."""
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            return len(platforms) > 0
        except ImportError:
            return False
    
    def _get_cuda_gpus(self) -> List[GPUInfo]:
        """Get CUDA GPU information."""
        gpus = []
        
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append(GPUInfo(
                        backend=GPUBackend.CUDA,
                        name=props.name,
                        memory_mb=props.total_memory // (1024 * 1024),
                        compute_capability=f"{props.major}.{props.minor}"
                    ))
        except Exception as e:
            logger.warning(f"Error getting CUDA GPU info: {e}")
        
        return gpus
    
    def _get_rocm_gpus(self) -> List[GPUInfo]:
        """Get ROCm GPU information."""
        gpus = []
        
        try:
            # Try to get info from rocm-smi
            result = subprocess.run(['rocm-smi', '--showproductname'], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'GPU' in line:
                        # Parse GPU name from rocm-smi output
                        parts = line.split()
                        if len(parts) > 2:
                            name = ' '.join(parts[2:])
                            gpus.append(GPUInfo(
                                backend=GPUBackend.ROCM,
                                name=name,
                                memory_mb=8192  # Default, would need more parsing
                            ))
        except Exception as e:
            logger.warning(f"Error getting ROCm GPU info: {e}")
        
        # Fallback: Add generic ROCm entry if detected but can't get details
        if not gpus and self._detect_rocm():
            gpus.append(GPUInfo(
                backend=GPUBackend.ROCM,
                name="AMD GPU (ROCm)",
                memory_mb=8192
            ))
        
        return gpus
    
    def _get_metal_gpus(self) -> List[GPUInfo]:
        """Get Metal GPU information (macOS)."""
        gpus = []
        
        try:
            # Parse system_profiler output for GPU info
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                current_gpu = None
                current_memory = 8192  # Default
                
                for line in lines:
                    line = line.strip()
                    if line.endswith(':') and ('GPU' in line or 'Graphics' in line):
                        if current_gpu:
                            gpus.append(GPUInfo(
                                backend=GPUBackend.METAL,
                                name=current_gpu,
                                memory_mb=current_memory
                            ))
                        current_gpu = line.rstrip(':')
                    elif 'VRAM' in line or 'Memory' in line:
                        # Try to extract memory size
                        try:
                            if 'GB' in line:
                                memory_str = line.split('GB')[0].split()[-1]
                                current_memory = int(float(memory_str) * 1024)
                            elif 'MB' in line:
                                memory_str = line.split('MB')[0].split()[-1]
                                current_memory = int(memory_str)
                        except (ValueError, IndexError):
                            pass
                
                if current_gpu:
                    gpus.append(GPUInfo(
                        backend=GPUBackend.METAL,
                        name=current_gpu,
                        memory_mb=current_memory
                    ))
        except Exception as e:
            logger.warning(f"Error getting Metal GPU info: {e}")
        
        # Fallback: Add generic Metal entry if detected but can't get details
        if not gpus and self._detect_metal():
            gpus.append(GPUInfo(
                backend=GPUBackend.METAL,
                name="Apple GPU (Metal)",
                memory_mb=8192
            ))
        
        return gpus
    
    def _get_opencl_gpus(self) -> List[GPUInfo]:
        """Get OpenCL GPU information."""
        gpus = []
        
        try:
            import pyopencl as cl
            for platform in cl.get_platforms():
                for device in platform.get_devices():
                    if device.type == cl.device_type.GPU:
                        gpus.append(GPUInfo(
                            backend=GPUBackend.OPENCL,
                            name=device.name.strip(),
                            memory_mb=device.global_mem_size // (1024 * 1024)
                        ))
        except Exception as e:
            logger.warning(f"Error getting OpenCL GPU info: {e}")
        
        return gpus
    
    def _get_system_memory_mb(self) -> int:
        """Get system memory in MB."""
        try:
            import psutil
            return psutil.virtual_memory().total // (1024 * 1024)
        except ImportError:
            return 8192  # Default fallback
