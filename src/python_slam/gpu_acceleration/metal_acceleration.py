"""
Metal GPU Acceleration for SLAM Operations

This module provides Metal Performance Shaders (MPS) acceleration
for Apple Silicon GPUs in SLAM computations.
"""

import numpy as np
import logging
import platform
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MetalConfig:
    """Configuration for Metal acceleration."""
    enable_mixed_precision: bool = True
    memory_pool_size_mb: int = 1024
    precision: str = "float32"  # float32, float16

class MetalAccelerator:
    """Metal Performance Shaders acceleration backend for Apple Silicon."""

    def __init__(self, config: Optional[MetalConfig] = None):
        self.config = config or MetalConfig()
        self.device = None
        self._initialized = False

        # Check if running on macOS
        self.is_macos = platform.system() == "Darwin"

        # Try to import Metal/MPS libraries
        self.torch = None
        self.mps_available = False

        self._import_libraries()

    def _import_libraries(self):
        """Import Metal/MPS libraries with fallback handling."""
        if not self.is_macos:
            logger.info("Metal acceleration only available on macOS")
            return

        try:
            import torch
            # Check if PyTorch was compiled with MPS support
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.torch = torch
                self.mps_available = True
                logger.info("PyTorch with Metal Performance Shaders (MPS) available")
            else:
                logger.warning("PyTorch available but MPS not supported")
        except ImportError:
            logger.warning("PyTorch not available for Metal acceleration")

    def initialize(self) -> bool:
        """Initialize Metal context and resources."""
        if self._initialized:
            return True

        if not self.is_macos:
            logger.error("Metal acceleration only available on macOS")
            return False

        try:
            if self.torch and self.mps_available:
                self.device = self.torch.device("mps")

                # Enable optimizations if available
                if hasattr(self.torch.backends.mps, 'enabled'):
                    self.torch.backends.mps.enabled = True

                logger.info("Metal Performance Shaders initialized")
                self._initialized = True
                return True

        except Exception as e:
            logger.error(f"Failed to initialize Metal: {e}")
            return False

        logger.error("No Metal backend available")
        return False

    def is_available(self) -> bool:
        """Check if Metal acceleration is available."""
        return self.is_macos and self.mps_available and self.torch

    def get_device_info(self) -> Dict[str, Any]:
        """Get Metal device information."""
        if not self._initialized or not self.is_macos:
            return {}

        try:
            # Get basic system info
            import subprocess
            result = subprocess.run(['system_profiler', 'SPHardwareDataType'],
                                 capture_output=True, text=True, timeout=10)

            info = {"backend": "Metal Performance Shaders"}

            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    line = line.strip()
                    if 'Chip:' in line:
                        info['chip'] = line.split('Chip:')[1].strip()
                    elif 'Total Number of Cores:' in line:
                        info['cores'] = line.split(':')[1].strip()
                    elif 'Memory:' in line:
                        info['memory'] = line.split(':')[1].strip()

            return info

        except Exception as e:
            logger.error(f"Error getting Metal device info: {e}")
            return {"backend": "Metal Performance Shaders"}

    def get_memory_info(self) -> Dict[str, int]:
        """Get GPU memory information for Metal."""
        # Metal shares system memory, so get system memory info
        try:
            import psutil
            mem = psutil.virtual_memory()

            # Estimate available memory for Metal (usually ~75% of system memory)
            metal_memory = int(mem.total * 0.75)
            metal_used = int(mem.used * 0.3)  # Rough estimate

            return {
                "total": metal_memory,
                "used": metal_used,
                "free": metal_memory - metal_used
            }
        except ImportError:
            return {"total": 16 * 1024 * 1024 * 1024, "used": 0, "free": 16 * 1024 * 1024 * 1024}  # 16GB default

    def feature_matching_gpu(self,
                           descriptors1: np.ndarray,
                           descriptors2: np.ndarray,
                           distance_threshold: float = 0.8) -> np.ndarray:
        """GPU-accelerated feature matching using Metal."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Metal not available for feature matching")

        try:
            return self._torch_feature_matching(descriptors1, descriptors2, distance_threshold)
        except Exception as e:
            logger.error(f"Metal feature matching failed: {e}")
            raise

    def _torch_feature_matching(self, desc1: np.ndarray, desc2: np.ndarray, threshold: float) -> np.ndarray:
        """PyTorch MPS-based feature matching."""
        # Convert to torch tensors
        dtype = self.torch.float16 if self.config.enable_mixed_precision else self.torch.float32
        d1 = self.torch.from_numpy(desc1).to(dtype).to(self.device)
        d2 = self.torch.from_numpy(desc2).to(dtype).to(self.device)

        # Compute distance matrix using Metal-optimized operations
        # MPS doesn't support autocast yet, so handle precision manually
        if self.config.enable_mixed_precision and dtype == self.torch.float16:
            # Use float16 for computation
            d1_norm = self.torch.nn.functional.normalize(d1, p=2, dim=1)
            d2_norm = self.torch.nn.functional.normalize(d2, p=2, dim=1)

            # Cosine similarity (converted to distance)
            similarity = self.torch.mm(d1_norm, d2_norm.t())
            distances = 1.0 - similarity
        else:
            # Use float32 for higher precision
            d1_norm = self.torch.nn.functional.normalize(d1, p=2, dim=1)
            d2_norm = self.torch.nn.functional.normalize(d2, p=2, dim=1)

            # Cosine similarity (converted to distance)
            similarity = self.torch.mm(d1_norm, d2_norm.t())
            distances = 1.0 - similarity

        # Find best matches
        min_distances, min_indices = self.torch.min(distances, dim=1)

        # Apply distance threshold
        valid_matches = min_distances < threshold

        # Create match pairs
        matches = []
        for i, (valid, idx, dist) in enumerate(zip(valid_matches, min_indices, min_distances)):
            if valid:
                matches.append([i, idx.item(), dist.item()])

        return np.array(matches)

    def matrix_operations_gpu(self,
                            matrix_a: np.ndarray,
                            matrix_b: np.ndarray,
                            operation: str = "multiply") -> np.ndarray:
        """GPU-accelerated matrix operations using Metal."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Metal not available for matrix operations")

        try:
            dtype = self.torch.float16 if self.config.enable_mixed_precision else self.torch.float32
            a = self.torch.from_numpy(matrix_a).to(dtype).to(self.device)
            b = self.torch.from_numpy(matrix_b).to(dtype).to(self.device)

            if operation == "multiply":
                result = self.torch.mm(a, b)
            elif operation == "add":
                result = self.torch.add(a, b)
            elif operation == "subtract":
                result = self.torch.sub(a, b)
            elif operation == "solve":
                # Linear system solver
                result = self.torch.linalg.solve(a, b)
            elif operation == "svd":
                # Singular Value Decomposition
                u, s, v = self.torch.linalg.svd(a)
                result = self.torch.mm(self.torch.mm(u, self.torch.diag(s)), v)
            elif operation == "eigendecomp":
                # Eigendecomposition
                eigenvals, eigenvecs = self.torch.linalg.eig(a)
                result = self.torch.mm(eigenvecs, self.torch.mm(self.torch.diag(eigenvals), eigenvecs.t()))
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            return result.cpu().numpy()

        except Exception as e:
            logger.error(f"Metal matrix operation failed: {e}")
            raise

    def convolution_gpu(self,
                       input_tensor: np.ndarray,
                       kernel: np.ndarray,
                       stride: int = 1,
                       padding: int = 0) -> np.ndarray:
        """GPU-accelerated convolution using Metal Performance Shaders."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Metal not available for convolution")

        try:
            # Convert to torch tensors with proper dimensions
            if len(input_tensor.shape) == 2:
                # Add batch and channel dimensions
                input_tensor = input_tensor[None, None, :, :]
            elif len(input_tensor.shape) == 3:
                # Add batch dimension
                input_tensor = input_tensor[None, :, :, :]

            if len(kernel.shape) == 2:
                # Add input and output channel dimensions
                kernel = kernel[None, None, :, :]
            elif len(kernel.shape) == 3:
                # Add output channel dimension
                kernel = kernel[None, :, :, :]

            dtype = self.torch.float16 if self.config.enable_mixed_precision else self.torch.float32
            input_torch = self.torch.from_numpy(input_tensor).to(dtype).to(self.device)
            kernel_torch = self.torch.from_numpy(kernel).to(dtype).to(self.device)

            # Perform convolution
            result = self.torch.nn.functional.conv2d(
                input_torch, kernel_torch, stride=stride, padding=padding
            )

            return result.squeeze().cpu().numpy()

        except Exception as e:
            logger.error(f"Metal convolution failed: {e}")
            raise

    def fft_gpu(self, signal: np.ndarray, inverse: bool = False) -> np.ndarray:
        """GPU-accelerated FFT using Metal."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Metal not available for FFT")

        try:
            dtype = self.torch.complex64 if self.config.enable_mixed_precision else self.torch.complex128
            signal_torch = self.torch.from_numpy(signal).to(dtype).to(self.device)

            if inverse:
                result = self.torch.fft.ifft(signal_torch)
            else:
                result = self.torch.fft.fft(signal_torch)

            return result.cpu().numpy()

        except Exception as e:
            logger.error(f"Metal FFT failed: {e}")
            raise

    def optimization_gpu(self,
                        objective_function,
                        initial_params: np.ndarray,
                        max_iterations: int = 1000,
                        learning_rate: float = 0.01) -> np.ndarray:
        """GPU-accelerated optimization using Metal."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Metal not available for optimization")

        try:
            # Convert parameters to torch tensor
            params = self.torch.from_numpy(initial_params).float().to(self.device)
            params.requires_grad_(True)

            # Use Adam optimizer (Metal optimized)
            optimizer = self.torch.optim.Adam([params], lr=learning_rate)

            for iteration in range(max_iterations):
                optimizer.zero_grad()

                # Compute objective function
                loss = objective_function(params)

                # Backward pass
                loss.backward()

                # Update parameters
                optimizer.step()

                # Check convergence (every 100 iterations)
                if iteration % 100 == 0:
                    logger.debug(f"Iteration {iteration}, Loss: {loss.item()}")

                    # Simple convergence check
                    if loss.item() < 1e-6:
                        logger.info(f"Converged at iteration {iteration}")
                        break

            return params.detach().cpu().numpy()

        except Exception as e:
            logger.error(f"Metal optimization failed: {e}")
            raise

    def memory_benchmark(self) -> Dict[str, float]:
        """Benchmark Metal memory bandwidth and compute performance."""
        if not self._initialized:
            if not self.initialize():
                return {}

        try:
            import time

            # Memory bandwidth test
            size = 1024 * 1024 * 50  # 50M elements (smaller for Metal shared memory)
            data = self.torch.randn(size, dtype=self.torch.float32, device=self.device)

            # Time memory operations
            start_time = time.time()

            for _ in range(10):
                result = data * 2.0
                # Metal operations are automatically synchronized

            end_time = time.time()
            memory_bandwidth = (size * 4 * 10) / (end_time - start_time) / 1e9  # GB/s

            # Compute performance test (smaller matrices for Metal)
            matrix_size = 1024
            a = self.torch.randn(matrix_size, matrix_size, dtype=self.torch.float32, device=self.device)
            b = self.torch.randn(matrix_size, matrix_size, dtype=self.torch.float32, device=self.device)

            start_time = time.time()

            for _ in range(10):
                c = self.torch.mm(a, b)
                # Metal operations are automatically synchronized

            end_time = time.time()
            flops = (2 * matrix_size**3 * 10) / (end_time - start_time) / 1e12  # TFLOPS

            return {
                "memory_bandwidth_gb_s": memory_bandwidth,
                "compute_performance_tflops": flops,
                "test_matrix_size": matrix_size
            }

        except Exception as e:
            logger.error(f"Metal benchmark failed: {e}")
            return {}

    def image_processing_gpu(self,
                           image: np.ndarray,
                           operation: str = "blur",
                           **kwargs) -> np.ndarray:
        """GPU-accelerated image processing using Metal."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("Metal not available for image processing")

        try:
            # Ensure image has proper dimensions
            if len(image.shape) == 2:
                image = image[None, None, :, :]  # Add batch and channel dims
            elif len(image.shape) == 3:
                image = image[None, :, :, :]  # Add batch dim

            dtype = self.torch.float32
            img_torch = self.torch.from_numpy(image).to(dtype).to(self.device)

            if operation == "blur":
                kernel_size = kwargs.get("kernel_size", 5)
                sigma = kwargs.get("sigma", 1.0)

                # Create Gaussian kernel
                kernel = self._create_gaussian_kernel(kernel_size, sigma)
                result = self.torch.nn.functional.conv2d(
                    img_torch, kernel, padding=kernel_size//2
                )

            elif operation == "sharpen":
                # Sharpening kernel
                kernel = self.torch.tensor([[[
                    [0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]
                ]]], dtype=dtype, device=self.device)

                result = self.torch.nn.functional.conv2d(
                    img_torch, kernel, padding=1
                )

            elif operation == "edge_detect":
                # Sobel edge detection
                sobel_x = self.torch.tensor([[[
                    [-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]
                ]]], dtype=dtype, device=self.device)

                sobel_y = self.torch.tensor([[[
                    [-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]
                ]]], dtype=dtype, device=self.device)

                grad_x = self.torch.nn.functional.conv2d(img_torch, sobel_x, padding=1)
                grad_y = self.torch.nn.functional.conv2d(img_torch, sobel_y, padding=1)

                result = self.torch.sqrt(grad_x**2 + grad_y**2)

            else:
                raise ValueError(f"Unsupported image operation: {operation}")

            return result.squeeze().cpu().numpy()

        except Exception as e:
            logger.error(f"Metal image processing failed: {e}")
            raise

    def _create_gaussian_kernel(self, kernel_size: int, sigma: float):
        """Create a Gaussian kernel for image processing."""
        coords = self.torch.arange(kernel_size, dtype=self.torch.float32, device=self.device)
        coords -= kernel_size // 2

        g = self.torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()

        # Create 2D kernel
        kernel_2d = g[:, None] * g[None, :]
        kernel_2d = kernel_2d[None, None, :, :]  # Add batch and channel dims

        return kernel_2d

    def cleanup(self):
        """Clean up Metal resources."""
        if self._initialized:
            try:
                # Metal automatically manages memory, but we can clear cache
                if self.torch and hasattr(self.torch.mps, 'empty_cache'):
                    self.torch.mps.empty_cache()

                logger.info("Metal resources cleaned up")

            except Exception as e:
                logger.error(f"Error cleaning up Metal: {e}")

            finally:
                self._initialized = False

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
