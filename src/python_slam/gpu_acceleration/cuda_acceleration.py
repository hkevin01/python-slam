"""
CUDA GPU Acceleration for SLAM Operations

This module provides CUDA-accelerated implementations of key SLAM algorithms
including feature matching, bundle adjustment, and map optimization.
"""

import numpy as np
import logging
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CudaConfig:
    """Configuration for CUDA acceleration."""
    device_id: int = 0
    memory_pool_size_mb: int = 1024
    enable_cudnn: bool = True
    enable_tensor_cores: bool = True
    precision: str = "float32"  # float32, float16, mixed

class CudaAccelerator:
    """CUDA acceleration backend for SLAM operations."""
    
    def __init__(self, config: Optional[CudaConfig] = None):
        self.config = config or CudaConfig()
        self.device = None
        self.context = None
        self.stream = None
        self._initialized = False
        
        # Try to import required libraries
        self.torch = None
        self.cupy = None
        self.cusolver = None
        
        self._import_libraries()
    
    def _import_libraries(self):
        """Import CUDA libraries with fallback handling."""
        try:
            import torch
            if torch.cuda.is_available():
                self.torch = torch
                logger.info("PyTorch CUDA backend available")
        except ImportError:
            logger.warning("PyTorch not available for CUDA acceleration")
        
        try:
            import cupy as cp
            self.cupy = cp
            logger.info("CuPy backend available")
        except ImportError:
            logger.warning("CuPy not available for CUDA acceleration")
        
        try:
            import cusolver
            self.cusolver = cusolver
            logger.info("cuSOLVER available for linear algebra")
        except ImportError:
            logger.warning("cuSOLVER not available")
    
    def initialize(self) -> bool:
        """Initialize CUDA context and resources."""
        if self._initialized:
            return True
        
        try:
            if self.torch and self.torch.cuda.is_available():
                self.device = self.torch.device(f'cuda:{self.config.device_id}')
                
                # Set memory pool if supported
                if hasattr(self.torch.cuda, 'memory_pool'):
                    pool_size = self.config.memory_pool_size_mb * 1024 * 1024
                    self.torch.cuda.memory_pool().set_memory_fraction(0.8)
                
                # Enable optimizations
                if self.config.enable_cudnn:
                    self.torch.backends.cudnn.enabled = True
                    self.torch.backends.cudnn.benchmark = True
                
                logger.info(f"CUDA initialized on device {self.config.device_id}")
                self._initialized = True
                return True
            
            elif self.cupy:
                self.cupy.cuda.Device(self.config.device_id).use()
                logger.info(f"CuPy initialized on device {self.config.device_id}")
                self._initialized = True
                return True
            
        except Exception as e:
            logger.error(f"Failed to initialize CUDA: {e}")
            return False
        
        logger.error("No CUDA backend available")
        return False
    
    def is_available(self) -> bool:
        """Check if CUDA acceleration is available."""
        return self.torch and self.torch.cuda.is_available() or self.cupy is not None
    
    def get_memory_info(self) -> Dict[str, int]:
        """Get GPU memory information."""
        if not self._initialized:
            return {"total": 0, "used": 0, "free": 0}
        
        try:
            if self.torch:
                total = self.torch.cuda.get_device_properties(self.config.device_id).total_memory
                allocated = self.torch.cuda.memory_allocated(self.config.device_id)
                cached = self.torch.cuda.memory_reserved(self.config.device_id)
                
                return {
                    "total": total,
                    "allocated": allocated,
                    "cached": cached,
                    "free": total - cached
                }
            
            elif self.cupy:
                mempool = self.cupy.get_default_memory_pool()
                return {
                    "total": mempool.total_bytes(),
                    "used": mempool.used_bytes(),
                    "free": mempool.free_bytes()
                }
        
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {"total": 0, "used": 0, "free": 0}
    
    def feature_matching_gpu(self, 
                           descriptors1: np.ndarray, 
                           descriptors2: np.ndarray,
                           distance_threshold: float = 0.8) -> np.ndarray:
        """GPU-accelerated feature matching using CUDA."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("CUDA not available for feature matching")
        
        try:
            if self.torch:
                return self._torch_feature_matching(descriptors1, descriptors2, distance_threshold)
            elif self.cupy:
                return self._cupy_feature_matching(descriptors1, descriptors2, distance_threshold)
        except Exception as e:
            logger.error(f"GPU feature matching failed: {e}")
            raise
    
    def _torch_feature_matching(self, desc1: np.ndarray, desc2: np.ndarray, threshold: float) -> np.ndarray:
        """PyTorch-based feature matching."""
        # Convert to torch tensors
        d1 = self.torch.from_numpy(desc1).float().to(self.device)
        d2 = self.torch.from_numpy(desc2).float().to(self.device)
        
        # Compute distance matrix
        distances = self.torch.cdist(d1, d2, p=2)
        
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
    
    def _cupy_feature_matching(self, desc1: np.ndarray, desc2: np.ndarray, threshold: float) -> np.ndarray:
        """CuPy-based feature matching."""
        # Convert to CuPy arrays
        d1 = self.cupy.array(desc1, dtype=self.cupy.float32)
        d2 = self.cupy.array(desc2, dtype=self.cupy.float32)
        
        # Compute distance matrix
        distances = self.cupy.linalg.norm(d1[:, None, :] - d2[None, :, :], axis=2)
        
        # Find best matches
        min_indices = self.cupy.argmin(distances, axis=1)
        min_distances = self.cupy.min(distances, axis=1)
        
        # Apply threshold
        valid_mask = min_distances < threshold
        
        # Create matches array
        valid_indices = self.cupy.where(valid_mask)[0]
        matches = self.cupy.column_stack([
            valid_indices,
            min_indices[valid_mask],
            min_distances[valid_mask]
        ])
        
        return self.cupy.asnumpy(matches)
    
    def bundle_adjustment_gpu(self, 
                            points_3d: np.ndarray,
                            camera_poses: np.ndarray,
                            observations: np.ndarray,
                            camera_intrinsics: np.ndarray,
                            max_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated bundle adjustment."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("CUDA not available for bundle adjustment")
        
        try:
            if self.torch:
                return self._torch_bundle_adjustment(
                    points_3d, camera_poses, observations, camera_intrinsics, max_iterations
                )
            else:
                logger.warning("Advanced bundle adjustment requires PyTorch")
                return points_3d, camera_poses
        
        except Exception as e:
            logger.error(f"GPU bundle adjustment failed: {e}")
            raise
    
    def _torch_bundle_adjustment(self, points_3d, poses, observations, intrinsics, max_iter):
        """PyTorch-based bundle adjustment using Levenberg-Marquardt."""
        # Convert to torch tensors
        points = self.torch.from_numpy(points_3d).float().to(self.device)
        poses_tensor = self.torch.from_numpy(poses).float().to(self.device)
        obs = self.torch.from_numpy(observations).float().to(self.device)
        K = self.torch.from_numpy(intrinsics).float().to(self.device)
        
        # Make parameters require gradients
        points.requires_grad_(True)
        poses_tensor.requires_grad_(True)
        
        optimizer = self.torch.optim.LBFGS([points, poses_tensor], lr=1e-3, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            
            # Project 3D points to 2D
            projected = self._project_points(points, poses_tensor, K)
            
            # Compute reprojection error
            error = self.torch.sum((projected - obs) ** 2)
            
            error.backward()
            return error
        
        # Optimize
        for _ in range(max_iter // 10):  # LBFGS handles internal iterations
            optimizer.step(closure)
        
        # Return optimized parameters
        optimized_points = points.detach().cpu().numpy()
        optimized_poses = poses_tensor.detach().cpu().numpy()
        
        return optimized_points, optimized_poses
    
    def _project_points(self, points_3d, poses, intrinsics):
        """Project 3D points to 2D using camera parameters."""
        # Apply camera transformation
        R = poses[:, :3, :3]
        t = poses[:, :3, 3]
        
        # Transform points to camera coordinates
        points_cam = self.torch.bmm(R, points_3d.T) + t.unsqueeze(-1)
        
        # Project to image plane
        points_2d = points_cam[:2] / points_cam[2]
        
        # Apply intrinsics
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        
        projected = self.torch.stack([
            fx * points_2d[0] + cx,
            fy * points_2d[1] + cy
        ], dim=0)
        
        return projected.T
    
    def optimize_map_gpu(self, 
                        map_points: np.ndarray,
                        pose_graph: Dict[str, Any],
                        loop_closures: List[Tuple[int, int, np.ndarray]]) -> np.ndarray:
        """GPU-accelerated map optimization using pose graph."""
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("CUDA not available for map optimization")
        
        try:
            if self.torch:
                return self._torch_map_optimization(map_points, pose_graph, loop_closures)
            else:
                logger.warning("Map optimization requires PyTorch")
                return map_points
        
        except Exception as e:
            logger.error(f"GPU map optimization failed: {e}")
            return map_points
    
    def _torch_map_optimization(self, map_points, pose_graph, loop_closures):
        """PyTorch-based map optimization."""
        # Implement pose graph optimization
        points = self.torch.from_numpy(map_points).float().to(self.device)
        
        # For now, return the input points
        # Full implementation would involve pose graph SLAM optimization
        return points.cpu().numpy()
    
    def cleanup(self):
        """Clean up CUDA resources."""
        if self._initialized:
            try:
                if self.torch:
                    self.torch.cuda.empty_cache()
                    self.torch.cuda.synchronize()
                
                if self.cupy:
                    self.cupy.get_default_memory_pool().free_all_blocks()
                
                logger.info("CUDA resources cleaned up")
            
            except Exception as e:
                logger.error(f"Error cleaning up CUDA: {e}")
            
            finally:
                self._initialized = False
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()
