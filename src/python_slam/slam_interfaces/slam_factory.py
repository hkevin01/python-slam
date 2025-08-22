#!/usr/bin/env python3
"""
SLAM Algorithm Factory for Multi-Algorithm Framework

This module provides a factory pattern for creating and managing different
SLAM algorithm instances with runtime switching capabilities, configuration
management, and automatic sensor type detection.
"""

import os
import yaml
import importlib
from typing import Dict, Type, Optional, List, Any
from enum import Enum
import logging

from .slam_interface import (
    SLAMInterface, SLAMConfiguration, SensorType, SLAMAlgorithmInfo
)


class SLAMAlgorithmType(Enum):
    """Available SLAM algorithm types."""
    ORB_SLAM3 = "orb_slam3"
    RTAB_MAP = "rtab_map"
    CARTOGRAPHER = "cartographer"
    OPENVSLAM = "openvslam"
    PYTHON_SLAM = "python_slam"
    VINS_FUSION = "vins_fusion"
    DSO = "dso"
    STEREO_DSO = "stereo_dso"


class SLAMFactory:
    """
    Factory for creating and managing SLAM algorithm instances.

    Provides runtime algorithm switching, configuration management,
    and automatic sensor type detection capabilities.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize SLAM factory.

        Args:
            config_dir: Directory containing algorithm configuration files
        """
        self.config_dir = config_dir or os.path.expanduser("~/.config/python_slam")
        self.logger = logging.getLogger(__name__)

        # Registry of available algorithms
        self._algorithm_registry: Dict[str, Type[SLAMInterface]] = {}
        self._algorithm_info: Dict[str, SLAMAlgorithmInfo] = {}

        # Current active algorithm
        self._active_algorithm: Optional[SLAMInterface] = None
        self._active_config: Optional[SLAMConfiguration] = None

        # Configuration cache
        self._config_cache: Dict[str, SLAMConfiguration] = {}

        # Initialize factory
        self._initialize_factory()

    def _initialize_factory(self):
        """Initialize the factory and register available algorithms."""
        try:
            self._register_builtin_algorithms()
            self._load_configurations()
            self.logger.info("SLAM Factory initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize SLAM Factory: {e}")
            raise

    def _register_builtin_algorithms(self):
        """Register built-in SLAM algorithms."""
        # Register algorithm information
        self._algorithm_info = {
            SLAMAlgorithmType.ORB_SLAM3.value: SLAMAlgorithmInfo(
                name="ORB-SLAM3",
                supported_sensors=[SensorType.MONOCULAR, SensorType.STEREO,
                                 SensorType.RGB_D, SensorType.VISUAL_INERTIAL],
                description="Feature-based visual SLAM with loop closure and relocalization",
                performance_rating=5
            ),
            SLAMAlgorithmType.RTAB_MAP.value: SLAMAlgorithmInfo(
                name="RTAB-Map",
                supported_sensors=[SensorType.STEREO, SensorType.RGB_D],
                description="RGB-D SLAM with appearance-based loop detection",
                performance_rating=4
            ),
            SLAMAlgorithmType.CARTOGRAPHER.value: SLAMAlgorithmInfo(
                name="Cartographer",
                supported_sensors=[SensorType.LIDAR_2D, SensorType.LIDAR_3D],
                description="Real-time 2D/3D SLAM for LiDAR sensors",
                performance_rating=5
            ),
            SLAMAlgorithmType.OPENVSLAM.value: SLAMAlgorithmInfo(
                name="OpenVSLAM",
                supported_sensors=[SensorType.MONOCULAR, SensorType.STEREO, SensorType.RGB_D],
                description="Versatile visual SLAM framework",
                performance_rating=4
            ),
            SLAMAlgorithmType.PYTHON_SLAM.value: SLAMAlgorithmInfo(
                name="Python SLAM",
                supported_sensors=[SensorType.MONOCULAR, SensorType.STEREO, SensorType.VISUAL_INERTIAL],
                description="Custom Python SLAM implementation",
                performance_rating=3
            ),
            SLAMAlgorithmType.VINS_FUSION.value: SLAMAlgorithmInfo(
                name="VINS-Fusion",
                supported_sensors=[SensorType.VISUAL_INERTIAL],
                description="Visual-inertial SLAM with tight IMU coupling",
                performance_rating=4
            ),
            SLAMAlgorithmType.DSO.value: SLAMAlgorithmInfo(
                name="DSO",
                supported_sensors=[SensorType.MONOCULAR],
                description="Direct sparse odometry for monocular cameras",
                performance_rating=4
            ),
            SLAMAlgorithmType.STEREO_DSO.value: SLAMAlgorithmInfo(
                name="Stereo DSO",
                supported_sensors=[SensorType.STEREO],
                description="Direct sparse odometry for stereo cameras",
                performance_rating=4
            )
        }

        # Register algorithm classes (lazy loading)
        self._register_algorithm_classes()

    def _register_algorithm_classes(self):
        """Register algorithm implementation classes."""
        try:
            # ORB-SLAM3
            from .algorithms.orb_slam3_wrapper import ORBSLAM3Wrapper
            self._algorithm_registry[SLAMAlgorithmType.ORB_SLAM3.value] = ORBSLAM3Wrapper

        except ImportError as e:
            self.logger.warning(f"ORB-SLAM3 not available: {e}")

        try:
            # RTAB-Map
            from .algorithms.rtab_map_wrapper import RTABMapWrapper
            self._algorithm_registry[SLAMAlgorithmType.RTAB_MAP.value] = RTABMapWrapper

        except ImportError as e:
            self.logger.warning(f"RTAB-Map not available: {e}")

        try:
            # Cartographer
            from .algorithms.cartographer_wrapper import CartographerWrapper
            self._algorithm_registry[SLAMAlgorithmType.CARTOGRAPHER.value] = CartographerWrapper

        except ImportError as e:
            self.logger.warning(f"Cartographer not available: {e}")

        try:
            # OpenVSLAM
            from .algorithms.openvslam_wrapper import OpenVSLAMWrapper
            self._algorithm_registry[SLAMAlgorithmType.OPENVSLAM.value] = OpenVSLAMWrapper

        except ImportError as e:
            self.logger.warning(f"OpenVSLAM not available: {e}")

        try:
            # Python SLAM (always available)
            from .algorithms.python_slam_wrapper import PythonSLAMWrapper
            self._algorithm_registry[SLAMAlgorithmType.PYTHON_SLAM.value] = PythonSLAMWrapper

        except ImportError as e:
            self.logger.error(f"Python SLAM not available: {e}")

        try:
            # VINS-Fusion
            from .algorithms.vins_fusion_wrapper import VINSFusionWrapper
            self._algorithm_registry[SLAMAlgorithmType.VINS_FUSION.value] = VINSFusionWrapper

        except ImportError as e:
            self.logger.warning(f"VINS-Fusion not available: {e}")

        try:
            # DSO
            from .algorithms.dso_wrapper import DSOWrapper
            self._algorithm_registry[SLAMAlgorithmType.DSO.value] = DSOWrapper

        except ImportError as e:
            self.logger.warning(f"DSO not available: {e}")

        try:
            # Stereo DSO
            from .algorithms.stereo_dso_wrapper import StereoDSOWrapper
            self._algorithm_registry[SLAMAlgorithmType.STEREO_DSO.value] = StereoDSOWrapper

        except ImportError as e:
            self.logger.warning(f"Stereo DSO not available: {e}")

    def _load_configurations(self):
        """Load algorithm configurations from files."""
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir, exist_ok=True)
            self._create_default_configs()

        for algorithm_type in SLAMAlgorithmType:
            config_file = os.path.join(self.config_dir, f"{algorithm_type.value}.yaml")
            if os.path.exists(config_file):
                try:
                    config = self._load_config_from_file(config_file, algorithm_type.value)
                    self._config_cache[algorithm_type.value] = config
                except Exception as e:
                    self.logger.warning(f"Failed to load config for {algorithm_type.value}: {e}")

    def _create_default_configs(self):
        """Create default configuration files."""
        default_configs = {
            SLAMAlgorithmType.ORB_SLAM3.value: {
                'algorithm_name': 'ORB-SLAM3',
                'sensor_type': 'monocular',
                'max_features': 1000,
                'quality_level': 0.01,
                'min_distance': 10.0,
                'enable_loop_closure': True,
                'enable_mapping': True,
                'enable_relocalization': True,
                'vocabulary_file': 'ORBvoc.txt',
                'custom_params': {
                    'ORBextractor.nFeatures': 1000,
                    'ORBextractor.scaleFactor': 1.2,
                    'ORBextractor.nLevels': 8
                }
            },
            SLAMAlgorithmType.RTAB_MAP.value: {
                'algorithm_name': 'RTAB-Map',
                'sensor_type': 'rgb_d',
                'enable_loop_closure': True,
                'enable_mapping': True,
                'map_resolution': 0.05,
                'custom_params': {
                    'RGBD.Enabled': True,
                    'RGBD.LinearUpdate': 0.1,
                    'RGBD.AngularUpdate': 0.1
                }
            },
            SLAMAlgorithmType.PYTHON_SLAM.value: {
                'algorithm_name': 'Python SLAM',
                'sensor_type': 'monocular',
                'max_features': 1000,
                'quality_level': 0.01,
                'min_distance': 10.0,
                'enable_loop_closure': True,
                'enable_mapping': True,
                'keyframe_distance': 1.0
            }
        }

        for algorithm_name, config_data in default_configs.items():
            config_file = os.path.join(self.config_dir, f"{algorithm_name}.yaml")
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)

    def _load_config_from_file(self, config_file: str, algorithm_name: str) -> SLAMConfiguration:
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        sensor_type = SensorType(config_data.get('sensor_type', 'monocular'))

        return SLAMConfiguration(
            algorithm_name=algorithm_name,
            sensor_type=sensor_type,
            max_features=config_data.get('max_features', 1000),
            quality_level=config_data.get('quality_level', 0.01),
            min_distance=config_data.get('min_distance', 10.0),
            enable_loop_closure=config_data.get('enable_loop_closure', True),
            enable_mapping=config_data.get('enable_mapping', True),
            enable_relocalization=config_data.get('enable_relocalization', True),
            map_resolution=config_data.get('map_resolution', 0.05),
            keyframe_distance=config_data.get('keyframe_distance', 1.0),
            config_file=config_data.get('config_file'),
            vocabulary_file=config_data.get('vocabulary_file'),
            custom_params=config_data.get('custom_params', {})
        )

    def create_algorithm(self, algorithm_type: str,
                        config: Optional[SLAMConfiguration] = None,
                        sensor_type: Optional[SensorType] = None) -> SLAMInterface:
        """
        Create SLAM algorithm instance.

        Args:
            algorithm_type: Type of SLAM algorithm to create
            config: Optional custom configuration
            sensor_type: Optional sensor type for auto-configuration

        Returns:
            SLAMInterface: Created SLAM algorithm instance

        Raises:
            ValueError: If algorithm type not available
            RuntimeError: If algorithm creation fails
        """
        if algorithm_type not in self._algorithm_registry:
            available_algorithms = list(self._algorithm_registry.keys())
            raise ValueError(f"Algorithm '{algorithm_type}' not available. "
                           f"Available: {available_algorithms}")

        # Use provided config or get from cache
        if config is None:
            config = self._get_or_create_config(algorithm_type, sensor_type)

        try:
            algorithm_class = self._algorithm_registry[algorithm_type]
            algorithm = algorithm_class(config)
            algorithm.set_logger(self.logger)

            self.logger.info(f"Created {algorithm_type} algorithm instance")
            return algorithm

        except Exception as e:
            self.logger.error(f"Failed to create {algorithm_type} algorithm: {e}")
            raise RuntimeError(f"Algorithm creation failed: {e}")

    def switch_algorithm(self, algorithm_type: str,
                        config: Optional[SLAMConfiguration] = None,
                        transfer_state: bool = True) -> SLAMInterface:
        """
        Switch to a different SLAM algorithm.

        Args:
            algorithm_type: New algorithm type
            config: Optional custom configuration
            transfer_state: Whether to transfer state from current algorithm

        Returns:
            SLAMInterface: New algorithm instance
        """
        # Stop current algorithm if running
        if self._active_algorithm is not None:
            try:
                current_pose = self._active_algorithm.get_pose()
                current_map = self._active_algorithm.get_map()
                self.logger.info(f"Stopping {self._active_config.algorithm_name}")
            except Exception as e:
                self.logger.warning(f"Error getting state from current algorithm: {e}")
                current_pose = None
                current_map = None

        # Create new algorithm
        new_algorithm = self.create_algorithm(algorithm_type, config)

        # Initialize new algorithm
        if new_algorithm.initialize():
            self._active_algorithm = new_algorithm
            self._active_config = new_algorithm.get_config()

            # Transfer state if requested and possible
            if transfer_state and current_pose is not None:
                try:
                    new_algorithm.relocalize(current_pose)
                    self.logger.info("State transferred to new algorithm")
                except Exception as e:
                    self.logger.warning(f"Failed to transfer state: {e}")

            self.logger.info(f"Switched to {algorithm_type}")
            return new_algorithm
        else:
            raise RuntimeError(f"Failed to initialize {algorithm_type}")

    def get_available_algorithms(self) -> List[str]:
        """Get list of available algorithm types."""
        return list(self._algorithm_registry.keys())

    def get_algorithm_info(self, algorithm_type: str) -> Optional[SLAMAlgorithmInfo]:
        """Get information about a specific algorithm."""
        return self._algorithm_info.get(algorithm_type)

    def get_algorithms_for_sensor(self, sensor_type: SensorType) -> List[str]:
        """Get algorithms that support a specific sensor type."""
        compatible_algorithms = []
        for algo_type, info in self._algorithm_info.items():
            if info.supports_sensor(sensor_type):
                compatible_algorithms.append(algo_type)
        return compatible_algorithms

    def auto_select_algorithm(self, sensor_type: SensorType,
                            performance_priority: bool = True) -> str:
        """
        Automatically select best algorithm for sensor type.

        Args:
            sensor_type: Available sensor type
            performance_priority: Whether to prioritize performance over availability

        Returns:
            str: Selected algorithm type
        """
        compatible_algorithms = self.get_algorithms_for_sensor(sensor_type)

        if not compatible_algorithms:
            raise ValueError(f"No algorithms available for sensor type: {sensor_type}")

        # Filter by availability
        available_algorithms = [algo for algo in compatible_algorithms
                              if algo in self._algorithm_registry]

        if not available_algorithms:
            raise RuntimeError(f"No compatible algorithms available for {sensor_type}")

        if performance_priority:
            # Select highest rated available algorithm
            best_algorithm = max(available_algorithms,
                               key=lambda x: self._algorithm_info[x].performance_rating)
        else:
            # Select first available algorithm
            best_algorithm = available_algorithms[0]

        self.logger.info(f"Auto-selected {best_algorithm} for {sensor_type}")
        return best_algorithm

    def detect_sensor_type(self, **kwargs) -> SensorType:
        """
        Automatically detect sensor type based on available inputs.

        Args:
            **kwargs: Available sensor inputs (has_stereo, has_depth, has_imu, has_lidar, etc.)

        Returns:
            SensorType: Detected sensor type
        """
        has_stereo = kwargs.get('has_stereo', False)
        has_depth = kwargs.get('has_depth', False)
        has_imu = kwargs.get('has_imu', False)
        has_lidar_2d = kwargs.get('has_lidar_2d', False)
        has_lidar_3d = kwargs.get('has_lidar_3d', False)
        has_camera = kwargs.get('has_camera', True)  # Assume camera available

        # Detection logic
        if has_lidar_3d:
            return SensorType.LIDAR_3D
        elif has_lidar_2d:
            return SensorType.LIDAR_2D
        elif has_depth and has_camera:
            return SensorType.RGB_D
        elif has_stereo and has_imu:
            return SensorType.VISUAL_INERTIAL
        elif has_stereo:
            return SensorType.STEREO
        elif has_camera and has_imu:
            return SensorType.VISUAL_INERTIAL
        elif has_camera:
            return SensorType.MONOCULAR
        else:
            raise ValueError("Cannot detect sensor type from available inputs")

    def _get_or_create_config(self, algorithm_type: str,
                             sensor_type: Optional[SensorType] = None) -> SLAMConfiguration:
        """Get configuration from cache or create default."""
        if algorithm_type in self._config_cache:
            config = self._config_cache[algorithm_type]
            # Update sensor type if provided
            if sensor_type is not None:
                config.sensor_type = sensor_type
            return config
        else:
            # Create default configuration
            if sensor_type is None:
                sensor_type = SensorType.MONOCULAR

            return SLAMConfiguration(
                algorithm_name=algorithm_type,
                sensor_type=sensor_type
            )

    def get_active_algorithm(self) -> Optional[SLAMInterface]:
        """Get currently active algorithm."""
        return self._active_algorithm

    def get_active_config(self) -> Optional[SLAMConfiguration]:
        """Get configuration of active algorithm."""
        return self._active_config

    def save_config(self, algorithm_type: str, config: SLAMConfiguration):
        """Save algorithm configuration to file."""
        config_file = os.path.join(self.config_dir, f"{algorithm_type}.yaml")

        config_data = {
            'algorithm_name': config.algorithm_name,
            'sensor_type': config.sensor_type.value,
            'max_features': config.max_features,
            'quality_level': config.quality_level,
            'min_distance': config.min_distance,
            'enable_loop_closure': config.enable_loop_closure,
            'enable_mapping': config.enable_mapping,
            'enable_relocalization': config.enable_relocalization,
            'map_resolution': config.map_resolution,
            'keyframe_distance': config.keyframe_distance,
            'config_file': config.config_file,
            'vocabulary_file': config.vocabulary_file,
            'custom_params': config.custom_params
        }

        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)

        # Update cache
        self._config_cache[algorithm_type] = config
        self.logger.info(f"Saved configuration for {algorithm_type}")

    def list_configurations(self) -> Dict[str, SLAMConfiguration]:
        """List all cached configurations."""
        return self._config_cache.copy()


# Global factory instance
_slam_factory: Optional[SLAMFactory] = None


def get_slam_factory(config_dir: Optional[str] = None) -> SLAMFactory:
    """Get global SLAM factory instance."""
    global _slam_factory
    if _slam_factory is None:
        _slam_factory = SLAMFactory(config_dir)
    return _slam_factory


def create_slam_algorithm(algorithm_type: str,
                         config: Optional[SLAMConfiguration] = None,
                         sensor_type: Optional[SensorType] = None) -> SLAMInterface:
    """Convenience function to create SLAM algorithm."""
    factory = get_slam_factory()
    return factory.create_algorithm(algorithm_type, config, sensor_type)
