"""
pySLAM Configuration Module

This module provides configuration management for pySLAM integration
within the python-slam project.
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class pySLAMFeatureConfig:
    """Configuration for pySLAM feature extraction."""
    detector: str = "ORB"
    descriptor: str = "ORB"
    matcher: str = "BF"
    max_features: int = 1000
    use_adaptive_threshold: bool = True

    # Advanced feature parameters
    orb_scale_factor: float = 1.2
    orb_n_levels: int = 8
    sift_n_features: int = 1000
    surf_hessian_threshold: float = 400.0


@dataclass
class pySLAMLoopClosureConfig:
    """Configuration for pySLAM loop closure detection."""
    method: str = "DBoW2"  # DBoW2, DBoW3, NetVLAD, iBoW, etc.
    vocabulary_path: str = ""
    similarity_threshold: float = 0.7
    min_matches: int = 50
    use_geometric_verification: bool = True


@dataclass
class pySLAMDepthConfig:
    """Configuration for pySLAM depth estimation."""
    enabled: bool = False
    method: str = "DepthAnything"  # DepthAnything, DepthPro, RAFT-Stereo, etc.
    use_in_frontend: bool = False
    use_in_backend: bool = True
    depth_scale_factor: float = 1000.0


@dataclass
class pySLAMSemanticConfig:
    """Configuration for pySLAM semantic mapping."""
    enabled: bool = False
    method: str = "DeepLabv3"  # DeepLabv3, Segformer, CLIP
    num_classes: int = 21
    confidence_threshold: float = 0.5


@dataclass
class pySLAMVolumetricConfig:
    """Configuration for pySLAM volumetric reconstruction."""
    enabled: bool = False
    method: str = "TSDF"  # TSDF, GAUSSIAN_SPLATTING
    voxel_size: float = 0.05
    truncation_distance: float = 0.3
    extract_mesh: bool = True


@dataclass
class pySLAMSystemConfig:
    """Main pySLAM system configuration."""
    pyslam_path: str = ""
    use_pyslam: bool = True
    fallback_to_opencv: bool = True

    # Sub-configurations
    features: pySLAMFeatureConfig = None
    loop_closure: pySLAMLoopClosureConfig = None
    depth: pySLAMDepthConfig = None
    semantic: pySLAMSemanticConfig = None
    volumetric: pySLAMVolumetricConfig = None

    def __post_init__(self):
        """Initialize sub-configurations if not provided."""
        if self.features is None:
            self.features = pySLAMFeatureConfig()
        if self.loop_closure is None:
            self.loop_closure = pySLAMLoopClosureConfig()
        if self.depth is None:
            self.depth = pySLAMDepthConfig()
        if self.semantic is None:
            self.semantic = pySLAMSemanticConfig()
        if self.volumetric is None:
            self.volumetric = pySLAMVolumetricConfig()


class pySLAMConfigManager:
    """Manager for pySLAM configuration files."""

    def __init__(self, config_dir: str = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            # Default to config directory in project root
            self.config_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'config'
            )
        else:
            self.config_dir = config_dir

        self.config_file = os.path.join(self.config_dir, 'pyslam_config.yaml')
        self._config: Optional[pySLAMSystemConfig] = None

    def load_config(self, config_file: str = None) -> pySLAMSystemConfig:
        """
        Load configuration from YAML file.

        Args:
            config_file: Path to configuration file

        Returns:
            Loaded configuration
        """
        try:
            file_path = config_file or self.config_file

            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    config_dict = yaml.safe_load(f)

                # Convert nested dictionaries to dataclasses
                self._config = self._dict_to_config(config_dict)
                logger.info(f"Loaded pySLAM configuration from {file_path}")

            else:
                logger.warning(f"Config file not found: {file_path}, using defaults")
                self._config = pySLAMSystemConfig()
                self.save_config()  # Save default configuration

            return self._config

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._config = pySLAMSystemConfig()
            return self._config

    def save_config(self, config: pySLAMSystemConfig = None, config_file: str = None):
        """
        Save configuration to YAML file.

        Args:
            config: Configuration to save (uses current if None)
            config_file: Path to configuration file
        """
        try:
            if config is None:
                config = self._config or pySLAMSystemConfig()

            file_path = config_file or self.config_file

            # Ensure config directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Convert to dictionary
            config_dict = self._config_to_dict(config)

            with open(file_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            logger.info(f"Saved pySLAM configuration to {file_path}")

        except Exception as e:
            logger.error(f"Error saving configuration: {e}")

    def get_config(self) -> pySLAMSystemConfig:
        """Get current configuration, loading if necessary."""
        if self._config is None:
            self.load_config()
        return self._config

    def update_config(self, **kwargs):
        """Update configuration with keyword arguments."""
        config = self.get_config()

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")

        self._config = config

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> pySLAMSystemConfig:
        """Convert dictionary to configuration dataclass."""
        try:
            # Extract sub-configurations
            features_dict = config_dict.get('features', {})
            loop_closure_dict = config_dict.get('loop_closure', {})
            depth_dict = config_dict.get('depth', {})
            semantic_dict = config_dict.get('semantic', {})
            volumetric_dict = config_dict.get('volumetric', {})

            # Create sub-configurations
            features = pySLAMFeatureConfig(**features_dict)
            loop_closure = pySLAMLoopClosureConfig(**loop_closure_dict)
            depth = pySLAMDepthConfig(**depth_dict)
            semantic = pySLAMSemanticConfig(**semantic_dict)
            volumetric = pySLAMVolumetricConfig(**volumetric_dict)

            # Create main configuration
            main_config = {k: v for k, v in config_dict.items()
                          if k not in ['features', 'loop_closure', 'depth', 'semantic', 'volumetric']}

            return pySLAMSystemConfig(
                features=features,
                loop_closure=loop_closure,
                depth=depth,
                semantic=semantic,
                volumetric=volumetric,
                **main_config
            )

        except Exception as e:
            logger.error(f"Error converting dict to config: {e}")
            return pySLAMSystemConfig()

    def _config_to_dict(self, config: pySLAMSystemConfig) -> Dict[str, Any]:
        """Convert configuration dataclass to dictionary."""
        try:
            config_dict = asdict(config)
            return config_dict

        except Exception as e:
            logger.error(f"Error converting config to dict: {e}")
            return {}

    def get_feature_config(self) -> pySLAMFeatureConfig:
        """Get feature extraction configuration."""
        return self.get_config().features

    def get_loop_closure_config(self) -> pySLAMLoopClosureConfig:
        """Get loop closure configuration."""
        return self.get_config().loop_closure

    def get_depth_config(self) -> pySLAMDepthConfig:
        """Get depth estimation configuration."""
        return self.get_config().depth

    def get_semantic_config(self) -> pySLAMSemanticConfig:
        """Get semantic mapping configuration."""
        return self.get_config().semantic

    def get_volumetric_config(self) -> pySLAMVolumetricConfig:
        """Get volumetric reconstruction configuration."""
        return self.get_config().volumetric


def create_default_config() -> pySLAMSystemConfig:
    """Create a default pySLAM configuration."""
    return pySLAMSystemConfig(
        pyslam_path="",
        use_pyslam=True,
        fallback_to_opencv=True,
        features=pySLAMFeatureConfig(
            detector="ORB",
            descriptor="ORB",
            matcher="BF",
            max_features=1000
        ),
        loop_closure=pySLAMLoopClosureConfig(
            method="DBoW2",
            similarity_threshold=0.7,
            min_matches=50
        ),
        depth=pySLAMDepthConfig(
            enabled=False,
            method="DepthAnything"
        ),
        semantic=pySLAMSemanticConfig(
            enabled=False,
            method="DeepLabv3"
        ),
        volumetric=pySLAMVolumetricConfig(
            enabled=False,
            method="TSDF"
        )
    )


# Global configuration manager instance
_config_manager = None

def get_config_manager() -> pySLAMConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = pySLAMConfigManager()
    return _config_manager


# Convenience functions
def load_pyslam_config(config_file: str = None) -> pySLAMSystemConfig:
    """Load pySLAM configuration."""
    return get_config_manager().load_config(config_file)


def save_pyslam_config(config: pySLAMSystemConfig, config_file: str = None):
    """Save pySLAM configuration."""
    get_config_manager().save_config(config, config_file)


def get_pyslam_config() -> pySLAMSystemConfig:
    """Get current pySLAM configuration."""
    return get_config_manager().get_config()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create and save default configuration
    config = create_default_config()
    manager = pySLAMConfigManager()
    manager.save_config(config)

    # Load and print configuration
    loaded_config = manager.load_config()
    print(f"pySLAM Configuration:")
    print(f"- Use pySLAM: {loaded_config.use_pyslam}")
    print(f"- Feature detector: {loaded_config.features.detector}")
    print(f"- Loop closure method: {loaded_config.loop_closure.method}")
    print(f"- Depth estimation: {loaded_config.depth.enabled}")
    print(f"- Semantic mapping: {loaded_config.semantic.enabled}")
    print(f"- Volumetric reconstruction: {loaded_config.volumetric.enabled}")
