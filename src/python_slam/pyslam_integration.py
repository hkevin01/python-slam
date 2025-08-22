"""
pySLAM Integration Module

This module provides integration between the existing python-slam project
and the pySLAM framework (https://github.com/luigifreda/pyslam).

pySLAM is a comprehensive Visual SLAM framework that supports:
- Multiple feature detectors (ORB, SIFT, SuperPoint, etc.)
- Advanced loop closure methods (DBoW2, DBoW3, NetVLAD, etc.)
- Volumetric reconstruction (TSDF, Gaussian Splatting)
- Depth prediction models
- Semantic mapping

Installation:
1. Clone pySLAM: git clone --recursive https://github.com/luigifreda/pyslam.git
2. Follow installation instructions in pySLAM documentation
3. Ensure pySLAM is in your Python path

Usage:
    from python_slam.pyslam_integration import pySLAMWrapper

    slam = pySLAMWrapper()
    if slam.is_available():
        # Use pySLAM features
        keypoints, descriptors = slam.extract_features(image)
    else:
        # Fallback to basic OpenCV features
        pass
"""

import os
import sys
import logging
import numpy as np
import cv2
from typing import Tuple, List, Optional, Any, Dict
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class pySLAMConfig:
    """Configuration for pySLAM integration"""
    feature_detector: str = "ORB"  # ORB, SIFT, SURF, SuperPoint, etc.
    feature_descriptor: str = "ORB"  # ORB, SIFT, SURF, SuperPoint, etc.
    matcher_type: str = "BF"  # BF (Brute Force), FLANN, LightGlue, etc.
    loop_detector: str = "DBoW2"  # DBoW2, DBoW3, NetVLAD, iBoW, etc.
    use_depth_estimation: bool = False
    depth_estimator: str = "DepthAnything"  # DepthAnything, DepthPro, etc.
    use_semantic_mapping: bool = False
    semantic_model: str = "DeepLabv3"
    use_volumetric_reconstruction: bool = False
    volumetric_method: str = "TSDF"  # TSDF, GAUSSIAN_SPLATTING


class pySLAMWrapper:
    """
    Wrapper class for pySLAM integration with fallback to OpenCV.

    This class provides a unified interface to pySLAM features while
    maintaining compatibility with the existing codebase.
    """

    def __init__(self, config: Optional[pySLAMConfig] = None):
        """
        Initialize pySLAM wrapper.

        Args:
            config: pySLAM configuration. If None, uses default config.
        """
        self.config = config or pySLAMConfig()
        self.available = False
        self.pyslam_modules = {}

        # Try to import pySLAM
        self._initialize_pyslam()

        # Initialize fallback OpenCV components
        self._initialize_opencv_fallback()

    def _initialize_pyslam(self):
        """Try to initialize pySLAM components"""
        try:
            # Check if pySLAM is in the path
            pyslam_paths = [
                '/opt/pyslam',  # Docker installation path
                '/usr/local/pyslam',  # System installation
                os.path.expanduser('~/pyslam'),  # User installation
                './pyslam',  # Local installation
            ]

            pyslam_path = None
            for path in pyslam_paths:
                if os.path.exists(path) and os.path.isdir(path):
                    pyslam_path = path
                    break

            if pyslam_path:
                # Add pySLAM to Python path
                if pyslam_path not in sys.path:
                    sys.path.insert(0, pyslam_path)

                # Try to import pySLAM modules
                from pyslam.config import Config
                from pyslam.local_features.feature_tracker import FeatureTracker
                from pyslam.slam.slam import Slam

                self.pyslam_modules = {
                    'Config': Config,
                    'FeatureTracker': FeatureTracker,
                    'Slam': Slam
                }

                self.available = True
                logger.info(f"pySLAM successfully loaded from {pyslam_path}")

                # Initialize pySLAM components
                self._setup_pyslam_components()

            else:
                logger.warning("pySLAM not found in standard paths")

        except ImportError as e:
            logger.warning(f"Failed to import pySLAM: {e}")
        except Exception as e:
            logger.error(f"Error initializing pySLAM: {e}")

    def _setup_pyslam_components(self):
        """Setup pySLAM components based on configuration"""
        try:
            if not self.available:
                return

            Config = self.pyslam_modules['Config']
            FeatureTracker = self.pyslam_modules['FeatureTracker']

            # Create a basic configuration
            self.config_dict = {
                'detector_type': self.config.feature_detector,
                'descriptor_type': self.config.feature_descriptor,
                'matcher_type': self.config.matcher_type
            }

            # Initialize feature tracker
            self.feature_tracker = FeatureTracker(self.config_dict)

            logger.info(f"pySLAM components initialized with detector: {self.config.feature_detector}")

        except Exception as e:
            logger.error(f"Failed to setup pySLAM components: {e}")
            self.available = False

    def _initialize_opencv_fallback(self):
        """Initialize OpenCV fallback components"""
        try:
            # Initialize basic OpenCV feature detector/descriptor
            if self.config.feature_detector.upper() == "ORB":
                self.cv_detector = cv2.ORB_create()
            elif self.config.feature_detector.upper() == "SIFT":
                self.cv_detector = cv2.SIFT_create()
            elif self.config.feature_detector.upper() == "SURF":
                self.cv_detector = cv2.xfeatures2d.SURF_create()
            else:
                self.cv_detector = cv2.ORB_create()  # Default fallback

            # Initialize matcher
            if self.config.matcher_type.upper() == "BF":
                if self.config.feature_detector.upper() == "ORB":
                    self.cv_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                else:
                    self.cv_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            else:
                self.cv_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            logger.info("OpenCV fallback components initialized")

        except Exception as e:
            logger.error(f"Failed to initialize OpenCV fallback: {e}")

    def is_available(self) -> bool:
        """Check if pySLAM is available"""
        return self.available

    def extract_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Extract features from image using pySLAM or OpenCV fallback.

        Args:
            image: Input image (grayscale or BGR)

        Returns:
            Tuple of (keypoints, descriptors)
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            if self.available and hasattr(self, 'feature_tracker'):
                # Use pySLAM feature extraction
                try:
                    keypoints, descriptors = self.feature_tracker.detectAndCompute(gray)
                    logger.debug(f"pySLAM extracted {len(keypoints)} features")
                    return keypoints, descriptors
                except Exception as e:
                    logger.warning(f"pySLAM feature extraction failed: {e}, falling back to OpenCV")

            # Fallback to OpenCV
            keypoints, descriptors = self.cv_detector.detectAndCompute(gray, None)
            logger.debug(f"OpenCV extracted {len(keypoints)} features")
            return keypoints, descriptors

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return [], np.array([])

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """
        Match features between two descriptor sets.

        Args:
            desc1: First set of descriptors
            desc2: Second set of descriptors

        Returns:
            List of matches
        """
        try:
            if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
                return []

            matches = self.cv_matcher.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)

            logger.debug(f"Matched {len(matches)} features")
            return matches

        except Exception as e:
            logger.error(f"Feature matching failed: {e}")
            return []

    def detect_loop_closure(self, current_descriptors: np.ndarray,
                           descriptor_database: List[np.ndarray]) -> Tuple[bool, int, float]:
        """
        Detect loop closure using pySLAM or simple similarity matching.

        Args:
            current_descriptors: Current frame descriptors
            descriptor_database: Database of previous descriptors

        Returns:
            Tuple of (loop_detected, frame_id, similarity_score)
        """
        try:
            if self.available:
                # TODO: Implement pySLAM loop closure when available
                logger.debug("pySLAM loop closure not yet implemented")

            # Simple fallback loop closure detection
            if len(descriptor_database) == 0:
                return False, -1, 0.0

            best_match_count = 0
            best_frame_id = -1

            for i, db_desc in enumerate(descriptor_database):
                if db_desc is None or len(db_desc) == 0:
                    continue

                matches = self.match_features(current_descriptors, db_desc)

                # Simple threshold-based loop detection
                if len(matches) > best_match_count and len(matches) > 50:
                    best_match_count = len(matches)
                    best_frame_id = i

            loop_detected = best_match_count > 100  # Threshold for loop detection
            similarity_score = best_match_count / max(len(current_descriptors), 1)

            if loop_detected:
                logger.info(f"Loop closure detected with frame {best_frame_id}, score: {similarity_score:.3f}")

            return loop_detected, best_frame_id, similarity_score

        except Exception as e:
            logger.error(f"Loop closure detection failed: {e}")
            return False, -1, 0.0

    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth from a single image using pySLAM depth estimation models.

        Args:
            image: Input RGB image

        Returns:
            Depth map or None if not available
        """
        try:
            if not self.available or not self.config.use_depth_estimation:
                logger.debug("Depth estimation not available or disabled")
                return None

            # TODO: Implement pySLAM depth estimation when available
            logger.debug("pySLAM depth estimation not yet implemented")
            return None

        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return None

    def segment_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Perform semantic segmentation using pySLAM models.

        Args:
            image: Input RGB image

        Returns:
            Segmentation mask or None if not available
        """
        try:
            if not self.available or not self.config.use_semantic_mapping:
                logger.debug("Semantic mapping not available or disabled")
                return None

            # TODO: Implement pySLAM semantic segmentation when available
            logger.debug("pySLAM semantic segmentation not yet implemented")
            return None

        except Exception as e:
            logger.error(f"Semantic segmentation failed: {e}")
            return None

    def get_supported_features(self) -> Dict[str, List[str]]:
        """
        Get list of supported features in pySLAM and OpenCV.

        Returns:
            Dictionary with supported detectors, descriptors, and matchers
        """
        opencv_features = {
            'detectors': ['ORB', 'SIFT', 'SURF', 'BRISK', 'AKAZE'],
            'descriptors': ['ORB', 'SIFT', 'SURF', 'BRISK', 'AKAZE'],
            'matchers': ['BF', 'FLANN']
        }

        if self.available:
            pyslam_features = {
                'detectors': [
                    'ORB', 'ORB2', 'SIFT', 'SURF', 'BRISK', 'AKAZE', 'KAZE',
                    'SuperPoint', 'D2Net', 'DELF', 'R2D2', 'KeyNet', 'DISK',
                    'ALIKED', 'Xfeat'
                ],
                'descriptors': [
                    'ORB', 'SIFT', 'SURF', 'BRISK', 'AKAZE', 'SuperPoint',
                    'Hardnet', 'GeoDesc', 'SOSNet', 'L2Net', 'D2Net', 'DELF',
                    'R2D2', 'DISK', 'ALIKED', 'Xfeat'
                ],
                'matchers': ['BF', 'FLANN', 'XFeat', 'LightGlue', 'LoFTR', 'MASt3R'],
                'loop_detectors': [
                    'DBoW2', 'DBoW3', 'VLAD', 'iBoW', 'OBIndex2', 'NetVLAD',
                    'CosPlace', 'EigenPlaces', 'Megaloc'
                ],
                'depth_estimators': [
                    'DepthPro', 'DepthAnything', 'RAFT-Stereo', 'CREStereo', 'MASt3R'
                ]
            }
            return {**opencv_features, **pyslam_features}

        return opencv_features

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about pySLAM integration status.

        Returns:
            Dictionary with integration information
        """
        info = {
            'pyslam_available': self.available,
            'current_config': {
                'detector': self.config.feature_detector,
                'descriptor': self.config.feature_descriptor,
                'matcher': self.config.matcher_type,
                'loop_detector': self.config.loop_detector,
                'depth_estimation': self.config.use_depth_estimation,
                'semantic_mapping': self.config.use_semantic_mapping,
                'volumetric_reconstruction': self.config.use_volumetric_reconstruction
            },
            'fallback_opencv': True,
            'python_path': sys.path[:3]  # First 3 paths for debugging
        }

        if self.available:
            info['pyslam_modules'] = list(self.pyslam_modules.keys())

        return info


def create_pyslam_wrapper(detector: str = "ORB",
                         descriptor: str = "ORB",
                         matcher: str = "BF") -> pySLAMWrapper:
    """
    Convenience function to create a pySLAM wrapper with specific configuration.

    Args:
        detector: Feature detector type
        descriptor: Feature descriptor type
        matcher: Feature matcher type

    Returns:
        Configured pySLAM wrapper
    """
    config = pySLAMConfig(
        feature_detector=detector,
        feature_descriptor=descriptor,
        matcher_type=matcher
    )
    return pySLAMWrapper(config)


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create pySLAM wrapper
    slam = pySLAMWrapper()

    # Print integration status
    info = slam.get_info()
    print("pySLAM Integration Status:")
    print(f"- Available: {info['pyslam_available']}")
    print(f"- Current detector: {info['current_config']['detector']}")
    print(f"- Fallback to OpenCV: {info['fallback_opencv']}")

    # Print supported features
    features = slam.get_supported_features()
    print(f"\nSupported detectors: {features['detectors'][:5]}...")  # Show first 5

    if slam.is_available():
        print("\n✅ pySLAM is available and ready to use!")
    else:
        print("\n⚠️ pySLAM not available, using OpenCV fallback")
        print("To install pySLAM:")
        print("1. git clone --recursive https://github.com/luigifreda/pyslam.git")
        print("2. Follow installation instructions in pySLAM documentation")
