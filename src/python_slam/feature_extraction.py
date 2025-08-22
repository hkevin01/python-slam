#!/usr/bin/env python3
"""
Enhanced Feature Extraction Module for Python SLAM
Supports both OpenCV and pySLAM feature detection and matching
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Any, Dict
import logging

# Import pySLAM integration
from .pyslam_integration import pySLAMWrapper, pySLAMConfig

# Configure logging
logger = logging.getLogger(__name__)


class FeatureExtraction:
    """
    Enhanced feature extraction class supporting both OpenCV and pySLAM features.
    Provides fallback to OpenCV when pySLAM is not available.
    """

    def __init__(self, method: str = 'ORB', max_features: int = 1000, use_pyslam: bool = True):
        """
        Initialize feature extractor.

        Args:
            method: Feature detection method ('ORB', 'SIFT', 'SURF', 'SuperPoint', etc.)
            max_features: Maximum number of features to detect
            use_pyslam: Whether to try using pySLAM (with OpenCV fallback)
        """
        self.method = method
        self.max_features = max_features
        self.use_pyslam = use_pyslam

        # Initialize pySLAM wrapper if requested
        self.pyslam_wrapper = None
        if use_pyslam:
            config = pySLAMConfig(
                feature_detector=method,
                feature_descriptor=method,
                matcher_type='BF'
            )
            self.pyslam_wrapper = pySLAMWrapper(config)

            if self.pyslam_wrapper.is_available():
                logger.info(f"pySLAM initialized with {method} features")
            else:
                logger.warning("pySLAM not available, using OpenCV fallback")

        # Initialize OpenCV feature detector (always available as fallback)
        self._initialize_opencv_detector()

        # Quality parameters
        self.min_match_distance = 30
        self.match_ratio_threshold = 0.75

        # Feature statistics
        self.stats = {
            'total_extractions': 0,
            'pyslam_extractions': 0,
            'opencv_extractions': 0,
            'average_features': 0
        }

    def _initialize_opencv_detector(self):
        """Initialize OpenCV feature detector and matcher."""
        try:
            if self.method.upper() == 'ORB':
                self.detector = cv2.ORB_create(
                    nfeatures=self.max_features,
                    scaleFactor=1.2,
                    nlevels=8,
                    edgeThreshold=31,
                    firstLevel=0,
                    WTA_K=2,
                    scoreType=cv2.ORB_HARRIS_SCORE,
                    patchSize=31,
                    fastThreshold=20
                )
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            elif self.method.upper() == 'SIFT':
                self.detector = cv2.SIFT_create(nfeatures=self.max_features)
                self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

            elif self.method.upper() == 'SURF':
                try:
                    self.detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
                    self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
                except AttributeError:
                    logger.warning("SURF not available, falling back to ORB")
                    self.method = 'ORB'
                    self._initialize_opencv_detector()
                    return

            elif self.method.upper() == 'BRISK':
                self.detector = cv2.BRISK_create()
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            elif self.method.upper() == 'AKAZE':
                self.detector = cv2.AKAZE_create()
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            else:
                logger.warning(f"Unsupported OpenCV method {self.method}, using ORB")
                self.method = 'ORB'
                self._initialize_opencv_detector()
                return

            logger.info(f"OpenCV {self.method} detector initialized")

        except Exception as e:
            logger.error(f"Error initializing OpenCV detector: {e}")
            # Final fallback to ORB
            self.method = 'ORB'
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Extract features from an image using pySLAM or OpenCV.

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

            # Try pySLAM first if available
            if self.pyslam_wrapper and self.pyslam_wrapper.is_available():
                try:
                    keypoints, descriptors = self.pyslam_wrapper.extract_features(gray)
                    if len(keypoints) > 0:
                        self.stats['pyslam_extractions'] += 1
                        self.stats['total_extractions'] += 1
                        self.stats['average_features'] = (
                            self.stats['average_features'] * (self.stats['total_extractions'] - 1) +
                            len(keypoints)
                        ) / self.stats['total_extractions']

                        logger.debug(f"pySLAM extracted {len(keypoints)} features")
                        return keypoints, descriptors
                except Exception as e:
                    logger.warning(f"pySLAM feature extraction failed: {e}, using OpenCV fallback")

            # Fallback to OpenCV
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)

            self.stats['opencv_extractions'] += 1
            self.stats['total_extractions'] += 1
            if keypoints:
                self.stats['average_features'] = (
                    self.stats['average_features'] * (self.stats['total_extractions'] - 1) +
                    len(keypoints)
                ) / self.stats['total_extractions']

            logger.debug(f"OpenCV extracted {len(keypoints) if keypoints else 0} features")
            return keypoints, descriptors

        except Exception as e:
            logger.error(f"Error in feature extraction: {e}")
            return [], None

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match features between two sets of descriptors using pySLAM or OpenCV.

        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image

        Returns:
            List of matches
        """
        try:
            if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
                return []

            # Try pySLAM matching first if available
            if self.pyslam_wrapper and self.pyslam_wrapper.is_available():
                try:
                    matches = self.pyslam_wrapper.match_features(desc1, desc2)
                    if matches:
                        logger.debug(f"pySLAM matched {len(matches)} features")
                        return matches
                except Exception as e:
                    logger.warning(f"pySLAM matching failed: {e}, using OpenCV fallback")

            # Fallback to OpenCV matching
            matches = self.matcher.match(desc1, desc2)

            # Filter matches by distance
            if len(matches) > 0:
                distances = [m.distance for m in matches]
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                distance_threshold = mean_distance + 0.5 * std_distance

                filtered_matches = [m for m in matches if m.distance < distance_threshold]

                # Sort by distance
                filtered_matches = sorted(filtered_matches, key=lambda x: x.distance)

                logger.debug(f"OpenCV matched {len(filtered_matches)} features (filtered from {len(matches)})")
                return filtered_matches

            return matches

        except Exception as e:
            logger.error(f"Error in feature matching: {e}")
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
        if self.pyslam_wrapper and self.pyslam_wrapper.is_available():
            return self.pyslam_wrapper.detect_loop_closure(current_descriptors, descriptor_database)

        # Simple fallback loop closure detection
        return self._simple_loop_detection(current_descriptors, descriptor_database)

    def _simple_loop_detection(self, current_desc: np.ndarray,
                              desc_db: List[np.ndarray]) -> Tuple[bool, int, float]:
        """Simple loop closure detection fallback."""
        try:
            if len(desc_db) == 0 or current_desc is None:
                return False, -1, 0.0

            best_match_count = 0
            best_frame_id = -1

            for i, db_desc in enumerate(desc_db):
                if db_desc is None or len(db_desc) == 0:
                    continue

                matches = self.match_features(current_desc, db_desc)

                if len(matches) > best_match_count and len(matches) > 50:
                    best_match_count = len(matches)
                    best_frame_id = i

            loop_detected = best_match_count > 100
            similarity_score = best_match_count / max(len(current_desc), 1)

            return loop_detected, best_frame_id, similarity_score

        except Exception as e:
            logger.error(f"Loop detection failed: {e}")
            return False, -1, 0.0

    def match_features_knn(self, desc1: np.ndarray, desc2: np.ndarray, k: int = 2) -> List[cv2.DMatch]:
        """
        Match features using k-nearest neighbors with ratio test.

        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            k: Number of nearest neighbors

        Returns:
            List of good matches after ratio test
        """
        try:
            if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
                return []

            # Create BF matcher without cross-check for KNN
            if self.method.upper() == 'ORB':
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            else:
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

            # Perform KNN matching
            knn_matches = matcher.knnMatch(desc1, desc2, k=k)

            # Apply ratio test
            good_matches = []
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.match_ratio_threshold * n.distance:
                        good_matches.append(m)

            logger.debug(f"KNN matching: {len(good_matches)} good matches from {len(knn_matches)} candidates")
            return good_matches

        except Exception as e:
            logger.error(f"Error in KNN feature matching: {e}")
            return []

    def extract_and_match(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], List[cv2.DMatch]]:
        """
        Extract features from two images and match them.

        Args:
            img1: First image
            img2: Second image

        Returns:
            Tuple of (keypoints1, keypoints2, matches)
        """
        # Extract features from both images
        kp1, desc1 = self.extract_features(img1)
        kp2, desc2 = self.extract_features(img2)

        # Match features
        matches = self.match_features(desc1, desc2)

        logger.debug(f"Extract and match: {len(kp1)} + {len(kp2)} keypoints, {len(matches)} matches")
        return kp1, kp2, matches

    def get_matched_points(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint],
                          matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract matched point coordinates.

        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: List of matches

        Returns:
            Tuple of (points1, points2) as numpy arrays
        """
        if len(matches) == 0:
            return np.array([]), np.array([])

        # Extract coordinates of matched points
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)

        return pts1, pts2

    def visualize_features(self, image: np.ndarray, keypoints: List[cv2.KeyPoint]) -> np.ndarray:
        """
        Visualize detected features on an image.

        Args:
            image: Input image
            keypoints: List of detected keypoints

        Returns:
            Image with drawn keypoints
        """
        try:
            # Create output image
            output_img = image.copy()

            # Draw keypoints
            cv2.drawKeypoints(
                output_img, keypoints, output_img,
                color=(0, 255, 0),
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            # Add text overlay with feature count
            text = f"Features: {len(keypoints)} ({self.method})"
            if self.pyslam_wrapper and self.pyslam_wrapper.is_available():
                text += " [pySLAM]"
            else:
                text += " [OpenCV]"

            cv2.putText(output_img, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            return output_img

        except Exception as e:
            logger.error(f"Error visualizing features: {e}")
            return image

    def visualize_matches(self, img1: np.ndarray, kp1: List[cv2.KeyPoint],
                         img2: np.ndarray, kp2: List[cv2.KeyPoint],
                         matches: List[cv2.DMatch], max_matches: int = 50) -> np.ndarray:
        """
        Visualize feature matches between two images.

        Args:
            img1: First image
            kp1: Keypoints from first image
            img2: Second image
            kp2: Keypoints from second image
            matches: List of matches
            max_matches: Maximum number of matches to draw

        Returns:
            Image showing feature matches
        """
        try:
            # Limit number of matches for cleaner visualization
            display_matches = matches[:max_matches] if len(matches) > max_matches else matches

            # Draw matches
            match_img = cv2.drawMatches(
                img1, kp1, img2, kp2, display_matches, None,
                matchColor=(0, 255, 0),
                singlePointColor=(255, 0, 0),
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

            # Add text overlay
            text = f"Matches: {len(matches)} (showing {len(display_matches)})"
            cv2.putText(match_img, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            return match_img

        except Exception as e:
            logger.error(f"Error visualizing matches: {e}")
            # Return side-by-side images if matching visualization fails
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            if len(img1.shape) == 3:
                combined[:h1, :w1] = img1
            else:
                combined[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            if len(img2.shape) == 3:
                combined[:h2, w1:w1+w2] = img2
            else:
                combined[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            return combined

    def set_feature_method(self, method: str) -> bool:
        """
        Change the feature detection method.

        Args:
            method: New feature method ('ORB', 'SIFT', 'SuperPoint', etc.)

        Returns:
            True if successfully changed, False otherwise
        """
        try:
            self.method = method

            # Update pySLAM configuration if available
            if self.pyslam_wrapper:
                config = pySLAMConfig(
                    feature_detector=method,
                    feature_descriptor=method,
                    matcher_type='BF'
                )
                self.pyslam_wrapper = pySLAMWrapper(config)

            # Update OpenCV detector
            self._initialize_opencv_detector()

            logger.info(f"Feature method changed to {method}")
            return True

        except Exception as e:
            logger.error(f"Error changing feature method: {e}")
            return False

    def get_supported_methods(self) -> Dict[str, List[str]]:
        """
        Get supported feature detection methods.

        Returns:
            Dictionary with supported methods for OpenCV and pySLAM
        """
        if self.pyslam_wrapper:
            return self.pyslam_wrapper.get_supported_features()
        else:
            return {
                'detectors': ['ORB', 'SIFT', 'SURF', 'BRISK', 'AKAZE'],
                'descriptors': ['ORB', 'SIFT', 'SURF', 'BRISK', 'AKAZE'],
                'matchers': ['BF', 'FLANN']
            }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get feature extraction statistics.

        Returns:
            Dictionary with extraction statistics
        """
        stats = self.stats.copy()
        stats.update({
            'current_method': self.method,
            'pyslam_available': self.pyslam_wrapper is not None and self.pyslam_wrapper.is_available(),
            'max_features': self.max_features,
            'pyslam_usage_percentage': (
                (stats['pyslam_extractions'] / stats['total_extractions'] * 100)
                if stats['total_extractions'] > 0 else 0
            )
        })

        if self.pyslam_wrapper:
            stats.update(self.pyslam_wrapper.get_info())

        return stats

    def reset_statistics(self):
        """Reset feature extraction statistics."""
        self.stats = {
            'total_extractions': 0,
            'pyslam_extractions': 0,
            'opencv_extractions': 0,
            'average_features': 0
        }
        logger.info("Feature extraction statistics reset")

    def __str__(self) -> str:
        """String representation of the feature extractor."""
        pyslam_status = "Available" if (self.pyslam_wrapper and self.pyslam_wrapper.is_available()) else "Not Available"
        return (f"FeatureExtraction(method={self.method}, max_features={self.max_features}, "
                f"pySLAM={pyslam_status})")

    def __repr__(self) -> str:
        """Detailed representation of the feature extractor."""
        return self.__str__()
        img_with_kp = cv2.drawKeypoints(
            image, keypoints, None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        return img_with_kp

    def visualize_matches(self, img1: np.ndarray, kp1: List[cv2.KeyPoint],
                         img2: np.ndarray, kp2: List[cv2.KeyPoint],
                         matches: List[cv2.DMatch]) -> np.ndarray:
        """
        Visualize feature matches between two images.

        Args:
            img1: First image
            kp1: Keypoints from first image
            img2: Second image
            kp2: Keypoints from second image
            matches: List of matches

        Returns:
            Image showing matches
        """
        # Draw matches
        match_img = cv2.drawMatches(
            img1, kp1, img2, kp2, matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        return match_img

    def filter_matches_by_fundamental_matrix(self, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter matches using fundamental matrix and RANSAC.

        Args:
            pts1: Points from first image
            pts2: Points from second image

        Returns:
            Tuple of filtered (points1, points2)
        """
        if len(pts1) < 8 or len(pts2) < 8:
            return pts1, pts2

        try:
            # Compute fundamental matrix using RANSAC
            F, mask = cv2.findFundamentalMat(
                pts1, pts2,
                method=cv2.FM_RANSAC,
                ransacReprojThreshold=1.0,
                confidence=0.99
            )

            if F is not None and mask is not None:
                # Filter points using the mask
                mask = mask.ravel().astype(bool)
                pts1_filtered = pts1[mask]
                pts2_filtered = pts2[mask]

                return pts1_filtered, pts2_filtered

        except Exception as e:
            print(f"Error in fundamental matrix filtering: {e}")

        return pts1, pts2

    def update_parameters(self, **kwargs):
        """
        Update feature extraction parameters.

        Args:
            **kwargs: Parameter name-value pairs
        """
        if 'max_features' in kwargs:
            self.max_features = kwargs['max_features']
            # Recreate detector with new parameters
            if self.method == 'ORB':
                self.detector = cv2.ORB_create(nfeatures=self.max_features)

        if 'match_ratio_threshold' in kwargs:
            self.match_ratio_threshold = kwargs['match_ratio_threshold']

        if 'min_match_distance' in kwargs:
            self.min_match_distance = kwargs['min_match_distance']


# Legacy compatibility
class FeatureExtractor(FeatureExtraction):
    """Legacy class name for backward compatibility."""
    pass


if __name__ == "__main__":
    # Demo usage
    print("Feature Extraction Module - Demo")

    # Create feature extractor
    fe = FeatureExtraction(method='ORB', max_features=1000)

    # Create dummy images for testing
    img1 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img2 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Extract and match features
    kp1, kp2, matches = fe.extract_and_match(img1, img2)

    print(f"Detected {len(kp1)} features in image 1")
    print(f"Detected {len(kp2)} features in image 2")
    print(f"Found {len(matches)} matches")

    if len(matches) > 0:
        # Get matched points
        pts1, pts2 = fe.get_matched_points(kp1, kp2, matches)
        print(f"Extracted {len(pts1)} matched point pairs")

        # Filter matches
        pts1_filtered, pts2_filtered = fe.filter_matches_by_fundamental_matrix(pts1, pts2)
        print(f"After filtering: {len(pts1_filtered)} good matches")
