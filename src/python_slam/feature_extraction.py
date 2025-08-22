#!/usr/bin/env python3
"""
Feature Extraction Module for Python SLAM
Implements ORB feature detection and matching for visual SLAM applications
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Any


class FeatureExtraction:
    """
    Feature extraction class using ORB (Oriented FAST and Rotated BRIEF) features.
    Optimized for real-time SLAM applications in aerial drone competitions.
    """

    def __init__(self, method: str = 'ORB', max_features: int = 1000):
        """
        Initialize feature extractor.

        Args:
            method: Feature detection method ('ORB', 'SIFT', 'SURF')
            max_features: Maximum number of features to detect
        """
        self.method = method
        self.max_features = max_features

        # Initialize feature detector based on method
        if method == 'ORB':
            self.detector = cv2.ORB_create(
                nfeatures=max_features,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=31,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31,
                fastThreshold=20
            )
        elif method == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=max_features)
        elif method == 'SURF':
            self.detector = cv2.SURF_create(hessianThreshold=400)
        else:
            raise ValueError(f"Unsupported feature detection method: {method}")

        # Initialize matcher
        if method == 'ORB':
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Quality parameters
        self.min_match_distance = 30
        self.match_ratio_threshold = 0.75

    def extract_features(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
        """
        Extract features from an image.

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

            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)

            return keypoints, descriptors

        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return [], None

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match features between two sets of descriptors.

        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image

        Returns:
            List of matches
        """
        try:
            if desc1 is None or desc2 is None:
                return []

            # Perform matching
            matches = self.matcher.match(desc1, desc2)

            # Filter matches by distance
            good_matches = []
            if len(matches) > 0:
                distances = [m.distance for m in matches]
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                threshold = mean_distance + std_distance

                good_matches = [m for m in matches if m.distance < threshold]

            # Sort by distance
            good_matches = sorted(good_matches, key=lambda x: x.distance)

            return good_matches

        except Exception as e:
            print(f"Error in feature matching: {e}")
            return []

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
            if desc1 is None or desc2 is None:
                return []

            # Create BF matcher without cross-check for KNN
            if self.method == 'ORB':
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

            return good_matches

        except Exception as e:
            print(f"Error in KNN feature matching: {e}")
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
        # Draw keypoints
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
