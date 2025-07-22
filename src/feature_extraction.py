"""
Module: feature_extraction.py
Purpose: Extract and match keypoints across frames.
"""


class FeatureExtractor:
    def __init__(self, method='ORB'):
        """Initialize feature extractor (ORB, FAST, BRIEF, etc.)."""
        self.method = method

    def extract_features(self, image):
        """Extract features from an image."""
        pass

    def match_features(self, desc1, desc2):
        """Match features between two sets of descriptors."""
        pass
