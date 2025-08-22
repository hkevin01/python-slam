import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import cv2

from feature_extraction import FeatureExtractor


def test_extract_features():
    extractor = FeatureExtractor()
    img = cv2.imread('data/frame1.png', cv2.IMREAD_GRAYSCALE)
    if img is not None:
        features = extractor.extract_features(img)
        assert features is not None
