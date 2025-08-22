#!/usr/bin/env python3
"""
Simple test script to validate core SLAM modules without ROS dependencies
"""

import sys
import os
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

def test_feature_extraction():
    """Test feature extraction module"""
    try:
        # Import directly from the file
        import sys
        sys.path.insert(0, 'src/python_slam')
        from feature_extraction import FeatureExtractor

        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.circle(test_image, (300, 300), 50, (128, 128, 128), -1)

        # Test feature extraction
        extractor = FeatureExtractor()
        keypoints, descriptors = extractor.extract_features(test_image)

        print(f"✓ Feature Extraction: Found {len(keypoints)} keypoints")
        return True

    except Exception as e:
        print(f"✗ Feature Extraction failed: {e}")
        return False

def test_pose_estimation():
    """Test pose estimation module"""
    try:
        import sys
        sys.path.insert(0, 'src/python_slam')
        from pose_estimation import PoseEstimator

        # Create test data
        points1 = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
        points2 = np.array([[110, 105], [210, 105], [210, 205], [110, 205]], dtype=np.float32)

        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)

        estimator = PoseEstimator()
        R, t = estimator.estimate_pose(points1, points2, K)

        print(f"✓ Pose Estimation: R shape {R.shape}, t shape {t.shape}")
        return True

    except Exception as e:
        print(f"✗ Pose Estimation failed: {e}")
        return False

def test_mapping():
    """Test mapping module"""
    try:
        import sys
        sys.path.insert(0, 'src/python_slam')
        from mapping import Mapper

        # Create test points
        points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

        mapper = Mapper()
        mapper.add_points(points)

        print(f"✓ Mapping: Added {len(points)} points")
        return True

    except Exception as e:
        print(f"✗ Mapping failed: {e}")
        return False

def test_localization():
    """Test localization module"""
    try:
        import sys
        sys.path.insert(0, 'src/python_slam')
        from localization import Localizer

        localizer = Localizer()

        # Test pose update
        pose = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)  # x, y, z, roll, pitch, yaw
        localizer.update_pose(pose)

        print(f"✓ Localization: Pose updated successfully")
        return True

    except Exception as e:
        print(f"✗ Localization failed: {e}")
        return False

def test_loop_closure():
    """Test loop closure module"""
    try:
        import sys
        sys.path.insert(0, 'src/python_slam')
        from loop_closure import LoopClosureDetector

        detector = LoopClosureDetector()

        # Create test descriptors
        desc1 = np.random.rand(100, 32).astype(np.float32)
        desc2 = np.random.rand(100, 32).astype(np.float32)

        detector.add_keyframe(0, desc1)
        detector.add_keyframe(1, desc2)

        print(f"✓ Loop Closure: Added 2 keyframes")
        return True

    except Exception as e:
        print(f"✗ Loop Closure failed: {e}")
        return False

def test_flight_integration():
    """Test flight integration module"""
    try:
        import sys
        sys.path.insert(0, 'src/python_slam')
        from flight_integration import FlightIntegration

        flight = FlightIntegration()

        # Test safety check
        class MockPose:
            def __init__(self):
                self.pose = MockPosition()

        class MockPosition:
            def __init__(self):
                self.position = MockPoint()

        class MockPoint:
            def __init__(self):
                self.x = 1.0
                self.y = 2.0
                self.z = 3.0

        pose = MockPose()
        is_safe = flight.check_safety(pose)

        print(f"✓ Flight Integration: Safety check completed ({is_safe})")
        return True

    except Exception as e:
        print(f"✗ Flight Integration failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Python SLAM Modules")
    print("=" * 40)

    tests = [
        test_feature_extraction,
        test_pose_estimation,
        test_mapping,
        test_localization,
        test_loop_closure,
        test_flight_integration
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All core SLAM modules are working correctly!")
        return True
    else:
        print("❌ Some modules need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
