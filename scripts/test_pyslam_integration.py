#!/usr/bin/env python3
"""
pySLAM Integration Example Script

This script demonstrates how to use the pySLAM integration within
the python-slam project. It shows:

1. Feature extraction using pySLAM vs OpenCV
2. Loop closure detection capabilities
3. Advanced SLAM features
4. Performance comparison

Usage:
    python test_pyslam_integration.py [--config CONFIG_FILE] [--images IMAGE_DIR]
"""

import os
import sys
import argparse
import logging
import time
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from python_slam.pyslam_integration import pySLAMWrapper, pySLAMConfig
from python_slam.feature_extraction import FeatureExtraction
from python_slam.pyslam_config import load_pyslam_config, create_default_config


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pyslam_integration_test.log')
        ]
    )


def test_feature_extraction(image_path: str, config):
    """Test feature extraction with pySLAM vs OpenCV."""
    logger = logging.getLogger(__name__)
    logger.info("Testing feature extraction...")

    # Load test image
    if not os.path.exists(image_path):
        logger.error(f"Test image not found: {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return

    logger.info(f"Loaded test image: {image.shape}")

    # Test with pySLAM enabled
    logger.info("Testing with pySLAM integration...")
    extractor_pyslam = FeatureExtraction(
        method=config.features.detector,
        max_features=config.features.max_features,
        use_pyslam=True
    )

    start_time = time.time()
    kp_pyslam, desc_pyslam = extractor_pyslam.extract_features(image)
    pyslam_time = time.time() - start_time

    # Test with OpenCV only
    logger.info("Testing with OpenCV only...")
    extractor_opencv = FeatureExtraction(
        method=config.features.detector,
        max_features=config.features.max_features,
        use_pyslam=False
    )

    start_time = time.time()
    kp_opencv, desc_opencv = extractor_opencv.extract_features(image)
    opencv_time = time.time() - start_time

    # Compare results
    logger.info(f"Feature Extraction Comparison:")
    logger.info(f"  pySLAM: {len(kp_pyslam)} features in {pyslam_time:.3f}s")
    logger.info(f"  OpenCV: {len(kp_opencv)} features in {opencv_time:.3f}s")

    if extractor_pyslam.pyslam_wrapper and extractor_pyslam.pyslam_wrapper.is_available():
        logger.info("  ✅ pySLAM integration working!")
    else:
        logger.warning("  ⚠️ pySLAM not available, used OpenCV fallback")

    # Create visualizations
    vis_pyslam = extractor_pyslam.visualize_features(image, kp_pyslam)
    vis_opencv = extractor_opencv.visualize_features(image, kp_opencv)

    # Save comparison images
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(str(output_dir / "features_pyslam.jpg"), vis_pyslam)
    cv2.imwrite(str(output_dir / "features_opencv.jpg"), vis_opencv)

    logger.info(f"Saved feature visualizations to {output_dir}")

    return {
        'pyslam_features': len(kp_pyslam),
        'opencv_features': len(kp_opencv),
        'pyslam_time': pyslam_time,
        'opencv_time': opencv_time,
        'pyslam_available': extractor_pyslam.pyslam_wrapper.is_available() if extractor_pyslam.pyslam_wrapper else False
    }


def test_feature_matching(image1_path: str, image2_path: str, config):
    """Test feature matching between two images."""
    logger = logging.getLogger(__name__)
    logger.info("Testing feature matching...")

    # Load images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        logger.error("Failed to load test images for matching")
        return

    # Extract and match features
    extractor = FeatureExtraction(
        method=config.features.detector,
        max_features=config.features.max_features,
        use_pyslam=True
    )

    start_time = time.time()
    kp1, kp2, matches = extractor.extract_and_match(img1, img2)
    match_time = time.time() - start_time

    logger.info(f"Feature Matching Results:")
    logger.info(f"  Image 1: {len(kp1)} features")
    logger.info(f"  Image 2: {len(kp2)} features")
    logger.info(f"  Matches: {len(matches)} in {match_time:.3f}s")

    # Create match visualization
    if len(matches) > 0:
        match_img = extractor.visualize_matches(img1, kp1, img2, kp2, matches)

        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(output_dir / "feature_matches.jpg"), match_img)

        logger.info(f"Saved match visualization to {output_dir}")

    return {
        'matches': len(matches),
        'match_time': match_time,
        'match_ratio': len(matches) / max(len(kp1), len(kp2), 1)
    }


def test_loop_closure(image_dir: str, config):
    """Test loop closure detection with multiple images."""
    logger = logging.getLogger(__name__)
    logger.info("Testing loop closure detection...")

    # Get test images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(image_dir).glob(ext))

    if len(image_files) < 3:
        logger.warning(f"Need at least 3 images for loop closure test, found {len(image_files)}")
        return

    # Extract features from multiple images
    extractor = FeatureExtraction(
        method=config.features.detector,
        max_features=config.features.max_features,
        use_pyslam=True
    )

    descriptor_database = []

    logger.info(f"Processing {len(image_files[:10])} images for loop closure test...")

    for i, img_file in enumerate(image_files[:10]):  # Limit to 10 images
        img = cv2.imread(str(img_file))
        if img is None:
            continue

        kp, desc = extractor.extract_features(img)
        if desc is not None:
            descriptor_database.append(desc)
            logger.debug(f"  Image {i}: {len(kp)} features")

    # Test loop closure detection with the last image
    if len(descriptor_database) > 2:
        current_desc = descriptor_database[-1]
        previous_desc = descriptor_database[:-1]

        start_time = time.time()
        loop_detected, frame_id, score = extractor.detect_loop_closure(current_desc, previous_desc)
        loop_time = time.time() - start_time

        logger.info(f"Loop Closure Results:")
        logger.info(f"  Loop detected: {loop_detected}")
        logger.info(f"  Best match frame: {frame_id}")
        logger.info(f"  Similarity score: {score:.3f}")
        logger.info(f"  Detection time: {loop_time:.3f}s")

        return {
            'loop_detected': loop_detected,
            'frame_id': frame_id,
            'score': score,
            'detection_time': loop_time
        }

    return {'error': 'Insufficient descriptors for loop closure test'}


def test_pyslam_capabilities():
    """Test pySLAM wrapper capabilities."""
    logger = logging.getLogger(__name__)
    logger.info("Testing pySLAM capabilities...")

    # Create pySLAM wrapper
    wrapper = pySLAMWrapper()

    # Get integration info
    info = wrapper.get_info()
    logger.info(f"pySLAM Integration Status:")
    logger.info(f"  Available: {info['pyslam_available']}")
    logger.info(f"  Fallback to OpenCV: {info['fallback_opencv']}")

    if info['pyslam_available']:
        logger.info(f"  Loaded modules: {info.get('pyslam_modules', [])}")

    # Get supported features
    features = wrapper.get_supported_features()
    logger.info(f"Supported Features:")
    logger.info(f"  Detectors: {features.get('detectors', [])[:5]}...")  # Show first 5
    logger.info(f"  Descriptors: {features.get('descriptors', [])[:5]}...")
    logger.info(f"  Matchers: {features.get('matchers', [])}")

    if 'loop_detectors' in features:
        logger.info(f"  Loop detectors: {features['loop_detectors']}")

    if 'depth_estimators' in features:
        logger.info(f"  Depth estimators: {features['depth_estimators']}")

    return info


def print_summary(results):
    """Print test summary."""
    logger = logging.getLogger(__name__)

    print("\n" + "="*60)
    print("pySLAM INTEGRATION TEST SUMMARY")
    print("="*60)

    if 'feature_extraction' in results:
        fe = results['feature_extraction']
        print(f"Feature Extraction:")
        print(f"  pySLAM available: {fe.get('pyslam_available', False)}")
        print(f"  pySLAM features: {fe.get('pyslam_features', 0)}")
        print(f"  OpenCV features: {fe.get('opencv_features', 0)}")
        print(f"  Performance ratio: {fe.get('opencv_time', 1) / fe.get('pyslam_time', 1):.2f}x")

    if 'feature_matching' in results:
        fm = results['feature_matching']
        print(f"Feature Matching:")
        print(f"  Matches found: {fm.get('matches', 0)}")
        print(f"  Match ratio: {fm.get('match_ratio', 0):.3f}")
        print(f"  Matching time: {fm.get('match_time', 0):.3f}s")

    if 'loop_closure' in results:
        lc = results['loop_closure']
        if 'error' not in lc:
            print(f"Loop Closure:")
            print(f"  Loop detected: {lc.get('loop_detected', False)}")
            print(f"  Similarity score: {lc.get('score', 0):.3f}")
            print(f"  Detection time: {lc.get('detection_time', 0):.3f}s")

    print("="*60)
    print("Check 'test_output/' directory for visualizations")
    print("Check 'pyslam_integration_test.log' for detailed logs")


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test pySLAM integration")
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--images', '-i', default='data', help='Directory containing test images')
    parser.add_argument('--image1', help='Path to first test image')
    parser.add_argument('--image2', help='Path to second test image')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')

    args = parser.parse_args()

    # Setup logging
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Starting pySLAM integration tests...")

    # Load configuration
    if args.config:
        config = load_pyslam_config(args.config)
    else:
        config = create_default_config()

    logger.info(f"Using configuration: detector={config.features.detector}")

    results = {}

    # Test pySLAM capabilities
    capabilities = test_pyslam_capabilities()
    results['capabilities'] = capabilities

    # Test feature extraction
    test_image = args.image1 or os.path.join(args.images, 'sample_image.jpg')
    if not os.path.exists(test_image):
        # Try to find any image in the directory
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files = list(Path(args.images).glob(ext))
            if image_files:
                test_image = str(image_files[0])
                break

    if os.path.exists(test_image):
        fe_results = test_feature_extraction(test_image, config)
        if fe_results:
            results['feature_extraction'] = fe_results
    else:
        logger.warning(f"No test image found at {test_image}")

    # Test feature matching
    img1 = args.image1 or test_image
    img2 = args.image2

    if not img2:
        # Try to find a second image
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files = list(Path(args.images).glob(ext))
            if len(image_files) >= 2:
                img1 = str(image_files[0])
                img2 = str(image_files[1])
                break

    if img1 and img2 and os.path.exists(img1) and os.path.exists(img2):
        fm_results = test_feature_matching(img1, img2, config)
        if fm_results:
            results['feature_matching'] = fm_results
    else:
        logger.warning("Need two images for matching test")

    # Test loop closure
    if os.path.exists(args.images):
        lc_results = test_loop_closure(args.images, config)
        if lc_results:
            results['loop_closure'] = lc_results

    # Print summary
    print_summary(results)

    logger.info("pySLAM integration tests completed!")


if __name__ == '__main__':
    main()
