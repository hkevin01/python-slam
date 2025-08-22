#!/usr/bin/env python3
"""
Standalone pySLAM Integration Test

This script tests the pySLAM integration without requiring ROS2 dependencies.
It demonstrates the basic functionality of the pySLAM wrapper.
"""

import os
import sys
import logging
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def test_basic_integration():
    """Test basic pySLAM integration."""
    logger = logging.getLogger(__name__)

    try:
        # Import just the pySLAM integration module
        from python_slam.pyslam_integration import pySLAMWrapper, pySLAMConfig
        logger.info("✅ Successfully imported pySLAM integration module")

        # Create wrapper
        wrapper = pySLAMWrapper()
        logger.info("✅ Successfully created pySLAM wrapper")

        # Check availability
        is_available = wrapper.is_available()
        logger.info(f"pySLAM availability: {is_available}")

        # Get integration info
        info = wrapper.get_info()
        logger.info(f"Integration info: {info}")

        # Get supported features
        features = wrapper.get_supported_features()
        logger.info(f"Supported detectors: {features.get('detectors', [])[:5]}...")

        # Test with a simple image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        keypoints, descriptors = wrapper.extract_features(test_image)
        logger.info(f"✅ Feature extraction test: {len(keypoints)} keypoints extracted")

        return True

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Test error: {e}")
        return False

def test_config_system():
    """Test the configuration system."""
    logger = logging.getLogger(__name__)

    try:
        from python_slam.pyslam_config import (
            create_default_config,
            pySLAMConfigManager,
            get_config_manager
        )
        logger.info("✅ Successfully imported configuration system")

        # Create default config
        config = create_default_config()
        logger.info(f"✅ Created default config: detector={config.features.detector}")

        # Test config manager
        manager = get_config_manager()
        logger.info("✅ Successfully created config manager")

        # Save and load config
        manager.save_config(config)
        loaded_config = manager.load_config()
        logger.info(f"✅ Config save/load test: {loaded_config.features.detector}")

        return True

    except ImportError as e:
        logger.error(f"❌ Config import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Config test error: {e}")
        return False

def main():
    """Main test function."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🚀 Starting pySLAM integration tests...")

    # Test basic integration
    integration_ok = test_basic_integration()

    # Test configuration system
    config_ok = test_config_system()

    # Summary
    print("\n" + "="*50)
    print("pySLAM INTEGRATION TEST RESULTS")
    print("="*50)
    print(f"Integration module: {'✅ PASS' if integration_ok else '❌ FAIL'}")
    print(f"Configuration system: {'✅ PASS' if config_ok else '❌ FAIL'}")

    if integration_ok and config_ok:
        print("\n🎉 All tests passed! pySLAM integration is ready.")
        print("\nTo install pySLAM for advanced features:")
        print("1. git clone --recursive https://github.com/luigifreda/pyslam.git")
        print("2. Follow pySLAM installation instructions")
        print("3. Run this test again to verify pySLAM detection")
    else:
        print("\n⚠️ Some tests failed. Check logs for details.")

    print("="*50)

    return 0 if (integration_ok and config_ok) else 1

if __name__ == '__main__':
    sys.exit(main())
