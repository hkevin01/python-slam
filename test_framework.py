#!/usr/bin/env python3
"""
Simple test script for the multi-algorithm SLAM framework.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from python_slam.slam_interfaces import SLAMInterface, SLAMConfiguration, SensorType
from python_slam.slam_interfaces import SLAMFactory, create_slam_system

class TestSLAMFramework(unittest.TestCase):
    """Test suite for the SLAM framework."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SLAMConfiguration(
            algorithm_name="python_slam",
            sensor_type=SensorType.MONOCULAR,
            max_features=500,
            enable_loop_closure=True
        )

    def test_slam_configuration(self):
        """Test SLAM configuration creation."""
        self.assertEqual(self.config.algorithm_name, "python_slam")
        self.assertEqual(self.config.sensor_type, SensorType.MONOCULAR)
        self.assertEqual(self.config.max_features, 500)
        self.assertTrue(self.config.enable_loop_closure)

    def test_slam_factory_available_algorithms(self):
        """Test factory algorithm availability."""
        factory = SLAMFactory()
        algorithms = factory.get_available_algorithms()

        # Should include at least our implemented algorithms
        expected_algorithms = ["python_slam", "orb_slam3", "rtabmap", "cartographer", "openvslam"]
        for alg in expected_algorithms:
            self.assertIn(alg, algorithms)

    def test_slam_factory_algorithm_creation(self):
        """Test algorithm creation through factory."""
        factory = SLAMFactory()

        # Test creating Python SLAM algorithm
        slam_system = factory.create_algorithm(self.config)
        self.assertIsNotNone(slam_system)
        self.assertIsInstance(slam_system, SLAMInterface)

    def test_create_slam_system_helper(self):
        """Test the create_slam_system helper function."""
        slam_system = create_slam_system(
            algorithm="python_slam",
            sensor_type=SensorType.MONOCULAR,
            max_features=1000
        )

        self.assertIsNotNone(slam_system)
        self.assertIsInstance(slam_system, SLAMInterface)

    @patch('cv2.imread')
    def test_slam_interface_methods(self, mock_imread):
        """Test SLAM interface method signatures."""
        # Mock image
        mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image

        slam_system = create_slam_system(
            algorithm="python_slam",
            sensor_type=SensorType.MONOCULAR
        )

        # Test interface methods exist
        self.assertTrue(hasattr(slam_system, 'initialize'))
        self.assertTrue(hasattr(slam_system, 'process_image'))
        self.assertTrue(hasattr(slam_system, 'get_pose'))
        self.assertTrue(hasattr(slam_system, 'get_map'))
        self.assertTrue(hasattr(slam_system, 'reset'))

    def test_sensor_type_enum(self):
        """Test sensor type enumeration."""
        # Test all sensor types are defined
        sensor_types = [
            SensorType.MONOCULAR,
            SensorType.STEREO,
            SensorType.RGBD,
            SensorType.VISUAL_INERTIAL,
            SensorType.LIDAR,
            SensorType.POINTCLOUD
        ]

        for sensor_type in sensor_types:
            self.assertIsInstance(sensor_type, SensorType)

    def test_algorithm_switching(self):
        """Test runtime algorithm switching."""
        factory = SLAMFactory()

        # Create initial algorithm
        initial_config = SLAMConfiguration(
            algorithm_name="python_slam",
            sensor_type=SensorType.MONOCULAR
        )
        slam1 = factory.create_algorithm(initial_config)

        # Switch to different algorithm
        new_config = SLAMConfiguration(
            algorithm_name="orb_slam3",
            sensor_type=SensorType.MONOCULAR
        )
        slam2 = factory.switch_algorithm(new_config)

        # Verify different instances
        self.assertIsNotNone(slam1)
        self.assertIsNotNone(slam2)

    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        # Test invalid algorithm name
        with self.assertRaises(ValueError):
            SLAMConfiguration(
                algorithm_name="invalid_algorithm",
                sensor_type=SensorType.MONOCULAR
            )

        # Test valid configuration with custom parameters
        config = SLAMConfiguration(
            algorithm_name="orb_slam3",
            sensor_type=SensorType.STEREO,
            custom_params={
                'camera': {'fx': 525.0, 'fy': 525.0},
                'orb_slam3': {'nFeatures': 2000}
            }
        )
        self.assertIsInstance(config.custom_params, dict)
        self.assertEqual(config.custom_params['camera']['fx'], 525.0)

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
