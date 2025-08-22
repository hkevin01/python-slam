from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'python_slam'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # Config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        # RViz config files
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'sensor_msgs',
        'geometry_msgs',
        'nav_msgs',
        'tf2_ros',
        'cv_bridge',
        'opencv-python',
        'numpy',
        'scipy',
        'matplotlib',
    ],
    zip_safe=True,
    maintainer='Python SLAM Team',
    maintainer_email='developer@python-slam.org',
    description='Advanced Python SLAM package for aerial drone competitions',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'slam_node = python_slam.slam_node:main',
            'feature_extraction_node = python_slam.feature_extraction_node:main',
            'pose_estimation_node = python_slam.pose_estimation_node:main',
            'mapping_node = python_slam.mapping_node:main',
            'localization_node = python_slam.localization_node:main',
            'loop_closure_node = python_slam.loop_closure_node:main',
            'flight_integration_node = python_slam.flight_integration_node:main',
            'slam_pipeline = python_slam.basic_slam_pipeline:main',
        ],
    },
)
