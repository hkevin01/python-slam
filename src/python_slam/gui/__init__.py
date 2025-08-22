"""
GUI module for Python SLAM visualization

This module provides advanced visualization capabilities for SLAM operations including:
- Real-time 3D point cloud and trajectory visualization
- Camera feed with feature tracking overlay
- Performance metrics and statistics monitoring
- Interactive controls for playback and analysis
"""

from .slam_visualizer import SLAMVisualizer, create_slam_gui, PYQT_AVAILABLE

__all__ = ['SLAMVisualizer', 'create_slam_gui', 'PYQT_AVAILABLE']
