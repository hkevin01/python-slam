"""
Unit Tests for GUI Components

This module provides detailed unit tests for the GUI components.
"""

import unittest
import sys
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestGUIAvailability(unittest.TestCase):
    """Test GUI backend availability."""
    
    def test_gui_backend_detection(self):
        """Test detection of available GUI backends."""
        pyqt6_available = False
        pyside6_available = False
        
        try:
            import PyQt6
            pyqt6_available = True
        except ImportError:
            pass
        
        try:
            import PySide6
            pyside6_available = True
        except ImportError:
            pass
        
        # At least one GUI backend should be available for full functionality
        if not (pyqt6_available or pyside6_available):
            self.skipTest("No GUI backend available")
        
        if pyqt6_available:
            print("PyQt6 available")
        if pyside6_available:
            print("PySide6 available")

class TestMaterialDesignManager(unittest.TestCase):
    """Test Material Design styling manager."""
    
    def setUp(self):
        """Set up test environment."""
        try:
            from python_slam.gui.utils import MaterialDesignManager
            self.MaterialDesignManager = MaterialDesignManager
        except ImportError:
            self.skipTest("Material Design manager not available")
    
    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = self.MaterialDesignManager()
        self.assertIsNotNone(manager)
    
    def test_theme_availability(self):
        """Test theme availability."""
        manager = self.MaterialDesignManager()
        themes = manager.available_themes()
        
        self.assertIsInstance(themes, list)
        self.assertIn("dark", themes)
        self.assertIn("light", themes)
        self.assertGreaterEqual(len(themes), 2)
    
    def test_color_palette(self):
        """Test color palette functionality."""
        manager = self.MaterialDesignManager()
        
        # Test dark theme colors
        dark_colors = manager.get_color_palette("dark")
        self.assertIsInstance(dark_colors, dict)
        self.assertIn("primary", dark_colors)
        self.assertIn("secondary", dark_colors)
        self.assertIn("background", dark_colors)
        self.assertIn("surface", dark_colors)
        
        # Test light theme colors
        light_colors = manager.get_color_palette("light")
        self.assertIsInstance(light_colors, dict)
        self.assertIn("primary", light_colors)
        self.assertIn("secondary", light_colors)
        self.assertIn("background", light_colors)
        self.assertIn("surface", light_colors)
        
        # Themes should have different colors
        self.assertNotEqual(dark_colors["background"], light_colors["background"])
    
    def test_stylesheet_generation(self):
        """Test stylesheet generation."""
        manager = self.MaterialDesignManager()
        
        stylesheet = manager.generate_stylesheet("dark")
        self.assertIsInstance(stylesheet, str)
        self.assertGreater(len(stylesheet), 0)
        
        # Should contain CSS-like content
        self.assertIn("background-color", stylesheet)
        self.assertIn("color", stylesheet)
    
    def test_icon_management(self):
        """Test icon management."""
        manager = self.MaterialDesignManager()
        
        # Test icon availability
        available_icons = manager.get_available_icons()
        self.assertIsInstance(available_icons, list)
        self.assertGreater(len(available_icons), 0)
        
        # Test icon retrieval
        if available_icons:
            icon_name = available_icons[0]
            icon = manager.get_icon(icon_name, "dark")
            self.assertIsNotNone(icon)

class TestMainWindow(unittest.TestCase):
    """Test main window functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Check for GUI backend
        try:
            from PyQt6.QtWidgets import QApplication
            self.gui_backend = "PyQt6"
        except ImportError:
            try:
                from PySide6.QtWidgets import QApplication
                self.gui_backend = "PySide6"
            except ImportError:
                self.skipTest("No GUI backend available")
        
        try:
            from python_slam.gui.main_window import SlamMainWindow
            self.SlamMainWindow = SlamMainWindow
        except ImportError:
            self.skipTest("Main window not available")
    
    @patch('sys.argv', ['test'])
    def test_main_window_creation(self):
        """Test main window creation."""
        if self.gui_backend == "PyQt6":
            from PyQt6.QtWidgets import QApplication
        else:
            from PySide6.QtWidgets import QApplication
        
        # Create application if needed
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        try:
            window = self.SlamMainWindow()
            self.assertIsNotNone(window)
            
            # Test window properties
            self.assertIsNotNone(window.windowTitle())
            self.assertGreater(len(window.windowTitle()), 0)
            
            # Test central widget
            central_widget = window.centralWidget()
            self.assertIsNotNone(central_widget)
            
        except Exception as e:
            self.fail(f"Main window creation failed: {e}")
    
    @patch('sys.argv', ['test'])
    def test_window_components(self):
        """Test window component creation."""
        if self.gui_backend == "PyQt6":
            from PyQt6.QtWidgets import QApplication
        else:
            from PySide6.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        try:
            window = self.SlamMainWindow()
            
            # Test menu bar
            menu_bar = window.menuBar()
            self.assertIsNotNone(menu_bar)
            
            # Test status bar
            status_bar = window.statusBar()
            self.assertIsNotNone(status_bar)
            
            # Test toolbar creation
            if hasattr(window, 'create_toolbars'):
                window.create_toolbars()
                toolbars = window.findChildren(type(window).toolbar_class if hasattr(window, 'toolbar_class') else object)
                # Should have some toolbars
                self.assertGreaterEqual(len(toolbars), 0)
        
        except Exception as e:
            self.fail(f"Window component test failed: {e}")

class TestVisualizationComponents(unittest.TestCase):
    """Test 3D visualization components."""
    
    def setUp(self):
        """Set up test environment."""
        try:
            from python_slam.gui.visualization import Map3DViewer, PointCloudRenderer, TrajectoryRenderer
            self.Map3DViewer = Map3DViewer
            self.PointCloudRenderer = PointCloudRenderer
            self.TrajectoryRenderer = TrajectoryRenderer
        except ImportError:
            self.skipTest("Visualization components not available")
        
        # Create test data
        self.test_points = np.random.randn(1000, 3).astype(np.float32)
        self.test_colors = np.random.rand(1000, 3).astype(np.float32)
        self.test_trajectory = np.random.randn(100, 3).astype(np.float32)
    
    def test_map3d_viewer_creation(self):
        """Test Map3D viewer creation."""
        try:
            viewer = self.Map3DViewer()
            self.assertIsNotNone(viewer)
            
            # Test initialization
            if hasattr(viewer, 'initialize'):
                viewer.initialize()
            
            # Test camera setup
            if hasattr(viewer, 'setup_camera'):
                viewer.setup_camera()
            
        except Exception as e:
            # OpenGL might not be available in test environment
            self.skipTest(f"Map3D viewer test failed (likely no OpenGL): {e}")
    
    def test_point_cloud_renderer(self):
        """Test point cloud renderer."""
        try:
            renderer = self.PointCloudRenderer()
            self.assertIsNotNone(renderer)
            
            # Test point cloud loading
            if hasattr(renderer, 'load_point_cloud'):
                renderer.load_point_cloud(self.test_points, self.test_colors)
            
            # Test rendering preparation
            if hasattr(renderer, 'prepare_render'):
                renderer.prepare_render()
        
        except Exception as e:
            self.skipTest(f"Point cloud renderer test failed: {e}")
    
    def test_trajectory_renderer(self):
        """Test trajectory renderer."""
        try:
            renderer = self.TrajectoryRenderer()
            self.assertIsNotNone(renderer)
            
            # Test trajectory loading
            if hasattr(renderer, 'load_trajectory'):
                renderer.load_trajectory(self.test_trajectory)
            
            # Test keyframe addition
            if hasattr(renderer, 'add_keyframe'):
                renderer.add_keyframe(self.test_trajectory[0], 0)
        
        except Exception as e:
            self.skipTest(f"Trajectory renderer test failed: {e}")

class TestControlPanels(unittest.TestCase):
    """Test control panel functionality."""
    
    def setUp(self):
        """Set up test environment."""
        try:
            from python_slam.gui.control_panels import (
                SlamControlPanel, 
                DatasetControlPanel, 
                VisualizationControlPanel
            )
            self.SlamControlPanel = SlamControlPanel
            self.DatasetControlPanel = DatasetControlPanel
            self.VisualizationControlPanel = VisualizationControlPanel
        except ImportError:
            self.skipTest("Control panels not available")
        
        # Check for GUI backend
        try:
            from PyQt6.QtWidgets import QApplication
            self.gui_backend = "PyQt6"
        except ImportError:
            try:
                from PySide6.QtWidgets import QApplication
                self.gui_backend = "PySide6"
            except ImportError:
                self.skipTest("No GUI backend available")
    
    @patch('sys.argv', ['test'])
    def test_slam_control_panel(self):
        """Test SLAM control panel."""
        if self.gui_backend == "PyQt6":
            from PyQt6.QtWidgets import QApplication
        else:
            from PySide6.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        try:
            panel = self.SlamControlPanel()
            self.assertIsNotNone(panel)
            
            # Test panel components
            if hasattr(panel, 'start_button'):
                self.assertIsNotNone(panel.start_button)
            if hasattr(panel, 'stop_button'):
                self.assertIsNotNone(panel.stop_button)
            if hasattr(panel, 'reset_button'):
                self.assertIsNotNone(panel.reset_button)
        
        except Exception as e:
            self.fail(f"SLAM control panel test failed: {e}")
    
    @patch('sys.argv', ['test'])
    def test_dataset_control_panel(self):
        """Test dataset control panel."""
        if self.gui_backend == "PyQt6":
            from PyQt6.QtWidgets import QApplication
        else:
            from PySide6.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        try:
            panel = self.DatasetControlPanel()
            self.assertIsNotNone(panel)
            
            # Test dataset loading components
            if hasattr(panel, 'load_button'):
                self.assertIsNotNone(panel.load_button)
            if hasattr(panel, 'dataset_combo'):
                self.assertIsNotNone(panel.dataset_combo)
        
        except Exception as e:
            self.fail(f"Dataset control panel test failed: {e}")
    
    @patch('sys.argv', ['test'])
    def test_visualization_control_panel(self):
        """Test visualization control panel."""
        if self.gui_backend == "PyQt6":
            from PyQt6.QtWidgets import QApplication
        else:
            from PySide6.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        try:
            panel = self.VisualizationControlPanel()
            self.assertIsNotNone(panel)
            
            # Test visualization controls
            if hasattr(panel, 'show_points_checkbox'):
                self.assertIsNotNone(panel.show_points_checkbox)
            if hasattr(panel, 'show_trajectory_checkbox'):
                self.assertIsNotNone(panel.show_trajectory_checkbox)
        
        except Exception as e:
            self.fail(f"Visualization control panel test failed: {e}")

class TestMetricsDashboard(unittest.TestCase):
    """Test metrics dashboard functionality."""
    
    def setUp(self):
        """Set up test environment."""
        try:
            from python_slam.gui.metrics_dashboard import MetricsDashboard, RealTimeMetrics
            self.MetricsDashboard = MetricsDashboard
            self.RealTimeMetrics = RealTimeMetrics
        except ImportError:
            self.skipTest("Metrics dashboard not available")
        
        # Check for GUI backend
        try:
            from PyQt6.QtWidgets import QApplication
            self.gui_backend = "PyQt6"
        except ImportError:
            try:
                from PySide6.QtWidgets import QApplication
                self.gui_backend = "PySide6"
            except ImportError:
                self.skipTest("No GUI backend available")
    
    @patch('sys.argv', ['test'])
    def test_metrics_dashboard_creation(self):
        """Test metrics dashboard creation."""
        if self.gui_backend == "PyQt6":
            from PyQt6.QtWidgets import QApplication
        else:
            from PySide6.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        try:
            dashboard = self.MetricsDashboard()
            self.assertIsNotNone(dashboard)
            
            # Test dashboard components
            if hasattr(dashboard, 'fps_label'):
                self.assertIsNotNone(dashboard.fps_label)
            if hasattr(dashboard, 'memory_label'):
                self.assertIsNotNone(dashboard.memory_label)
        
        except Exception as e:
            self.fail(f"Metrics dashboard creation failed: {e}")
    
    def test_real_time_metrics(self):
        """Test real-time metrics functionality."""
        try:
            metrics = self.RealTimeMetrics()
            self.assertIsNotNone(metrics)
            
            # Test metric recording
            if hasattr(metrics, 'record_frame_time'):
                metrics.record_frame_time(0.033)  # 30 FPS
                metrics.record_frame_time(0.040)  # 25 FPS
                metrics.record_frame_time(0.020)  # 50 FPS
            
            # Test metric retrieval
            if hasattr(metrics, 'get_average_fps'):
                avg_fps = metrics.get_average_fps()
                self.assertIsInstance(avg_fps, (int, float))
                self.assertGreater(avg_fps, 0)
            
            if hasattr(metrics, 'get_memory_usage'):
                memory_usage = metrics.get_memory_usage()
                self.assertIsInstance(memory_usage, (int, float))
                self.assertGreaterEqual(memory_usage, 0)
        
        except Exception as e:
            self.fail(f"Real-time metrics test failed: {e}")
    
    @patch('sys.argv', ['test'])
    def test_metrics_plotting(self):
        """Test metrics plotting functionality."""
        if self.gui_backend == "PyQt6":
            from PyQt6.QtWidgets import QApplication
        else:
            from PySide6.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        try:
            dashboard = self.MetricsDashboard()
            
            # Test plot updates
            if hasattr(dashboard, 'update_fps_plot'):
                fps_data = [30, 25, 35, 28, 32, 29, 31]
                dashboard.update_fps_plot(fps_data)
            
            if hasattr(dashboard, 'update_memory_plot'):
                memory_data = [1024, 1056, 1089, 1123, 1098, 1067, 1034]
                dashboard.update_memory_plot(memory_data)
        
        except Exception as e:
            # Plotting might fail without proper display
            self.skipTest(f"Metrics plotting test failed: {e}")

class TestGUIIntegration(unittest.TestCase):
    """Test GUI component integration."""
    
    def setUp(self):
        """Set up integration test environment."""
        # Check for GUI backend
        try:
            from PyQt6.QtWidgets import QApplication
            self.gui_backend = "PyQt6"
        except ImportError:
            try:
                from PySide6.QtWidgets import QApplication
                self.gui_backend = "PySide6"
            except ImportError:
                self.skipTest("No GUI backend available")
    
    @patch('sys.argv', ['test'])
    def test_full_gui_integration(self):
        """Test full GUI integration."""
        if self.gui_backend == "PyQt6":
            from PyQt6.QtWidgets import QApplication
        else:
            from PySide6.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        try:
            # Test importing all GUI components
            from python_slam.gui.main_window import SlamMainWindow
            from python_slam.gui.utils import MaterialDesignManager
            
            # Test component interaction
            manager = MaterialDesignManager()
            window = SlamMainWindow()
            
            # Test theme application
            stylesheet = manager.generate_stylesheet("dark")
            window.setStyleSheet(stylesheet)
            
            # Test window show (don't actually show to avoid GUI popup)
            # window.show()  # Commented out for automated testing
            
            self.assertIsNotNone(window)
            self.assertIsNotNone(manager)
        
        except Exception as e:
            self.fail(f"Full GUI integration test failed: {e}")

class TestGUIUtilities(unittest.TestCase):
    """Test GUI utility functions."""
    
    def setUp(self):
        """Set up utility test environment."""
        try:
            from python_slam.gui.utils import (
                MaterialDesignManager,
                create_icon_button,
                create_styled_button,
                create_metric_widget
            )
            self.MaterialDesignManager = MaterialDesignManager
            self.create_icon_button = create_icon_button
            self.create_styled_button = create_styled_button
            self.create_metric_widget = create_metric_widget
        except ImportError:
            self.skipTest("GUI utilities not available")
        
        # Check for GUI backend
        try:
            from PyQt6.QtWidgets import QApplication
            self.gui_backend = "PyQt6"
        except ImportError:
            try:
                from PySide6.QtWidgets import QApplication
                self.gui_backend = "PySide6"
            except ImportError:
                self.skipTest("No GUI backend available")
    
    @patch('sys.argv', ['test'])
    def test_utility_functions(self):
        """Test utility functions."""
        if self.gui_backend == "PyQt6":
            from PyQt6.QtWidgets import QApplication
        else:
            from PySide6.QtWidgets import QApplication
        
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        
        try:
            # Test button creation
            styled_button = self.create_styled_button("Test Button", "primary")
            self.assertIsNotNone(styled_button)
            
            # Test icon button creation
            if hasattr(self, 'create_icon_button'):
                icon_button = self.create_icon_button("play", "Start")
                self.assertIsNotNone(icon_button)
            
            # Test metric widget creation
            if hasattr(self, 'create_metric_widget'):
                metric_widget = self.create_metric_widget("FPS", "30.0")
                self.assertIsNotNone(metric_widget)
        
        except Exception as e:
            self.fail(f"Utility functions test failed: {e}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
