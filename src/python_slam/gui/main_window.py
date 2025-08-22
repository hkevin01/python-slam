"""
Main Window for Python SLAM GUI

Modern PyQt6/PySide6 implementation with Material Design styling.
"""

import sys
import os
from typing import Optional, Dict, Any, List
import json
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtOpenGL import *
    PYQT_VERSION = 6
except ImportError:
    try:
        from PyQt5.QtWidgets import *
        from PyQt5.QtCore import *
        from PyQt5.QtGui import *
        from PyQt5.QtOpenGL import *
        PYQT_VERSION = 5
    except ImportError:
        try:
            from PySide6.QtWidgets import *
            from PySide6.QtCore import *
            from PySide6.QtGui import *
            from PySide6.QtOpenGL import *
            PYQT_VERSION = 6
        except ImportError:
            try:
                from PySide2.QtWidgets import *
                from PySide2.QtCore import *
                from PySide2.QtGui import *
                from PySide2.QtOpenGL import *
                PYQT_VERSION = 5
            except ImportError:
                raise ImportError("No PyQt or PySide installation found")

import numpy as np
import cv2

# Import SLAM interfaces
from ..slam_interfaces import SLAMFactory, SLAMConfiguration, SensorType
from .visualization import Map3DViewer, PointCloudRenderer, TrajectoryViewer
from .control_panels import AlgorithmControlPanel, ParameterTuningPanel, RecordingControlPanel
from .metrics_dashboard import MetricsDashboard
from .utils import MaterialDesign, ThemeManager


class SlamMainWindow(QMainWindow):
    """
    Main window for Python SLAM GUI with modern Material Design interface.

    Features:
    - 3D visualization with OpenGL
    - Real-time control panels
    - Metrics dashboard
    - Multi-view layouts
    - Algorithm benchmarking
    """

    def __init__(self, config_file: Optional[str] = None):
        super().__init__()

        # Initialize core components
        self.slam_factory = SLAMFactory()
        self.current_slam = None
        self.theme_manager = ThemeManager()
        self.config = self._load_config(config_file)

        # Initialize UI
        self.init_ui()
        self.setup_connections()
        self.apply_theme()

        # Initialize timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_visualization)
        self.update_timer.start(33)  # ~30 FPS

        # Status tracking
        self.is_recording = False
        self.is_playing = False
        self.current_dataset = None

    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load GUI configuration from file."""
        default_config = {
            "window": {
                "width": 1600,
                "height": 1000,
                "title": "Python SLAM - Advanced Visualization",
                "theme": "dark"
            },
            "visualization": {
                "point_size": 2.0,
                "trajectory_width": 3.0,
                "keyframe_size": 5.0,
                "update_rate": 30
            },
            "algorithms": {
                "default": "orb_slam3",
                "available": ["orb_slam3", "rtabmap", "cartographer", "openvslam", "python_slam"]
            }
        }

        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Could not load config file {config_file}: {e}")

        return default_config

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle(self.config["window"]["title"])
        self.setGeometry(100, 100,
                        self.config["window"]["width"],
                        self.config["window"]["height"])

        # Create central widget with splitter layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)

        # Left panel (controls)
        self.left_panel = self.create_left_panel()
        main_splitter.addWidget(self.left_panel)

        # Center panel (visualization)
        self.center_panel = self.create_center_panel()
        main_splitter.addWidget(self.center_panel)

        # Right panel (metrics)
        self.right_panel = self.create_right_panel()
        main_splitter.addWidget(self.right_panel)

        # Set splitter proportions
        main_splitter.setSizes([300, 1000, 300])

        # Create menu bar
        self.create_menu_bar()

        # Create status bar
        self.create_status_bar()

        # Create toolbars
        self.create_toolbars()

    def create_left_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        panel.setFixedWidth(300)
        panel.setStyleSheet(MaterialDesign.get_panel_style())

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Algorithm Control Panel
        self.algorithm_panel = AlgorithmControlPanel(self.slam_factory)
        layout.addWidget(self.algorithm_panel)

        # Parameter Tuning Panel
        self.parameter_panel = ParameterTuningPanel()
        layout.addWidget(self.parameter_panel)

        # Recording Control Panel
        self.recording_panel = RecordingControlPanel()
        layout.addWidget(self.recording_panel)

        # Dataset Selection Panel
        self.dataset_panel = self.create_dataset_panel()
        layout.addWidget(self.dataset_panel)

        layout.addStretch()

        return panel

    def create_center_panel(self) -> QWidget:
        """Create the center visualization panel."""
        panel = QWidget()
        panel.setStyleSheet(MaterialDesign.get_panel_style())

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Create tab widget for different views
        self.view_tabs = QTabWidget()
        self.view_tabs.setStyleSheet(MaterialDesign.get_tab_style())

        # 3D Map View
        self.map_3d_viewer = Map3DViewer()
        self.view_tabs.addTab(self.map_3d_viewer, "3D Map")

        # Split View (Camera + Map)
        self.split_view = self.create_split_view()
        self.view_tabs.addTab(self.split_view, "Split View")

        # Point Cloud View
        self.pointcloud_viewer = PointCloudRenderer()
        self.view_tabs.addTab(self.pointcloud_viewer, "Point Cloud")

        # Trajectory View
        self.trajectory_viewer = TrajectoryViewer()
        self.view_tabs.addTab(self.trajectory_viewer, "Trajectory")

        # AR View
        self.ar_view = self.create_ar_view()
        self.view_tabs.addTab(self.ar_view, "AR Mode")

        layout.addWidget(self.view_tabs)

        # Add view controls
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(8, 8, 8, 8)

        # View mode buttons
        self.fullscreen_btn = QPushButton("Fullscreen")
        self.fullscreen_btn.setStyleSheet(MaterialDesign.get_button_style())
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        controls_layout.addWidget(self.fullscreen_btn)

        self.pip_btn = QPushButton("Picture-in-Picture")
        self.pip_btn.setStyleSheet(MaterialDesign.get_button_style())
        self.pip_btn.clicked.connect(self.toggle_pip_mode)
        controls_layout.addWidget(self.pip_btn)

        controls_layout.addStretch()

        # Synchronized views checkbox
        self.sync_views_cb = QCheckBox("Synchronized Views")
        self.sync_views_cb.setStyleSheet(MaterialDesign.get_checkbox_style())
        self.sync_views_cb.setChecked(True)
        controls_layout.addWidget(self.sync_views_cb)

        layout.addLayout(controls_layout)

        return panel

    def create_right_panel(self) -> QWidget:
        """Create the right metrics panel."""
        panel = QWidget()
        panel.setFixedWidth(300)
        panel.setStyleSheet(MaterialDesign.get_panel_style())

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Metrics Dashboard
        self.metrics_dashboard = MetricsDashboard()
        layout.addWidget(self.metrics_dashboard)

        return panel

    def create_split_view(self) -> QWidget:
        """Create split-screen camera/map view."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Create splitter for camera and map
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Camera view with feature overlay
        self.camera_view = QLabel()
        self.camera_view.setMinimumHeight(200)
        self.camera_view.setStyleSheet("border: 1px solid #333; background: black;")
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setText("Camera Feed")
        splitter.addWidget(self.camera_view)

        # Map view
        map_container = QWidget()
        map_layout = QVBoxLayout(map_container)
        map_layout.setContentsMargins(0, 0, 0, 0)

        self.mini_map_viewer = Map3DViewer()
        map_layout.addWidget(self.mini_map_viewer)

        splitter.addWidget(map_container)

        layout.addWidget(splitter)

        return widget

    def create_ar_view(self) -> QWidget:
        """Create AR visualization mode."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)

        # AR view label
        ar_label = QLabel("Augmented Reality View")
        ar_label.setStyleSheet(MaterialDesign.get_label_style())
        ar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ar_label)

        # AR canvas
        self.ar_canvas = QLabel()
        self.ar_canvas.setStyleSheet("border: 1px solid #333; background: black;")
        self.ar_canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ar_canvas.setText("AR Visualization")
        layout.addWidget(self.ar_canvas)

        # AR controls
        ar_controls = QHBoxLayout()

        self.ar_overlay_cb = QCheckBox("Show Overlay")
        self.ar_overlay_cb.setStyleSheet(MaterialDesign.get_checkbox_style())
        self.ar_overlay_cb.setChecked(True)
        ar_controls.addWidget(self.ar_overlay_cb)

        self.ar_tracking_cb = QCheckBox("Feature Tracking")
        self.ar_tracking_cb.setStyleSheet(MaterialDesign.get_checkbox_style())
        self.ar_tracking_cb.setChecked(True)
        ar_controls.addWidget(self.ar_tracking_cb)

        ar_controls.addStretch()

        layout.addLayout(ar_controls)

        return widget

    def create_dataset_panel(self) -> QWidget:
        """Create dataset selection panel."""
        panel = QGroupBox("Dataset")
        panel.setStyleSheet(MaterialDesign.get_groupbox_style())

        layout = QVBoxLayout(panel)

        # Dataset selection combo
        self.dataset_combo = QComboBox()
        self.dataset_combo.setStyleSheet(MaterialDesign.get_combo_style())
        self.dataset_combo.addItems([
            "Live Camera",
            "KITTI Dataset",
            "TUM Dataset",
            "EuRoC Dataset",
            "Custom Dataset"
        ])
        layout.addWidget(self.dataset_combo)

        # Browse button
        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet(MaterialDesign.get_button_style())
        browse_btn.clicked.connect(self.browse_dataset)
        layout.addWidget(browse_btn)

        # Dataset info
        self.dataset_info = QTextEdit()
        self.dataset_info.setMaximumHeight(80)
        self.dataset_info.setStyleSheet(MaterialDesign.get_textedit_style())
        self.dataset_info.setPlainText("No dataset loaded")
        layout.addWidget(self.dataset_info)

        return panel

    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')

        new_action = QAction('New Session', self)
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.new_session)
        file_menu.addAction(new_action)

        open_action = QAction('Open Dataset', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.browse_dataset)
        file_menu.addAction(open_action)

        save_action = QAction('Save Map', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_map)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        export_action = QAction('Export Results', self)
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)

        # View menu
        view_menu = menubar.addMenu('View')

        theme_action = QAction('Toggle Theme', self)
        theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(theme_action)

        layout_action = QAction('Reset Layout', self)
        layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(layout_action)

        # Tools menu
        tools_menu = menubar.addMenu('Tools')

        benchmark_action = QAction('Benchmark', self)
        benchmark_action.triggered.connect(self.start_benchmark)
        tools_menu.addAction(benchmark_action)

        calibration_action = QAction('Camera Calibration', self)
        calibration_action.triggered.connect(self.camera_calibration)
        tools_menu.addAction(calibration_action)

        # Help menu
        help_menu = menubar.addMenu('Help')

        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_status_bar(self):
        """Create the status bar."""
        self.status_bar = self.statusBar()

        # Algorithm status
        self.algorithm_status = QLabel("Algorithm: None")
        self.status_bar.addWidget(self.algorithm_status)

        # FPS indicator
        self.fps_status = QLabel("FPS: 0")
        self.status_bar.addPermanentWidget(self.fps_status)

        # Processing status
        self.processing_status = QLabel("Status: Ready")
        self.status_bar.addPermanentWidget(self.processing_status)

    def create_toolbars(self):
        """Create toolbars."""
        # Main toolbar
        main_toolbar = self.addToolBar('Main')

        # Play/Pause action
        self.play_action = QAction('Play', self)
        self.play_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_action.triggered.connect(self.toggle_playback)
        main_toolbar.addAction(self.play_action)

        # Stop action
        stop_action = QAction('Stop', self)
        stop_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        stop_action.triggered.connect(self.stop_playback)
        main_toolbar.addAction(stop_action)

        main_toolbar.addSeparator()

        # Record action
        self.record_action = QAction('Record', self)
        self.record_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        self.record_action.triggered.connect(self.toggle_recording)
        main_toolbar.addAction(self.record_action)

        main_toolbar.addSeparator()

        # Reset action
        reset_action = QAction('Reset', self)
        reset_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        reset_action.triggered.connect(self.reset_slam)
        main_toolbar.addAction(reset_action)

    def setup_connections(self):
        """Setup signal-slot connections."""
        # Algorithm panel connections
        self.algorithm_panel.algorithm_changed.connect(self.on_algorithm_changed)

        # Parameter panel connections
        self.parameter_panel.parameter_changed.connect(self.on_parameter_changed)

        # Recording panel connections
        self.recording_panel.recording_started.connect(self.start_recording)
        self.recording_panel.recording_stopped.connect(self.stop_recording)
        self.recording_panel.playback_started.connect(self.start_playback)
        self.recording_panel.playback_stopped.connect(self.stop_playback)

        # Dataset combo connection
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)

    def apply_theme(self):
        """Apply the current theme."""
        theme = self.config["window"]["theme"]
        self.theme_manager.apply_theme(self, theme)

    def update_visualization(self):
        """Update visualization components."""
        if not self.current_slam:
            return

        try:
            # Update 3D viewer
            if hasattr(self.map_3d_viewer, 'update_data'):
                pose = self.current_slam.get_pose()
                map_points = self.current_slam.get_map()
                self.map_3d_viewer.update_data(pose, map_points)

            # Update metrics
            if hasattr(self.metrics_dashboard, 'update_metrics'):
                metrics = self.current_slam.get_performance_metrics()
                self.metrics_dashboard.update_metrics(metrics)

            # Update status bar
            if hasattr(self.current_slam, 'get_performance_metrics'):
                metrics = self.current_slam.get_performance_metrics()
                self.fps_status.setText(f"FPS: {metrics.get('fps', 0):.1f}")

        except Exception as e:
            print(f"Error updating visualization: {e}")

    # Event handlers
    def on_algorithm_changed(self, algorithm_name: str):
        """Handle algorithm change."""
        try:
            config = SLAMConfiguration(
                algorithm_name=algorithm_name,
                sensor_type=SensorType.MONOCULAR  # Default, can be changed
            )
            self.current_slam = self.slam_factory.create_algorithm(config)
            self.algorithm_status.setText(f"Algorithm: {algorithm_name}")
            self.processing_status.setText("Status: Algorithm Changed")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to change algorithm: {e}")

    def on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter change."""
        if self.current_slam and hasattr(self.current_slam, 'update_parameter'):
            self.current_slam.update_parameter(param_name, value)

    def on_dataset_changed(self, dataset_name: str):
        """Handle dataset change."""
        self.dataset_info.setPlainText(f"Selected: {dataset_name}")
        # Implementation depends on dataset loading logic

    def toggle_playback(self):
        """Toggle playback state."""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_action.setText('Pause')
            self.play_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            self.play_action.setText('Play')
            self.play_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def stop_playback(self):
        """Stop playback."""
        self.is_playing = False
        self.play_action.setText('Play')
        self.play_action.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))

    def toggle_recording(self):
        """Toggle recording state."""
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.record_action.setText('Stop Recording')
            self.processing_status.setText("Status: Recording")
        else:
            self.record_action.setText('Record')
            self.processing_status.setText("Status: Ready")

    def reset_slam(self):
        """Reset SLAM system."""
        if self.current_slam:
            self.current_slam.reset()
            self.processing_status.setText("Status: Reset")

    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def toggle_pip_mode(self):
        """Toggle picture-in-picture mode."""
        # Implementation for PiP mode
        pass

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        current_theme = self.config["window"]["theme"]
        new_theme = "light" if current_theme == "dark" else "dark"
        self.config["window"]["theme"] = new_theme
        self.apply_theme()

    def reset_layout(self):
        """Reset window layout to default."""
        # Implementation for layout reset
        pass

    def start_benchmark(self):
        """Start benchmarking process."""
        # Implementation for benchmarking
        pass

    def camera_calibration(self):
        """Open camera calibration dialog."""
        # Implementation for camera calibration
        pass

    def browse_dataset(self):
        """Browse for dataset file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "All Files (*)"
        )
        if file_path:
            self.current_dataset = file_path
            self.dataset_info.setPlainText(f"Loaded: {os.path.basename(file_path)}")

    def new_session(self):
        """Start new SLAM session."""
        self.reset_slam()
        self.processing_status.setText("Status: New Session")

    def save_map(self):
        """Save current map."""
        if self.current_slam:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getSaveFileName(
                self,
                "Save Map",
                "map.dat",
                "Map Files (*.dat)"
            )
            if file_path:
                self.current_slam.save_map(file_path)
                self.processing_status.setText("Status: Map Saved")

    def export_results(self):
        """Export results to file."""
        # Implementation for results export
        pass

    def start_recording(self):
        """Start recording session."""
        self.is_recording = True
        self.processing_status.setText("Status: Recording")

    def stop_recording(self):
        """Stop recording session."""
        self.is_recording = False
        self.processing_status.setText("Status: Ready")

    def start_playback(self):
        """Start playback of recorded session."""
        self.is_playing = True
        self.processing_status.setText("Status: Playing")

    def stop_playback(self):
        """Stop playback."""
        self.is_playing = False
        self.processing_status.setText("Status: Ready")

    def show_about(self):
        """Show about dialog."""
        about_text = """
        <h2>Python SLAM</h2>
        <p>Advanced Multi-Algorithm SLAM Framework</p>
        <p>Version 1.0.0</p>
        <p>Built with PyQt6 and Modern Material Design</p>
        """
        QMessageBox.about(self, "About Python SLAM", about_text)

    def closeEvent(self, event):
        """Handle application close."""
        if self.current_slam:
            self.current_slam.reset()
        event.accept()


def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Python SLAM")
    app.setApplicationVersion("1.0.0")

    # Set application icon
    app.setWindowIcon(QIcon())  # Add icon path when available

    # Create and show main window
    window = SlamMainWindow()
    window.show()

    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
