"""
Enhanced 3D SLAM Visualizer using PyQt5 and PyOpenGL

This module provides a comprehensive GUI for visualizing SLAM operations including:
- Real-time 3D point cloud visualization
- Camera trajectory tracking
- Feature matching visualization
- Performance metrics monitoring
- Interactive controls for playback and analysis
"""

import sys
import numpy as np
from typing import Optional, List, Tuple
import cv2
import threading
import time
from collections import deque

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                QHBoxLayout, QPushButton, QLabel, QSlider, QTabWidget,
                                QProgressBar, QTextEdit, QGridLayout, QGroupBox,
                                QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
    from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg
    PYQT_AVAILABLE = True
except ImportError:
    print("PyQt5 not available. GUI features will be disabled.")
    PYQT_AVAILABLE = False

class SLAMDataProcessor(QThread):
    """Background thread for processing SLAM data"""

    # Signals for updating GUI
    pointcloud_updated = pyqtSignal(np.ndarray, np.ndarray)
    trajectory_updated = pyqtSignal(np.ndarray)
    frame_updated = pyqtSignal(np.ndarray, np.ndarray)
    metrics_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.running = False
        self.slam_system = None
        self.data_queue = deque(maxlen=1000)

    def set_slam_system(self, slam_system):
        """Set the SLAM system to process data from"""
        self.slam_system = slam_system

    def add_data(self, data_type, data):
        """Add data to processing queue"""
        self.data_queue.append((data_type, data, time.time()))

    def run(self):
        """Main processing loop"""
        self.running = True
        while self.running:
            if self.data_queue:
                data_type, data, timestamp = self.data_queue.popleft()
                self.process_data(data_type, data)
            else:
                self.msleep(10)  # Sleep for 10ms if no data

    def process_data(self, data_type, data):
        """Process different types of SLAM data"""
        if data_type == 'pointcloud':
            points, colors = data
            self.pointcloud_updated.emit(points, colors)
        elif data_type == 'trajectory':
            poses = data
            self.trajectory_updated.emit(poses)
        elif data_type == 'frame':
            frame, features = data
            self.frame_updated.emit(frame, features)
        elif data_type == 'metrics':
            metrics = data
            self.metrics_updated.emit(metrics)

    def stop(self):
        """Stop the processing thread"""
        self.running = False
        self.wait()

class SLAMVisualizer(QMainWindow):
    """Main SLAM visualization window"""

    def __init__(self):
        super().__init__()
        if not PYQT_AVAILABLE:
            raise ImportError("PyQt5 not available for GUI")

        self.setWindowTitle("Python-SLAM Advanced Visualizer")
        self.setGeometry(100, 100, 1600, 1000)

        # Initialize data storage
        self.trajectory_data = []
        self.pointcloud_data = None
        self.current_frame = None
        self.metrics_history = {
            'pose_error': [],
            'tracked_features': [],
            'processing_time': [],
            'timestamps': []
        }

        # Initialize background processor
        self.data_processor = SLAMDataProcessor()
        self.data_processor.pointcloud_updated.connect(self.update_pointcloud)
        self.data_processor.trajectory_updated.connect(self.update_trajectory)
        self.data_processor.frame_updated.connect(self.update_camera_frame)
        self.data_processor.metrics_updated.connect(self.update_metrics)

        # Setup UI
        self.setup_ui()
        self.setup_styles()

        # Timer for regular updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.regular_update)
        self.update_timer.start(50)  # 20 FPS

        # State
        self.is_playing = False
        self.playback_speed = 1.0

    def setup_ui(self):
        """Setup the main UI components"""
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create tab widget for different views
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Setup different visualization tabs
        self.setup_3d_view_tab()
        self.setup_camera_view_tab()
        self.setup_metrics_tab()
        self.setup_control_tab()

        # Control panel at bottom
        self.setup_control_panel(layout)

        # Status bar
        self.setup_status_bar()

    def setup_styles(self):
        """Setup custom styles for the application"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #555555;
                background-color: #3b3b3b;
            }
            QTabBar::tab {
                background-color: #555555;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #0078d4;
            }
            QPushButton {
                background-color: #0078d4;
                color: #ffffff;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QLabel {
                color: #ffffff;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 8px;
                background: #3b3b3b;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 1px solid #0078d4;
                width: 18px;
                border-radius: 9px;
            }
        """)

    def setup_3d_view_tab(self):
        """Setup 3D point cloud and trajectory visualization"""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Left side: 3D view
        view_layout = QVBoxLayout()

        # 3D view widget
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=50, azimuth=45, elevation=30)
        self.gl_widget.setMinimumSize(800, 600)
        view_layout.addWidget(self.gl_widget)

        # 3D view controls
        controls_layout = QHBoxLayout()

        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_3d_view)
        controls_layout.addWidget(self.reset_view_btn)

        self.follow_camera_cb = QCheckBox("Follow Camera")
        self.follow_camera_cb.setChecked(True)
        controls_layout.addWidget(self.follow_camera_cb)

        self.show_grid_cb = QCheckBox("Show Grid")
        self.show_grid_cb.setChecked(True)
        self.show_grid_cb.toggled.connect(self.toggle_grid)
        controls_layout.addWidget(self.show_grid_cb)

        controls_layout.addStretch()
        view_layout.addLayout(controls_layout)
        layout.addLayout(view_layout)

        # Right side: 3D view settings
        settings_layout = QVBoxLayout()

        # Point cloud settings
        pc_group = QGroupBox("Point Cloud")
        pc_layout = QGridLayout(pc_group)

        pc_layout.addWidget(QLabel("Point Size:"), 0, 0)
        self.point_size_spin = QDoubleSpinBox()
        self.point_size_spin.setRange(0.1, 5.0)
        self.point_size_spin.setValue(0.5)
        self.point_size_spin.setSingleStep(0.1)
        pc_layout.addWidget(self.point_size_spin, 0, 1)

        pc_layout.addWidget(QLabel("Max Points:"), 1, 0)
        self.max_points_spin = QSpinBox()
        self.max_points_spin.setRange(1000, 100000)
        self.max_points_spin.setValue(10000)
        self.max_points_spin.setSingleStep(1000)
        pc_layout.addWidget(self.max_points_spin, 1, 1)

        self.show_pointcloud_cb = QCheckBox("Show Point Cloud")
        self.show_pointcloud_cb.setChecked(True)
        pc_layout.addWidget(self.show_pointcloud_cb, 2, 0, 1, 2)

        settings_layout.addWidget(pc_group)

        # Trajectory settings
        traj_group = QGroupBox("Trajectory")
        traj_layout = QGridLayout(traj_group)

        traj_layout.addWidget(QLabel("Line Width:"), 0, 0)
        self.traj_width_spin = QDoubleSpinBox()
        self.traj_width_spin.setRange(1.0, 10.0)
        self.traj_width_spin.setValue(2.0)
        self.traj_width_spin.setSingleStep(0.5)
        traj_layout.addWidget(self.traj_width_spin, 0, 1)

        traj_layout.addWidget(QLabel("Max Length:"), 1, 0)
        self.max_traj_spin = QSpinBox()
        self.max_traj_spin.setRange(100, 10000)
        self.max_traj_spin.setValue(1000)
        traj_layout.addWidget(self.max_traj_spin, 1, 1)

        self.show_trajectory_cb = QCheckBox("Show Trajectory")
        self.show_trajectory_cb.setChecked(True)
        traj_layout.addWidget(self.show_trajectory_cb, 2, 0, 1, 2)

        settings_layout.addWidget(traj_group)

        settings_layout.addStretch()
        layout.addLayout(settings_layout)

        # Initialize 3D items
        self.setup_3d_items()

        self.tabs.addTab(widget, "3D View")

    def setup_3d_items(self):
        """Initialize 3D visualization items"""
        # Grid
        self.grid = gl.GLGridItem()
        self.grid.scale(2, 2, 1)
        self.grid.setColor((0.5, 0.5, 0.5, 0.3))
        self.gl_widget.addItem(self.grid)

        # Coordinate axes
        self.axes = gl.GLAxisItem()
        self.axes.setSize(x=5, y=5, z=5)
        self.gl_widget.addItem(self.axes)

        # Point cloud scatter
        self.point_cloud_item = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]]),
            color=(1, 1, 1, 0.5),
            size=0.5
        )
        self.gl_widget.addItem(self.point_cloud_item)

        # Trajectory line
        self.trajectory_item = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0]]),
            color=(0, 1, 0, 1),
            width=2
        )
        self.gl_widget.addItem(self.trajectory_item)

        # Current camera pose
        self.camera_item = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0]]),
            color=(1, 0, 0, 1),
            width=3
        )
        self.gl_widget.addItem(self.camera_item)

    def setup_camera_view_tab(self):
        """Setup camera feed with feature tracking"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Image display area
        images_layout = QHBoxLayout()

        # Current frame view
        current_group = QGroupBox("Current Frame")
        current_layout = QVBoxLayout(current_group)
        self.current_frame_label = QLabel()
        self.current_frame_label.setMinimumSize(640, 480)
        self.current_frame_label.setStyleSheet("border: 1px solid #555555; background-color: #1e1e1e;")
        self.current_frame_label.setAlignment(Qt.AlignCenter)
        self.current_frame_label.setText("No Frame")
        current_layout.addWidget(self.current_frame_label)
        images_layout.addWidget(current_group)

        # Feature tracking view
        feature_group = QGroupBox("Feature Tracking")
        feature_layout = QVBoxLayout(feature_group)
        self.feature_frame_label = QLabel()
        self.feature_frame_label.setMinimumSize(640, 480)
        self.feature_frame_label.setStyleSheet("border: 1px solid #555555; background-color: #1e1e1e;")
        self.feature_frame_label.setAlignment(Qt.AlignCenter)
        self.feature_frame_label.setText("No Features")
        feature_layout.addWidget(self.feature_frame_label)
        images_layout.addWidget(feature_group)

        layout.addLayout(images_layout)

        # Camera controls
        controls_layout = QHBoxLayout()

        self.save_frame_btn = QPushButton("Save Frame")
        controls_layout.addWidget(self.save_frame_btn)

        self.show_features_cb = QCheckBox("Show Features")
        self.show_features_cb.setChecked(True)
        controls_layout.addWidget(self.show_features_cb)

        self.show_matches_cb = QCheckBox("Show Matches")
        self.show_matches_cb.setChecked(True)
        controls_layout.addWidget(self.show_matches_cb)

        controls_layout.addStretch()

        controls_layout.addWidget(QLabel("Feature Type:"))
        self.feature_type_combo = QComboBox()
        self.feature_type_combo.addItems(["ORB", "SIFT", "SURF", "FAST"])
        controls_layout.addWidget(self.feature_type_combo)

        layout.addLayout(controls_layout)

        self.tabs.addTab(widget, "Camera View")

    def setup_metrics_tab(self):
        """Setup metrics and statistics visualization"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create plot widgets for metrics
        plots_layout = QGridLayout()

        # Pose error plot
        self.pose_error_plot = pg.PlotWidget(title="Pose Error Over Time")
        self.pose_error_plot.setLabel('left', 'Error (m)')
        self.pose_error_plot.setLabel('bottom', 'Frame')
        self.pose_error_plot.setBackground('#2b2b2b')
        plots_layout.addWidget(self.pose_error_plot, 0, 0)

        # Tracked features plot
        self.tracking_plot = pg.PlotWidget(title="Tracked Features")
        self.tracking_plot.setLabel('left', 'Features')
        self.tracking_plot.setLabel('bottom', 'Frame')
        self.tracking_plot.setBackground('#2b2b2b')
        plots_layout.addWidget(self.tracking_plot, 0, 1)

        # Processing time plot
        self.timing_plot = pg.PlotWidget(title="Processing Time")
        self.timing_plot.setLabel('left', 'Time (ms)')
        self.timing_plot.setLabel('bottom', 'Frame')
        self.timing_plot.setBackground('#2b2b2b')
        plots_layout.addWidget(self.timing_plot, 1, 0)

        # System resources plot
        self.resources_plot = pg.PlotWidget(title="System Resources")
        self.resources_plot.setLabel('left', 'Usage (%)')
        self.resources_plot.setLabel('bottom', 'Time')
        self.resources_plot.setBackground('#2b2b2b')
        plots_layout.addWidget(self.resources_plot, 1, 1)

        layout.addLayout(plots_layout)

        # Statistics panel
        stats_layout = QHBoxLayout()

        # Current statistics
        current_stats_group = QGroupBox("Current Statistics")
        current_stats_layout = QGridLayout(current_stats_group)

        self.current_frame_label_stats = QLabel("Frame: 0")
        current_stats_layout.addWidget(self.current_frame_label_stats, 0, 0)

        self.current_features_label = QLabel("Features: 0")
        current_stats_layout.addWidget(self.current_features_label, 0, 1)

        self.current_fps_label = QLabel("FPS: 0.0")
        current_stats_layout.addWidget(self.current_fps_label, 1, 0)

        self.current_pose_label = QLabel("Pose: [0, 0, 0]")
        current_stats_layout.addWidget(self.current_pose_label, 1, 1)

        stats_layout.addWidget(current_stats_group)

        # Overall statistics
        overall_stats_group = QGroupBox("Overall Statistics")
        overall_stats_layout = QGridLayout(overall_stats_group)

        self.total_frames_label = QLabel("Total Frames: 0")
        overall_stats_layout.addWidget(self.total_frames_label, 0, 0)

        self.avg_features_label = QLabel("Avg Features: 0")
        overall_stats_layout.addWidget(self.avg_features_label, 0, 1)

        self.total_distance_label = QLabel("Distance: 0.0m")
        overall_stats_layout.addWidget(self.total_distance_label, 1, 0)

        self.avg_fps_label = QLabel("Avg FPS: 0.0")
        overall_stats_layout.addWidget(self.avg_fps_label, 1, 1)

        stats_layout.addWidget(overall_stats_group)

        layout.addLayout(stats_layout)

        self.tabs.addTab(widget, "Metrics")

    def setup_control_tab(self):
        """Setup control and configuration tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # SLAM parameters
        slam_group = QGroupBox("SLAM Parameters")
        slam_layout = QGridLayout(slam_group)

        slam_layout.addWidget(QLabel("Max Features:"), 0, 0)
        self.max_features_spin = QSpinBox()
        self.max_features_spin.setRange(100, 5000)
        self.max_features_spin.setValue(1000)
        slam_layout.addWidget(self.max_features_spin, 0, 1)

        slam_layout.addWidget(QLabel("Quality Level:"), 1, 0)
        self.quality_level_spin = QDoubleSpinBox()
        self.quality_level_spin.setRange(0.001, 0.1)
        self.quality_level_spin.setValue(0.01)
        self.quality_level_spin.setDecimals(3)
        slam_layout.addWidget(self.quality_level_spin, 1, 1)

        slam_layout.addWidget(QLabel("Min Distance:"), 2, 0)
        self.min_distance_spin = QSpinBox()
        self.min_distance_spin.setRange(1, 50)
        self.min_distance_spin.setValue(10)
        slam_layout.addWidget(self.min_distance_spin, 2, 1)

        self.loop_closure_cb = QCheckBox("Enable Loop Closure")
        self.loop_closure_cb.setChecked(True)
        slam_layout.addWidget(self.loop_closure_cb, 3, 0, 1, 2)

        layout.addWidget(slam_group)

        # Recording controls
        recording_group = QGroupBox("Recording")
        recording_layout = QGridLayout(recording_group)

        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.toggle_recording)
        recording_layout.addWidget(self.record_btn, 0, 0)

        self.save_map_btn = QPushButton("Save Map")
        recording_layout.addWidget(self.save_map_btn, 0, 1)

        self.load_map_btn = QPushButton("Load Map")
        recording_layout.addWidget(self.load_map_btn, 1, 0)

        self.export_traj_btn = QPushButton("Export Trajectory")
        recording_layout.addWidget(self.export_traj_btn, 1, 1)

        layout.addWidget(recording_group)

        layout.addStretch()

        self.tabs.addTab(widget, "Controls")

    def setup_control_panel(self, parent_layout):
        """Setup control buttons and sliders at bottom"""
        control_layout = QHBoxLayout()

        # Playback controls
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        control_layout.addWidget(self.play_button)

        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(self.step_frame)
        control_layout.addWidget(self.step_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_slam)
        control_layout.addWidget(self.reset_button)

        # Speed control
        control_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 50)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.speed_changed)
        control_layout.addWidget(self.speed_slider)

        self.speed_label = QLabel("1.0x")
        control_layout.addWidget(self.speed_label)

        # Progress bar
        control_layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        control_layout.addWidget(self.progress_bar)

        # Add stretch to push controls to the left
        control_layout.addStretch()

        parent_layout.addLayout(control_layout)

    def setup_status_bar(self):
        """Setup status bar at bottom"""
        self.statusBar().showMessage("Ready")
        self.statusBar().setStyleSheet("background-color: #3b3b3b; color: #ffffff;")

    @pyqtSlot(np.ndarray, np.ndarray)
    def update_pointcloud(self, points, colors):
        """Update 3D point cloud"""
        if len(points) > 0 and self.show_pointcloud_cb.isChecked():
            # Limit number of points for performance
            max_points = self.max_points_spin.value()
            if len(points) > max_points:
                indices = np.random.choice(len(points), max_points, replace=False)
                points = points[indices]
                colors = colors[indices]

            # Update point size
            size = self.point_size_spin.value()
            self.point_cloud_item.setData(pos=points, color=colors, size=size)

    @pyqtSlot(np.ndarray)
    def update_trajectory(self, poses):
        """Update camera trajectory"""
        if len(poses) > 0 and self.show_trajectory_cb.isChecked():
            # Limit trajectory length
            max_length = self.max_traj_spin.value()
            if len(poses) > max_length:
                poses = poses[-max_length:]

            # Update line width
            width = self.traj_width_spin.value()
            self.trajectory_item.setData(pos=poses, width=width)

            # Update camera pose visualization
            if len(poses) > 0:
                current_pose = poses[-1]
                self.update_camera_pose(current_pose)

            # Follow camera if enabled
            if self.follow_camera_cb.isChecked() and len(poses) > 0:
                self.follow_camera(poses[-1])

    @pyqtSlot(np.ndarray, np.ndarray)
    def update_camera_frame(self, frame, features):
        """Update camera view with current frame"""
        if frame is not None:
            # Convert frame to QImage and display
            height, width = frame.shape[:2]
            if len(frame.shape) == 3:
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            else:
                bytes_per_line = width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

            # Scale image to fit label
            pixmap = QPixmap.fromImage(q_image).scaled(
                self.current_frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.current_frame_label.setPixmap(pixmap)

            # Update feature frame if features are available
            if features is not None and len(features) > 0 and self.show_features_cb.isChecked():
                self.update_feature_frame(frame, features)

    def update_feature_frame(self, frame, features):
        """Update feature tracking visualization"""
        feature_frame = frame.copy()

        # Draw features
        for feature in features:
            cv2.circle(feature_frame, tuple(feature.astype(int)), 3, (0, 255, 0), -1)

        # Convert to QImage and display
        height, width = feature_frame.shape[:2]
        if len(feature_frame.shape) == 3:
            bytes_per_line = 3 * width
            q_image = QImage(feature_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:
            bytes_per_line = width
            q_image = QImage(feature_frame.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_image).scaled(
            self.feature_frame_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.feature_frame_label.setPixmap(pixmap)

    @pyqtSlot(dict)
    def update_metrics(self, metrics):
        """Update metrics plots and statistics"""
        timestamp = time.time()

        # Update metrics history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

        self.metrics_history['timestamps'].append(timestamp)

        # Limit history length
        max_history = 1000
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > max_history:
                self.metrics_history[key] = self.metrics_history[key][-max_history:]

        # Update plots
        self.update_metrics_plots()

        # Update statistics labels
        self.update_statistics_labels(metrics)

    def update_metrics_plots(self):
        """Update all metrics plots"""
        timestamps = self.metrics_history['timestamps']

        if len(timestamps) > 1:
            # Pose error plot
            if self.metrics_history['pose_error']:
                self.pose_error_plot.clear()
                self.pose_error_plot.plot(
                    timestamps, self.metrics_history['pose_error'],
                    pen='y', name='Pose Error'
                )

            # Tracked features plot
            if self.metrics_history['tracked_features']:
                self.tracking_plot.clear()
                self.tracking_plot.plot(
                    timestamps, self.metrics_history['tracked_features'],
                    pen='g', name='Features'
                )

            # Processing time plot
            if self.metrics_history['processing_time']:
                self.timing_plot.clear()
                self.timing_plot.plot(
                    timestamps, self.metrics_history['processing_time'],
                    pen='r', name='Processing Time'
                )

    def update_statistics_labels(self, metrics):
        """Update statistics labels"""
        frame_count = len(self.metrics_history['timestamps'])

        # Current statistics
        self.current_frame_label_stats.setText(f"Frame: {frame_count}")
        self.current_features_label.setText(f"Features: {metrics.get('tracked_features', 0)}")
        self.current_fps_label.setText(f"FPS: {metrics.get('fps', 0.0):.1f}")

        if 'pose' in metrics:
            pose = metrics['pose']
            self.current_pose_label.setText(f"Pose: [{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}]")

        # Overall statistics
        self.total_frames_label.setText(f"Total Frames: {frame_count}")

        if self.metrics_history['tracked_features']:
            avg_features = np.mean(self.metrics_history['tracked_features'])
            self.avg_features_label.setText(f"Avg Features: {avg_features:.0f}")

    def update_camera_pose(self, pose):
        """Update camera pose visualization in 3D view"""
        # Create camera frustum visualization
        frustum_points = self.create_camera_frustum(pose)
        self.camera_item.setData(pos=frustum_points, color=(1, 0, 0, 1))

    def create_camera_frustum(self, pose):
        """Create camera frustum points for visualization"""
        # Simple camera frustum (pyramid)
        scale = 1.0
        points = np.array([
            [0, 0, 0],  # Camera center
            [-scale, -scale, scale],  # Top-left
            [scale, -scale, scale],   # Top-right
            [scale, scale, scale],    # Bottom-right
            [-scale, scale, scale],   # Bottom-left
            [0, 0, 0],  # Back to center
        ])

        # Transform points by camera pose
        # Note: This is simplified - in practice, you'd use the full pose matrix
        transformed_points = points + pose[:3]

        return transformed_points

    def follow_camera(self, pose):
        """Update 3D view to follow camera"""
        center = pose[:3]
        self.gl_widget.setCameraPosition(pos=center, distance=20)

    def reset_3d_view(self):
        """Reset 3D view to default position"""
        self.gl_widget.setCameraPosition(distance=50, azimuth=45, elevation=30)

    def toggle_grid(self, show):
        """Toggle grid visibility"""
        self.grid.setVisible(show)

    def toggle_playback(self):
        """Toggle SLAM playback"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_button.setText("Pause")
            if not self.data_processor.isRunning():
                self.data_processor.start()
        else:
            self.play_button.setText("Play")

    def step_frame(self):
        """Process single frame"""
        # Implement single frame stepping
        pass

    def reset_slam(self):
        """Reset SLAM system"""
        # Clear visualizations
        self.point_cloud_item.setData(pos=np.array([[0, 0, 0]]))
        self.trajectory_item.setData(pos=np.array([[0, 0, 0]]))

        # Clear metrics history
        for key in self.metrics_history:
            self.metrics_history[key].clear()

        # Clear plots
        self.pose_error_plot.clear()
        self.tracking_plot.clear()
        self.timing_plot.clear()

        # Reset labels
        self.current_frame_label.setText("No Frame")
        self.feature_frame_label.setText("No Features")

        self.statusBar().showMessage("SLAM Reset")

    def speed_changed(self, value):
        """Handle speed slider change"""
        self.playback_speed = value / 10.0
        self.speed_label.setText(f"{self.playback_speed:.1f}x")

    def toggle_recording(self):
        """Toggle recording state"""
        if self.record_btn.text() == "Start Recording":
            self.record_btn.setText("Stop Recording")
            self.statusBar().showMessage("Recording...")
        else:
            self.record_btn.setText("Start Recording")
            self.statusBar().showMessage("Recording stopped")

    def regular_update(self):
        """Regular update function called by timer"""
        # Update progress bar based on some metric
        # This is a placeholder - implement based on your SLAM system
        pass

    def closeEvent(self, event):
        """Handle window close event"""
        if self.data_processor.isRunning():
            self.data_processor.stop()
        event.accept()

def create_slam_gui(slam_system=None):
    """Factory function to create SLAM GUI"""
    if not PYQT_AVAILABLE:
        print("PyQt5 not available. Cannot create GUI.")
        return None

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    visualizer = SLAMVisualizer()

    if slam_system is not None:
        visualizer.data_processor.set_slam_system(slam_system)

    return visualizer, app

if __name__ == "__main__":
    # Test the GUI
    visualizer, app = create_slam_gui()
    if visualizer is not None:
        visualizer.show()
        sys.exit(app.exec_())
