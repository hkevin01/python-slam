#!/usr/bin/env python3
"""
Enhanced Visualization Node - Advanced GUI for Defense SLAM System
Provides comprehensive visualization with defense-oriented capabilities
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import sys
import json
import threading
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import Image, PointCloud2

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                                QHBoxLayout, QWidget, QLabel, QTextEdit,
                                QPushButton, QTabWidget, QGridLayout,
                                QProgressBar, QStatusBar, QFrame)
    from PyQt5.QtCore import QTimer, pyqtSignal, QObject
    from PyQt5.QtGui import QFont, QColor, QPalette
    PyQt5_available = True
except ImportError:
    PyQt5_available = False
    print("Warning: PyQt5 not available - visualization disabled")

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.patches as patches
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    print("Warning: Matplotlib not available - plotting disabled")

import numpy as np


class DefenseGUI(QMainWindow):
    """Enhanced GUI for defense SLAM operations"""

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.init_ui()
        self.setup_timers()

        # Data storage
        self.pose_history = []
        self.threat_data = []
        self.mission_status = "STANDBY"
        self.classification_level = "UNCLASSIFIED"

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("Defense SLAM System - Enhanced Visualization")
        self.setGeometry(100, 100, 1400, 900)

        # Set dark theme for defense applications
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: #ffffff; }
            QTabWidget::pane { border: 1px solid #555555; }
            QTabBar::tab { background-color: #404040; color: #ffffff; padding: 8px; }
            QTabBar::tab:selected { background-color: #005500; }
            QLabel { color: #ffffff; }
            QTextEdit { background-color: #404040; color: #00ff00; font-family: monospace; }
            QPushButton { background-color: #005500; color: #ffffff; padding: 8px; }
            QPushButton:hover { background-color: #007700; }
            QProgressBar { background-color: #404040; }
            QProgressBar::chunk { background-color: #00aa00; }
        """)

        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Classification banner
        self.classification_label = QLabel(f"CLASSIFICATION: {self.classification_level}")
        self.classification_label.setStyleSheet("""
            QLabel {
                background-color: #ff0000;
                color: #ffffff;
                font-weight: bold;
                font-size: 16px;
                padding: 10px;
                text-align: center;
            }
        """)
        layout.addWidget(self.classification_label)

        # Tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Create tabs
        self.create_overview_tab()
        self.create_mapping_tab()
        self.create_mission_tab()
        self.create_threats_tab()
        self.create_telemetry_tab()
        self.create_system_tab()

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Defense SLAM System - Operational")

    def create_overview_tab(self):
        """Create system overview tab"""
        tab = QWidget()
        layout = QGridLayout(tab)

        # System status
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.Box)
        status_layout = QVBoxLayout(status_frame)

        status_layout.addWidget(QLabel("System Status"))
        self.system_status_text = QTextEdit()
        self.system_status_text.setMaximumHeight(150)
        status_layout.addWidget(self.system_status_text)

        layout.addWidget(status_frame, 0, 0)

        # Position display
        pos_frame = QFrame()
        pos_frame.setFrameStyle(QFrame.Box)
        pos_layout = QVBoxLayout(pos_frame)

        pos_layout.addWidget(QLabel("Current Position"))
        self.position_text = QTextEdit()
        self.position_text.setMaximumHeight(150)
        pos_layout.addWidget(self.position_text)

        layout.addWidget(pos_frame, 0, 1)

        # Mission status
        mission_frame = QFrame()
        mission_frame.setFrameStyle(QFrame.Box)
        mission_layout = QVBoxLayout(mission_frame)

        mission_layout.addWidget(QLabel("Mission Status"))
        self.mission_status_text = QTextEdit()
        self.mission_status_text.setMaximumHeight(150)
        mission_layout.addWidget(self.mission_status_text)

        layout.addWidget(mission_frame, 1, 0)

        # Controls
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.Box)
        control_layout = QVBoxLayout(control_frame)

        control_layout.addWidget(QLabel("System Controls"))

        self.emergency_btn = QPushButton("EMERGENCY STOP")
        self.emergency_btn.setStyleSheet("QPushButton { background-color: #cc0000; }")
        control_layout.addWidget(self.emergency_btn)

        self.reset_btn = QPushButton("Reset SLAM")
        control_layout.addWidget(self.reset_btn)

        self.calibrate_btn = QPushButton("Calibrate Sensors")
        control_layout.addWidget(self.calibrate_btn)

        layout.addWidget(control_frame, 1, 1)

        self.tab_widget.addTab(tab, "Overview")

    def create_mapping_tab(self):
        """Create mapping visualization tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        if matplotlib_available:
            # Create matplotlib figure for map display
            self.map_figure = Figure(figsize=(12, 8), facecolor='black')
            self.map_canvas = FigureCanvas(self.map_figure)
            self.map_axes = self.map_figure.add_subplot(111, facecolor='black')

            self.map_axes.set_xlabel('X (meters)', color='white')
            self.map_axes.set_ylabel('Y (meters)', color='white')
            self.map_axes.set_title('SLAM Map with Trajectory', color='white')
            self.map_axes.tick_params(colors='white')

            layout.addWidget(self.map_canvas)
        else:
            layout.addWidget(QLabel("Mapping visualization requires matplotlib"))

        self.tab_widget.addTab(tab, "Mapping")

    def create_mission_tab(self):
        """Create mission management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Mission planning area
        mission_frame = QFrame()
        mission_frame.setFrameStyle(QFrame.Box)
        mission_layout = QVBoxLayout(mission_frame)

        mission_layout.addWidget(QLabel("Mission Planning"))
        self.mission_text = QTextEdit()
        mission_layout.addWidget(self.mission_text)

        # Mission controls
        mission_controls = QHBoxLayout()
        self.load_mission_btn = QPushButton("Load Mission")
        self.start_mission_btn = QPushButton("Start Mission")
        self.abort_mission_btn = QPushButton("Abort Mission")
        self.abort_mission_btn.setStyleSheet("QPushButton { background-color: #cc6600; }")

        mission_controls.addWidget(self.load_mission_btn)
        mission_controls.addWidget(self.start_mission_btn)
        mission_controls.addWidget(self.abort_mission_btn)

        mission_layout.addLayout(mission_controls)
        layout.addWidget(mission_frame)

        # Progress indicators
        progress_frame = QFrame()
        progress_frame.setFrameStyle(QFrame.Box)
        progress_layout = QVBoxLayout(progress_frame)

        progress_layout.addWidget(QLabel("Mission Progress"))

        self.waypoint_progress = QProgressBar()
        progress_layout.addWidget(QLabel("Waypoint Progress"))
        progress_layout.addWidget(self.waypoint_progress)

        self.mission_progress = QProgressBar()
        progress_layout.addWidget(QLabel("Overall Mission"))
        progress_layout.addWidget(self.mission_progress)

        layout.addWidget(progress_frame)

        self.tab_widget.addTab(tab, "Mission")

    def create_threats_tab(self):
        """Create threat monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Threat display
        threat_frame = QFrame()
        threat_frame.setFrameStyle(QFrame.Box)
        threat_layout = QVBoxLayout(threat_frame)

        threat_layout.addWidget(QLabel("Threat Assessment"))
        self.threat_text = QTextEdit()
        self.threat_text.setStyleSheet("QTextEdit { color: #ffaa00; }")
        threat_layout.addWidget(self.threat_text)

        layout.addWidget(threat_frame)

        # Threat controls
        threat_controls = QHBoxLayout()
        self.acknowledge_btn = QPushButton("Acknowledge Threats")
        self.evasive_btn = QPushButton("Evasive Maneuvers")
        self.evasive_btn.setStyleSheet("QPushButton { background-color: #cc6600; }")

        threat_controls.addWidget(self.acknowledge_btn)
        threat_controls.addWidget(self.evasive_btn)

        layout.addLayout(threat_controls)

        self.tab_widget.addTab(tab, "Threats")

    def create_telemetry_tab(self):
        """Create telemetry monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Telemetry display
        telemetry_frame = QFrame()
        telemetry_frame.setFrameStyle(QFrame.Box)
        telemetry_layout = QVBoxLayout(telemetry_frame)

        telemetry_layout.addWidget(QLabel("System Telemetry"))
        self.telemetry_text = QTextEdit()
        self.telemetry_text.setStyleSheet("QTextEdit { color: #00aaff; }")
        telemetry_layout.addWidget(self.telemetry_text)

        layout.addWidget(telemetry_frame)

        self.tab_widget.addTab(tab, "Telemetry")

    def create_system_tab(self):
        """Create system diagnostics tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # System diagnostics
        diag_frame = QFrame()
        diag_frame.setFrameStyle(QFrame.Box)
        diag_layout = QVBoxLayout(diag_frame)

        diag_layout.addWidget(QLabel("System Diagnostics"))
        self.diagnostics_text = QTextEdit()
        diag_layout.addWidget(self.diagnostics_text)

        layout.addWidget(diag_frame)

        # System controls
        system_controls = QHBoxLayout()
        self.restart_slam_btn = QPushButton("Restart SLAM")
        self.restart_px4_btn = QPushButton("Restart PX4")
        self.restart_uci_btn = QPushButton("Restart UCI")

        system_controls.addWidget(self.restart_slam_btn)
        system_controls.addWidget(self.restart_px4_btn)
        system_controls.addWidget(self.restart_uci_btn)

        layout.addLayout(system_controls)

        self.tab_widget.addTab(tab, "System")

    def setup_timers(self):
        """Setup GUI update timers"""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_displays)
        self.update_timer.start(100)  # Update every 100ms

    def update_displays(self):
        """Update all display elements"""
        try:
            # Update system status
            status_text = f"SLAM Status: {self.mission_status}\n"
            status_text += f"Classification: {self.classification_level}\n"
            status_text += f"Pose History: {len(self.pose_history)} points\n"
            status_text += f"Active Threats: {len(self.threat_data)}\n"
            self.system_status_text.setPlainText(status_text)

            # Update position display
            if self.pose_history:
                latest_pose = self.pose_history[-1]
                pos_text = f"X: {latest_pose[0]:.3f} m\n"
                pos_text += f"Y: {latest_pose[1]:.3f} m\n"
                pos_text += f"Z: {latest_pose[2]:.3f} m\n"
                self.position_text.setPlainText(pos_text)

            # Update map visualization
            if matplotlib_available and hasattr(self, 'map_axes'):
                self.update_map_display()

        except Exception as e:
            print(f"GUI update error: {e}")

    def update_map_display(self):
        """Update the map visualization"""
        try:
            if not self.pose_history:
                return

            self.map_axes.clear()

            # Plot trajectory
            if len(self.pose_history) > 1:
                x_coords = [pose[0] for pose in self.pose_history]
                y_coords = [pose[1] for pose in self.pose_history]

                self.map_axes.plot(x_coords, y_coords, 'g-', linewidth=2, label='Trajectory')
                self.map_axes.plot(x_coords[-1], y_coords[-1], 'ro', markersize=8, label='Current Position')
                self.map_axes.plot(x_coords[0], y_coords[0], 'bo', markersize=8, label='Start Position')

            # Plot threats if any
            for threat in self.threat_data:
                if 'position' in threat:
                    pos = threat['position']
                    circle = patches.Circle((pos[0], pos[1]), 10, color='red', alpha=0.3)
                    self.map_axes.add_patch(circle)

            self.map_axes.set_xlabel('X (meters)', color='white')
            self.map_axes.set_ylabel('Y (meters)', color='white')
            self.map_axes.set_title('SLAM Map with Trajectory', color='white')
            self.map_axes.tick_params(colors='white')
            self.map_axes.legend()
            self.map_axes.grid(True, alpha=0.3)

            self.map_canvas.draw()

        except Exception as e:
            print(f"Map display update error: {e}")

    def add_pose(self, pose):
        """Add a new pose to the history"""
        self.pose_history.append([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
        if len(self.pose_history) > 1000:  # Limit history
            self.pose_history.pop(0)

    def add_threat(self, threat_data):
        """Add threat information"""
        self.threat_data.append(threat_data)

        # Update threat display
        threat_text = self.threat_text.toPlainText()
        threat_text += f"[{threat_data.get('timestamp', 'N/A')}] {threat_data.get('type', 'UNKNOWN')}: {threat_data.get('description', 'No description')}\n"
        self.threat_text.setPlainText(threat_text)

    def update_classification(self, level):
        """Update security classification"""
        self.classification_level = level
        self.classification_label.setText(f"CLASSIFICATION: {level}")

        # Update color based on classification
        colors = {
            'UNCLASSIFIED': '#008800',
            'CONFIDENTIAL': '#0088ff',
            'SECRET': '#ff8800',
            'TOP SECRET': '#ff0000'
        }
        color = colors.get(level, '#ff0000')
        self.classification_label.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: #ffffff;
                font-weight: bold;
                font-size: 16px;
                padding: 10px;
                text-align: center;
            }}
        """)


class EnhancedVisualizationNode(Node):
    """ROS2 node for enhanced defense visualization"""

    def __init__(self):
        super().__init__('enhanced_visualization_node')

        # Parameters
        self.declare_parameter('enable_3d_viz', True)
        self.declare_parameter('enable_feature_viz', True)
        self.declare_parameter('enable_metrics_viz', True)
        self.declare_parameter('enable_defense_viz', True)

        # QoS profiles
        self.mission_critical_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )

        # Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/slam/pose',
            self.pose_callback,
            self.mission_critical_qos
        )

        self.threat_sub = self.create_subscription(
            String,
            '/uci/threats',
            self.threat_callback,
            self.mission_critical_qos
        )

        self.status_sub = self.create_subscription(
            String,
            '/slam/status',
            self.status_callback,
            self.mission_critical_qos
        )

        # Initialize GUI if available
        self.gui = None
        if PyQt5_available:
            self.app = QApplication.instance()
            if self.app is None:
                self.app = QApplication(sys.argv)

            self.gui = DefenseGUI(self)
            self.gui.show()

        self.get_logger().info("Enhanced Visualization Node started")

    def pose_callback(self, msg: PoseStamped):
        """Handle pose updates"""
        if self.gui:
            self.gui.add_pose(msg)

    def threat_callback(self, msg: String):
        """Handle threat updates"""
        try:
            threat_data = json.loads(msg.data)
            if self.gui:
                self.gui.add_threat(threat_data)
        except json.JSONDecodeError:
            self.get_logger().warning("Invalid threat data received")

    def status_callback(self, msg: String):
        """Handle status updates"""
        try:
            status_data = json.loads(msg.data)
            if self.gui and 'classification' in status_data:
                self.gui.update_classification(status_data['classification'])
        except json.JSONDecodeError:
            pass  # Ignore non-JSON status messages


def main(args=None):
    rclpy.init(args=args)

    if not PyQt5_available:
        print("Enhanced visualization requires PyQt5 - install with: pip install PyQt5")
        return

    try:
        node = EnhancedVisualizationNode()

        # Run both ROS2 and Qt event loops
        executor = rclpy.executors.MultiThreadedExecutor()
        executor.add_node(node)

        # Start ROS2 executor in separate thread
        ros_thread = threading.Thread(target=executor.spin, daemon=True)
        ros_thread.start()

        # Run Qt event loop
        if node.gui:
            sys.exit(node.app.exec_())

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Enhanced Visualization Node error: {e}")
    finally:
        if 'executor' in locals():
            executor.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
