"""
Metrics Dashboard for Python SLAM GUI

Real-time performance monitoring and metrics visualization.
"""

import sys
import time
import psutil
from typing import Dict, Any, List, Optional
from collections import deque
import numpy as np

try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    PYQT_VERSION = 6
except ImportError:
    try:
        from PyQt5.QtWidgets import *
        from PyQt5.QtCore import *
        from PyQt5.QtGui import *
        PYQT_VERSION = 5
    except ImportError:
        raise ImportError("PyQt6 or PyQt5 required")

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib not available, using basic charts")

from .utils import MaterialDesign


class MetricsDashboard(QWidget):
    """
    Main metrics dashboard showing real-time SLAM performance.

    Features:
    - FPS and processing time monitoring
    - Memory and CPU usage graphs
    - Feature tracking statistics
    - Loop closure detection info
    - Pose uncertainty visualization
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data storage
        self.max_history = 100
        self.metrics_history = {
            'fps': deque(maxlen=self.max_history),
            'processing_time': deque(maxlen=self.max_history),
            'memory_usage': deque(maxlen=self.max_history),
            'cpu_usage': deque(maxlen=self.max_history),
            'feature_count': deque(maxlen=self.max_history),
            'loop_closures': deque(maxlen=self.max_history),
            'timestamps': deque(maxlen=self.max_history)
        }

        self.current_metrics = {}

        self.init_ui()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_system_metrics)
        self.update_timer.start(1000)  # Update every second

    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Performance Monitor
        self.performance_monitor = PerformanceMonitor()
        layout.addWidget(self.performance_monitor)

        # Resource Monitor
        self.resource_monitor = ResourceMonitor()
        layout.addWidget(self.resource_monitor)

        # SLAM Metrics
        self.slam_metrics = SlamMetricsWidget()
        layout.addWidget(self.slam_metrics)

        # Uncertainty Visualization
        self.uncertainty_widget = UncertaintyWidget()
        layout.addWidget(self.uncertainty_widget)

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update dashboard with new metrics."""
        self.current_metrics = metrics

        # Update timestamp
        current_time = time.time()
        self.metrics_history['timestamps'].append(current_time)

        # Update metric history
        for key in ['fps', 'processing_time', 'feature_count', 'loop_closures']:
            value = metrics.get(key, 0)
            self.metrics_history[key].append(value)

        # Update individual components
        self.performance_monitor.update_metrics(metrics)
        self.slam_metrics.update_metrics(metrics)
        self.uncertainty_widget.update_metrics(metrics)

    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            self.metrics_history['cpu_usage'].append(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.metrics_history['memory_usage'].append(memory_percent)

            # Update resource monitor
            self.resource_monitor.update_metrics({
                'cpu_usage': cpu_percent,
                'memory_usage': memory_percent,
                'memory_available': memory.available / (1024**3),  # GB
                'memory_total': memory.total / (1024**3)  # GB
            })

        except Exception as e:
            print(f"Error updating system metrics: {e}")


class PerformanceMonitor(QGroupBox):
    """
    Performance monitoring widget showing FPS and processing times.
    """

    def __init__(self, parent=None):
        super().__init__("Performance", parent)
        self.setStyleSheet(MaterialDesign.get_groupbox_style())

        self.init_ui()

    def init_ui(self):
        """Initialize the UI."""
        layout = QGridLayout(self)

        # FPS display
        fps_label = QLabel("FPS:")
        fps_label.setStyleSheet(MaterialDesign.get_label_style())
        layout.addWidget(fps_label, 0, 0)

        self.fps_value = QLabel("0.0")
        self.fps_value.setStyleSheet(MaterialDesign.get_value_label_style())
        layout.addWidget(self.fps_value, 0, 1)

        # Processing time
        proc_label = QLabel("Proc Time:")
        proc_label.setStyleSheet(MaterialDesign.get_label_style())
        layout.addWidget(proc_label, 1, 0)

        self.proc_time_value = QLabel("0 ms")
        self.proc_time_value.setStyleSheet(MaterialDesign.get_value_label_style())
        layout.addWidget(self.proc_time_value, 1, 1)

        # Average FPS
        avg_fps_label = QLabel("Avg FPS:")
        avg_fps_label.setStyleSheet(MaterialDesign.get_label_style())
        layout.addWidget(avg_fps_label, 2, 0)

        self.avg_fps_value = QLabel("0.0")
        self.avg_fps_value.setStyleSheet(MaterialDesign.get_value_label_style())
        layout.addWidget(self.avg_fps_value, 2, 1)

        # Frame drops
        drops_label = QLabel("Drops:")
        drops_label.setStyleSheet(MaterialDesign.get_label_style())
        layout.addWidget(drops_label, 3, 0)

        self.drops_value = QLabel("0")
        self.drops_value.setStyleSheet(MaterialDesign.get_value_label_style())
        layout.addWidget(self.drops_value, 3, 1)

        # Performance chart
        if MATPLOTLIB_AVAILABLE:
            self.chart = PerformanceChart()
            layout.addWidget(self.chart, 4, 0, 1, 2)

        self.fps_history = deque(maxlen=50)
        self.frame_drops = 0

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update performance metrics."""
        fps = metrics.get('fps', 0.0)
        processing_time = metrics.get('processing_time', 0.0)

        # Update displays
        self.fps_value.setText(f"{fps:.1f}")
        self.proc_time_value.setText(f"{processing_time:.1f} ms")

        # Track FPS history
        self.fps_history.append(fps)

        # Calculate average FPS
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            self.avg_fps_value.setText(f"{avg_fps:.1f}")

        # Count frame drops (FPS < 20)
        if fps < 20 and fps > 0:
            self.frame_drops += 1
            self.drops_value.setText(str(self.frame_drops))

        # Update chart
        if MATPLOTLIB_AVAILABLE and hasattr(self, 'chart'):
            self.chart.update_data(list(self.fps_history))


class ResourceMonitor(QGroupBox):
    """
    System resource monitoring widget.
    """

    def __init__(self, parent=None):
        super().__init__("System Resources", parent)
        self.setStyleSheet(MaterialDesign.get_groupbox_style())

        self.init_ui()

    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)

        # CPU usage
        cpu_layout = QHBoxLayout()
        cpu_label = QLabel("CPU:")
        cpu_label.setStyleSheet(MaterialDesign.get_label_style())
        cpu_layout.addWidget(cpu_label)

        self.cpu_progress = QProgressBar()
        self.cpu_progress.setStyleSheet(MaterialDesign.get_progressbar_style())
        cpu_layout.addWidget(self.cpu_progress)

        self.cpu_value = QLabel("0%")
        self.cpu_value.setStyleSheet(MaterialDesign.get_value_label_style())
        cpu_layout.addWidget(self.cpu_value)

        layout.addLayout(cpu_layout)

        # Memory usage
        mem_layout = QHBoxLayout()
        mem_label = QLabel("Memory:")
        mem_label.setStyleSheet(MaterialDesign.get_label_style())
        mem_layout.addWidget(mem_label)

        self.mem_progress = QProgressBar()
        self.mem_progress.setStyleSheet(MaterialDesign.get_progressbar_style())
        mem_layout.addWidget(self.mem_progress)

        self.mem_value = QLabel("0%")
        self.mem_value.setStyleSheet(MaterialDesign.get_value_label_style())
        mem_layout.addWidget(self.mem_value)

        layout.addLayout(mem_layout)

        # Memory details
        self.mem_details = QLabel("Available: 0 GB / Total: 0 GB")
        self.mem_details.setStyleSheet(MaterialDesign.get_small_label_style())
        layout.addWidget(self.mem_details)

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update resource metrics."""
        cpu_usage = metrics.get('cpu_usage', 0)
        memory_usage = metrics.get('memory_usage', 0)
        memory_available = metrics.get('memory_available', 0)
        memory_total = metrics.get('memory_total', 0)

        # Update CPU
        self.cpu_progress.setValue(int(cpu_usage))
        self.cpu_value.setText(f"{cpu_usage:.1f}%")

        # Update Memory
        self.mem_progress.setValue(int(memory_usage))
        self.mem_value.setText(f"{memory_usage:.1f}%")
        self.mem_details.setText(f"Available: {memory_available:.1f} GB / Total: {memory_total:.1f} GB")


class SlamMetricsWidget(QGroupBox):
    """
    SLAM-specific metrics widget.
    """

    def __init__(self, parent=None):
        super().__init__("SLAM Metrics", parent)
        self.setStyleSheet(MaterialDesign.get_groupbox_style())

        self.init_ui()

    def init_ui(self):
        """Initialize the UI."""
        layout = QGridLayout(self)

        # Features tracked
        features_label = QLabel("Features:")
        features_label.setStyleSheet(MaterialDesign.get_label_style())
        layout.addWidget(features_label, 0, 0)

        self.features_value = QLabel("0")
        self.features_value.setStyleSheet(MaterialDesign.get_value_label_style())
        layout.addWidget(self.features_value, 0, 1)

        # Keyframes
        keyframes_label = QLabel("Keyframes:")
        keyframes_label.setStyleSheet(MaterialDesign.get_label_style())
        layout.addWidget(keyframes_label, 1, 0)

        self.keyframes_value = QLabel("0")
        self.keyframes_value.setStyleSheet(MaterialDesign.get_value_label_style())
        layout.addWidget(self.keyframes_value, 1, 1)

        # Map points
        points_label = QLabel("Map Points:")
        points_label.setStyleSheet(MaterialDesign.get_label_style())
        layout.addWidget(points_label, 2, 0)

        self.points_value = QLabel("0")
        self.points_value.setStyleSheet(MaterialDesign.get_value_label_style())
        layout.addWidget(self.points_value, 2, 1)

        # Loop closures
        loops_label = QLabel("Loop Closures:")
        loops_label.setStyleSheet(MaterialDesign.get_label_style())
        layout.addWidget(loops_label, 3, 0)

        self.loops_value = QLabel("0")
        self.loops_value.setStyleSheet(MaterialDesign.get_value_label_style())
        layout.addWidget(self.loops_value, 3, 1)

        # Tracking status
        status_label = QLabel("Status:")
        status_label.setStyleSheet(MaterialDesign.get_label_style())
        layout.addWidget(status_label, 4, 0)

        self.status_value = QLabel("Not Tracking")
        self.status_value.setStyleSheet(MaterialDesign.get_value_label_style())
        layout.addWidget(self.status_value, 4, 1)

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update SLAM metrics."""
        features = metrics.get('feature_count', 0)
        keyframes = metrics.get('keyframe_count', 0)
        map_points = metrics.get('map_point_count', 0)
        loop_closures = metrics.get('loop_closures', 0)
        tracking_status = metrics.get('tracking_status', 'Unknown')

        self.features_value.setText(str(features))
        self.keyframes_value.setText(str(keyframes))
        self.points_value.setText(str(map_points))
        self.loops_value.setText(str(loop_closures))
        self.status_value.setText(tracking_status)

        # Color code status
        if tracking_status == "Tracking":
            self.status_value.setStyleSheet(MaterialDesign.get_value_label_style("success"))
        elif tracking_status == "Lost":
            self.status_value.setStyleSheet(MaterialDesign.get_value_label_style("danger"))
        else:
            self.status_value.setStyleSheet(MaterialDesign.get_value_label_style())


class UncertaintyWidget(QGroupBox):
    """
    Pose uncertainty visualization widget.
    """

    def __init__(self, parent=None):
        super().__init__("Pose Uncertainty", parent)
        self.setStyleSheet(MaterialDesign.get_groupbox_style())

        self.init_ui()

    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)

        # Uncertainty canvas
        self.canvas = QLabel()
        self.canvas.setMinimumHeight(150)
        self.canvas.setStyleSheet("border: 1px solid #333; background: black;")
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.canvas)

        # Uncertainty values
        values_layout = QGridLayout()

        # Translation uncertainty
        trans_label = QLabel("Translation:")
        trans_label.setStyleSheet(MaterialDesign.get_label_style())
        values_layout.addWidget(trans_label, 0, 0)

        self.trans_value = QLabel("N/A")
        self.trans_value.setStyleSheet(MaterialDesign.get_value_label_style())
        values_layout.addWidget(self.trans_value, 0, 1)

        # Rotation uncertainty
        rot_label = QLabel("Rotation:")
        rot_label.setStyleSheet(MaterialDesign.get_label_style())
        values_layout.addWidget(rot_label, 1, 0)

        self.rot_value = QLabel("N/A")
        self.rot_value.setStyleSheet(MaterialDesign.get_value_label_style())
        values_layout.addWidget(self.rot_value, 1, 1)

        layout.addLayout(values_layout)

        self.uncertainty_history = deque(maxlen=50)

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update uncertainty metrics."""
        translation_uncertainty = metrics.get('translation_uncertainty', 0.0)
        rotation_uncertainty = metrics.get('rotation_uncertainty', 0.0)

        self.trans_value.setText(f"{translation_uncertainty:.3f} m")
        self.rot_value.setText(f"{rotation_uncertainty:.3f} rad")

        # Store history
        self.uncertainty_history.append((translation_uncertainty, rotation_uncertainty))

        # Update visualization
        self.render_uncertainty()

    def render_uncertainty(self):
        """Render uncertainty visualization."""
        if not self.uncertainty_history:
            return

        # Create pixmap for drawing
        pixmap = QPixmap(self.canvas.size())
        pixmap.fill(QColor(25, 25, 25))

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw uncertainty over time
        if len(self.uncertainty_history) > 1:
            width = self.canvas.width()
            height = self.canvas.height()

            # Extract translation uncertainties
            trans_uncertainties = [u[0] for u in self.uncertainty_history]

            if max(trans_uncertainties) > 0:
                # Normalize values
                max_uncertainty = max(trans_uncertainties)
                normalized = [u / max_uncertainty * (height - 20) for u in trans_uncertainties]

                # Draw line
                painter.setPen(QPen(QColor(255, 165, 0), 2))  # Orange

                for i in range(len(normalized) - 1):
                    x1 = int(i * width / len(normalized))
                    y1 = int(height - 10 - normalized[i])
                    x2 = int((i + 1) * width / len(normalized))
                    y2 = int(height - 10 - normalized[i + 1])

                    painter.drawLine(x1, y1, x2, y2)

        painter.end()
        self.canvas.setPixmap(pixmap)


if MATPLOTLIB_AVAILABLE:
    class PerformanceChart(FigureCanvas):
        """
        Matplotlib-based performance chart.
        """

        def __init__(self, parent=None):
            self.figure = Figure(figsize=(4, 2), dpi=80)
            super().__init__(self.figure)
            self.setParent(parent)

            self.axes = self.figure.add_subplot(111)
            self.axes.set_facecolor('#1a1a1a')
            self.figure.patch.set_facecolor('#1a1a1a')

            self.axes.set_xlabel('Time', color='white')
            self.axes.set_ylabel('FPS', color='white')
            self.axes.tick_params(colors='white')

            self.line, = self.axes.plot([], [], 'g-', linewidth=2)
            self.axes.set_ylim(0, 60)

        def update_data(self, fps_data: List[float]):
            """Update chart with new FPS data."""
            if not fps_data:
                return

            x_data = list(range(len(fps_data)))
            self.line.set_data(x_data, fps_data)

            self.axes.set_xlim(0, max(len(fps_data), 10))
            self.axes.relim()
            self.axes.autoscale_view()

            self.draw()
else:
    class PerformanceChart(QLabel):
        """
        Fallback performance chart without matplotlib.
        """

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setText("Performance Chart\n(Matplotlib not available)")
            self.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setStyleSheet("border: 1px solid #333; background: black; color: white;")
            self.setMinimumHeight(100)

        def update_data(self, fps_data: List[float]):
            """Placeholder update method."""
            pass
