"""
Control Panels for Python SLAM GUI

Interactive control panels for algorithm selection, parameter tuning,
recording controls, and benchmarking.
"""

import sys
from typing import Dict, Any, List, Optional, Callable
import json
import os
from pathlib import Path

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

# Import SLAM interfaces
from ..slam_interfaces import SLAMFactory, SLAMConfiguration, SensorType
from .utils import MaterialDesign


class AlgorithmControlPanel(QGroupBox):
    """
    Control panel for algorithm selection and configuration.

    Features:
    - Algorithm selection dropdown
    - Sensor type selection
    - Quick algorithm switching
    - Algorithm status display
    """

    # Signals
    algorithm_changed = pyqtSignal(str)
    sensor_type_changed = pyqtSignal(object)
    algorithm_switched = pyqtSignal(str, str)  # from_algorithm, to_algorithm

    def __init__(self, slam_factory: SLAMFactory, parent=None):
        super().__init__("Algorithm Control", parent)

        self.slam_factory = slam_factory
        self.current_algorithm = None
        self.current_sensor_type = SensorType.MONOCULAR

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        """Initialize the UI."""
        self.setStyleSheet(MaterialDesign.get_groupbox_style())

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Algorithm selection
        algo_layout = QFormLayout()

        self.algorithm_combo = QComboBox()
        self.algorithm_combo.setStyleSheet(MaterialDesign.get_combo_style())

        # Populate with available algorithms
        available_algorithms = self.slam_factory.get_available_algorithms()
        self.algorithm_combo.addItems(available_algorithms)

        algo_layout.addRow("Algorithm:", self.algorithm_combo)

        # Sensor type selection
        self.sensor_combo = QComboBox()
        self.sensor_combo.setStyleSheet(MaterialDesign.get_combo_style())
        self.sensor_combo.addItems([
            "Monocular",
            "Stereo",
            "RGB-D",
            "Visual-Inertial",
            "LiDAR",
            "Point Cloud"
        ])

        algo_layout.addRow("Sensor Type:", self.sensor_combo)

        layout.addLayout(algo_layout)

        # Quick switch buttons
        switch_layout = QHBoxLayout()

        self.orb_btn = QPushButton("ORB-SLAM3")
        self.orb_btn.setStyleSheet(MaterialDesign.get_button_style())
        switch_layout.addWidget(self.orb_btn)

        self.rtab_btn = QPushButton("RTAB-Map")
        self.rtab_btn.setStyleSheet(MaterialDesign.get_button_style())
        switch_layout.addWidget(self.rtab_btn)

        layout.addLayout(switch_layout)

        switch_layout2 = QHBoxLayout()

        self.cartographer_btn = QPushButton("Cartographer")
        self.cartographer_btn.setStyleSheet(MaterialDesign.get_button_style())
        switch_layout2.addWidget(self.cartographer_btn)

        self.python_slam_btn = QPushButton("Python SLAM")
        self.python_slam_btn.setStyleSheet(MaterialDesign.get_button_style())
        switch_layout2.addWidget(self.python_slam_btn)

        layout.addLayout(switch_layout2)

        # Algorithm status
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet(MaterialDesign.get_label_style())
        layout.addWidget(self.status_label)

        # Performance info
        self.performance_text = QTextEdit()
        self.performance_text.setMaximumHeight(80)
        self.performance_text.setStyleSheet(MaterialDesign.get_textedit_style())
        self.performance_text.setPlainText("No algorithm selected")
        layout.addWidget(self.performance_text)

    def setup_connections(self):
        """Setup signal-slot connections."""
        self.algorithm_combo.currentTextChanged.connect(self.on_algorithm_changed)
        self.sensor_combo.currentTextChanged.connect(self.on_sensor_changed)

        # Quick switch buttons
        self.orb_btn.clicked.connect(lambda: self.quick_switch("orb_slam3"))
        self.rtab_btn.clicked.connect(lambda: self.quick_switch("rtabmap"))
        self.cartographer_btn.clicked.connect(lambda: self.quick_switch("cartographer"))
        self.python_slam_btn.clicked.connect(lambda: self.quick_switch("python_slam"))

    def on_algorithm_changed(self, algorithm_name: str):
        """Handle algorithm selection change."""
        old_algorithm = self.current_algorithm
        self.current_algorithm = algorithm_name

        self.status_label.setText(f"Status: Switching to {algorithm_name}")
        self.algorithm_changed.emit(algorithm_name)

        if old_algorithm:
            self.algorithm_switched.emit(old_algorithm, algorithm_name)

        self.update_performance_info(algorithm_name)

    def on_sensor_changed(self, sensor_name: str):
        """Handle sensor type change."""
        sensor_map = {
            "Monocular": SensorType.MONOCULAR,
            "Stereo": SensorType.STEREO,
            "RGB-D": SensorType.RGBD,
            "Visual-Inertial": SensorType.VISUAL_INERTIAL,
            "LiDAR": SensorType.LIDAR,
            "Point Cloud": SensorType.POINTCLOUD
        }

        self.current_sensor_type = sensor_map.get(sensor_name, SensorType.MONOCULAR)
        self.sensor_type_changed.emit(self.current_sensor_type)

    def quick_switch(self, algorithm_name: str):
        """Quick switch to specified algorithm."""
        index = self.algorithm_combo.findText(algorithm_name)
        if index >= 0:
            self.algorithm_combo.setCurrentIndex(index)

    def update_performance_info(self, algorithm_name: str):
        """Update performance information display."""
        info_text = f"Algorithm: {algorithm_name}\n"
        info_text += f"Sensor: {self.sensor_combo.currentText()}\n"
        info_text += "Ready to process"

        self.performance_text.setPlainText(info_text)

    def set_algorithm_status(self, status: str):
        """Set algorithm status display."""
        self.status_label.setText(f"Status: {status}")


class ParameterTuningPanel(QGroupBox):
    """
    Panel for real-time parameter tuning.

    Features:
    - Dynamic parameter sliders
    - Parameter presets
    - Real-time parameter updates
    - Parameter saving/loading
    """

    # Signals
    parameter_changed = pyqtSignal(str, object)  # parameter_name, value
    preset_loaded = pyqtSignal(str)  # preset_name

    def __init__(self, parent=None):
        super().__init__("Parameter Tuning", parent)

        self.parameters = {}
        self.parameter_widgets = {}
        self.presets = {}

        self.init_ui()
        self.load_default_parameters()

    def init_ui(self):
        """Initialize the UI."""
        self.setStyleSheet(MaterialDesign.get_groupbox_style())

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Preset selection
        preset_layout = QHBoxLayout()

        self.preset_combo = QComboBox()
        self.preset_combo.setStyleSheet(MaterialDesign.get_combo_style())
        self.preset_combo.addItems(["Default", "High Quality", "Fast", "Custom"])
        preset_layout.addWidget(self.preset_combo)

        save_preset_btn = QPushButton("Save")
        save_preset_btn.setStyleSheet(MaterialDesign.get_button_style())
        save_preset_btn.setMaximumWidth(60)
        save_preset_btn.clicked.connect(self.save_preset)
        preset_layout.addWidget(save_preset_btn)

        layout.addLayout(preset_layout)

        # Scrollable parameter area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(300)

        self.param_widget = QWidget()
        self.param_layout = QVBoxLayout(self.param_widget)

        scroll_area.setWidget(self.param_widget)
        layout.addWidget(scroll_area)

        # Parameter controls
        controls_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset")
        reset_btn.setStyleSheet(MaterialDesign.get_button_style())
        reset_btn.clicked.connect(self.reset_parameters)
        controls_layout.addWidget(reset_btn)

        apply_btn = QPushButton("Apply")
        apply_btn.setStyleSheet(MaterialDesign.get_button_style())
        apply_btn.clicked.connect(self.apply_parameters)
        controls_layout.addWidget(apply_btn)

        layout.addLayout(controls_layout)

    def load_default_parameters(self):
        """Load default parameter set."""
        default_params = {
            "max_features": {"value": 1000, "min": 100, "max": 5000, "type": "int"},
            "scale_factor": {"value": 1.2, "min": 1.1, "max": 2.0, "type": "float"},
            "pyramid_levels": {"value": 8, "min": 4, "max": 16, "type": "int"},
            "edge_threshold": {"value": 19, "min": 5, "max": 50, "type": "int"},
            "patch_size": {"value": 31, "min": 15, "max": 63, "type": "int"},
            "loop_closure_threshold": {"value": 0.7, "min": 0.1, "max": 1.0, "type": "float"},
            "keyframe_threshold": {"value": 0.8, "min": 0.1, "max": 1.0, "type": "float"},
        }

        self.add_parameters(default_params)

    def add_parameters(self, params: Dict[str, Dict[str, Any]]):
        """Add parameters to the panel."""
        for param_name, param_info in params.items():
            self.add_parameter(param_name, param_info)

    def add_parameter(self, name: str, info: Dict[str, Any]):
        """Add a single parameter control."""
        self.parameters[name] = info

        # Create parameter row
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        # Parameter label
        label = QLabel(name.replace("_", " ").title())
        label.setMinimumWidth(120)
        label.setStyleSheet(MaterialDesign.get_label_style())
        row_layout.addWidget(label)

        # Parameter control
        if info["type"] == "int":
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(info["min"])
            slider.setMaximum(info["max"])
            slider.setValue(info["value"])
            slider.valueChanged.connect(
                lambda value, param=name: self.on_parameter_changed(param, value)
            )
            row_layout.addWidget(slider)

            value_label = QLabel(str(info["value"]))
            value_label.setMinimumWidth(40)
            value_label.setStyleSheet(MaterialDesign.get_label_style())
            row_layout.addWidget(value_label)

            self.parameter_widgets[name] = {"slider": slider, "label": value_label}

        elif info["type"] == "float":
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(int(info["min"] * 100))
            slider.setMaximum(int(info["max"] * 100))
            slider.setValue(int(info["value"] * 100))
            slider.valueChanged.connect(
                lambda value, param=name: self.on_parameter_changed(param, value / 100.0)
            )
            row_layout.addWidget(slider)

            value_label = QLabel(f"{info['value']:.2f}")
            value_label.setMinimumWidth(40)
            value_label.setStyleSheet(MaterialDesign.get_label_style())
            row_layout.addWidget(value_label)

            self.parameter_widgets[name] = {"slider": slider, "label": value_label}

        self.param_layout.addWidget(row_widget)

    def on_parameter_changed(self, param_name: str, value: Any):
        """Handle parameter value change."""
        self.parameters[param_name]["value"] = value

        # Update label
        if param_name in self.parameter_widgets:
            widget_info = self.parameter_widgets[param_name]
            if "label" in widget_info:
                if isinstance(value, float):
                    widget_info["label"].setText(f"{value:.2f}")
                else:
                    widget_info["label"].setText(str(value))

        # Emit signal
        self.parameter_changed.emit(param_name, value)

    def reset_parameters(self):
        """Reset all parameters to default values."""
        self.load_default_parameters()
        self.update_ui()

    def apply_parameters(self):
        """Apply current parameter values."""
        for param_name, param_info in self.parameters.items():
            self.parameter_changed.emit(param_name, param_info["value"])

    def save_preset(self):
        """Save current parameters as a preset."""
        preset_name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if ok and preset_name:
            self.presets[preset_name] = self.parameters.copy()
            self.preset_combo.addItem(preset_name)

    def update_ui(self):
        """Update UI to reflect current parameter values."""
        for param_name, widget_info in self.parameter_widgets.items():
            if param_name in self.parameters:
                param_info = self.parameters[param_name]
                value = param_info["value"]

                if param_info["type"] == "float":
                    widget_info["slider"].setValue(int(value * 100))
                    widget_info["label"].setText(f"{value:.2f}")
                else:
                    widget_info["slider"].setValue(value)
                    widget_info["label"].setText(str(value))


class RecordingControlPanel(QGroupBox):
    """
    Panel for recording and playback controls.

    Features:
    - Recording start/stop
    - Playback controls
    - Recording status
    - File management
    """

    # Signals
    recording_started = pyqtSignal()
    recording_stopped = pyqtSignal()
    playback_started = pyqtSignal()
    playback_stopped = pyqtSignal()
    file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("Recording & Playback", parent)

        self.is_recording = False
        self.is_playing = False
        self.current_file = None

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        """Initialize the UI."""
        self.setStyleSheet(MaterialDesign.get_groupbox_style())

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Recording controls
        record_layout = QHBoxLayout()

        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setStyleSheet(MaterialDesign.get_button_style("success"))
        record_layout.addWidget(self.record_btn)

        self.stop_record_btn = QPushButton("Stop")
        self.stop_record_btn.setStyleSheet(MaterialDesign.get_button_style("danger"))
        self.stop_record_btn.setEnabled(False)
        record_layout.addWidget(self.stop_record_btn)

        layout.addLayout(record_layout)

        # File selection
        file_layout = QHBoxLayout()

        self.file_edit = QLineEdit()
        self.file_edit.setStyleSheet(MaterialDesign.get_lineedit_style())
        self.file_edit.setPlaceholderText("Select recording file...")
        file_layout.addWidget(self.file_edit)

        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(MaterialDesign.get_button_style())
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)

        layout.addLayout(file_layout)

        # Playback controls
        playback_layout = QHBoxLayout()

        self.play_btn = QPushButton("Play")
        self.play_btn.setStyleSheet(MaterialDesign.get_button_style("primary"))
        playback_layout.addWidget(self.play_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setStyleSheet(MaterialDesign.get_button_style())
        self.pause_btn.setEnabled(False)
        playback_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(MaterialDesign.get_button_style())
        self.stop_btn.setEnabled(False)
        playback_layout.addWidget(self.stop_btn)

        layout.addLayout(playback_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(MaterialDesign.get_progressbar_style())
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet(MaterialDesign.get_label_style())
        layout.addWidget(self.status_label)

    def setup_connections(self):
        """Setup signal-slot connections."""
        self.record_btn.clicked.connect(self.toggle_recording)
        self.stop_record_btn.clicked.connect(self.stop_recording)

        self.play_btn.clicked.connect(self.start_playback)
        self.pause_btn.clicked.connect(self.pause_playback)
        self.stop_btn.clicked.connect(self.stop_playback)

    def toggle_recording(self):
        """Toggle recording state."""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start recording session."""
        self.is_recording = True

        self.record_btn.setText("Recording...")
        self.record_btn.setEnabled(False)
        self.stop_record_btn.setEnabled(True)

        self.status_label.setText("Status: Recording")
        self.recording_started.emit()

    def stop_recording(self):
        """Stop recording session."""
        self.is_recording = False

        self.record_btn.setText("Start Recording")
        self.record_btn.setEnabled(True)
        self.stop_record_btn.setEnabled(False)

        self.status_label.setText("Status: Recording Stopped")
        self.recording_stopped.emit()

    def start_playback(self):
        """Start playback."""
        if not self.current_file:
            QMessageBox.warning(self, "Warning", "Please select a file first")
            return

        self.is_playing = True

        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

        self.status_label.setText("Status: Playing")
        self.playback_started.emit()

    def pause_playback(self):
        """Pause playback."""
        self.is_playing = False

        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)

        self.status_label.setText("Status: Paused")

    def stop_playback(self):
        """Stop playback."""
        self.is_playing = False

        self.play_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Stopped")
        self.playback_stopped.emit()

    def browse_file(self):
        """Browse for recording file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Recording File",
            "",
            "Bag Files (*.bag);;All Files (*)"
        )

        if file_path:
            self.current_file = file_path
            self.file_edit.setText(file_path)
            self.file_selected.emit(file_path)

    def update_progress(self, progress: float):
        """Update playback progress."""
        self.progress_bar.setValue(int(progress * 100))


class BenchmarkControlPanel(QGroupBox):
    """
    Panel for benchmarking controls.

    Features:
    - Benchmark mode toggle
    - Metric selection
    - Test configuration
    - Results display
    """

    # Signals
    benchmark_started = pyqtSignal(dict)  # config
    benchmark_stopped = pyqtSignal()
    metric_selected = pyqtSignal(str, bool)  # metric_name, enabled

    def __init__(self, parent=None):
        super().__init__("Benchmark", parent)

        self.benchmark_active = False
        self.selected_metrics = set()

        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        """Initialize the UI."""
        self.setStyleSheet(MaterialDesign.get_groupbox_style())

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Benchmark toggle
        self.benchmark_cb = QCheckBox("Enable Benchmark Mode")
        self.benchmark_cb.setStyleSheet(MaterialDesign.get_checkbox_style())
        layout.addWidget(self.benchmark_cb)

        # Metrics selection
        metrics_label = QLabel("Metrics to Track:")
        metrics_label.setStyleSheet(MaterialDesign.get_label_style())
        layout.addWidget(metrics_label)

        metrics_layout = QVBoxLayout()

        self.metrics_checkboxes = {}
        metrics = [
            "Absolute Trajectory Error",
            "Relative Pose Error",
            "Processing Time",
            "Memory Usage",
            "Feature Count",
            "Loop Closures",
            "Map Quality"
        ]

        for metric in metrics:
            cb = QCheckBox(metric)
            cb.setStyleSheet(MaterialDesign.get_checkbox_style())
            self.metrics_checkboxes[metric] = cb
            metrics_layout.addWidget(cb)

        layout.addLayout(metrics_layout)

        # Benchmark controls
        controls_layout = QHBoxLayout()

        self.start_benchmark_btn = QPushButton("Start Benchmark")
        self.start_benchmark_btn.setStyleSheet(MaterialDesign.get_button_style("success"))
        controls_layout.addWidget(self.start_benchmark_btn)

        self.stop_benchmark_btn = QPushButton("Stop")
        self.stop_benchmark_btn.setStyleSheet(MaterialDesign.get_button_style("danger"))
        self.stop_benchmark_btn.setEnabled(False)
        controls_layout.addWidget(self.stop_benchmark_btn)

        layout.addLayout(controls_layout)

        # Results display
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(100)
        self.results_text.setStyleSheet(MaterialDesign.get_textedit_style())
        self.results_text.setPlainText("No benchmark results")
        layout.addWidget(self.results_text)

    def setup_connections(self):
        """Setup signal-slot connections."""
        self.benchmark_cb.toggled.connect(self.on_benchmark_toggled)
        self.start_benchmark_btn.clicked.connect(self.start_benchmark)
        self.stop_benchmark_btn.clicked.connect(self.stop_benchmark)

        # Metrics checkboxes
        for metric, cb in self.metrics_checkboxes.items():
            cb.toggled.connect(
                lambda checked, m=metric: self.on_metric_toggled(m, checked)
            )

    def on_benchmark_toggled(self, enabled: bool):
        """Handle benchmark mode toggle."""
        self.benchmark_active = enabled

        if enabled:
            self.start_benchmark_btn.setEnabled(True)
        else:
            self.start_benchmark_btn.setEnabled(False)
            if self.stop_benchmark_btn.isEnabled():
                self.stop_benchmark()

    def on_metric_toggled(self, metric: str, enabled: bool):
        """Handle metric selection."""
        if enabled:
            self.selected_metrics.add(metric)
        else:
            self.selected_metrics.discard(metric)

        self.metric_selected.emit(metric, enabled)

    def start_benchmark(self):
        """Start benchmark process."""
        if not self.selected_metrics:
            QMessageBox.warning(self, "Warning", "Please select at least one metric")
            return

        config = {
            "metrics": list(self.selected_metrics),
            "timestamp": True,
            "export_format": "csv"
        }

        self.start_benchmark_btn.setEnabled(False)
        self.stop_benchmark_btn.setEnabled(True)

        self.results_text.setPlainText("Benchmark started...")
        self.benchmark_started.emit(config)

    def stop_benchmark(self):
        """Stop benchmark process."""
        self.start_benchmark_btn.setEnabled(True)
        self.stop_benchmark_btn.setEnabled(False)

        self.results_text.setPlainText("Benchmark stopped")
        self.benchmark_stopped.emit()

    def update_results(self, results: Dict[str, Any]):
        """Update benchmark results display."""
        results_text = "Benchmark Results:\n"
        for metric, value in results.items():
            results_text += f"- {metric}: {value}\n"

        self.results_text.setPlainText(results_text)
