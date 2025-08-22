"""
3D Visualization Components for Python SLAM

OpenGL-based 3D viewers for maps, point clouds, trajectories, and keyframes.
"""

import sys
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import time

try:
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtOpenGL import *
    from PyQt6.QtOpenGLWidgets import QOpenGLWidget
    PYQT_VERSION = 6
except ImportError:
    try:
        from PyQt5.QtWidgets import *
        from PyQt5.QtCore import *
        from PyQt5.QtGui import *
        from PyQt5.QtOpenGL import *
        PYQT_VERSION = 5
    except ImportError:
        raise ImportError("PyQt6 or PyQt5 required for visualization")

try:
    import OpenGL.GL as gl
    import OpenGL.GLU as glu
    from OpenGL.arrays import vbo
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("Warning: OpenGL not available, using fallback rendering")

# Import SLAM data structures
from ..slam_interfaces import SLAMPose, SLAMMapPoint, SLAMTrajectory


@dataclass
class RenderSettings:
    """Rendering settings for 3D visualization."""
    point_size: float = 2.0
    trajectory_width: float = 3.0
    keyframe_size: float = 5.0
    background_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
    point_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    trajectory_color: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    keyframe_color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    enable_lighting: bool = True
    enable_depth_test: bool = True


class Map3DViewer(QOpenGLWidget):
    """
    3D Map viewer with OpenGL rendering.
    
    Features:
    - Real-time point cloud rendering
    - Camera trajectory visualization
    - Keyframe display
    - Interactive camera controls
    - Heatmap overlay for mapping density
    """
    
    # Signals
    pose_clicked = pyqtSignal(object)  # Emitted when a pose is clicked
    point_clicked = pyqtSignal(object)  # Emitted when a point is clicked
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Rendering data
        self.map_points: List[SLAMMapPoint] = []
        self.trajectory: Optional[SLAMTrajectory] = None
        self.current_pose: Optional[SLAMPose] = None
        self.keyframes: List[SLAMPose] = []
        
        # Rendering settings
        self.settings = RenderSettings()
        
        # Camera controls
        self.camera_distance = 10.0
        self.camera_azimuth = 0.0
        self.camera_elevation = 30.0
        self.camera_target = np.array([0.0, 0.0, 0.0])
        
        # Mouse interaction
        self.last_mouse_pos = QPoint()
        self.mouse_buttons = Qt.MouseButton.NoButton
        
        # VBOs for efficient rendering
        self.point_vbo = None
        self.trajectory_vbo = None
        self.keyframe_vbo = None
        
        # Rendering flags
        self.show_points = True
        self.show_trajectory = True
        self.show_keyframes = True
        self.show_current_pose = True
        self.show_heatmap = False
        
        # Performance tracking
        self.last_frame_time = time.time()
        self.frame_count = 0
        
    def initializeGL(self):
        """Initialize OpenGL settings."""
        if not OPENGL_AVAILABLE:
            return
            
        # Enable depth testing
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LESS)
        
        # Enable blending for transparency
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        # Enable point size control
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        
        # Set background color
        bg_color = self.settings.background_color
        gl.glClearColor(bg_color[0], bg_color[1], bg_color[2], 1.0)
        
        # Setup lighting
        if self.settings.enable_lighting:
            self.setup_lighting()
    
    def setup_lighting(self):
        """Setup OpenGL lighting."""
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_LIGHT0)
        
        # Light position
        light_pos = [5.0, 5.0, 5.0, 1.0]
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_pos)
        
        # Light colors
        white_light = [1.0, 1.0, 1.0, 1.0]
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, white_light)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, white_light)
        
        # Material properties
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
        gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 50.0)
    
    def resizeGL(self, width, height):
        """Handle window resize."""
        if not OPENGL_AVAILABLE:
            return
            
        gl.glViewport(0, 0, width, height)
        
        # Set perspective projection
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        aspect_ratio = width / height if height > 0 else 1.0
        glu.gluPerspective(45.0, aspect_ratio, 0.1, 1000.0)
        
        gl.glMatrixMode(gl.GL_MODELVIEW)
    
    def paintGL(self):
        """Render the 3D scene."""
        if not OPENGL_AVAILABLE:
            self.render_fallback()
            return
        
        # Clear buffers
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # Setup camera
        gl.glLoadIdentity()
        self.setup_camera()
        
        # Render components
        if self.show_points and self.map_points:
            self.render_map_points()
        
        if self.show_trajectory and self.trajectory:
            self.render_trajectory()
        
        if self.show_keyframes and self.keyframes:
            self.render_keyframes()
        
        if self.show_current_pose and self.current_pose:
            self.render_current_pose()
        
        if self.show_heatmap:
            self.render_heatmap()
        
        # Render coordinate axes
        self.render_axes()
        
        # Update performance metrics
        self.update_performance_metrics()
    
    def setup_camera(self):
        """Setup camera transformation."""
        # Calculate camera position
        camera_x = self.camera_distance * np.cos(np.radians(self.camera_elevation)) * np.cos(np.radians(self.camera_azimuth))
        camera_y = self.camera_distance * np.cos(np.radians(self.camera_elevation)) * np.sin(np.radians(self.camera_azimuth))
        camera_z = self.camera_distance * np.sin(np.radians(self.camera_elevation))
        
        camera_pos = self.camera_target + np.array([camera_x, camera_y, camera_z])
        
        # Set camera view
        glu.gluLookAt(
            camera_pos[0], camera_pos[1], camera_pos[2],  # Camera position
            self.camera_target[0], self.camera_target[1], self.camera_target[2],  # Target
            0.0, 0.0, 1.0  # Up vector
        )
    
    def render_map_points(self):
        """Render map points as colored dots."""
        if not self.map_points:
            return
        
        gl.glPointSize(self.settings.point_size)
        gl.glColor3f(*self.settings.point_color)
        
        gl.glBegin(gl.GL_POINTS)
        for point in self.map_points:
            # Color based on confidence
            confidence = point.confidence
            gl.glColor3f(confidence, confidence, 1.0)
            gl.glVertex3f(point.position[0], point.position[1], point.position[2])
        gl.glEnd()
    
    def render_trajectory(self):
        """Render camera trajectory as a line."""
        if not self.trajectory or len(self.trajectory.poses) < 2:
            return
        
        gl.glLineWidth(self.settings.trajectory_width)
        gl.glColor3f(*self.settings.trajectory_color)
        
        gl.glBegin(gl.GL_LINE_STRIP)
        for pose in self.trajectory.poses:
            gl.glVertex3f(pose.position[0], pose.position[1], pose.position[2])
        gl.glEnd()
    
    def render_keyframes(self):
        """Render keyframes as larger colored points."""
        if not self.keyframes:
            return
        
        gl.glPointSize(self.settings.keyframe_size)
        gl.glColor3f(*self.settings.keyframe_color)
        
        gl.glBegin(gl.GL_POINTS)
        for keyframe in self.keyframes:
            gl.glVertex3f(keyframe.position[0], keyframe.position[1], keyframe.position[2])
        gl.glEnd()
    
    def render_current_pose(self):
        """Render current camera pose with orientation."""
        if not self.current_pose:
            return
        
        pos = self.current_pose.position
        
        # Draw camera frustum
        gl.glPushMatrix()
        gl.glTranslatef(pos[0], pos[1], pos[2])
        
        # Apply rotation from quaternion
        quat = self.current_pose.orientation
        # Convert quaternion to rotation matrix and apply
        # (Implementation depends on specific quaternion format)
        
        # Draw camera frustum lines
        gl.glColor3f(1.0, 1.0, 0.0)  # Yellow
        gl.glBegin(gl.GL_LINES)
        
        # Frustum corners (scaled for visibility)
        scale = 0.5
        corners = [
            [0, 0, 0], [scale, scale, scale],
            [0, 0, 0], [scale, -scale, scale],
            [0, 0, 0], [-scale, -scale, scale],
            [0, 0, 0], [-scale, scale, scale]
        ]
        
        for corner in corners:
            gl.glVertex3f(corner[0], corner[1], corner[2])
        
        gl.glEnd()
        gl.glPopMatrix()
    
    def render_heatmap(self):
        """Render density heatmap overlay."""
        # Implementation for density heatmap
        # This would involve calculating point density in grid cells
        # and rendering colored quads or using texture mapping
        pass
    
    def render_axes(self):
        """Render coordinate system axes."""
        axis_length = 1.0
        
        gl.glBegin(gl.GL_LINES)
        
        # X axis (red)
        gl.glColor3f(1.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(axis_length, 0.0, 0.0)
        
        # Y axis (green)
        gl.glColor3f(0.0, 1.0, 0.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, axis_length, 0.0)
        
        # Z axis (blue)
        gl.glColor3f(0.0, 0.0, 1.0)
        gl.glVertex3f(0.0, 0.0, 0.0)
        gl.glVertex3f(0.0, 0.0, axis_length)
        
        gl.glEnd()
    
    def render_fallback(self):
        """Fallback rendering when OpenGL is not available."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(25, 25, 25))
        
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 
                        "3D Visualization\n(OpenGL not available)")
    
    def update_performance_metrics(self):
        """Update performance tracking."""
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_frame_time > 1.0:  # Update every second
            fps = self.frame_count / (current_time - self.last_frame_time)
            self.last_frame_time = current_time
            self.frame_count = 0
            
            # Emit performance signal if needed
            # self.performance_updated.emit({'fps': fps})
    
    def update_data(self, pose: Optional[SLAMPose], map_points: List[SLAMMapPoint]):
        """Update visualization data."""
        self.current_pose = pose
        self.map_points = map_points
        
        # Update trajectory
        if pose:
            if not self.trajectory:
                self.trajectory = SLAMTrajectory()
            self.trajectory.poses.append(pose)
            
            # Limit trajectory length for performance
            if len(self.trajectory.poses) > 1000:
                self.trajectory.poses = self.trajectory.poses[-1000:]
        
        self.update()
    
    def set_keyframes(self, keyframes: List[SLAMPose]):
        """Set keyframes for visualization."""
        self.keyframes = keyframes
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse press events."""
        self.last_mouse_pos = event.pos()
        self.mouse_buttons = event.buttons()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events for camera control."""
        if not (self.mouse_buttons & Qt.MouseButton.LeftButton):
            return
        
        dx = event.pos().x() - self.last_mouse_pos.x()
        dy = event.pos().y() - self.last_mouse_pos.y()
        
        # Rotate camera
        self.camera_azimuth += dx * 0.5
        self.camera_elevation += dy * 0.5
        
        # Clamp elevation
        self.camera_elevation = max(-89, min(89, self.camera_elevation))
        
        self.last_mouse_pos = event.pos()
        self.update()
    
    def wheelEvent(self, event):
        """Handle mouse wheel events for zoom."""
        zoom_factor = 1.1
        if event.angleDelta().y() > 0:
            self.camera_distance /= zoom_factor
        else:
            self.camera_distance *= zoom_factor
        
        self.camera_distance = max(1.0, min(100.0, self.camera_distance))
        self.update()
    
    def set_render_settings(self, settings: RenderSettings):
        """Update render settings."""
        self.settings = settings
        self.update()
    
    def toggle_component(self, component: str, visible: bool):
        """Toggle visibility of rendering components."""
        if component == "points":
            self.show_points = visible
        elif component == "trajectory":
            self.show_trajectory = visible
        elif component == "keyframes":
            self.show_keyframes = visible
        elif component == "current_pose":
            self.show_current_pose = visible
        elif component == "heatmap":
            self.show_heatmap = visible
        
        self.update()


class PointCloudRenderer(Map3DViewer):
    """
    Specialized renderer for point clouds with additional features.
    
    Features:
    - Color-coded point clouds
    - Point cloud segmentation visualization
    - Density visualization
    - Point filtering and selection
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Point cloud specific settings
        self.point_cloud_colors = []
        self.segmentation_labels = []
        self.show_segmentation = False
        self.point_size_mode = "uniform"  # "uniform", "distance", "confidence"
    
    def render_map_points(self):
        """Enhanced point cloud rendering with color coding."""
        if not self.map_points:
            return
        
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        
        gl.glBegin(gl.GL_POINTS)
        for i, point in enumerate(self.map_points):
            # Determine point color
            if self.show_segmentation and i < len(self.segmentation_labels):
                # Color by segmentation label
                label = self.segmentation_labels[i]
                color = self.get_segmentation_color(label)
                gl.glColor3f(*color)
            elif i < len(self.point_cloud_colors):
                # Use provided colors
                gl.glColor3f(*self.point_cloud_colors[i])
            else:
                # Color by confidence
                confidence = point.confidence
                gl.glColor3f(confidence, confidence * 0.8, 1.0)
            
            # Determine point size
            if self.point_size_mode == "confidence":
                size = self.settings.point_size * point.confidence
                gl.glPointSize(size)
            elif self.point_size_mode == "distance":
                # Size based on distance from camera
                dist = np.linalg.norm(point.position - self.camera_target)
                size = max(1.0, self.settings.point_size * 10.0 / dist)
                gl.glPointSize(size)
            
            gl.glVertex3f(point.position[0], point.position[1], point.position[2])
        gl.glEnd()
    
    def get_segmentation_color(self, label: int) -> Tuple[float, float, float]:
        """Get color for segmentation label."""
        colors = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyan
            (1.0, 0.5, 0.0),  # Orange
            (0.5, 0.0, 1.0),  # Purple
        ]
        return colors[label % len(colors)]
    
    def set_point_cloud_colors(self, colors: List[Tuple[float, float, float]]):
        """Set custom colors for point cloud."""
        self.point_cloud_colors = colors
        self.update()
    
    def set_segmentation_labels(self, labels: List[int]):
        """Set segmentation labels for points."""
        self.segmentation_labels = labels
        self.update()
    
    def toggle_segmentation_view(self, enabled: bool):
        """Toggle segmentation visualization."""
        self.show_segmentation = enabled
        self.update()


class TrajectoryViewer(QWidget):
    """
    2D trajectory viewer with timeline controls.
    
    Features:
    - Top-down trajectory view
    - Timeline scrubbing
    - Pose selection
    - Trajectory statistics
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.trajectory: Optional[SLAMTrajectory] = None
        self.current_pose_index = 0
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Trajectory canvas
        self.canvas = QLabel()
        self.canvas.setMinimumHeight(400)
        self.canvas.setStyleSheet("border: 1px solid #333; background: black;")
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.canvas)
        
        # Timeline controls
        controls_layout = QHBoxLayout()
        
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.valueChanged.connect(self.on_timeline_changed)
        controls_layout.addWidget(self.timeline_slider)
        
        self.pose_label = QLabel("Pose: 0/0")
        controls_layout.addWidget(self.pose_label)
        
        layout.addLayout(controls_layout)
    
    def update_trajectory(self, trajectory: SLAMTrajectory):
        """Update trajectory data."""
        self.trajectory = trajectory
        
        if trajectory and trajectory.poses:
            self.timeline_slider.setMaximum(len(trajectory.poses) - 1)
            self.render_trajectory()
    
    def render_trajectory(self):
        """Render trajectory on canvas."""
        if not self.trajectory or not self.trajectory.poses:
            return
        
        # Create pixmap for drawing
        pixmap = QPixmap(self.canvas.size())
        pixmap.fill(QColor(25, 25, 25))
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Extract 2D positions (X-Y plane)
        positions = np.array([pose.position[:2] for pose in self.trajectory.poses])
        
        if len(positions) > 1:
            # Normalize to canvas size with margin
            margin = 50
            canvas_width = self.canvas.width() - 2 * margin
            canvas_height = self.canvas.height() - 2 * margin
            
            min_pos = positions.min(axis=0)
            max_pos = positions.max(axis=0)
            pos_range = max_pos - min_pos
            
            if pos_range.max() > 0:
                scale = min(canvas_width / pos_range[0], canvas_height / pos_range[1]) * 0.8
                
                # Convert to canvas coordinates
                canvas_positions = (positions - min_pos) * scale
                canvas_positions[:, 1] = canvas_height - canvas_positions[:, 1]  # Flip Y
                canvas_positions += margin
                
                # Draw trajectory
                painter.setPen(QPen(QColor(0, 255, 0), 2))
                for i in range(len(canvas_positions) - 1):
                    start = QPointF(canvas_positions[i][0], canvas_positions[i][1])
                    end = QPointF(canvas_positions[i+1][0], canvas_positions[i+1][1])
                    painter.drawLine(start, end)
                
                # Draw current pose
                if self.current_pose_index < len(canvas_positions):
                    current_pos = canvas_positions[self.current_pose_index]
                    painter.setPen(QPen(QColor(255, 0, 0), 3))
                    painter.setBrush(QBrush(QColor(255, 0, 0)))
                    painter.drawEllipse(QPointF(current_pos[0], current_pos[1]), 5, 5)
        
        painter.end()
        self.canvas.setPixmap(pixmap)
    
    def on_timeline_changed(self, value: int):
        """Handle timeline slider change."""
        self.current_pose_index = value
        
        if self.trajectory:
            total_poses = len(self.trajectory.poses)
            self.pose_label.setText(f"Pose: {value + 1}/{total_poses}")
        
        self.render_trajectory()


class KeyframeViewer(QWidget):
    """
    Keyframe viewer with thumbnail grid.
    
    Features:
    - Keyframe thumbnail display
    - Keyframe selection
    - Feature overlay
    - Keyframe statistics
    """
    
    keyframe_selected = pyqtSignal(int)  # Emitted when keyframe is selected
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.keyframes: List[SLAMPose] = []
        self.keyframe_images: List[np.ndarray] = []
        self.selected_keyframe = -1
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the UI."""
        layout = QVBoxLayout(self)
        
        # Keyframe grid
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        
        scroll_area.setWidget(self.grid_widget)
        layout.addWidget(scroll_area)
        
        # Keyframe info
        self.info_label = QLabel("No keyframes")
        layout.addWidget(self.info_label)
    
    def update_keyframes(self, keyframes: List[SLAMPose], images: List[np.ndarray]):
        """Update keyframe data."""
        self.keyframes = keyframes
        self.keyframe_images = images
        
        self.update_grid()
        self.info_label.setText(f"Keyframes: {len(keyframes)}")
    
    def update_grid(self):
        """Update keyframe grid display."""
        # Clear existing items
        for i in reversed(range(self.grid_layout.count())):
            self.grid_layout.itemAt(i).widget().setParent(None)
        
        # Add keyframe thumbnails
        cols = 4
        for i, (keyframe, image) in enumerate(zip(self.keyframes, self.keyframe_images)):
            row = i // cols
            col = i % cols
            
            thumbnail = self.create_thumbnail(image, i)
            self.grid_layout.addWidget(thumbnail, row, col)
    
    def create_thumbnail(self, image: np.ndarray, index: int) -> QLabel:
        """Create thumbnail widget for keyframe."""
        # Convert numpy array to QPixmap
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Color image
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            # Grayscale image
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        # Scale to thumbnail size
        pixmap = QPixmap.fromImage(q_image).scaled(150, 100, Qt.AspectRatioMode.KeepAspectRatio)
        
        # Create label
        label = QLabel()
        label.setPixmap(pixmap)
        label.setStyleSheet("border: 2px solid #333; margin: 2px;")
        label.mousePressEvent = lambda event, idx=index: self.on_keyframe_clicked(idx)
        
        return label
    
    def on_keyframe_clicked(self, index: int):
        """Handle keyframe selection."""
        self.selected_keyframe = index
        self.keyframe_selected.emit(index)
        
        # Update visual selection
        self.update_selection_visual()
    
    def update_selection_visual(self):
        """Update visual indication of selected keyframe."""
        # Implementation for highlighting selected keyframe
        pass
