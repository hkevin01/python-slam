"""
Utilities for Python SLAM GUI

Material Design styling, theme management, and utility functions.
"""

from typing import Dict, Any, Optional, Tuple
import json
import os
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


class MaterialDesign:
    """
    Material Design styling for Qt widgets.
    
    Provides consistent styling across the application with Material Design principles.
    """
    
    # Color palette
    COLORS = {
        'primary': '#2196F3',
        'primary_dark': '#1976D2',
        'primary_light': '#BBDEFB',
        'accent': '#FF5722',
        'background': '#121212',
        'surface': '#1E1E1E',
        'surface_variant': '#2A2A2A',
        'on_background': '#FFFFFF',
        'on_surface': '#E0E0E0',
        'on_surface_variant': '#BDBDBD',
        'success': '#4CAF50',
        'warning': '#FF9800',
        'danger': '#F44336',
        'info': '#2196F3'
    }
    
    # Typography
    FONTS = {
        'body1': 'font-size: 14px; font-weight: 400;',
        'body2': 'font-size: 12px; font-weight: 400;',
        'subtitle1': 'font-size: 16px; font-weight: 500;',
        'subtitle2': 'font-size: 14px; font-weight: 500;',
        'caption': 'font-size: 10px; font-weight: 400;',
        'button': 'font-size: 14px; font-weight: 500; text-transform: uppercase;'
    }
    
    @classmethod
    def get_button_style(cls, variant: str = 'default') -> str:
        """Get button styling."""
        base_style = f"""
            QPushButton {{
                {cls.FONTS['button']}
                color: {cls.COLORS['on_surface']};
                background-color: {cls.COLORS['surface_variant']};
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                min-height: 36px;
            }}
            QPushButton:hover {{
                background-color: {cls.COLORS['primary']};
            }}
            QPushButton:pressed {{
                background-color: {cls.COLORS['primary_dark']};
            }}
            QPushButton:disabled {{
                background-color: {cls.COLORS['surface']};
                color: {cls.COLORS['on_surface_variant']};
            }}
        """
        
        if variant == 'primary':
            base_style = base_style.replace(
                f"background-color: {cls.COLORS['surface_variant']};",
                f"background-color: {cls.COLORS['primary']};"
            )
        elif variant == 'success':
            base_style = base_style.replace(
                f"background-color: {cls.COLORS['surface_variant']};",
                f"background-color: {cls.COLORS['success']};"
            )
        elif variant == 'danger':
            base_style = base_style.replace(
                f"background-color: {cls.COLORS['surface_variant']};",
                f"background-color: {cls.COLORS['danger']};"
            )
        elif variant == 'warning':
            base_style = base_style.replace(
                f"background-color: {cls.COLORS['surface_variant']};",
                f"background-color: {cls.COLORS['warning']};"
            )
        
        return base_style
    
    @classmethod
    def get_groupbox_style(cls) -> str:
        """Get group box styling."""
        return f"""
            QGroupBox {{
                {cls.FONTS['subtitle1']}
                color: {cls.COLORS['on_surface']};
                background-color: {cls.COLORS['surface']};
                border: 1px solid {cls.COLORS['surface_variant']};
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                color: {cls.COLORS['primary']};
                background-color: transparent;
            }}
        """
    
    @classmethod
    def get_panel_style(cls) -> str:
        """Get panel styling."""
        return f"""
            QWidget {{
                background-color: {cls.COLORS['surface']};
                color: {cls.COLORS['on_surface']};
                border-radius: 8px;
            }}
        """
    
    @classmethod
    def get_combo_style(cls) -> str:
        """Get combo box styling."""
        return f"""
            QComboBox {{
                {cls.FONTS['body1']}
                color: {cls.COLORS['on_surface']};
                background-color: {cls.COLORS['surface_variant']};
                border: 1px solid {cls.COLORS['surface_variant']};
                border-radius: 4px;
                padding: 6px 12px;
                min-height: 24px;
            }}
            QComboBox:hover {{
                border-color: {cls.COLORS['primary']};
            }}
            QComboBox:focus {{
                border-color: {cls.COLORS['primary']};
                border-width: 2px;
            }}
            QComboBox::drop-down {{
                border: none;
                background: transparent;
                width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid {cls.COLORS['on_surface']};
                margin-right: 5px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {cls.COLORS['surface_variant']};
                color: {cls.COLORS['on_surface']};
                border: 1px solid {cls.COLORS['primary']};
                border-radius: 4px;
                selection-background-color: {cls.COLORS['primary']};
            }}
        """
    
    @classmethod
    def get_checkbox_style(cls) -> str:
        """Get checkbox styling."""
        return f"""
            QCheckBox {{
                {cls.FONTS['body1']}
                color: {cls.COLORS['on_surface']};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border: 2px solid {cls.COLORS['on_surface_variant']};
                border-radius: 3px;
                background-color: transparent;
            }}
            QCheckBox::indicator:hover {{
                border-color: {cls.COLORS['primary']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {cls.COLORS['primary']};
                border-color: {cls.COLORS['primary']};
                image: none;
            }}
            QCheckBox::indicator:checked {{
                background-color: {cls.COLORS['primary']};
                border-color: {cls.COLORS['primary']};
            }}
        """
    
    @classmethod
    def get_label_style(cls) -> str:
        """Get label styling."""
        return f"""
            QLabel {{
                {cls.FONTS['body1']}
                color: {cls.COLORS['on_surface']};
                background: transparent;
            }}
        """
    
    @classmethod
    def get_value_label_style(cls, variant: str = 'default') -> str:
        """Get value label styling."""
        color = cls.COLORS['on_surface']
        
        if variant == 'success':
            color = cls.COLORS['success']
        elif variant == 'danger':
            color = cls.COLORS['danger']
        elif variant == 'warning':
            color = cls.COLORS['warning']
        
        return f"""
            QLabel {{
                {cls.FONTS['subtitle2']}
                color: {color};
                background: transparent;
                font-weight: bold;
            }}
        """
    
    @classmethod
    def get_small_label_style(cls) -> str:
        """Get small label styling."""
        return f"""
            QLabel {{
                {cls.FONTS['caption']}
                color: {cls.COLORS['on_surface_variant']};
                background: transparent;
            }}
        """
    
    @classmethod
    def get_textedit_style(cls) -> str:
        """Get text edit styling."""
        return f"""
            QTextEdit {{
                {cls.FONTS['body2']}
                color: {cls.COLORS['on_surface']};
                background-color: {cls.COLORS['surface_variant']};
                border: 1px solid {cls.COLORS['surface_variant']};
                border-radius: 4px;
                padding: 8px;
            }}
            QTextEdit:focus {{
                border-color: {cls.COLORS['primary']};
                border-width: 2px;
            }}
        """
    
    @classmethod
    def get_lineedit_style(cls) -> str:
        """Get line edit styling."""
        return f"""
            QLineEdit {{
                {cls.FONTS['body1']}
                color: {cls.COLORS['on_surface']};
                background-color: {cls.COLORS['surface_variant']};
                border: 1px solid {cls.COLORS['surface_variant']};
                border-radius: 4px;
                padding: 6px 12px;
                min-height: 24px;
            }}
            QLineEdit:hover {{
                border-color: {cls.COLORS['primary']};
            }}
            QLineEdit:focus {{
                border-color: {cls.COLORS['primary']};
                border-width: 2px;
            }}
        """
    
    @classmethod
    def get_progressbar_style(cls) -> str:
        """Get progress bar styling."""
        return f"""
            QProgressBar {{
                border: 1px solid {cls.COLORS['surface_variant']};
                border-radius: 4px;
                background-color: {cls.COLORS['surface_variant']};
                text-align: center;
                color: {cls.COLORS['on_surface']};
                font-size: 12px;
            }}
            QProgressBar::chunk {{
                background-color: {cls.COLORS['primary']};
                border-radius: 3px;
            }}
        """
    
    @classmethod
    def get_tab_style(cls) -> str:
        """Get tab widget styling."""
        return f"""
            QTabWidget::pane {{
                border: 1px solid {cls.COLORS['surface_variant']};
                background-color: {cls.COLORS['surface']};
                border-radius: 4px;
            }}
            QTabBar::tab {{
                {cls.FONTS['body1']}
                color: {cls.COLORS['on_surface_variant']};
                background-color: {cls.COLORS['surface_variant']};
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                color: {cls.COLORS['primary']};
                background-color: {cls.COLORS['surface']};
                border-bottom: 2px solid {cls.COLORS['primary']};
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {cls.COLORS['surface']};
                color: {cls.COLORS['on_surface']};
            }}
        """
    
    @classmethod
    def get_slider_style(cls) -> str:
        """Get slider styling."""
        return f"""
            QSlider::groove:horizontal {{
                border: none;
                height: 4px;
                background: {cls.COLORS['surface_variant']};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {cls.COLORS['primary']};
                border: none;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {cls.COLORS['primary_light']};
            }}
            QSlider::sub-page:horizontal {{
                background: {cls.COLORS['primary']};
                border-radius: 2px;
            }}
        """


class ThemeManager:
    """
    Theme management for the application.
    
    Handles theme switching and persistence.
    """
    
    def __init__(self):
        self.current_theme = 'dark'
        self.themes = {
            'dark': {
                'primary': '#2196F3',
                'background': '#121212',
                'surface': '#1E1E1E',
                'on_background': '#FFFFFF',
                'on_surface': '#E0E0E0'
            },
            'light': {
                'primary': '#1976D2',
                'background': '#FAFAFA',
                'surface': '#FFFFFF',
                'on_background': '#000000',
                'on_surface': '#212121'
            }
        }
    
    def set_theme(self, theme_name: str):
        """Set the current theme."""
        if theme_name in self.themes:
            self.current_theme = theme_name
            # Update MaterialDesign colors
            theme_colors = self.themes[theme_name]
            for key, value in theme_colors.items():
                if key in MaterialDesign.COLORS:
                    MaterialDesign.COLORS[key] = value
    
    def get_current_theme(self) -> str:
        """Get the current theme name."""
        return self.current_theme
    
    def apply_theme(self, widget: QWidget, theme_name: Optional[str] = None):
        """Apply theme to a widget."""
        if theme_name:
            self.set_theme(theme_name)
        
        # Apply global stylesheet
        app_stylesheet = self.get_application_stylesheet()
        if hasattr(widget, 'setStyleSheet'):
            widget.setStyleSheet(app_stylesheet)
    
    def get_application_stylesheet(self) -> str:
        """Get the global application stylesheet."""
        return f"""
            QMainWindow {{
                background-color: {MaterialDesign.COLORS['background']};
                color: {MaterialDesign.COLORS['on_background']};
            }}
            QWidget {{
                background-color: {MaterialDesign.COLORS['background']};
                color: {MaterialDesign.COLORS['on_background']};
            }}
            QMenuBar {{
                background-color: {MaterialDesign.COLORS['surface']};
                color: {MaterialDesign.COLORS['on_surface']};
                border-bottom: 1px solid {MaterialDesign.COLORS['surface_variant']};
            }}
            QMenuBar::item {{
                padding: 4px 8px;
                background: transparent;
            }}
            QMenuBar::item:selected {{
                background-color: {MaterialDesign.COLORS['primary']};
            }}
            QMenu {{
                background-color: {MaterialDesign.COLORS['surface']};
                color: {MaterialDesign.COLORS['on_surface']};
                border: 1px solid {MaterialDesign.COLORS['surface_variant']};
                border-radius: 4px;
            }}
            QMenu::item {{
                padding: 6px 16px;
            }}
            QMenu::item:selected {{
                background-color: {MaterialDesign.COLORS['primary']};
            }}
            QStatusBar {{
                background-color: {MaterialDesign.COLORS['surface']};
                color: {MaterialDesign.COLORS['on_surface']};
                border-top: 1px solid {MaterialDesign.COLORS['surface_variant']};
            }}
            QToolBar {{
                background-color: {MaterialDesign.COLORS['surface']};
                border: none;
                spacing: 2px;
            }}
            QToolBar QToolButton {{
                background: transparent;
                border: none;
                padding: 6px;
                border-radius: 4px;
            }}
            QToolBar QToolButton:hover {{
                background-color: {MaterialDesign.COLORS['surface_variant']};
            }}
            QSplitter::handle {{
                background-color: {MaterialDesign.COLORS['surface_variant']};
            }}
            QSplitter::handle:horizontal {{
                width: 4px;
            }}
            QSplitter::handle:vertical {{
                height: 4px;
            }}
            QScrollBar:vertical {{
                background-color: {MaterialDesign.COLORS['surface']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background-color: {MaterialDesign.COLORS['surface_variant']};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {MaterialDesign.COLORS['primary']};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                border: none;
                background: none;
            }}
        """
    
    def save_theme_preferences(self, file_path: str):
        """Save theme preferences to file."""
        try:
            preferences = {
                'theme': self.current_theme,
                'custom_colors': {}
            }
            with open(file_path, 'w') as f:
                json.dump(preferences, f, indent=2)
        except Exception as e:
            print(f"Error saving theme preferences: {e}")
    
    def load_theme_preferences(self, file_path: str):
        """Load theme preferences from file."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    preferences = json.load(f)
                    theme = preferences.get('theme', 'dark')
                    self.set_theme(theme)
        except Exception as e:
            print(f"Error loading theme preferences: {e}")


class IconProvider:
    """
    Icon provider for the application.
    
    Provides consistent icons throughout the application.
    """
    
    @staticmethod
    def get_icon(name: str, color: str = '#FFFFFF') -> QIcon:
        """Get an icon by name."""
        # This would typically load SVG icons and apply colors
        # For now, return a placeholder icon
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(color))
        return QIcon(pixmap)
    
    @staticmethod
    def get_material_icon(name: str, size: int = 24, color: str = '#FFFFFF') -> QIcon:
        """Get a Material Design icon."""
        # Implementation would use Material Design icons
        # For now, return a placeholder
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(color))
        return QIcon(pixmap)


class AnimationHelper:
    """
    Helper for creating smooth animations.
    """
    
    @staticmethod
    def fade_widget(widget: QWidget, fade_in: bool = True, duration: int = 300):
        """Fade a widget in or out."""
        effect = QGraphicsOpacityEffect()
        widget.setGraphicsEffect(effect)
        
        animation = QPropertyAnimation(effect, b"opacity")
        animation.setDuration(duration)
        
        if fade_in:
            animation.setStartValue(0.0)
            animation.setEndValue(1.0)
        else:
            animation.setStartValue(1.0)
            animation.setEndValue(0.0)
        
        animation.start()
        return animation
    
    @staticmethod
    def slide_widget(widget: QWidget, direction: str = 'left', duration: int = 300):
        """Slide a widget in from a direction."""
        animation = QPropertyAnimation(widget, b"pos")
        animation.setDuration(duration)
        
        start_pos = widget.pos()
        if direction == 'left':
            end_pos = QPoint(start_pos.x() - widget.width(), start_pos.y())
        elif direction == 'right':
            end_pos = QPoint(start_pos.x() + widget.width(), start_pos.y())
        elif direction == 'up':
            end_pos = QPoint(start_pos.x(), start_pos.y() - widget.height())
        else:  # down
            end_pos = QPoint(start_pos.x(), start_pos.y() + widget.height())
        
        animation.setStartValue(end_pos)
        animation.setEndValue(start_pos)
        animation.start()
        return animation


class GeometryHelper:
    """
    Helper for geometry calculations and transformations.
    """
    
    @staticmethod
    def screen_to_world(screen_pos: Tuple[int, int], camera_matrix: 'np.ndarray', 
                       depth: float) -> Tuple[float, float, float]:
        """Convert screen coordinates to world coordinates."""
        # Implementation for 3D coordinate transformation
        # This would use camera intrinsics and depth information
        pass
    
    @staticmethod
    def world_to_screen(world_pos: Tuple[float, float, float], 
                       camera_matrix: 'np.ndarray') -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        # Implementation for 3D to 2D projection
        pass
