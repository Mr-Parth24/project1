"""
Enhanced Trajectory Widget for Agricultural SLAM System
Provides 3D trajectory visualization with agricultural mapping features
Optimized for real-time performance with interactive navigation
"""

import numpy as np
import time
import math
from typing import Dict, List, Optional, Tuple
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QCheckBox, QComboBox, QGroupBox, QGridLayout, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QBrush, QPolygon, QPixmap
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtOpenGL import QOpenGLVersionProfile
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("OpenGL not available, using 2D visualization only")

class Agricultural3DViewer(QOpenGLWidget if OPENGL_AVAILABLE else QWidget):
    """
    3D OpenGL viewer for agricultural trajectory visualization
    Provides interactive 3D navigation and agricultural feature rendering
    """
    
    def __init__(self):
        super().__init__()
        
        # 3D view parameters
        self.camera_distance = 10.0
        self.camera_elevation = 30.0  # degrees
        self.camera_azimuth = 45.0    # degrees
        self.camera_target = np.array([0.0, 0.0, 0.0])
        
        # Trajectory data
        self.trajectory_points = []
        self.trajectory_colors = []
        self.agricultural_features = {
            'crop_rows': [],
            'field_boundaries': [],
            'ground_plane': None,
            'coverage_areas': []
        }
        
        # Rendering options
        self.show_trajectory = True
        self.show_grid = True
        self.show_axes = True
        self.show_agricultural_features = True
        self.trajectory_line_width = 3.0
        self.point_size = 5.0
        
        # Performance optimization
        self.max_trajectory_points = 10000
        self.update_frequency = 30  # Hz
        
        # Mouse interaction
        self.last_mouse_pos = None
        self.mouse_sensitivity = 0.5
        
        if OPENGL_AVAILABLE:
            self.setMinimumSize(600, 400)
        
        print("Agricultural 3D Viewer initialized")
    
    def initializeGL(self):
        """Initialize OpenGL settings"""
        if not OPENGL_AVAILABLE:
            return
        
        try:
            # Enable depth testing
            glEnable(GL_DEPTH_TEST)
            glDepthFunc(GL_LESS)
            
            # Enable blending for transparency
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            # Enable anti-aliasing
            glEnable(GL_LINE_SMOOTH)
            glEnable(GL_POINT_SMOOTH)
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
            glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
            
            # Set background color (sky blue for agricultural scenes)
            glClearColor(0.6, 0.8, 1.0, 1.0)
            
            # Enable lighting
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            
            # Set light properties
            light_pos = [10.0, 10.0, 10.0, 1.0]
            light_ambient = [0.3, 0.3, 0.3, 1.0]
            light_diffuse = [0.8, 0.8, 0.8, 1.0]
            
            glLightfv(GL_LIGHT0, GL_POSITION, light_pos)
            glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
            glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
            
            print("OpenGL initialized successfully")
            
        except Exception as e:
            print(f"OpenGL initialization error: {e}")
    
    def resizeGL(self, width, height):
        """Handle window resize"""
        if not OPENGL_AVAILABLE:
            return
        
        try:
            if height == 0:
                height = 1
            
            glViewport(0, 0, width, height)
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            
            # Set perspective projection
            aspect_ratio = width / height
            gluPerspective(45.0, aspect_ratio, 0.1, 1000.0)
            
            glMatrixMode(GL_MODELVIEW)
            
        except Exception as e:
            print(f"OpenGL resize error: {e}")
    
    def paintGL(self):
        """Render the 3D scene"""
        if not OPENGL_AVAILABLE:
            return
        
        try:
            # Clear buffers
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            
            # Set up camera
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()
            
            # Position camera
            camera_x = self.camera_target[0] + self.camera_distance * math.cos(math.radians(self.camera_elevation)) * math.cos(math.radians(self.camera_azimuth))
            camera_y = self.camera_target[1] + self.camera_distance * math.sin(math.radians(self.camera_elevation))
            camera_z = self.camera_target[2] + self.camera_distance * math.cos(math.radians(self.camera_elevation)) * math.sin(math.radians(self.camera_azimuth))
            
            gluLookAt(camera_x, camera_y, camera_z,
                     self.camera_target[0], self.camera_target[1], self.camera_target[2],
                     0, 1, 0)
            
            # Render scene elements
            if self.show_grid:
                self._render_grid()
            
            if self.show_axes:
                self._render_axes()
            
            if self.show_agricultural_features:
                self._render_agricultural_features()
            
            if self.show_trajectory:
                self._render_trajectory()
            
            # Render HUD information
            self._render_hud()
            
        except Exception as e:
            print(f"OpenGL rendering error: {e}")
    
    def _render_grid(self):
        """Render ground grid"""
        try:
            glDisable(GL_LIGHTING)
            glColor3f(0.7, 0.7, 0.7)
            glLineWidth(1.0)
            
            grid_size = 20
            grid_spacing = 1.0
            
            glBegin(GL_LINES)
            
            # Grid lines parallel to X axis
            for i in range(-grid_size, grid_size + 1):
                z = i * grid_spacing
                glVertex3f(-grid_size * grid_spacing, 0, z)
                glVertex3f(grid_size * grid_spacing, 0, z)
            
            # Grid lines parallel to Z axis
            for i in range(-grid_size, grid_size + 1):
                x = i * grid_spacing
                glVertex3f(x, 0, -grid_size * grid_spacing)
                glVertex3f(x, 0, grid_size * grid_spacing)
            
            glEnd()
            glEnable(GL_LIGHTING)
            
        except Exception as e:
            print(f"Grid rendering error: {e}")
    
    def _render_axes(self):
        """Render coordinate axes"""
        try:
            glDisable(GL_LIGHTING)
            glLineWidth(3.0)
            
            axis_length = 2.0
            
            glBegin(GL_LINES)
            
            # X axis (red)
            glColor3f(1.0, 0.0, 0.0)
            glVertex3f(0, 0, 0)
            glVertex3f(axis_length, 0, 0)
            
            # Y axis (green)
            glColor3f(0.0, 1.0, 0.0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, axis_length, 0)
            
            # Z axis (blue)
            glColor3f(0.0, 0.0, 1.0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, axis_length)
            
            glEnd()
            glEnable(GL_LIGHTING)
            
        except Exception as e:
            print(f"Axes rendering error: {e}")
    
    def _render_trajectory(self):
        """Render trajectory path"""
        try:
            if len(self.trajectory_points) < 2:
                return
            
            glDisable(GL_LIGHTING)
            
            # Render trajectory line
            glColor3f(0.0, 1.0, 0.0)  # Green
            glLineWidth(self.trajectory_line_width)
            
            glBegin(GL_LINE_STRIP)
            for point in self.trajectory_points:
                glVertex3f(point[0], point[1], point[2])
            glEnd()
            
            # Render trajectory points
            glPointSize(self.point_size)
            glBegin(GL_POINTS)
            
            for i, point in enumerate(self.trajectory_points):
                # Color gradient from blue to red
                ratio = i / max(len(self.trajectory_points) - 1, 1)
                glColor3f(ratio, 0.0, 1.0 - ratio)
                glVertex3f(point[0], point[1], point[2])
            
            glEnd()
            
            # Render current position (larger red dot)
            if self.trajectory_points:
                current_pos = self.trajectory_points[-1]
                glColor3f(1.0, 0.0, 0.0)
                glPointSize(self.point_size * 2)
                glBegin(GL_POINTS)
                glVertex3f(current_pos[0], current_pos[1], current_pos[2])
                glEnd()
            
            glEnable(GL_LIGHTING)
            
        except Exception as e:
            print(f"Trajectory rendering error: {e}")
    
    def _render_agricultural_features(self):
        """Render agricultural features"""
        try:
            glDisable(GL_LIGHTING)
            
            # Render crop rows
            crop_rows = self.agricultural_features.get('crop_rows', [])
            if crop_rows:
                glColor3f(0.0, 0.8, 0.0)  # Bright green
                glLineWidth(4.0)
                
                for crop_row in crop_rows:
                    if hasattr(crop_row, 'start_point') and hasattr(crop_row, 'end_point'):
                        start = crop_row.start_point
                        end = crop_row.end_point
                        
                        glBegin(GL_LINES)
                        glVertex3f(start[0], 0.1, start[1])  # Slightly above ground
                        glVertex3f(end[0], 0.1, end[1])
                        glEnd()
            
            # Render field boundaries
            boundaries = self.agricultural_features.get('field_boundaries', [])
            if len(boundaries) > 2:
                glColor3f(1.0, 1.0, 0.0)  # Yellow
                glLineWidth(2.0)
                
                glBegin(GL_LINE_STRIP)
                for boundary in boundaries[-100:]:  # Last 100 points
                    if len(boundary) >= 3:
                        glVertex3f(boundary[0], 0.05, boundary[2])
                glEnd()
            
            # Render ground plane
            ground_plane = self.agricultural_features.get('ground_plane')
            if ground_plane:
                glColor4f(0.8, 0.6, 0.4, 0.3)  # Semi-transparent brown
                
                # Render ground plane as a quad
                plane_size = 10.0
                glBegin(GL_QUADS)
                glVertex3f(-plane_size, 0, -plane_size)
                glVertex3f(plane_size, 0, -plane_size)
                glVertex3f(plane_size, 0, plane_size)
                glVertex3f(-plane_size, 0, plane_size)
                glEnd()
            
            glEnable(GL_LIGHTING)
            
        except Exception as e:
            print(f"Agricultural features rendering error: {e}")
    
    def _render_hud(self):
        """Render heads-up display information"""
        try:
            # Switch to 2D rendering for HUD
            glMatrixMode(GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity()
            glOrtho(0, self.width(), self.height(), 0, -1, 1)
            
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glLoadIdentity()
            
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_LIGHTING)
            
            # Render text information (simplified - would use proper text rendering)
            # For now, just render colored rectangles as indicators
            
            # Trajectory info indicator
            if self.trajectory_points:
                glColor3f(0.0, 1.0, 0.0)
                glBegin(GL_QUADS)
                glVertex2f(10, 10)
                glVertex2f(30, 10)
                glVertex2f(30, 30)
                glVertex2f(10, 30)
                glEnd()
            
            # Restore 3D rendering state
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            
            glPopMatrix()
            glMatrixMode(GL_PROJECTION)
            glPopMatrix()
            glMatrixMode(GL_MODELVIEW)
            
        except Exception as e:
            print(f"HUD rendering error: {e}")
    
    def mousePressEvent(self, event):
        """Handle mouse press for 3D navigation"""
        self.last_mouse_pos = event.pos()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for 3D navigation"""
        if self.last_mouse_pos and (event.buttons() & Qt.MouseButton.LeftButton):
            dx = event.pos().x() - self.last_mouse_pos.x()
            dy = event.pos().y() - self.last_mouse_pos.y()
            
            # Update camera azimuth and elevation
            self.camera_azimuth += dx * self.mouse_sensitivity
            self.camera_elevation -= dy * self.mouse_sensitivity
            
            # Clamp elevation
            self.camera_elevation = max(-90, min(90, self.camera_elevation))
            
            self.update()
        
        self.last_mouse_pos = event.pos()
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom"""
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        
        self.camera_distance *= zoom_factor
        self.camera_distance = max(1.0, min(100.0, self.camera_distance))
        
        self.update()
    
    def add_trajectory_point(self, point: np.ndarray):
        """Add point to trajectory"""
        if len(point) >= 3:
            self.trajectory_points.append(point[:3].copy())
            
            # Limit trajectory length for performance
            if len(self.trajectory_points) > self.max_trajectory_points:
                self.trajectory_points.pop(0)
            
            self.update()
    
    def update_agricultural_features(self, features: Dict):
        """Update agricultural features"""
        self.agricultural_features.update(features)
        self.update()
    
    def clear_trajectory(self):
        """Clear trajectory data"""
        self.trajectory_points.clear()
        self.trajectory_colors.clear()
        self.update()
    
    def reset_camera(self):
        """Reset camera to default position"""
        self.camera_distance = 10.0
        self.camera_elevation = 30.0
        self.camera_azimuth = 45.0
        self.camera_target = np.array([0.0, 0.0, 0.0])
        self.update()

class Trajectory2DWidget(QWidget):
    """
    2D trajectory visualization widget (fallback when OpenGL unavailable)
    Provides top-down view of trajectory and agricultural features
    """
    
    def __init__(self):
        super().__init__()
        
        # 2D view parameters
        self.scale = 50.0  # pixels per meter
        self.center_x = 0.0
        self.center_z = 0.0
        self.trajectory_points = []
        self.agricultural_features = {}
        
        # Rendering options
        self.show_trajectory = True
        self.show_grid = True
        self.show_agricultural_features = True
        self.auto_scale = True
        
        self.setMinimumSize(600, 400)
        print("2D Trajectory Widget initialized")
    
    def paintEvent(self, event):
        """Paint 2D trajectory view"""
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Clear background
            painter.fillRect(self.rect(), QColor(240, 248, 255))  # Light blue
            
            # Set up coordinate system
            painter.translate(self.width() // 2, self.height() // 2)
            painter.scale(1, -1)  # Flip Y axis
            
            # Draw grid
            if self.show_grid:
                self._draw_grid_2d(painter)
            
            # Draw agricultural features
            if self.show_agricultural_features:
                self._draw_agricultural_features_2d(painter)
            
            # Draw trajectory
            if self.show_trajectory and len(self.trajectory_points) > 1:
                self._draw_trajectory_2d(painter)
            
            # Draw current position
            if self.trajectory_points:
                self._draw_current_position_2d(painter)
            
            # Draw scale and info
            painter.scale(1, -1)  # Flip back for text
            self._draw_info_2d(painter)
            
        except Exception as e:
            print(f"2D painting error: {e}")
    
    def _draw_grid_2d(self, painter):
        """Draw grid in 2D view"""
        try:
            pen = QPen(QColor(200, 200, 200), 1)
            painter.setPen(pen)
            
            grid_spacing = 1.0  # meters
            pixel_spacing = int(grid_spacing * self.scale)
            
            if pixel_spacing > 10:  # Only draw if spacing is reasonable
                # Vertical lines
                for x in range(-self.width()//2, self.width()//2, pixel_spacing):
                    painter.drawLine(x, -self.height()//2, x, self.height()//2)
                
                # Horizontal lines
                for y in range(-self.height()//2, self.height()//2, pixel_spacing):
                    painter.drawLine(-self.width()//2, y, self.width()//2, y)
        
        except Exception as e:
            print(f"2D grid drawing error: {e}")
    
    def _draw_trajectory_2d(self, painter):
        """Draw trajectory in 2D view"""
        try:
            # Draw trajectory line
            pen = QPen(QColor(0, 255, 0), 3)  # Green
            painter.setPen(pen)
            
            points = []
            for point in self.trajectory_points:
                if len(point) >= 3:
                    x = int((point[0] - self.center_x) * self.scale)
                    z = int((point[2] - self.center_z) * self.scale)
                    points.append((x, z))
            
            if len(points) > 1:
                for i in range(1, len(points)):
                    painter.drawLine(points[i-1][0], points[i-1][1], 
                                   points[i][0], points[i][1])
        
        except Exception as e:
            print(f"2D trajectory drawing error: {e}")
    
    def _draw_current_position_2d(self, painter):
        """Draw current position marker"""
        try:
            if self.trajectory_points:
                current = self.trajectory_points[-1]
                x = int((current[0] - self.center_x) * self.scale)
                z = int((current[2] - self.center_z) * self.scale)
                
                # Draw current position as red circle
                pen = QPen(QColor(255, 0, 0), 2)
                brush = QBrush(QColor(255, 0, 0))
                painter.setPen(pen)
                painter.setBrush(brush)
                painter.drawEllipse(x-5, z-5, 10, 10)
        
        except Exception as e:
            print(f"2D current position drawing error: {e}")
    
    def _draw_agricultural_features_2d(self, painter):
        """Draw agricultural features in 2D view"""
        try:
            # Draw crop rows
            crop_rows = self.agricultural_features.get('crop_rows', [])
            if crop_rows:
                pen = QPen(QColor(0, 200, 0), 2)  # Green
                painter.setPen(pen)
                
                for crop_row in crop_rows:
                    if hasattr(crop_row, 'start_point') and hasattr(crop_row, 'end_point'):
                        start = crop_row.start_point
                        end = crop_row.end_point
                        
                        x1 = int((start[0] - self.center_x) * self.scale)
                        z1 = int((start[1] - self.center_z) * self.scale)
                        x2 = int((end[0] - self.center_x) * self.scale)
                        z2 = int((end[1] - self.center_z) * self.scale)
                        
                        painter.drawLine(x1, z1, x2, z2)
            
            # Draw field boundaries
            boundaries = self.agricultural_features.get('field_boundaries', [])
            if len(boundaries) > 2:
                pen = QPen(QColor(255, 255, 0), 2)  # Yellow
                painter.setPen(pen)
                
                boundary_points = []
                for boundary in boundaries[-100:]:  # Last 100 points
                    if len(boundary) >= 3:
                        x = int((boundary[0] - self.center_x) * self.scale)
                        z = int((boundary[2] - self.center_z) * self.scale)
                        boundary_points.append((x, z))
                
                if len(boundary_points) > 1:
                    for i in range(1, len(boundary_points)):
                        painter.drawLine(boundary_points[i-1][0], boundary_points[i-1][1],
                                       boundary_points[i][0], boundary_points[i][1])
        
        except Exception as e:
            print(f"2D agricultural features drawing error: {e}")
    
    def _draw_info_2d(self, painter):
        """Draw information overlay"""
        try:
            font = QFont("Arial", 10)
            painter.setFont(font)
            pen = QPen(QColor(0, 0, 0))
            painter.setPen(pen)
            
            info_text = f"Points: {len(self.trajectory_points)} | Scale: {self.scale:.1f} px/m"
            painter.drawText(10, -self.height()//2 + 20, info_text)
            
            if self.trajectory_points:
                current = self.trajectory_points[-1]
                pos_text = f"Position: ({current[0]:.2f}, {current[2]:.2f})"
                painter.drawText(10, -self.height()//2 + 40, pos_text)
        
        except Exception as e:
            print(f"2D info drawing error: {e}")
    
    def add_trajectory_point(self, point: np.ndarray):
        """Add point to 2D trajectory"""
        if len(point) >= 3:
            self.trajectory_points.append(point[:3].copy())
            
            # Auto-scale and center
            if self.auto_scale and len(self.trajectory_points) > 1:
                self._auto_scale_view()
            
            self.update()
    
    def _auto_scale_view(self):
        """Auto-scale view to fit trajectory"""
        try:
            if len(self.trajectory_points) < 2:
                return
            
            # Calculate trajectory bounds
            xs = [p[0] for p in self.trajectory_points]
            zs = [p[2] for p in self.trajectory_points]
            
            min_x, max_x = min(xs), max(xs)
            min_z, max_z = min(zs), max(zs)
            
            # Calculate center
            self.center_x = (min_x + max_x) / 2
            self.center_z = (min_z + max_z) / 2
            
            # Calculate scale to fit trajectory
            width_m = max_x - min_x + 2  # Add margin
            height_m = max_z - min_z + 2
            
            if width_m > 0 and height_m > 0:
                scale_x = (self.width() * 0.8) / width_m
                scale_z = (self.height() * 0.8) / height_m
                self.scale = min(scale_x, scale_z)
                self.scale = max(10, min(self.scale, 200))  # Clamp scale
        
        except Exception as e:
            print(f"Auto-scale error: {e}")
    
    def update_agricultural_features(self, features: Dict):
        """Update agricultural features for 2D view"""
        self.agricultural_features.update(features)
        self.update()
    
    def clear_trajectory(self):
        """Clear 2D trajectory"""
        self.trajectory_points.clear()
        self.update()
    
    def wheelEvent(self, event):
        """Handle zoom in 2D view"""
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        
        self.scale *= zoom_factor
        self.scale = max(5, min(self.scale, 500))
        
        self.update()

class EnhancedTrajectoryWidget(QWidget):
    """
    Enhanced Trajectory Widget combining 2D and 3D visualization
    Provides comprehensive trajectory and agricultural feature visualization
    """
    
    # Signals
    view_changed = pyqtSignal(str)  # 2D or 3D
    trajectory_cleared = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # Current view mode
        self.view_mode = '2D'
        self.current_viewer = None
        
        # Data storage
        self.trajectory_data = []
        self.agricultural_data = {}
        self.session_stats = {
            'total_distance': 0.0,
            'max_distance_from_origin': 0.0,
            'field_coverage_area': 0.0
        }
        
        # Initialize UI
        self.init_ui()
        
        # Set initial view
        self.set_view_mode('2D')
        
        print("Enhanced Trajectory Widget initialized")
    
    def init_ui(self):
        """Initialize enhanced trajectory widget UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Control panel
        control_panel = self.create_control_panel()
        layout.addWidget(control_panel)
        
        # Viewer container
        self.viewer_container = QWidget()
        self.viewer_layout = QVBoxLayout(self.viewer_container)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.viewer_container)
        
        # Statistics panel
        stats_panel = self.create_statistics_panel()
        layout.addWidget(stats_panel)
        
        # Set layout proportions
        layout.setStretch(0, 0)  # Control panel - fixed size
        layout.setStretch(1, 1)  # Viewer - expandable
        layout.setStretch(2, 0)  # Stats panel - fixed size
    
    def create_control_panel(self) -> QGroupBox:
        """Create trajectory control panel"""
        panel = QGroupBox("Trajectory Controls")
        layout = QHBoxLayout(panel)
        
        # View mode selection
        self.view_2d_btn = QPushButton("2D View")
        self.view_2d_btn.setCheckable(True)
        self.view_2d_btn.setChecked(True)
        self.view_2d_btn.clicked.connect(lambda: self.set_view_mode('2D'))
        layout.addWidget(self.view_2d_btn)
        
        self.view_3d_btn = QPushButton("3D View")
        self.view_3d_btn.setCheckable(True)
        self.view_3d_btn.clicked.connect(lambda: self.set_view_mode('3D'))
        layout.addWidget(self.view_3d_btn)
        
        layout.addWidget(QLabel("|"))
        
        # Display options
        self.show_trajectory_cb = QCheckBox("Trajectory")
        self.show_trajectory_cb.setChecked(True)
        self.show_trajectory_cb.stateChanged.connect(self.update_display_options)
        layout.addWidget(self.show_trajectory_cb)
        
        self.show_grid_cb = QCheckBox("Grid")
        self.show_grid_cb.setChecked(True)
        self.show_grid_cb.stateChanged.connect(self.update_display_options)
        layout.addWidget(self.show_grid_cb)
        
        self.show_agricultural_cb = QCheckBox("Agricultural")
        self.show_agricultural_cb.setChecked(True)
        self.show_agricultural_cb.stateChanged.connect(self.update_display_options)
        layout.addWidget(self.show_agricultural_cb)
        
        layout.addWidget(QLabel("|"))
        
        # Action buttons
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_trajectory)
        layout.addWidget(self.clear_btn)
        
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        layout.addWidget(self.reset_view_btn)
        
        layout.addStretch()
        
        return panel
    
    def create_statistics_panel(self) -> QGroupBox:
        """Create statistics display panel"""
        panel = QGroupBox("Trajectory Statistics")
        layout = QGridLayout(panel)
        
        # Distance statistics
        layout.addWidget(QLabel("Total Distance:"), 0, 0)
        self.total_distance_label = QLabel("0.000 m")
        self.total_distance_label.setStyleSheet("font-weight: bold; color: blue;")
        layout.addWidget(self.total_distance_label, 0, 1)
        
        layout.addWidget(QLabel("Max Distance:"), 0, 2)
        self.max_distance_label = QLabel("0.000 m")
        layout.addWidget(self.max_distance_label, 0, 3)
        
        # Point statistics
        layout.addWidget(QLabel("Trajectory Points:"), 1, 0)
        self.point_count_label = QLabel("0")
        layout.addWidget(self.point_count_label, 1, 1)
        
        layout.addWidget(QLabel("Field Coverage:"), 1, 2)
        self.coverage_label = QLabel("0.0 m²")
        layout.addWidget(self.coverage_label, 1, 3)
        
        return panel
    
    def set_view_mode(self, mode: str):
        """Set visualization mode (2D or 3D)"""
        try:
            if mode == self.view_mode:
                return
            
            self.view_mode = mode
            
            # Clear current viewer
            if self.current_viewer:
                self.viewer_layout.removeWidget(self.current_viewer)
                self.current_viewer.setParent(None)
                self.current_viewer = None
            
            # Create new viewer
            if mode == '3D' and OPENGL_AVAILABLE:
                self.current_viewer = Agricultural3DViewer()
            else:
                self.current_viewer = Trajectory2DWidget()
                if mode == '3D':
                    print("OpenGL not available, using 2D view")
                    mode = '2D'  # Fallback to 2D
            
            # Add new viewer
            self.viewer_layout.addWidget(self.current_viewer)
            
            # Update button states
            self.view_2d_btn.setChecked(mode == '2D')
            self.view_3d_btn.setChecked(mode == '3D')
            
            # Transfer existing data
            if self.trajectory_data:
                for point in self.trajectory_data:
                    self.current_viewer.add_trajectory_point(point)
            
            if self.agricultural_data:
                self.current_viewer.update_agricultural_features(self.agricultural_data)
            
            # Update display options
            self.update_display_options()
            
            # Emit signal
            self.view_changed.emit(mode)
            
            print(f"Switched to {mode} view")
            
        except Exception as e:
            print(f"View mode switch error: {e}")
    
    def add_point(self, position: np.ndarray):
        """Add point to trajectory"""
        try:
            if len(position) >= 3:
                # Store point
                point = position[:3].copy()
                self.trajectory_data.append(point)
                
                # Update viewer
                if self.current_viewer:
                    self.current_viewer.add_trajectory_point(point)
                
                # Update statistics
                self._update_statistics()
            
        except Exception as e:
            print(f"Add point error: {e}")
    
    def update_agricultural_features(self, features: Dict):
        """Update agricultural features display"""
        try:
            self.agricultural_data.update(features)
            
            if self.current_viewer:
                self.current_viewer.update_agricultural_features(features)
        
        except Exception as e:
            print(f"Agricultural features update error: {e}")
    
    def update_display_options(self):
        """Update display options for current viewer"""
        try:
            if not self.current_viewer:
                return
            
            # Update common options
            if hasattr(self.current_viewer, 'show_trajectory'):
                self.current_viewer.show_trajectory = self.show_trajectory_cb.isChecked()
            
            if hasattr(self.current_viewer, 'show_grid'):
                self.current_viewer.show_grid = self.show_grid_cb.isChecked()
            
            if hasattr(self.current_viewer, 'show_agricultural_features'):
                self.current_viewer.show_agricultural_features = self.show_agricultural_cb.isChecked()
            
            # Trigger update
            if hasattr(self.current_viewer, 'update'):
                self.current_viewer.update()
        
        except Exception as e:
            print(f"Display options update error: {e}")
    
    def clear_trajectory(self):
        """Clear trajectory data"""
        try:
            self.trajectory_data.clear()
            self.agricultural_data.clear()
            
            if self.current_viewer:
                self.current_viewer.clear_trajectory()
            
            # Reset statistics
            self.session_stats = {
                'total_distance': 0.0,
                'max_distance_from_origin': 0.0,
                'field_coverage_area': 0.0
            }
            self._update_statistics_display()
            
            self.trajectory_cleared.emit()
            print("Trajectory cleared")
        
        except Exception as e:
            print(f"Clear trajectory error: {e}")
    
    def reset_view(self):
        """Reset viewer to default settings"""
        try:
            if self.current_viewer and hasattr(self.current_viewer, 'reset_camera'):
                self.current_viewer.reset_camera()
            elif self.current_viewer and hasattr(self.current_viewer, '_auto_scale_view'):
                self.current_viewer._auto_scale_view()
        
        except Exception as e:
            print(f"Reset view error: {e}")
    
    def _update_statistics(self):
        """Update trajectory statistics"""
        try:
            if len(self.trajectory_data) < 2:
                return
            
            # Calculate total distance
            total_distance = 0.0
            for i in range(1, len(self.trajectory_data)):
                diff = self.trajectory_data[i] - self.trajectory_data[i-1]
                distance = np.linalg.norm(diff)
                total_distance += distance
            
            self.session_stats['total_distance'] = total_distance
            
            # Calculate max distance from origin
            distances_from_origin = [np.linalg.norm(point) for point in self.trajectory_data]
            self.session_stats['max_distance_from_origin'] = max(distances_from_origin)
            
            # Calculate approximate field coverage (bounding box area)
            if len(self.trajectory_data) > 2:
                xs = [p[0] for p in self.trajectory_data]
                zs = [p[2] for p in self.trajectory_data]
                
                width = max(xs) - min(xs)
                height = max(zs) - min(zs)
                self.session_stats['field_coverage_area'] = width * height
            
            # Update display
            self._update_statistics_display()
        
        except Exception as e:
            print(f"Statistics update error: {e}")
    
    def _update_statistics_display(self):
        """Update statistics display labels"""
        try:
            self.total_distance_label.setText(f"{self.session_stats['total_distance']:.3f} m")
            self.max_distance_label.setText(f"{self.session_stats['max_distance_from_origin']:.3f} m")
            self.point_count_label.setText(str(len(self.trajectory_data)))
            self.coverage_label.setText(f"{self.session_stats['field_coverage_area']:.1f} m²")
        
        except Exception as e:
            print(f"Statistics display update error: {e}")
    
    def get_trajectory_data(self) -> np.ndarray:
        """Get trajectory data as numpy array"""
        return np.array(self.trajectory_data) if self.trajectory_data else np.array([])
    
    def get_session_statistics(self) -> Dict:
        """Get session statistics"""
        return self.session_stats.copy()

# Compatibility class for existing code
class TrajectoryWidget(EnhancedTrajectoryWidget):
    """
    Compatibility wrapper for existing code
    Maintains the original interface while using enhanced implementation
    """
    
    def __init__(self):
        super().__init__()
    
    def add_trajectory_point(self, position: np.ndarray):
        """Add trajectory point (compatibility method)"""
        self.add_point(position)
    
    def update_trajectory(self, trajectory: np.ndarray):
        """Update entire trajectory (compatibility method)"""
        self.clear_trajectory()
        for point in trajectory:
            self.add_point(point)

# Test function
def test_enhanced_trajectory_widget():
    """Test enhanced trajectory widget"""
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    widget = EnhancedTrajectoryWidget()
    widget.show()
    
    # Add test trajectory
    for i in range(100):
        t = i * 0.1
        x = t
        y = 0.1 * np.sin(t)
        z = 0.5 * t
        widget.add_point(np.array([x, y, z]))
    
    print("Test trajectory widget displayed")
    sys.exit(app.exec())

if __name__ == "__main__":
    test_enhanced_trajectory_widget()