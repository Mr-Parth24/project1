"""
3D Trajectory visualization widget using PyQtGraph
"""

import numpy as np
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import logging

class TrajectoryWidget(QWidget):
    """3D trajectory visualization widget"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Trajectory data
        self.trajectory_points = []
        self.current_position = np.array([0, 0, 0])
        
        # Visualization objects
        self.trajectory_line = None
        self.current_pos_marker = None
        self.start_pos_marker = None
        self.coordinate_system = None
        
        # Display settings
        self.max_points = 5000
        self.view_mode = "3D"  # "3D", "2D_XY", "2D_XZ", "2D_YZ"
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        # View mode buttons
        self.view_3d_btn = QPushButton("3D View")
        self.view_3d_btn.clicked.connect(lambda: self.set_view_mode("3D"))
        control_layout.addWidget(self.view_3d_btn)
        
        self.view_xy_btn = QPushButton("Top (XY)")
        self.view_xy_btn.clicked.connect(lambda: self.set_view_mode("2D_XY"))
        control_layout.addWidget(self.view_xy_btn)
        
        self.view_xz_btn = QPushButton("Side (XZ)")
        self.view_xz_btn.clicked.connect(lambda: self.set_view_mode("2D_XZ"))
        control_layout.addWidget(self.view_xz_btn)
        
        # Control buttons
        control_layout.addStretch()
        
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(self.reset_view)
        control_layout.addWidget(self.reset_view_btn)
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_trajectory)
        control_layout.addWidget(self.clear_btn)
        
        layout.addLayout(control_layout)
        
        # 3D visualization widget
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setMinimumSize(600, 400)
        layout.addWidget(self.gl_widget)
        
        # Initialize 3D scene
        self.setup_scene()
        
        # Information panel
        info_layout = QHBoxLayout()
        
        self.points_label = QLabel("Points: 0")
        info_layout.addWidget(self.points_label)
        
        info_layout.addStretch()
        
        self.bounds_label = QLabel("Bounds: --")
        info_layout.addWidget(self.bounds_label)
        
        layout.addLayout(info_layout)
        
        # Update button states
        self.update_button_states()
    
    def setup_scene(self):
        """Setup the 3D scene"""
        # Set background color
        self.gl_widget.setBackgroundColor('#2c3e50')
        
        # Set initial camera position
        self.gl_widget.setCameraPosition(distance=10, elevation=20, azimuth=45)
        
        # Add coordinate system
        self.coordinate_system = gl.GLAxisItem()
        self.coordinate_system.setSize(2, 2, 2)
        self.gl_widget.addItem(self.coordinate_system)
        
        # Add grid
        grid = gl.GLGridItem()
        grid.scale(0.5, 0.5, 1)
        grid.setColor((100, 100, 100, 100))
        self.gl_widget.addItem(grid)
    
    def set_view_mode(self, mode: str):
        """Set visualization view mode"""
        self.view_mode = mode
        self.update_button_states()
        self.update_camera_view()
    
    def update_button_states(self):
        """Update button visual states"""
        buttons = {
            "3D": self.view_3d_btn,
            "2D_XY": self.view_xy_btn,
            "2D_XZ": self.view_xz_btn
        }
        
        for mode, button in buttons.items():
            if mode == self.view_mode:
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #e74c3c;
                        border: 2px solid #c0392b;
                        color: #ffffff;
                        font-weight: bold;
                        padding: 5px;
                        border-radius: 3px;
                    }
                """)
            else:
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #34495e;
                        border: 2px solid #2c3e50;
                        color: #bdc3c7;
                        padding: 5px;
                        border-radius: 3px;
                    }
                    QPushButton:hover {
                        background-color: #4a5f7a;
                    }
                """)
    
    def update_camera_view(self):
        """Update camera view based on view mode"""
        if self.view_mode == "3D":
            self.gl_widget.setCameraPosition(distance=10, elevation=20, azimuth=45)
        elif self.view_mode == "2D_XY":
            self.gl_widget.setCameraPosition(distance=15, elevation=90, azimuth=0)
        elif self.view_mode == "2D_XZ":
            self.gl_widget.setCameraPosition(distance=15, elevation=0, azimuth=0)
    
    def update_trajectory(self, points: np.ndarray, current_pos: np.ndarray):
        """Update trajectory visualization"""
        if len(points) < 2:
            return
        
        try:
            # Store trajectory data
            self.trajectory_points = points.copy()
            self.current_position = current_pos.copy()
            
            # Limit number of points for performance
            if len(self.trajectory_points) > self.max_points:
                self.trajectory_points = self.trajectory_points[-self.max_points:]
            
            # Remove old trajectory line
            if self.trajectory_line is not None:
                self.gl_widget.removeItem(self.trajectory_line)
            
            # Remove old markers
            if self.current_pos_marker is not None:
                self.gl_widget.removeItem(self.current_pos_marker)
            if self.start_pos_marker is not None:
                self.gl_widget.removeItem(self.start_pos_marker)
            
            # Create new trajectory line
            self.trajectory_line = gl.GLLinePlotItem(
                pos=self.trajectory_points,
                color=(0.2, 0.8, 1.0, 1.0),  # Light blue
                width=3,
                antialias=True
            )
            self.gl_widget.addItem(self.trajectory_line)
            
            # Add start position marker (green sphere)
            start_pos = self.trajectory_points[0:1]  # First point as array
            self.start_pos_marker = gl.GLScatterPlotItem(
                pos=start_pos,
                color=(0.2, 0.8, 0.2, 1.0),  # Green
                size=15,
                pxMode=False
            )
            self.gl_widget.addItem(self.start_pos_marker)
            
            # Add current position marker (red sphere)
            current_pos_array = np.array([current_pos])
            self.current_pos_marker = gl.GLScatterPlotItem(
                pos=current_pos_array,
                color=(0.8, 0.2, 0.2, 1.0),  # Red
                size=12,
                pxMode=False
            )
            self.gl_widget.addItem(self.current_pos_marker)
            
            # Update information
            self.update_info()
            
        except Exception as e:
            self.logger.error(f"Trajectory update error: {e}")
    
    def update_info(self):
        """Update information display"""
        if len(self.trajectory_points) > 0:
            self.points_label.setText(f"Points: {len(self.trajectory_points)}")
            
            # Calculate bounds
            min_coords = np.min(self.trajectory_points, axis=0)
            max_coords = np.max(self.trajectory_points, axis=0)
            
            bounds_text = f"X: [{min_coords[0]:.2f}, {max_coords[0]:.2f}] "
            bounds_text += f"Y: [{min_coords[1]:.2f}, {max_coords[1]:.2f}] "
            bounds_text += f"Z: [{min_coords[2]:.2f}, {max_coords[2]:.2f}]"
            
            self.bounds_label.setText(f"Bounds: {bounds_text}")
        else:
            self.points_label.setText("Points: 0")
            self.bounds_label.setText("Bounds: --")
    
    def reset_view(self):
        """Reset camera view"""
        self.update_camera_view()
        
        # Auto-fit trajectory if points exist
        if len(self.trajectory_points) > 0:
            center = np.mean(self.trajectory_points, axis=0)
            bounds = np.max(self.trajectory_points, axis=0) - np.min(self.trajectory_points, axis=0)
            max_bound = np.max(bounds)
            
            if max_bound > 0:
                distance = max_bound * 2
                self.gl_widget.setCameraPosition(
                    pos=center,
                    distance=distance,
                    elevation=20,
                    azimuth=45
                )
    
    def clear_trajectory(self):
        """Clear trajectory visualization"""
        # Remove trajectory elements
        if self.trajectory_line is not None:
            self.gl_widget.removeItem(self.trajectory_line)
            self.trajectory_line = None
        
        if self.current_pos_marker is not None:
            self.gl_widget.removeItem(self.current_pos_marker)
            self.current_pos_marker = None
        
        if self.start_pos_marker is not None:
            self.gl_widget.removeItem(self.start_pos_marker)
            self.start_pos_marker = None
        
        # Clear data
        self.trajectory_points = []
        self.current_position = np.array([0, 0, 0])
        
        # Update info
        self.update_info()
    
    def export_view(self, filename: str):
        """Export current view as image"""
        try:
            # Capture the GL widget content
            pixmap = self.gl_widget.grabFrameBuffer()
            pixmap.save(filename)
            self.logger.info(f"View exported to {filename}")
        except Exception as e:
            self.logger.error(f"Export view error: {e}")