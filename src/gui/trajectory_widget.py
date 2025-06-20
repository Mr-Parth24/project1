"""
Trajectory Widget for 2D trajectory visualization
Real-time display of camera movement path
"""

import sys
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox
from PyQt6.QtCore import QTimer, pyqtSignal, Qt
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont
import math

class TrajectoryWidget(QWidget):
    """
    Widget for displaying 2D trajectory visualization
    Shows camera path, current position, and distance metrics
    """
    
    # Signals
    trajectory_updated = pyqtSignal(float)  # distance_traveled
    
    def __init__(self, width=400, height=300):
        super().__init__()
        
        # Widget properties
        self.plot_width = width
        self.plot_height = height
        self.setFixedSize(width + 50, height + 100)  # Extra space for controls
        
        # Trajectory data
        self.trajectory_points = [np.array([0.0, 0.0, 0.0])]  # 3D points
        self.trajectory_2d = [(int(width // 2), int(height // 2))]  # Screen coordinates (start at center)
        
        # Display settings
        self.scale = 100.0  # pixels per meter
        self.center_x = width // 2
        self.center_y = height // 2
        self.auto_scale = True
        self.show_grid = True
        self.show_distance = True
        
        # Colors
        self.bg_color = QColor(30, 30, 30)
        self.grid_color = QColor(70, 70, 70)
        self.trajectory_color = QColor(0, 255, 0)
        self.current_pos_color = QColor(255, 0, 0)
        self.text_color = QColor(255, 255, 255)
        self.start_pos_color = QColor(0, 0, 255)
        
        # Distance tracking
        self.total_distance = 0.0
        self.current_position = np.array([0.0, 0.0, 0.0])
        
        # Initialize UI
        self._init_ui()
        
        print("Trajectory Widget initialized")
    
    def _init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # Control panel
        control_layout = QHBoxLayout()
        
        # Auto-scale checkbox
        self.autoscale_checkbox = QCheckBox("Auto Scale")
        self.autoscale_checkbox.setChecked(self.auto_scale)
        self.autoscale_checkbox.stateChanged.connect(self._toggle_autoscale)
        control_layout.addWidget(self.autoscale_checkbox)
        
        # Grid checkbox
        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(self.show_grid)
        self.grid_checkbox.stateChanged.connect(self._toggle_grid)
        control_layout.addWidget(self.grid_checkbox)
        
        # Reset button
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_trajectory)
        control_layout.addWidget(self.reset_button)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Distance display
        self.distance_label = QLabel("Distance: 0.00 m")
        self.distance_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(self.distance_label)
        
        # Position display
        self.position_label = QLabel("Position: (0.00, 0.00)")
        layout.addWidget(self.position_label)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Set background
        self.setStyleSheet("background-color: #1e1e1e; color: white;")
    
    def _toggle_autoscale(self, state):
        """Toggle auto-scaling"""
        self.auto_scale = state == Qt.CheckState.Checked.value
        if self.auto_scale:
            self._update_scale()
        self.update()
    
    def _toggle_grid(self, state):
        """Toggle grid display"""
        self.show_grid = state == Qt.CheckState.Checked.value
        self.update()
    
    def add_trajectory_point(self, position: np.ndarray):
        """
        Add a new point to the trajectory
        
        Args:
            position: 3D position [x, y, z]
        """
        if len(position) != 3:
            return
        
        # Store 3D point
        self.trajectory_points.append(position.copy())
        self.current_position = position.copy()
        
        # Convert to 2D screen coordinates
        screen_x, screen_y = self._world_to_screen(position[0], position[2])  # Use X and Z
        self.trajectory_2d.append((screen_x, screen_y))
        
        # Update distance
        if len(self.trajectory_points) > 1:
            prev_pos = self.trajectory_points[-2]
            distance = np.linalg.norm(position - prev_pos)
            self.total_distance += distance
        
        # Update scale if auto-scaling
        if self.auto_scale:
            self._update_scale()
        
        # Update displays
        self._update_labels()
        self.update()  # Trigger repaint
        
        # Emit signal
        self.trajectory_updated.emit(self.total_distance)
    
    def _world_to_screen(self, world_x: float, world_z: float) -> tuple:
        """
        Convert world coordinates to screen coordinates
        
        Args:
            world_x: X coordinate in meters
            world_z: Z coordinate in meters (forward direction)
            
        Returns:
            Tuple of (screen_x, screen_y) as integers
        """
        # Convert meters to pixels and center on screen
        screen_x = int(self.center_x + world_x * self.scale)
        screen_y = int(self.center_y - world_z * self.scale)  # Flip Y axis
        
        return screen_x, screen_y
    
    def _update_scale(self):
        """Update scale to fit all trajectory points"""
        if len(self.trajectory_points) < 2:
            return
        
        # Find bounding box of trajectory
        positions = np.array(self.trajectory_points)
        x_coords = positions[:, 0]
        z_coords = positions[:, 2]
        
        x_range = np.max(x_coords) - np.min(x_coords)
        z_range = np.max(z_coords) - np.min(z_coords)
        
        if x_range == 0 and z_range == 0:
            return
        
        # Calculate scale to fit in 80% of display area
        margin = 0.8
        max_range = max(x_range, z_range)
        
        if max_range > 0:
            scale_x = (self.plot_width * margin) / max_range
            scale_y = (self.plot_height * margin) / max_range
            self.scale = min(scale_x, scale_y, 200.0)  # Cap at 200 pixels/meter
        
        # Recalculate all screen coordinates
        self.trajectory_2d = []
        for pos in self.trajectory_points:
            screen_x, screen_y = self._world_to_screen(pos[0], pos[2])
            self.trajectory_2d.append((screen_x, screen_y))
    
    def _update_labels(self):
        """Update distance and position labels"""
        self.distance_label.setText(f"Distance: {self.total_distance:.2f} m")
        self.position_label.setText(
            f"Position: ({self.current_position[0]:.2f}, {self.current_position[2]:.2f})"
        )
    
    def paintEvent(self, event):
        """Paint the trajectory visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Clear background
        painter.fillRect(self.rect(), self.bg_color)
        
        # Calculate plot area
        plot_rect = self.rect().adjusted(10, 30, -10, -70)
        painter.setClipRect(plot_rect)
        
        # Draw grid
        if self.show_grid:
            self._draw_grid(painter, plot_rect)
        
        # Draw coordinate axes
        self._draw_axes(painter, plot_rect)
        
        # Draw trajectory
        self._draw_trajectory(painter)
        
        # Draw current position
        self._draw_current_position(painter)
        
        # Draw start position
        self._draw_start_position(painter)
        
        # Draw scale info
        painter.setClipRect(self.rect())
        self._draw_scale_info(painter)
    
    def _draw_grid(self, painter, rect):
        """Draw background grid"""
        painter.setPen(QPen(self.grid_color, 1))
        
        # Grid spacing in meters
        grid_spacing_m = 1.0
        grid_spacing_px = grid_spacing_m * self.scale
        
        # Adjust grid spacing if too dense or sparse
        if grid_spacing_px < 20:
            grid_spacing_m = 5.0
            grid_spacing_px = grid_spacing_m * self.scale
        elif grid_spacing_px > 100:
            grid_spacing_m = 0.5
            grid_spacing_px = grid_spacing_m * self.scale
        
        # Vertical lines
        x = int(self.center_x)
        while x < rect.right():
            painter.drawLine(x, rect.top(), x, rect.bottom())
            x += int(grid_spacing_px)
        
        x = int(self.center_x) - int(grid_spacing_px)
        while x > rect.left():
            painter.drawLine(x, rect.top(), x, rect.bottom())
            x -= int(grid_spacing_px)
        
        # Horizontal lines
        y = int(self.center_y)
        while y < rect.bottom():
            painter.drawLine(rect.left(), y, rect.right(), y)
            y += int(grid_spacing_px)
        
        y = int(self.center_y) - int(grid_spacing_px)
        while y > rect.top():
            painter.drawLine(rect.left(), y, rect.right(), y)
            y -= int(grid_spacing_px)
    
    def _draw_axes(self, painter, rect):
        """Draw coordinate axes"""
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        
        # X axis (horizontal)
        painter.drawLine(rect.left(), int(self.center_y), rect.right(), int(self.center_y))
        
        # Z axis (vertical)
        painter.drawLine(int(self.center_x), rect.top(), int(self.center_x), rect.bottom())
        
        # Draw axis labels
        painter.setPen(QPen(self.text_color, 1))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(rect.right() - 20, int(self.center_y) - 5, "X")
        painter.drawText(int(self.center_x) + 5, rect.top() + 15, "Z")
    
    def _draw_trajectory(self, painter):
        """Draw the trajectory path"""
        if len(self.trajectory_2d) < 2:
            return
        
        # Draw trajectory line
        painter.setPen(QPen(self.trajectory_color, 2))
        
        for i in range(1, len(self.trajectory_2d)):
            x1, y1 = self.trajectory_2d[i-1]
            x2, y2 = self.trajectory_2d[i]
            # Convert to integers
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        
        # Draw trajectory points
        painter.setPen(QPen(self.trajectory_color, 1))
        painter.setBrush(QBrush(self.trajectory_color))
        
        for x, y in self.trajectory_2d[::5]:  # Every 5th point to avoid clutter
            painter.drawEllipse(int(x)-2, int(y)-2, 4, 4)
    
    def _draw_current_position(self, painter):
        """Draw current position marker"""
        if len(self.trajectory_2d) == 0:
            return
        
        x, y = self.trajectory_2d[-1]
        
        # Draw circle for current position
        painter.setPen(QPen(self.current_pos_color, 2))
        painter.setBrush(QBrush(self.current_pos_color))
        painter.drawEllipse(x-6, y-6, 12, 12)
        
        # Draw direction arrow if we have previous point
        if len(self.trajectory_2d) > 1:
            prev_x, prev_y = self.trajectory_2d[-2]
            dx = x - prev_x
            dy = y - prev_y
            length = math.sqrt(dx*dx + dy*dy)
            
            if length > 0:
                # Normalize and scale
                dx = dx / length * 15
                dy = dy / length * 15
                
                # Convert to integers for drawing
                x_end = int(x + dx)
                y_end = int(y + dy)
                
                # Draw arrow
                painter.setPen(QPen(self.current_pos_color, 3))
                painter.drawLine(int(x), int(y), x_end, y_end)
                
                # Arrow head
                angle = math.atan2(dy, dx)
                head_len = 8
                head_angle = 0.5
                
                x1 = int(x_end - head_len * math.cos(angle - head_angle))
                y1 = int(y_end - head_len * math.sin(angle - head_angle))
                x2 = int(x_end - head_len * math.cos(angle + head_angle))
                y2 = int(y_end - head_len * math.sin(angle + head_angle))
                
                painter.drawLine(x_end, y_end, x1, y1)
                painter.drawLine(x_end, y_end, x2, y2)
    
    def _draw_start_position(self, painter):
        """Draw start position marker"""
        if len(self.trajectory_2d) == 0:
            return
        
        x, y = self.trajectory_2d[0]
        
        # Draw square for start position
        painter.setPen(QPen(self.start_pos_color, 2))
        painter.setBrush(QBrush(self.start_pos_color))
        painter.drawRect(int(x)-4, int(y)-4, 8, 8)
    
    def _draw_scale_info(self, painter):
        """Draw scale information"""
        painter.setPen(QPen(self.text_color, 1))
        painter.setFont(QFont("Arial", 8))
        
        scale_text = f"Scale: {self.scale:.0f} px/m"
        painter.drawText(10, self.height() - 10, scale_text)
    
    def reset_trajectory(self):
        """Reset trajectory to start"""
        self.trajectory_points = [np.array([0.0, 0.0, 0.0])]
        self.trajectory_2d = [(int(self.center_x), int(self.center_y))]
        self.total_distance = 0.0
        self.current_position = np.array([0.0, 0.0, 0.0])
        
        self._update_labels()
        self.update()
        
        print("Trajectory reset")
    
    def get_trajectory_data(self) -> dict:
        """Get trajectory data for export"""
        return {
            'points_3d': np.array(self.trajectory_points),
            'total_distance': self.total_distance,
            'num_points': len(self.trajectory_points)
        }

# Test function
def test_trajectory_widget():
    """Test the trajectory widget"""
    from PyQt6.QtWidgets import QApplication
    import time
    import math
    
    app = QApplication(sys.argv)
    
    widget = TrajectoryWidget()
    widget.show()
    
    # Simulate trajectory
    def add_test_points():
        for i in range(100):
            t = i * 0.1
            x = math.cos(t) * 2
            z = math.sin(t) * 2
            y = 0
            
            widget.add_trajectory_point(np.array([x, y, z]))
            app.processEvents()
            time.sleep(0.05)
    
    # Start adding points after a short delay
    from PyQt6.QtCore import QTimer
    timer = QTimer()
    timer.singleShot(1000, add_test_points)
    
    return app.exec()

if __name__ == "__main__":
    test_trajectory_widget()