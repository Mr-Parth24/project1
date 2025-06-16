"""
Main GUI Application for Visual Odometry & SLAM System
Beautiful interface with camera feed, trajectory plot, and controls all in one screen
"""

import sys
import cv2
import numpy as np
import threading
import time
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl

from modules.realsense_manager import RealSenseManager
from modules.visual_odometry import VisualOdometry
from modules.data_exporter import DataExporter
import os

class VideoWidget(QLabel):
    """Custom widget for displaying camera feed"""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 2px solid #3498db; border-radius: 10px; background-color: #2c3e50;")
        self.setAlignment(Qt.AlignCenter)
        self.setText("Camera Feed\nConnecting...")
        self.setScaledContents(True)
        
    def update_frame(self, cv_img):
        """Convert OpenCV image to QPixmap and display"""
        if cv_img is not None:
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.setPixmap(pixmap)

class TrajectoryWidget(gl.GLViewWidget):
    """3D Trajectory visualization widget"""
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 400)
        
        # Set up the 3D view
        self.setCameraPosition(distance=10, elevation=20, azimuth=45)
        
        # Add grid
        grid = gl.GLGridItem()
        grid.scale(0.5, 0.5, 0.5)
        self.addItem(grid)
        
        # Add axes
        axis = gl.GLAxisItem()
        axis.setSize(2, 2, 2)
        self.addItem(axis)
        
        # Trajectory line
        self.trajectory_line = None
        self.current_pos_marker = None
        self.start_pos_marker = None
        
        # Data
        self.trajectory_points = []
        
    def update_trajectory(self, points, current_pos):
        """Update the 3D trajectory"""
        if len(points) < 2:
            return
            
        # Remove old items
        if self.trajectory_line is not None:
            self.removeItem(self.trajectory_line)
        if self.current_pos_marker is not None:
            self.removeItem(self.current_pos_marker)
        if self.start_pos_marker is not None:
            self.removeItem(self.start_pos_marker)
            
        # Convert to numpy array
        trajectory_array = np.array(points)
        
        # Create trajectory line
        self.trajectory_line = gl.GLLinePlotItem(
            pos=trajectory_array,
            color=(0.2, 0.6, 1.0, 1.0),
            width=3,
            antialias=True
        )
        self.addItem(self.trajectory_line)
        
        # Start position marker (green)
        start_pos = np.array([[trajectory_array[0]]])
        self.start_pos_marker = gl.GLScatterPlotItem(
            pos=start_pos,
            color=(0.2, 0.8, 0.2, 1.0),
            size=10
        )
        self.addItem(self.start_pos_marker)
        
        # Current position marker (red)
        curr_pos = np.array([[current_pos]])
        self.current_pos_marker = gl.GLScatterPlotItem(
            pos=curr_pos,
            color=(1.0, 0.2, 0.2, 1.0),
            size=12
        )
        self.addItem(self.current_pos_marker)
        
    def clear_trajectory(self):
        """Clear all trajectory data"""
        if self.trajectory_line is not None:
            self.removeItem(self.trajectory_line)
        if self.current_pos_marker is not None:
            self.removeItem(self.current_pos_marker)
        if self.start_pos_marker is not None:
            self.removeItem(self.start_pos_marker)
            
        self.trajectory_line = None
        self.current_pos_marker = None
        self.start_pos_marker = None

class SLAMTrackerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎯 Visual Odometry & SLAM System - Intel RealSense D435i")
        self.setWindowIcon(QIcon("icon.png"))  # Optional: add an icon
        self.resize(1400, 900)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            QPushButton {
                background-color: #3498db;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
            QLabel {
                color: #ecf0f1;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #34495e;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #3498db;
            }
        """)
        
        # Initialize components
        self.realsense = RealSenseManager()
        self.visual_odometry = VisualOdometry()
        self.exporter = DataExporter()
        
        # Tracking state
        self.is_running = False
        self.is_paused = False
        self.trajectory = []
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.total_distance = 0.0
        self.start_time = None
        self.frame_count = 0
        
        # Create data directory
        os.makedirs('data/output_paths', exist_ok=True)
        
        self.setup_ui()
        self.setup_timer()
        
    def setup_ui(self):
        """Set up the user interface"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Left panel (Camera feed and controls)
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel (Trajectory and info)
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)
        
    def create_left_panel(self):
        """Create left panel with camera feed and controls"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Camera feed group
        camera_group = QGroupBox("📷 Camera Feed")
        camera_layout = QVBoxLayout(camera_group)
        
        self.video_widget = VideoWidget()
        camera_layout.addWidget(self.video_widget)
        
        left_layout.addWidget(camera_group)
        
        # Controls group
        controls_group = QGroupBox("🎮 Controls")
        controls_layout = QGridLayout(controls_group)
        
        # Main control buttons
        self.start_btn = QPushButton("🚀 Start Tracking")
        self.start_btn.clicked.connect(self.start_tracking)
        self.start_btn.setStyleSheet("QPushButton { background-color: #27ae60; }")
        
        self.pause_btn = QPushButton("⏸️ Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet("QPushButton { background-color: #f39c12; }")
        
        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self.stop_tracking)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #e74c3c; }")
        
        self.reset_btn = QPushButton("🔄 Reset")
        self.reset_btn.clicked.connect(self.reset_trajectory)
        
        self.save_btn = QPushButton("💾 Save Data")
        self.save_btn.clicked.connect(self.save_trajectory)
        self.save_btn.setStyleSheet("QPushButton { background-color: #8e44ad; }")
        
        # Add buttons to grid
        controls_layout.addWidget(self.start_btn, 0, 0)
        controls_layout.addWidget(self.pause_btn, 0, 1)
        controls_layout.addWidget(self.stop_btn, 1, 0)
        controls_layout.addWidget(self.reset_btn, 1, 1)
        controls_layout.addWidget(self.save_btn, 2, 0, 1, 2)
        
        left_layout.addWidget(controls_group)
        
        # Status group
        status_group = QGroupBox("📊 Status Information")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Status: Ready")
        self.fps_label = QLabel("FPS: 0.0")
        self.features_label = QLabel("Features: 0")
        self.position_label = QLabel("Position: (0.00, 0.00, 0.00)")
        self.distance_label = QLabel("Distance: 0.00 m")
        self.points_label = QLabel("Trajectory Points: 0")
        self.time_label = QLabel("Time: 00:00:00")
        
        for label in [self.status_label, self.fps_label, self.features_label, 
                     self.position_label, self.distance_label, self.points_label, self.time_label]:
            label.setStyleSheet("QLabel { font-size: 11px; padding: 2px; }")
            status_layout.addWidget(label)
            
        left_layout.addWidget(status_group)
        
        return left_widget
        
    def create_right_panel(self):
        """Create right panel with trajectory plot"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Trajectory group
        trajectory_group = QGroupBox("🗺️ 3D Trajectory Visualization")
        trajectory_layout = QVBoxLayout(trajectory_group)
        
        self.trajectory_widget = TrajectoryWidget()
        trajectory_layout.addWidget(self.trajectory_widget)
        
        # Trajectory info
        info_layout = QHBoxLayout()
        
        self.loop_closure_label = QLabel("Loop Closure: Not Detected")
        self.drift_label = QLabel("Estimated Drift: 0.00 m")
        
        info_layout.addWidget(self.loop_closure_label)
        info_layout.addWidget(self.drift_label)
        
        trajectory_layout.addLayout(info_layout)
        
        right_layout.addWidget(trajectory_group)
        
        # Data export group
        export_group = QGroupBox("💾 Data Export")
        export_layout = QVBoxLayout(export_group)
        
        export_info_layout = QHBoxLayout()
        
        self.last_save_label = QLabel("Last Save: Never")
        self.file_count_label = QLabel("Saved Files: 0")
        
        export_info_layout.addWidget(self.last_save_label)
        export_info_layout.addWidget(self.file_count_label)
        
        export_layout.addLayout(export_info_layout)
        
        # Export buttons
        export_buttons_layout = QHBoxLayout()
        
        self.export_csv_btn = QPushButton("📄 Export CSV")
        self.export_csv_btn.clicked.connect(lambda: self.export_data('csv'))
        
        self.export_json_btn = QPushButton("📋 Export JSON")
        self.export_json_btn.clicked.connect(lambda: self.export_data('json'))
        
        self.open_folder_btn = QPushButton("📁 Open Folder")
        self.open_folder_btn.clicked.connect(self.open_data_folder)
        
        export_buttons_layout.addWidget(self.export_csv_btn)
        export_buttons_layout.addWidget(self.export_json_btn)
        export_buttons_layout.addWidget(self.open_folder_btn)
        
        export_layout.addLayout(export_buttons_layout)
        
        right_layout.addWidget(export_group)
        
        return right_widget
        
    def setup_timer(self):
        """Set up update timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        
    def start_tracking(self):
        """Start SLAM tracking"""
        if not self.realsense.start():
            QMessageBox.critical(self, "Error", "Failed to start RealSense camera!\n\nMake sure your Intel RealSense D435i is connected.")
            return
            
        self.is_running = True
        self.is_paused = False
        self.start_time = time.time()
        self.frame_count = 0
        
        # Update button states
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)
        
        # Start timer
        self.timer.start(33)  # ~30 FPS
        
        self.update_status("Tracking Started")
        
    def toggle_pause(self):
        """Toggle pause state"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.setText("▶️ Resume")
            self.update_status("Paused")
        else:
            self.pause_btn.setText("⏸️ Pause")
            self.update_status("Tracking")
            
    def stop_tracking(self):
        """Stop SLAM tracking"""
        self.is_running = False
        self.timer.stop()
        self.realsense.stop()
        
        # Update button states
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setText("⏸️ Pause")
        
        self.update_status("Stopped")
        
    def reset_trajectory(self):
        """Reset trajectory data"""
        self.trajectory = []
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.total_distance = 0.0
        self.frame_count = 0
        self.visual_odometry.reset()
        self.trajectory_widget.clear_trajectory()
        
        self.update_status("Trajectory Reset")
        
    def update_display(self):
        """Main update loop"""
        if not self.is_running:
            return
            
        # Get camera frames
        color_frame, depth_frame = self.realsense.get_frames()
        if color_frame is None:
            return
            
        self.frame_count += 1
        
        # Process frame for visual odometry
        if not self.is_paused:
            movement = self.visual_odometry.process_frame(color_frame, depth_frame)
            
            if movement is not None:
                # Update position
                self.current_position += movement
                self.trajectory.append(self.current_position.copy())
                
                # Calculate distance
                if len(self.trajectory) > 1:
                    distance = np.linalg.norm(movement)
                    self.total_distance += distance
                
                # Update trajectory visualization
                if len(self.trajectory) > 1:
                    self.trajectory_widget.update_trajectory(self.trajectory, self.current_position)
                    
                # Check for loop closure
                self.check_loop_closure()
        
        # Draw overlays on camera frame
        self.draw_overlays(color_frame)
        
        # Update video display
        self.video_widget.update_frame(color_frame)
        
        # Update labels
        self.update_labels()
        
    def draw_overlays(self, frame):
        """Draw information overlays on camera frame"""
        # Get current features
        features = self.visual_odometry.get_current_features()
        
        # Draw feature points
        for point in features[:50]:  # Limit for performance
            cv2.circle(frame, tuple(map(int, point)), 3, (0, 255, 255), -1)
            
        # Draw status overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Status text
        status_text = "PAUSED" if self.is_paused else "TRACKING"
        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Features: {len(features)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Distance: {self.total_distance:.2f}m", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
    def update_labels(self):
        """Update information labels"""
        # Calculate FPS
        elapsed_time = time.time() - self.start_time if self.start_time else 1
        fps = self.frame_count / elapsed_time
        
        # Update labels
        status = "Paused" if self.is_paused else ("Tracking" if self.is_running else "Ready")
        self.status_label.setText(f"Status: {status}")
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
        features = self.visual_odometry.get_current_features()
        self.features_label.setText(f"Features: {len(features)}")
        
        self.position_label.setText(f"Position: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f})")
        self.distance_label.setText(f"Distance: {self.total_distance:.2f} m")
        self.points_label.setText(f"Trajectory Points: {len(self.trajectory)}")
        
        # Format time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        self.time_label.setText(f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
    def check_loop_closure(self):
        """Check for loop closure"""
        if len(self.trajectory) < 10:
            return
            
        current_pos = self.current_position
        start_pos = self.trajectory[0]
        
        distance_to_start = np.linalg.norm(current_pos - start_pos)
        
        if distance_to_start < 0.5:  # Within 50cm of start
            self.loop_closure_label.setText("Loop Closure: ✅ Detected")
            self.loop_closure_label.setStyleSheet("QLabel { color: #27ae60; }")
        else:
            self.loop_closure_label.setText(f"Loop Closure: Distance to start: {distance_to_start:.2f}m")
            self.loop_closure_label.setStyleSheet("QLabel { color: #e74c3c; }")
            
        # Calculate drift
        if len(self.trajectory) > 1:
            net_displacement = np.linalg.norm(current_pos - start_pos)
            self.drift_label.setText(f"Estimated Drift: {net_displacement:.2f} m")
            
    def save_trajectory(self):
        """Save trajectory data"""
        if len(self.trajectory) == 0:
            QMessageBox.warning(self, "Warning", "No trajectory data to save!")
            return
            
        try:
            filename = self.exporter.save_trajectory(self.trajectory, self.total_distance)
            self.last_save_label.setText(f"Last Save: {datetime.now().strftime('%H:%M:%S')}")
            
            # Count files in output directory
            file_count = len([f for f in os.listdir('data/output_paths') if f.endswith('.json')])
            self.file_count_label.setText(f"Saved Files: {file_count}")
            
            QMessageBox.information(self, "Success", f"Trajectory saved successfully!\n\nFile: {os.path.basename(filename)}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save trajectory:\n{str(e)}")
            
    def export_data(self, format_type):
        """Export data in specific format"""
        if len(self.trajectory) == 0:
            QMessageBox.warning(self, "Warning", "No trajectory data to export!")
            return
            
        file_dialog = QFileDialog()
        if format_type == 'csv':
            filename, _ = file_dialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
        else:
            filename, _ = file_dialog.getSaveFileName(self, "Export JSON", "", "JSON Files (*.json)")
            
        if filename:
            try:
                if format_type == 'csv':
                    self.exporter.export_csv(self.trajectory, filename)
                else:
                    self.exporter.export_json(self.trajectory, self.total_distance, filename)
                    
                QMessageBox.information(self, "Success", f"Data exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")
                
    def open_data_folder(self):
        """Open data folder in file explorer"""
        import subprocess
        import platform
        
        folder_path = os.path.abspath('data/output_paths')
        
        if platform.system() == 'Windows':
            subprocess.run(['explorer', folder_path])
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', folder_path])
        else:  # Linux
            subprocess.run(['xdg-open', folder_path])
            
    def update_status(self, message):
        """Update status message"""
        self.status_label.setText(f"Status: {message}")
        
    def closeEvent(self, event):
        """Handle application close"""
        if self.is_running:
            self.stop_tracking()
        event.accept()

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("SLAM Tracker")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("Visual Odometry Systems")
    
    # Create and show main window
    window = SLAMTrackerGUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()