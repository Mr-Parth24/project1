"""
Main Window for Agricultural SLAM System - Phase 2
Includes SLAM processing and trajectory visualization
"""

import sys
import os
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QMenuBar, QStatusBar, QMessageBox, 
                            QSplitter, QTextEdit, QLabel, QPushButton, QGroupBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QAction
import yaml
import numpy as np

# Import our custom widgets and SLAM system
from src.gui.camera_widget import CameraWidget
from src.gui.trajectory_widget import TrajectoryWidget
from src.core.camera_manager import CameraManager
from src.algorithms.custom_visual_slam import CustomVisualSLAM

class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, config_path="config/camera_config.yaml"):
        super().__init__()
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize camera manager
        self.camera_manager = CameraManager(config_path)
        
        # Initialize SLAM system
        self.slam_system = CustomVisualSLAM()
        
        # SLAM state tracking
        self.slam_active = False
        self.trajectory_points = 0
        
        # Set up the main window
        self._init_ui()
        self._create_menu_bar()
        self._create_status_bar()
        
        # Window properties
        self.setWindowTitle(self.config.get('gui', {}).get('window_title', 'Agricultural SLAM System'))
        self.setGeometry(100, 100, 
                        self.config.get('gui', {}).get('window_width', 1400),
                        self.config.get('gui', {}).get('window_height', 900))
        
        # Connect SLAM signals
        self._connect_slam_signals()
        
        print("Main Window initialized with SLAM integration")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                print(f"Main window config loaded from {config_path}")
                return config
        except Exception as e:
            print(f"Error loading config: {e}")
            return {
                'gui': {
                    'window_title': 'Agricultural SLAM System',
                    'window_width': 1200,
                    'window_height': 800
                }
            }
    
    def _init_ui(self):
        """Initialize the user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Camera and controls
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # Center panel - Trajectory and SLAM visualization
        center_panel = self._create_center_panel()
        splitter.addWidget(center_panel)
        
        # Right panel - Information and logs
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions (40% left, 35% center, 25% right)
        splitter.setSizes([560, 490, 350])
    
    def _create_left_panel(self):
        """Create the left panel with camera feed"""
        left_widget = QWidget()
        layout = QVBoxLayout(left_widget)
        
        # Title
        title = QLabel("Live Camera Feed")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Camera widget with SLAM integration
        self.camera_widget = CameraWidget(self.camera_manager, self.slam_system)
        layout.addWidget(self.camera_widget)
        
        # Connect camera widget signals
        self.camera_widget.frame_processed.connect(self._on_frame_processed)
        self.camera_widget.slam_results_ready.connect(self._on_slam_results)
        
        return left_widget
    
    def _create_center_panel(self):
        """Create the center panel with trajectory visualization"""
        center_widget = QWidget()
        layout = QVBoxLayout(center_widget)
        
        # Trajectory section
        traj_group = QGroupBox("Trajectory Visualization")
        traj_layout = QVBoxLayout(traj_group)
        
        # Trajectory widget
        self.trajectory_widget = TrajectoryWidget(width=450, height=350)
        traj_layout.addWidget(self.trajectory_widget)
        
        # Connect trajectory signals
        self.trajectory_widget.trajectory_updated.connect(self._on_trajectory_updated)
        
        layout.addWidget(traj_group)
        
        # SLAM Controls section
        slam_group = QGroupBox("SLAM Controls")
        slam_layout = QVBoxLayout(slam_group)
        
        # SLAM status display
        self.slam_status_detailed = QTextEdit()
        self.slam_status_detailed.setMaximumHeight(120)
        self.slam_status_detailed.setReadOnly(True)
        self.slam_status_detailed.setPlainText("SLAM Status:\nNot initialized\nWaiting for camera...")
        slam_layout.addWidget(self.slam_status_detailed)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.reset_slam_button = QPushButton("Reset SLAM")
        self.reset_slam_button.clicked.connect(self._reset_slam)
        button_layout.addWidget(self.reset_slam_button)
        
        self.save_trajectory_button = QPushButton("Save Trajectory")
        self.save_trajectory_button.clicked.connect(self._save_trajectory)
        button_layout.addWidget(self.save_trajectory_button)
        
        slam_layout.addLayout(button_layout)
        layout.addWidget(slam_group)
        
        return center_widget
    
    def _create_right_panel(self):
        """Create the right panel with information and controls"""
        right_widget = QWidget()
        layout = QVBoxLayout(right_widget)
        
        # System Information
        info_label = QLabel("System Information")
        info_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(info_label)
        
        # System status
        self.system_info = QTextEdit()
        self.system_info.setMaximumHeight(150)
        self.system_info.setReadOnly(True)
        self.system_info.setPlainText(self._get_system_info())
        layout.addWidget(self.system_info)
        
        # SLAM Metrics
        metrics_label = QLabel("SLAM Metrics")
        metrics_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(metrics_label)
        
        self.slam_metrics = QTextEdit()
        self.slam_metrics.setMaximumHeight(200)
        self.slam_metrics.setReadOnly(True)
        self.slam_metrics.setPlainText("Distance: 0.00 m\nFeatures: 0\nKeyframes: 0\nMap Points: 0")
        layout.addWidget(self.slam_metrics)
        
        # Performance Monitor
        perf_label = QLabel("Performance")
        perf_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        layout.addWidget(perf_label)
        
        self.performance_display = QTextEdit()
        self.performance_display.setMaximumHeight(120)
        self.performance_display.setReadOnly(True)
        self.performance_display.setPlainText("Processing Time: 0.0 ms\nCamera FPS: 0\nSLAM FPS: 0")
        layout.addWidget(self.performance_display)
        
        # Log display
        log_label = QLabel("System Logs")
        log_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        layout.addWidget(log_label)
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        layout.addWidget(self.log_display)
        
        # Add initial log entries
        self._log_message("Phase 2 SLAM system initialized")
        self._log_message("Ready to start camera and SLAM")
        
        return right_widget
    
    def _create_menu_bar(self):
        """Create the menu bar"""
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu('File')
        
        # Save trajectory action
        save_traj_action = QAction('Save Trajectory', self)
        save_traj_action.setShortcut('Ctrl+S')
        save_traj_action.triggered.connect(self._save_trajectory)
        file_menu.addAction(save_traj_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Camera Menu
        camera_menu = menubar.addMenu('Camera')
        
        # Start camera action
        start_camera_action = QAction('Start Camera', self)
        start_camera_action.setShortcut('Ctrl+C')
        start_camera_action.triggered.connect(self._start_camera)
        camera_menu.addAction(start_camera_action)
        
        # Stop camera action
        stop_camera_action = QAction('Stop Camera', self)
        stop_camera_action.setShortcut('Ctrl+T')
        stop_camera_action.triggered.connect(self._stop_camera)
        camera_menu.addAction(stop_camera_action)
        
        # SLAM Menu
        slam_menu = menubar.addMenu('SLAM')
        
        # Start SLAM action
        start_slam_action = QAction('Enable SLAM', self)
        start_slam_action.setShortcut('Ctrl+R')
        start_slam_action.triggered.connect(self._toggle_slam)
        slam_menu.addAction(start_slam_action)
        
        # Reset SLAM action
        reset_slam_action = QAction('Reset SLAM', self)
        reset_slam_action.setShortcut('Ctrl+E')
        reset_slam_action.triggered.connect(self._reset_slam)
        slam_menu.addAction(reset_slam_action)
        
        # Help Menu
        help_menu = menubar.addMenu('Help')
        
        # About action
        about_action = QAction('About', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_status_bar(self):
        """Create the status bar"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage('Ready')
        
        # Add permanent widgets to status bar
        self.connection_status = QLabel("Camera: Disconnected")
        self.status_bar.addPermanentWidget(self.connection_status)
    
    def _get_system_info(self):
        """Get system information text"""
        try:
            import platform
            info = f"""System: {platform.system()} {platform.release()}
Python: {platform.python_version()}
Camera: Intel RealSense D435i
Mode: Visual SLAM (No IMU)
Resolution: {self.config.get('camera', {}).get('color_width', 640)}x{self.config.get('camera', {}).get('color_height', 480)}
FPS: {self.config.get('camera', {}).get('fps', 30)}
SLAM: Custom Implementation
Features: ORB + Visual Odometry
"""
            return info
        except Exception as e:
            return f"Error getting system info: {e}"
    
    def _connect_slam_signals(self):
        """Connect SLAM-related signals"""
        # Timer for updating SLAM displays
        self.slam_update_timer = QTimer()
        self.slam_update_timer.timeout.connect(self._update_slam_displays)
        self.slam_update_timer.start(1000)  # Update every second
    
    def _toggle_slam(self):
        """Toggle SLAM processing"""
        if hasattr(self.camera_widget, 'slam_checkbox'):
            current_state = self.camera_widget.slam_checkbox.isChecked()
            self.camera_widget.slam_checkbox.setChecked(not current_state)
    
    def _start_slam(self):
        """Start SLAM processing"""
        if hasattr(self.camera_widget, 'slam_checkbox'):
            self.camera_widget.slam_checkbox.setChecked(True)
    
    def _reset_slam(self):
        """Reset SLAM system"""
        try:
            if self.slam_system:
                self.slam_system.reset()
                self.trajectory_widget.reset_trajectory()
                self.trajectory_points = 0
                self._log_message("SLAM system reset")
                self._update_slam_displays()
        except Exception as e:
            self._log_message(f"Error resetting SLAM: {e}")
    
    def _save_trajectory(self):
        """Save trajectory data"""
        try:
            trajectory_data = self.trajectory_widget.get_trajectory_data()
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.npz"
            
            # Create data directory if it doesn't exist
            os.makedirs("data/trajectories", exist_ok=True)
            filepath = os.path.join("data/trajectories", filename)
            
            # Save trajectory data
            np.savez(filepath, 
                    points=trajectory_data['points_3d'],
                    distance=trajectory_data['total_distance'],
                    num_points=trajectory_data['num_points'])
            
            self._log_message(f"Trajectory saved to {filepath}")
            QMessageBox.information(self, "Trajectory Saved", 
                                  f"Trajectory data saved to:\n{filepath}")
            
        except Exception as e:
            self._log_message(f"Error saving trajectory: {e}")
            QMessageBox.warning(self, "Save Error", f"Failed to save trajectory:\n{str(e)}")
    
    def _on_slam_results(self, slam_results):
        """Handle SLAM processing results"""
        try:
            if slam_results.get('pose_estimated', False):
                # Update trajectory visualization
                position = slam_results.get('position', np.array([0, 0, 0]))
                self.trajectory_widget.add_trajectory_point(position)
                self.trajectory_points += 1
                
                # Log successful tracking occasionally
                if self.trajectory_points % 30 == 0:
                    distance = slam_results.get('distance_traveled', 0.0)
                    self._log_message(f"SLAM tracking: {self.trajectory_points} points, {distance:.2f}m traveled")
            else:
                # Log tracking issues occasionally
                if hasattr(self, '_tracking_fail_count'):
                    self._tracking_fail_count += 1
                else:
                    self._tracking_fail_count = 1
                
                if self._tracking_fail_count % 60 == 0:  # Every 2 seconds
                    debug_info = slam_results.get('debug_info', 'No debug info')
                    self._log_message(f"SLAM tracking issue: {debug_info}")
            
            # Update SLAM status
            self.slam_active = slam_results.get('tracking', False)
            
        except Exception as e:
            print(f"Error handling SLAM results: {e}")
            self._log_message(f"SLAM result error: {e}")
    
    def _on_trajectory_updated(self, distance):
        """Handle trajectory updates"""
        self.status_bar.showMessage(f"Distance traveled: {distance:.2f} m")
    
    def _update_slam_displays(self):
        """Update SLAM-related displays"""
        try:
            if self.slam_system:
                # Get SLAM statistics
                stats = self.slam_system.get_statistics()
                
                # Update detailed SLAM status
                slam_status_text = f"""SLAM Status: {'TRACKING' if stats['tracking'] else 'LOST' if stats['initialized'] else 'NOT STARTED'}
Initialized: {'Yes' if stats['initialized'] else 'No'}
Frames Processed: {stats['frame_count']}
Keyframes: {stats['num_keyframes']}
Distance: {stats['distance_traveled']:.2f} m"""
                
                self.slam_status_detailed.setPlainText(slam_status_text)
                
                # Update metrics display
                metrics_text = f"""Distance: {stats['distance_traveled']:.2f} m
Features: {stats['avg_features']:.0f} avg
Matches: {stats['avg_matches']:.0f} avg
Keyframes: {stats['num_keyframes']}
Map Points: {stats['num_map_points']}
Trajectory Points: {stats['trajectory_length']}"""
                
                self.slam_metrics.setPlainText(metrics_text)
                
                # Update performance display
                perf_text = f"""Processing Time: {stats['avg_processing_time']*1000:.1f} ms
SLAM FPS: {1.0/max(stats['avg_processing_time'], 0.001):.1f}
Tracking: {'Active' if stats['tracking'] else 'Inactive'}
Quality: {'Good' if stats['avg_matches'] > 50 else 'Fair' if stats['avg_matches'] > 20 else 'Poor'}"""
                
                self.performance_display.setPlainText(perf_text)
                
        except Exception as e:
            print(f"Error updating SLAM displays: {e}")
    
    def _start_camera(self):
        """Start camera from menu"""
        if hasattr(self.camera_widget, 'start_button'):
            if not self.camera_manager.is_streaming:
                self.camera_widget._start_camera()
    
    def _stop_camera(self):
        """Stop camera from menu"""
        if hasattr(self.camera_widget, 'start_button'):
            if self.camera_manager.is_streaming:
                self.camera_widget._stop_camera()
    
    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About Agricultural SLAM System",
                         "Agricultural SLAM System v2.0 - Phase 2\n\n"
                         "Visual SLAM system using Intel RealSense D435i\n"
                         "for agricultural equipment tracking.\n\n"
                         "Features:\n"
                         "• Real-time visual odometry\n"
                         "• Feature detection and tracking\n"
                         "• 2D trajectory visualization\n"
                         "• Distance measurement\n"
                         "• No IMU dependency\n\n"
                         "Built with Python, PyQt6, OpenCV, and custom SLAM algorithms")
    
    def _on_frame_processed(self, color_frame, depth_frame):
        """Handle processed frames from camera widget"""
        # Update connection status
        if color_frame is not None:
            self.connection_status.setText("Camera: Connected")
            self.status_bar.showMessage(f"Streaming - Frame size: {color_frame.shape}")
        else:
            self.connection_status.setText("Camera: Error")
            self.status_bar.showMessage("Frame processing error")
        
        # Update frame count for display
        if hasattr(self, '_frame_count'):
            self._frame_count += 1
        else:
            self._frame_count = 1
    
    def _log_message(self, message):
        """Add a message to the log display"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.log_display.append(log_entry)
        print(log_entry)  # Also print to console
    
    def closeEvent(self, event):
        """Handle application close event"""
        self._log_message("Shutting down system...")
        
        # Stop SLAM system
        if self.slam_system:
            self.slam_system.reset()
        
        # Stop camera if running
        if self.camera_manager.is_streaming:
            self.camera_manager.stop_streaming()
        
        self._log_message("System shutdown complete")
        event.accept()

# Test function for the main window
def test_main_window():
    """Test the main window with SLAM integration"""
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    return app.exec()

if __name__ == "__main__":
    test_main_window()