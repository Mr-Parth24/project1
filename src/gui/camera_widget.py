"""
Camera Widget for displaying live RealSense camera feed with SLAM features
"""

import sys
import numpy as np
import cv2
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox
from PyQt6.QtCore import QTimer, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
import yaml

class CameraWidget(QWidget):
    """Widget for displaying live camera feed from RealSense D435i with SLAM overlay"""
    
    # Signals
    frame_processed = pyqtSignal(np.ndarray, np.ndarray)  # color_frame, depth_frame
    slam_results_ready = pyqtSignal(dict)  # slam_results
    
    def __init__(self, camera_manager, slam_system=None, config_path="config/camera_config.yaml"):
        super().__init__()
        self.camera_manager = camera_manager
        self.slam_system = slam_system
        self.config = self._load_config(config_path)
        
        # Display settings
        self.display_width = self.config.get('gui', {}).get('camera_display_width', 640)
        self.display_height = self.config.get('gui', {}).get('camera_display_height', 480)
        
        # Frame processing
        self.latest_color_frame = None
        self.latest_depth_frame = None
        self.latest_slam_results = None
        self.show_depth = False
        self.show_features = True
        self.slam_enabled = False
        
        # Initialize UI
        self._init_ui()
        
        # Timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_frame)
        self.timer.start(33)  # ~30 FPS (33ms)
        
        print("Camera Widget initialized with SLAM integration")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()
        
        # Control panel
        control_layout = QHBoxLayout()
        
        # Start/Stop button
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self._toggle_camera)
        control_layout.addWidget(self.start_button)
        
        # Depth toggle
        self.depth_checkbox = QCheckBox("Show Depth")
        self.depth_checkbox.stateChanged.connect(self._toggle_depth_view)
        control_layout.addWidget(self.depth_checkbox)
        
        # Features toggle
        self.features_checkbox = QCheckBox("Show Features")
        self.features_checkbox.setChecked(self.show_features)
        self.features_checkbox.stateChanged.connect(self._toggle_features)
        control_layout.addWidget(self.features_checkbox)
        
        # SLAM toggle
        self.slam_checkbox = QCheckBox("Enable SLAM")
        self.slam_checkbox.stateChanged.connect(self._toggle_slam)
        control_layout.addWidget(self.slam_checkbox)
        
        # FPS label
        self.fps_label = QLabel("FPS: 0")
        control_layout.addWidget(self.fps_label)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(self.display_width, self.display_height)
        self.camera_label.setStyleSheet("border: 1px solid black; background-color: #2b2b2b;")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setText("Camera Not Started")
        layout.addWidget(self.camera_label)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
        # Frame rate calculation
        self.frame_count = 0
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self._calculate_fps)
        self.fps_timer.start(1000)  # Update FPS every second
    
    def _toggle_camera(self):
        """Start or stop the camera"""
        if not self.camera_manager.is_streaming:
            self._start_camera()
        else:
            self._stop_camera()
    
    def _start_camera(self):
        """Start camera streaming"""
        try:
            self.status_label.setText("Status: Initializing camera...")
            
            if not self.camera_manager.initialize_camera():
                self.status_label.setText("Status: Failed to initialize camera")
                return
            
            if not self.camera_manager.start_streaming():
                self.status_label.setText("Status: Failed to start streaming")
                return
            
            self.start_button.setText("Stop Camera")
            self.status_label.setText("Status: Camera running")
            
        except Exception as e:
            self.status_label.setText(f"Status: Error - {str(e)}")
            print(f"Error starting camera: {e}")
    
    def _stop_camera(self):
        """Stop camera streaming"""
        try:
            self.camera_manager.stop_streaming()
            self.start_button.setText("Start Camera")
            self.status_label.setText("Status: Camera stopped")
            self.camera_label.setText("Camera Stopped")
            
        except Exception as e:
            self.status_label.setText(f"Status: Error stopping camera - {str(e)}")
            print(f"Error stopping camera: {e}")
    
    def _toggle_depth_view(self, state):
        """Toggle between color and depth view"""
        self.show_depth = state == Qt.CheckState.Checked.value
        print(f"Depth view: {'ON' if self.show_depth else 'OFF'}")
    
    def _toggle_features(self, state):
        """Toggle feature overlay"""
        self.show_features = state == Qt.CheckState.Checked.value
        print(f"Feature overlay: {'ON' if self.show_features else 'OFF'}")
    
    def _toggle_slam(self, state):
        """Toggle SLAM processing"""
        self.slam_enabled = state == Qt.CheckState.Checked.value
        if self.slam_system and self.slam_enabled:
            print("SLAM processing: ON")
        else:
            print("SLAM processing: OFF")
    
    def _update_frame(self):
        """Update the camera frame display"""
        if not self.camera_manager.is_streaming:
            return
        
        try:
            # Get frames from camera
            frames = self.camera_manager.get_frames()
            if frames is None:
                return
            
            color_frame, depth_frame = frames
            self.latest_color_frame = color_frame
            self.latest_depth_frame = depth_frame
            
            # Process with SLAM if enabled
            slam_results = None
            if self.slam_system and self.slam_enabled:
                try:
                    slam_results = self.slam_system.process_frame(color_frame, depth_frame)
                    self.latest_slam_results = slam_results
                except Exception as e:
                    print(f"SLAM processing error: {e}")
            
            # Choose which frame to display
            if self.show_depth and depth_frame is not None:
                display_frame = self._process_depth_frame(depth_frame)
            else:
                display_frame = color_frame.copy()
                
                # Add feature overlay if enabled
                if self.show_features and slam_results:
                    # Get feature results from SLAM
                    if hasattr(self.slam_system, 'feature_detector'):
                        # Create mock feature results for overlay
                        feature_results = {
                            'keypoints': getattr(self.slam_system.feature_detector, 'prev_keypoints', []),
                            'num_features': slam_results.get('num_features', 0),
                            'matches': []
                        }
                        
                        if feature_results['keypoints']:
                            display_frame = self.slam_system.draw_features_overlay(display_frame, feature_results)
            
            # Convert to Qt format and display
            self._display_frame(display_frame)
            
            # Emit signals for other components
            self.frame_processed.emit(color_frame, depth_frame)
            if slam_results:
                self.slam_results_ready.emit(slam_results)
            
            # Update frame count for FPS calculation
            self.frame_count += 1
            
        except Exception as e:
            print(f"Error updating frame: {e}")
            self.status_label.setText(f"Status: Frame error - {str(e)}")
    
    def _process_depth_frame(self, depth_frame):
        """Process depth frame for visualization"""
        try:
            # Convert depth to colormap
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_frame, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            return depth_colormap
        except Exception as e:
            print(f"Error processing depth frame: {e}")
            return depth_frame
    
    def _display_frame(self, frame):
        """Display frame in the QLabel"""
        try:
            if frame is None:
                return
            
            # Resize frame to fit display
            frame_resized = cv2.resize(frame, (self.display_width, self.display_height))
            
            # Convert BGR to RGB for Qt
            if len(frame_resized.shape) == 3:
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                height, width, channel = frame_rgb.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            else:
                # Grayscale image
                height, width = frame_resized.shape
                bytes_per_line = width
                q_image = QImage(frame_resized.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
            
            # Convert to pixmap and display
            pixmap = QPixmap.fromImage(q_image)
            self.camera_label.setPixmap(pixmap)
            
        except Exception as e:
            print(f"Error displaying frame: {e}")
    
    def _calculate_fps(self):
        """Calculate and display FPS"""
        fps = self.frame_count
        self.fps_label.setText(f"FPS: {fps}")
        self.frame_count = 0
    
    def get_latest_frames(self):
        """Get the latest color and depth frames"""
        return self.latest_color_frame, self.latest_depth_frame
    
    def closeEvent(self, event):
        """Handle widget close event"""
        self._stop_camera()
        event.accept()