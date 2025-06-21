#!/usr/bin/env python3
"""
IMMEDIATE GUI FIX for Camera Display
Mr-Parth24 | 2025-06-20 23:40:01 UTC
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QLabel, QPushButton
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QPixmap, QImage
import cv2
import numpy as np
from src.core.camera_manager import CameraManager

class FixedCameraGUI(QMainWindow):
    """Fixed GUI with working camera display"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mr-Parth24 FIXED Camera GUI - 2025-06-20 23:40:01")
        self.setGeometry(100, 100, 1400, 600)
        
        # Initialize camera
        self.camera = CameraManager()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.camera_active = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup WORKING UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        
        # === LEFT: CAMERA DISPLAY (THE FIX!) ===
        camera_panel = QWidget()
        camera_panel.setMinimumSize(660, 550)
        camera_panel.setStyleSheet("""
            QWidget {
                border: 3px solid #4CAF50;
                border-radius: 10px;
                background-color: #f0f0f0;
            }
        """)
        
        camera_layout = QVBoxLayout(camera_panel)
        
        # Camera display label
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 2px solid #333;
                background-color: #000;
                color: #4CAF50;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setText("Mr-Parth24 Camera Display\n2025-06-20 23:40:01 UTC\n\nClick Start Camera")
        
        # Camera controls
        self.start_button = QPushButton("Start Camera")
        self.start_button.clicked.connect(self.toggle_camera)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
        """)
        
        camera_layout.addWidget(self.camera_label)
        camera_layout.addWidget(self.start_button)
        
        # === RIGHT: INFO PANEL ===
        info_panel = QWidget()
        info_panel.setMinimumSize(300, 550)
        info_panel.setStyleSheet("""
            QWidget {
                border: 2px solid #2196F3;
                border-radius: 10px;
                background-color: #f9f9f9;
            }
        """)
        
        info_layout = QVBoxLayout(info_panel)
        info_label = QLabel("""
        FIXED CAMERA GUI
        
        User: Mr-Parth24
        Date: 2025-06-20 23:40:01 UTC
        
        ✅ Camera: Working
        ✅ Display: Fixed
        ✅ Layout: Corrected
        
        This shows how the camera
        display should appear in
        the main GUI.
        
        If this works, we can apply
        the same fix to the main
        application.
        """)
        info_label.setStyleSheet("""
            QLabel {
                padding: 20px;
                font-size: 12px;
                color: #333;
            }
        """)
        info_layout.addWidget(info_label)
        
        # Add panels to main layout
        main_layout.addWidget(camera_panel, 2)  # Camera gets more space
        main_layout.addWidget(info_panel, 1)    # Info panel smaller
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        if not self.camera_active:
            if self.camera.initialize_camera() and self.camera.start_streaming():
                self.timer.start(33)  # 30 FPS
                self.camera_active = True
                self.start_button.setText("Stop Camera")
                print("✅ Camera started in FIXED GUI")
            else:
                print("❌ Camera start failed")
        else:
            self.timer.stop()
            self.camera.stop_streaming()
            self.camera_active = False
            self.start_button.setText("Start Camera")
            self.camera_label.setText("Mr-Parth24 Camera Display\n2025-06-20 23:40:01 UTC\n\nClick Start Camera")
    
    def update_frame(self):
        """Update camera frame"""
        try:
            frames = self.camera.get_frames()
            if frames:
                color_image, depth_image = frames
                
                # Add overlay
                cv2.putText(color_image, "Mr-Parth24 FIXED GUI", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(color_image, "Camera Display Working!", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(color_image, f"Frame: {self.timer.remainingTime()}", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                
                # Convert to QPixmap
                height, width, channel = color_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(color_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_image)
                
                # Scale and display
                scaled_pixmap = pixmap.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.camera_label.setPixmap(scaled_pixmap)
                
        except Exception as e:
            print(f"Frame update error: {e}")
    
    def closeEvent(self, event):
        """Cleanup on close"""
        if self.camera_active:
            self.timer.stop()
            self.camera.stop_streaming()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = FixedCameraGUI()
    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())