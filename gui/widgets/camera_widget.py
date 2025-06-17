"""
Camera display widget for live video feed and depth visualization
"""

import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import logging

class CameraWidget(QWidget):
    """Widget for displaying camera feed and depth information"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Display parameters
        self.display_width = 640
        self.display_height = 480
        
        # Current images
        self.current_color_image = None
        self.current_depth_image = None
        
        # Display mode
        self.display_mode = "color"  # "color", "depth", "split"
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.color_btn = QPushButton("Color")
        self.color_btn.clicked.connect(lambda: self.set_display_mode("color"))
        button_layout.addWidget(self.color_btn)
        
        self.depth_btn = QPushButton("Depth")
        self.depth_btn.clicked.connect(lambda: self.set_display_mode("depth"))
        button_layout.addWidget(self.depth_btn)
        
        self.split_btn = QPushButton("Split")
        self.split_btn.clicked.connect(lambda: self.set_display_mode("split"))
        button_layout.addWidget(self.split_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # Camera display label
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(self.display_width, self.display_height)
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 2px solid #3498db;
                border-radius: 10px;
                background-color: #2c3e50;
                color: #ffffff;
            }
        """)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setText("Camera Feed\nWaiting for data...")
        self.camera_label.setScaledContents(True)
        
        layout.addWidget(self.camera_label)
        
        # Information panel
        info_layout = QHBoxLayout()
        
        self.info_label = QLabel("Resolution: Not connected")
        self.info_label.setStyleSheet("color: #bdc3c7;")
        info_layout.addWidget(self.info_label)
        
        info_layout.addStretch()
        
        self.timestamp_label = QLabel("Timestamp: --")
        self.timestamp_label.setStyleSheet("color: #bdc3c7;")
        info_layout.addWidget(self.timestamp_label)
        
        layout.addLayout(info_layout)
        
        # Update button states
        self.update_button_states()
    
    def set_display_mode(self, mode: str):
        """Set display mode"""
        self.display_mode = mode
        self.update_button_states()
        self.update_display()
    
    def update_button_states(self):
        """Update button visual states"""
        buttons = {
            "color": self.color_btn,
            "depth": self.depth_btn,
            "split": self.split_btn
        }
        
        for mode, button in buttons.items():
            if mode == self.display_mode:
                button.setStyleSheet("""
                    QPushButton {
                        background-color: #3498db;
                        border: 2px solid #2980b9;
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
    
    def update_frame(self, color_image: np.ndarray, depth_image: np.ndarray):
        """Update display with new frame data"""
        self.current_color_image = color_image.copy() if color_image is not None else None
        self.current_depth_image = depth_image.copy() if depth_image is not None else None
        
        self.update_display()
        self.update_info()
    
    def update_display(self):
        """Update the display based on current mode"""
        if self.current_color_image is None:
            return
        
        try:
            if self.display_mode == "color":
                display_image = self.current_color_image
            elif self.display_mode == "depth":
                display_image = self.create_depth_visualization()
            elif self.display_mode == "split":
                display_image = self.create_split_visualization()
            else:
                display_image = self.current_color_image
            
            # Convert to QPixmap and display
            qt_image = self.numpy_to_qt_image(display_image)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Scale to fit widget while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            
            self.camera_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.logger.error(f"Display update error: {e}")
    
    def create_depth_visualization(self) -> np.ndarray:
        """Create colorized depth visualization"""
        if self.current_depth_image is None:
            return self.current_color_image
        
        # Normalize depth values for visualization
        depth_normalized = cv2.normalize(
            self.current_depth_image, 
            None, 
            0, 255, 
            cv2.NORM_MINMAX, 
            dtype=cv2.CV_8U
        )
        
        # Apply colormap
        depth_colorized = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        return depth_colorized
    
    def create_split_visualization(self) -> np.ndarray:
        """Create split view with color and depth"""
        if self.current_color_image is None or self.current_depth_image is None:
            return self.current_color_image
        
        # Resize images to half width
        height, width = self.current_color_image.shape[:2]
        half_width = width // 2
        
        color_half = cv2.resize(self.current_color_image, (half_width, height))
        depth_colorized = self.create_depth_visualization()
        depth_half = cv2.resize(depth_colorized, (half_width, height))
        
        # Combine horizontally
        split_image = np.hstack([color_half, depth_half])
        
        # Add labels
        cv2.putText(split_image, "Color", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(split_image, "Depth", (half_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return split_image
    
    def numpy_to_qt_image(self, cv_image: np.ndarray) -> QImage:
        """Convert OpenCV image to QImage"""
        height, width, channel = cv_image.shape
        bytes_per_line = 3 * width
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        qt_image = QImage(
            rgb_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )
        
        return qt_image
    
    def update_info(self):
        """Update information display"""
        if self.current_color_image is not None:
            height, width = self.current_color_image.shape[:2]
            self.info_label.setText(f"Resolution: {width}x{height}")
            
            # Update timestamp
            from datetime import datetime
            self.timestamp_label.setText(f"Updated: {datetime.now().strftime('%H:%M:%S')}")
        else:
            self.info_label.setText("Resolution: Not connected")
            self.timestamp_label.setText("Timestamp: --")
    
    def clear_display(self):
        """Clear the display"""
        self.camera_label.clear()
        self.camera_label.setText("Camera Feed\nWaiting for data...")
        self.current_color_image = None
        self.current_depth_image = None
        self.update_info()