"""
Enhanced Camera Widget for Agricultural SLAM System
Displays live camera feed with agricultural feature overlays
Optimized for real-time performance with interactive features
"""

import numpy as np
import cv2
import math
from typing import Dict, List, Optional, Tuple
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QCheckBox, QSlider
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont, QBrush

class EnhancedCameraWidget(QWidget):
    """
    Enhanced Camera Widget with Agricultural Feature Overlays
    Provides real-time visualization of SLAM features and agricultural scene analysis
    """
    
    # Signals for interaction
    point_clicked = pyqtSignal(int, int)  # x, y coordinates clicked
    roi_selected = pyqtSignal(int, int, int, int)  # x, y, width, height
    
    def __init__(self, width: int = 640, height: int = 480):
        super().__init__()
        
        # Display parameters
        self.display_width = width
        self.display_height = height
        self.aspect_ratio = width / height
        
        # Current frame data
        self.current_frame = None
        self.current_depth = None
        self.current_features = []
        self.current_matches = []
        self.agricultural_features = {}
        
        # Overlay settings
        self.show_features = True
        self.show_matches = True
        self.show_agricultural = True
        self.show_depth_overlay = False
        self.show_grid = False
        self.show_center_cross = True
        
        # Agricultural visualization settings
        self.crop_row_color = QColor(0, 255, 0)  # Green
        self.ground_plane_color = QColor(255, 255, 0)  # Yellow
        self.feature_color = QColor(0, 255, 255)  # Cyan
        self.match_color = QColor(255, 0, 255)  # Magenta
        
        # Performance optimization
        self.frame_skip_counter = 0
        self.frame_skip_threshold = 1  # Process every frame
        self.last_update_time = 0
        self.target_fps = 30
        
        # Interactive features
        self.mouse_tracking = True
        self.current_mouse_pos = (0, 0)
        self.click_tolerance = 10  # pixels
        
        # Statistical tracking
        self.frame_count = 0
        self.feature_count_history = []
        self.agricultural_score_history = []
        
        # Initialize UI
        self.init_ui()
        
        print(f"Enhanced Camera Widget initialized: {width}x{height}")
    
    def init_ui(self):
        """Initialize the enhanced camera widget UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Camera display label
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(self.display_width, self.display_height)
        self.camera_label.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                border-radius: 4px;
                background-color: #000000;
            }
        """)
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setText("Camera Feed\nWaiting for frames...")
        self.camera_label.setStyleSheet("color: white; font-size: 14px;")
        
        # Enable mouse tracking
        self.camera_label.setMouseTracking(True)
        self.camera_label.mousePressEvent = self.on_mouse_click
        self.camera_label.mouseMoveEvent = self.on_mouse_move
        
        layout.addWidget(self.camera_label)
        
        # Control panel
        controls_layout = QHBoxLayout()
        
        # Feature overlay controls
        self.features_checkbox = QCheckBox("Features")
        self.features_checkbox.setChecked(self.show_features)
        self.features_checkbox.stateChanged.connect(self.toggle_features)
        controls_layout.addWidget(self.features_checkbox)
        
        self.agricultural_checkbox = QCheckBox("Agricultural")
        self.agricultural_checkbox.setChecked(self.show_agricultural)
        self.agricultural_checkbox.stateChanged.connect(self.toggle_agricultural)
        controls_layout.addWidget(self.agricultural_checkbox)
        
        self.depth_checkbox = QCheckBox("Depth")
        self.depth_checkbox.setChecked(self.show_depth_overlay)
        self.depth_checkbox.stateChanged.connect(self.toggle_depth_overlay)
        controls_layout.addWidget(self.depth_checkbox)
        
        self.grid_checkbox = QCheckBox("Grid")
        self.grid_checkbox.setChecked(self.show_grid)
        self.grid_checkbox.stateChanged.connect(self.toggle_grid)
        controls_layout.addWidget(self.grid_checkbox)
        
        controls_layout.addStretch()
        
        # Information display
        self.info_label = QLabel("Ready")
        self.info_label.setStyleSheet("color: #666666; font-size: 10px;")
        controls_layout.addWidget(self.info_label)
        
        layout.addLayout(controls_layout)
    
    def update_frame(self, color_frame: np.ndarray, depth_frame: np.ndarray = None, 
                    features: List = None, matches: List = None, 
                    agricultural_features: Dict = None):
        """
        Update camera display with new frame and overlay data
        
        Args:
            color_frame: RGB color frame
            depth_frame: Optional depth frame
            features: List of detected features
            matches: List of feature matches
            agricultural_features: Dictionary with agricultural scene data
        """
        try:
            # Performance optimization - frame skipping
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if current_time - self.last_update_time < 1.0 / self.target_fps:
                return
            
            self.frame_count += 1
            self.last_update_time = current_time
            
            # Store frame data
            self.current_frame = color_frame.copy()
            if depth_frame is not None:
                self.current_depth = depth_frame.copy()
            
            # Store feature data
            if features is not None:
                self.current_features = features
                self.feature_count_history.append(len(features))
                if len(self.feature_count_history) > 100:
                    self.feature_count_history.pop(0)
            
            if matches is not None:
                self.current_matches = matches
            
            # Store agricultural features
            if agricultural_features is not None:
                self.agricultural_features = agricultural_features
                agri_score = agricultural_features.get('agricultural_score', 0.0)
                self.agricultural_score_history.append(agri_score)
                if len(self.agricultural_score_history) > 100:
                    self.agricultural_score_history.pop(0)
            
            # Create display frame with overlays
            display_frame = self._create_display_frame()
            
            # Convert to QPixmap and display
            self._display_frame(display_frame)
            
            # Update info label
            self._update_info_display()
            
        except Exception as e:
            print(f"Camera widget update error: {e}")
    
    def _create_display_frame(self) -> np.ndarray:
        """Create frame with all overlays applied"""
        if self.current_frame is None:
            return np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        # Start with current frame
        display_frame = self.current_frame.copy()
        
        # Resize to display size if needed
        if (display_frame.shape[1] != self.display_width or 
            display_frame.shape[0] != self.display_height):
            display_frame = cv2.resize(display_frame, (self.display_width, self.display_height))
        
        # Apply depth overlay if enabled
        if self.show_depth_overlay and self.current_depth is not None:
            display_frame = self._apply_depth_overlay(display_frame)
        
        # Apply grid overlay if enabled
        if self.show_grid:
            display_frame = self._apply_grid_overlay(display_frame)
        
        # Apply feature overlays
        if self.show_features and self.current_features:
            display_frame = self._apply_feature_overlay(display_frame)
        
        # Apply match overlays
        if self.show_matches and self.current_matches:
            display_frame = self._apply_match_overlay(display_frame)
        
        # Apply agricultural overlays
        if self.show_agricultural and self.agricultural_features:
            display_frame = self._apply_agricultural_overlay(display_frame)
        
        # Apply center cross
        if self.show_center_cross:
            display_frame = self._apply_center_cross(display_frame)
        
        # Apply mouse cursor indicator
        if self.mouse_tracking:
            display_frame = self._apply_mouse_indicator(display_frame)
        
        return display_frame
    
    def _apply_depth_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Apply depth information overlay"""
        try:
            if self.current_depth is None:
                return frame
            
            # Resize depth to match display
            depth_resized = cv2.resize(self.current_depth, (self.display_width, self.display_height))
            
            # Create depth colormap
            depth_normalized = cv2.convertScaleAbs(depth_resized, alpha=0.03)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
            
            # Blend with original frame
            alpha = 0.3
            blended = cv2.addWeighted(frame, 1-alpha, depth_colormap, alpha, 0)
            
            return blended
            
        except Exception as e:
            print(f"Depth overlay error: {e}")
            return frame
    
    def _apply_grid_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Apply grid overlay for reference"""
        try:
            # Draw grid lines
            grid_color = (128, 128, 128)
            grid_spacing = 50
            
            # Vertical lines
            for x in range(0, self.display_width, grid_spacing):
                cv2.line(frame, (x, 0), (x, self.display_height), grid_color, 1)
            
            # Horizontal lines
            for y in range(0, self.display_height, grid_spacing):
                cv2.line(frame, (0, y), (self.display_width, y), grid_color, 1)
            
            return frame
            
        except Exception as e:
            print(f"Grid overlay error: {e}")
            return frame
    
    def _apply_feature_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Apply feature detection overlay"""
        try:
            for i, feature in enumerate(self.current_features):
                if hasattr(feature, 'pt'):
                    # ORB keypoint
                    x, y = int(feature.pt[0]), int(feature.pt[1])
                    response = getattr(feature, 'response', 50)
                    
                    # Scale circle size based on feature response
                    radius = max(2, min(8, int(response / 10)))
                    
                    # Color based on feature strength
                    intensity = min(255, int(response * 5))
                    color = (0, intensity, 255 - intensity)  # Blue to green gradient
                    
                elif len(feature) >= 2:
                    # Point coordinates
                    x, y = int(feature[0]), int(feature[1])
                    radius = 3
                    color = (0, 255, 255)  # Cyan
                else:
                    continue
                
                # Ensure coordinates are within bounds
                if 0 <= x < self.display_width and 0 <= y < self.display_height:
                    # Draw feature circle
                    cv2.circle(frame, (x, y), radius, color, 2)
                    
                    # Draw feature ID for first 20 features
                    if i < 20:
                        cv2.putText(frame, str(i), (x+radius+2, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            return frame
            
        except Exception as e:
            print(f"Feature overlay error: {e}")
            return frame
    
    def _apply_match_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Apply feature match overlay"""
        try:
            # This would show matches between frames if available
            # For now, just indicate matching quality
            if len(self.current_matches) > 0:
                match_text = f"Matches: {len(self.current_matches)}"
                cv2.putText(frame, match_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            return frame
            
        except Exception as e:
            print(f"Match overlay error: {e}")
            return frame
    
    def _apply_agricultural_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Apply agricultural features overlay"""
        try:
            # Draw crop rows
            crop_rows = self.agricultural_features.get('crop_rows', [])
            for i, crop_row in enumerate(crop_rows):
                if hasattr(crop_row, 'start_point') and hasattr(crop_row, 'end_point'):
                    start = crop_row.start_point
                    end = crop_row.end_point
                    confidence = getattr(crop_row, 'confidence', 1.0)
                    
                    # Scale coordinates to display size
                    start_scaled = self._scale_coordinates(start)
                    end_scaled = self._scale_coordinates(end)
                    
                    # Color intensity based on confidence
                    intensity = int(255 * confidence)
                    color = (0, intensity, 0)  # Green
                    
                    # Draw crop row line
                    cv2.line(frame, start_scaled, end_scaled, color, 3)
                    
                    # Draw crop row label
                    mid_point = ((start_scaled[0] + end_scaled[0]) // 2,
                                (start_scaled[1] + end_scaled[1]) // 2)
                    cv2.putText(frame, f"Row{i}", mid_point, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw ground plane indicator
            ground_plane = self.agricultural_features.get('ground_plane')
            if ground_plane:
                confidence = getattr(ground_plane, 'confidence', 0.0)
                
                # Draw ground plane indicator at bottom of frame
                plane_text = f"Ground Plane: {confidence:.2f}"
                cv2.putText(frame, plane_text, (10, self.display_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Draw ground plane horizon line if visible
                if confidence > 0.5:
                    horizon_y = int(self.display_height * 0.6)  # Approximate horizon
                    cv2.line(frame, (0, horizon_y), (self.display_width, horizon_y), 
                            (255, 255, 0), 2, cv2.LINE_TYPE_AA)
            
            # Draw scene analysis information
            scene_type = self.agricultural_features.get('scene_type', 'unknown')
            agricultural_score = self.agricultural_features.get('agricultural_score', 0.0)
            
            # Scene info overlay
            scene_text = f"Scene: {scene_type.title()} ({agricultural_score:.2f})"
            cv2.putText(frame, scene_text, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Scene complexity and lighting
            complexity = self.agricultural_features.get('scene_complexity', 0.0)
            lighting = self.agricultural_features.get('lighting_quality', 0.0)
            
            quality_text = f"Complexity: {complexity:.2f} | Lighting: {lighting:.2f}"
            cv2.putText(frame, quality_text, (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            return frame
            
        except Exception as e:
            print(f"Agricultural overlay error: {e}")
            return frame
    
    def _apply_center_cross(self, frame: np.ndarray) -> np.ndarray:
        """Apply center cross indicator"""
        try:
            center_x = self.display_width // 2
            center_y = self.display_height // 2
            cross_size = 20
            cross_color = (255, 255, 255)
            
            # Draw cross
            cv2.line(frame, 
                    (center_x - cross_size, center_y), 
                    (center_x + cross_size, center_y), 
                    cross_color, 2)
            cv2.line(frame, 
                    (center_x, center_y - cross_size), 
                    (center_x, center_y + cross_size), 
                    cross_color, 2)
            
            return frame
            
        except Exception as e:
            print(f"Center cross error: {e}")
            return frame
    
    def _apply_mouse_indicator(self, frame: np.ndarray) -> np.ndarray:
        """Apply mouse position indicator"""
        try:
            if self.current_mouse_pos:
                x, y = self.current_mouse_pos
                
                # Draw mouse position circle
                cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)
                
                # Draw coordinates text
                coord_text = f"({x},{y})"
                cv2.putText(frame, coord_text, (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            print(f"Mouse indicator error: {e}")
            return frame
    
    def _scale_coordinates(self, coords) -> Tuple[int, int]:
        """Scale coordinates to display size"""
        try:
            if len(coords) >= 2:
                # Assume original coordinates are in source frame size
                # Scale to display size
                scale_x = self.display_width / 640  # Assume 640 width source
                scale_y = self.display_height / 480  # Assume 480 height source
                
                x = int(coords[0] * scale_x)
                y = int(coords[1] * scale_y)
                
                # Clamp to display bounds
                x = max(0, min(x, self.display_width - 1))
                y = max(0, min(y, self.display_height - 1))
                
                return (x, y)
            
            return (0, 0)
            
        except Exception as e:
            print(f"Coordinate scaling error: {e}")
            return (0, 0)
    
    def _display_frame(self, frame: np.ndarray):
        """Convert frame to QPixmap and display"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create QImage
            height, width, channel = rgb_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Convert to QPixmap
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.camera_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Display
            self.camera_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"Frame display error: {e}")
    
    def _update_info_display(self):
        """Update information display"""
        try:
            info_parts = []
            
            # Frame count
            info_parts.append(f"Frame: {self.frame_count}")
            
            # Feature count
            if self.feature_count_history:
                avg_features = np.mean(self.feature_count_history[-10:])
                info_parts.append(f"Features: {avg_features:.0f}")
            
            # Agricultural score
            if self.agricultural_score_history:
                avg_agri_score = np.mean(self.agricultural_score_history[-10:])
                info_parts.append(f"Agri: {avg_agri_score:.2f}")
            
            # Mouse position
            if self.current_mouse_pos:
                x, y = self.current_mouse_pos
                info_parts.append(f"Mouse: ({x},{y})")
            
            info_text = " | ".join(info_parts)
            self.info_label.setText(info_text)
            
        except Exception as e:
            print(f"Info display update error: {e}")
    
    def toggle_features(self):
        """Toggle feature overlay display"""
        self.show_features = self.features_checkbox.isChecked()
    
    def toggle_agricultural(self):
        """Toggle agricultural overlay display"""
        self.show_agricultural = self.agricultural_checkbox.isChecked()
    
    def toggle_depth_overlay(self):
        """Toggle depth overlay display"""
        self.show_depth_overlay = self.depth_checkbox.isChecked()
    
    def toggle_grid(self):
        """Toggle grid overlay display"""
        self.show_grid = self.grid_checkbox.isChecked()
    
    def on_mouse_click(self, event):
        """Handle mouse click events"""
        try:
            x = event.pos().x()
            y = event.pos().y()
            
            # Scale coordinates to actual image coordinates
            label_size = self.camera_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                # Calculate scaling factors
                scale_x = self.display_width / label_size.width()
                scale_y = self.display_height / label_size.height()
                
                # Convert to image coordinates
                image_x = int(x * scale_x)
                image_y = int(y * scale_y)
                
                # Emit signal
                self.point_clicked.emit(image_x, image_y)
                
                print(f"Camera click: ({image_x}, {image_y})")
            
        except Exception as e:
            print(f"Mouse click handling error: {e}")
    
    def on_mouse_move(self, event):
        """Handle mouse move events"""
        try:
            if self.mouse_tracking:
                x = event.pos().x()
                y = event.pos().y()
                
                # Scale to image coordinates
                label_size = self.camera_label.size()
                if label_size.width() > 0 and label_size.height() > 0:
                    scale_x = self.display_width / label_size.width()
                    scale_y = self.display_height / label_size.height()
                    
                    image_x = int(x * scale_x)
                    image_y = int(y * scale_y)
                    
                    self.current_mouse_pos = (image_x, image_y)
            
        except Exception as e:
            print(f"Mouse move handling error: {e}")
    
    def set_display_size(self, width: int, height: int):
        """Set display size"""
        self.display_width = width
        self.display_height = height
        self.aspect_ratio = width / height
        self.camera_label.setMinimumSize(width, height)
    
    def set_target_fps(self, fps: int):
        """Set target FPS for performance optimization"""
        self.target_fps = max(1, min(fps, 60))
    
    def clear_display(self):
        """Clear camera display"""
        self.camera_label.clear()
        self.camera_label.setText("Camera Feed\nNo signal")
        self.current_frame = None
        self.current_depth = None
        self.current_features = []
        self.agricultural_features = {}
        self.frame_count = 0
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'frame_count': self.frame_count,
            'avg_features': np.mean(self.feature_count_history) if self.feature_count_history else 0,
            'avg_agricultural_score': np.mean(self.agricultural_score_history) if self.agricultural_score_history else 0,
            'target_fps': self.target_fps,
            'display_size': (self.display_width, self.display_height)
        }

# Compatibility class for existing code
class CameraWidget(EnhancedCameraWidget):
    """
    Compatibility wrapper for existing code
    Maintains the original interface while using enhanced implementation
    """
    
    def __init__(self, width: int = 640, height: int = 480):
        super().__init__(width, height)
    
    def update_frame(self, frame: np.ndarray):
        """Update with single frame (compatibility method)"""
        super().update_frame(frame)
    
    def set_show_features(self, show: bool):
        """Set feature display (compatibility method)"""
        self.show_features = show
        self.features_checkbox.setChecked(show)
    
    def set_feature_overlay_data(self, features: List, matches: List = None):
        """Set feature overlay data (compatibility method)"""
        self.current_features = features
        if matches:
            self.current_matches = matches

# Test function
def test_enhanced_camera_widget():
    """Test enhanced camera widget with sample data"""
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    widget = EnhancedCameraWidget()
    widget.show()
    
    # Create test frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create test features
    test_features = []
    for i in range(50):
        feature = type('Feature', (), {})()
        feature.pt = (np.random.randint(0, 640), np.random.randint(0, 480))
        feature.response = np.random.uniform(10, 100)
        test_features.append(feature)
    
    # Create test agricultural features
    test_agricultural = {
        'scene_type': 'crop_rows',
        'agricultural_score': 0.8,
        'scene_complexity': 0.6,
        'lighting_quality': 0.7,
        'crop_rows': [],
        'ground_plane': None
    }
    
    # Update widget
    widget.update_frame(test_frame, features=test_features, 
                       agricultural_features=test_agricultural)
    
    print("Test camera widget displayed")
    sys.exit(app.exec())

if __name__ == "__main__":
    test_enhanced_camera_widget()