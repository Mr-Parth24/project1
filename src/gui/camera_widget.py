"""
Enhanced Camera Widget for Agricultural SLAM System
Displays live camera feed with agricultural feature overlays
FIXED: Coordinate scaling, frame format handling, and agricultural overlays
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
    FIXED: Dynamic coordinate scaling and proper agricultural feature rendering
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
        
        # FIXED: Dynamic source frame dimensions (no hardcoded assumptions)
        self.source_frame_width = 640  # Will be updated dynamically
        self.source_frame_height = 480  # Will be updated dynamically
        
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
        
        # FIXED: Frame format tracking
        self.last_frame_format = None  # 'BGR', 'RGB', or 'GRAY'
        self.frame_format_detection_enabled = True
        
        # Initialize UI
        self.init_ui()
        
        print(f"🎥 Enhanced Camera Widget initialized: {width}x{height} (FIXED VERSION)")
        print(f"   - Dynamic coordinate scaling: ✅")
        print(f"   - Agricultural feature compatibility: ✅")
        print(f"   - Frame format auto-detection: ✅")
    
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
        FIXED: Update camera display with proper frame handling and coordinate scaling
        
        Args:
            color_frame: RGB/BGR color frame
            depth_frame: Optional depth frame
            features: List of detected features
            matches: List of feature matches
            agricultural_features: Dictionary with agricultural scene data (FIXED FORMAT)
        """
        try:
            # Performance optimization - frame skipping
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if current_time - self.last_update_time < 1.0 / self.target_fps:
                return
            
            self.frame_count += 1
            self.last_update_time = current_time
            
            # FIXED: Detect and update source frame dimensions dynamically
            if color_frame is not None:
                self.source_frame_height, self.source_frame_width = color_frame.shape[:2]
                
                # Detect frame format for proper conversion
                if self.frame_format_detection_enabled:
                    self.last_frame_format = self._detect_frame_format(color_frame)
            
            # Store frame data
            self.current_frame = color_frame.copy() if color_frame is not None else None
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
            
            # FIXED: Store agricultural features with proper validation
            if agricultural_features is not None:
                self.agricultural_features = agricultural_features
                agri_score = agricultural_features.get('agricultural_score', 0.0)
                self.agricultural_score_history.append(agri_score)
                if len(self.agricultural_score_history) > 100:
                    self.agricultural_score_history.pop(0)
                    
                # Debug log for agricultural features
                if self.frame_count % 50 == 0:
                    crop_count = len(agricultural_features.get('crop_rows', []))
                    ground_detected = agricultural_features.get('ground_plane') is not None
                    print(f"🎥 Agricultural features: {crop_count} crop rows, ground: {ground_detected}")
            
            # Create display frame with overlays
            display_frame = self._create_display_frame()
            
            if display_frame is not None:
                # Convert to QPixmap and display
                self._display_frame(display_frame)
                
                # Update info label
                self._update_info_display()
            
        except Exception as e:
            print(f"❌ Camera widget update error: {e}")
            # Set error display
            self.camera_label.setText(f"Display Error\n{str(e)}")
    
    def _detect_frame_format(self, frame: np.ndarray) -> str:
        """FIXED: Detect frame format (BGR, RGB, or GRAY) for proper conversion"""
        try:
            if len(frame.shape) == 2:
                return 'GRAY'
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                # Heuristic: Check if blue channel is generally higher (BGR) or red channel (RGB)
                # Agricultural scenes typically have more green, so we check red vs blue
                mean_red = np.mean(frame[:, :, 0])
                mean_blue = np.mean(frame[:, :, 2])
                
                # OpenCV typically uses BGR, so if blue channel is higher, likely BGR
                if mean_blue > mean_red * 1.2:
                    return 'BGR'
                else:
                    return 'RGB'
            else:
                return 'BGR'  # Default assumption
                
        except Exception as e:
            print(f"Frame format detection error: {e}")
            return 'BGR'
    
    def _create_display_frame(self) -> Optional[np.ndarray]:
        """FIXED: Create frame with all overlays applied and proper coordinate scaling"""
        if self.current_frame is None:
            return np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
        
        try:
            # Start with current frame
            display_frame = self.current_frame.copy()
            
            # FIXED: Proper frame format conversion
            display_frame = self._ensure_bgr_format(display_frame)
            
            # Resize to display size if needed with proper aspect ratio handling
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
            
            # FIXED: Apply agricultural overlays with proper coordinate handling
            if self.show_agricultural and self.agricultural_features:
                display_frame = self._apply_agricultural_overlay_fixed(display_frame)
            
            # Apply center cross
            if self.show_center_cross:
                display_frame = self._apply_center_cross(display_frame)
            
            # Apply mouse cursor indicator
            if self.mouse_tracking:
                display_frame = self._apply_mouse_indicator(display_frame)
            
            return display_frame
            
        except Exception as e:
            print(f"❌ Display frame creation error: {e}")
            return self.current_frame
    
    def _ensure_bgr_format(self, frame: np.ndarray) -> np.ndarray:
        """FIXED: Ensure frame is in BGR format for OpenCV operations"""
        try:
            if len(frame.shape) == 2:
                # Grayscale to BGR
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                if self.last_frame_format == 'RGB':
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    return frame  # Already BGR
            else:
                return frame
                
        except Exception as e:
            print(f"Frame format conversion error: {e}")
            return frame
    
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
        """FIXED: Apply feature detection overlay with proper coordinate scaling"""
        try:
            for i, feature in enumerate(self.current_features):
                if hasattr(feature, 'pt'):
                    # ORB keypoint - scale coordinates
                    x_orig, y_orig = feature.pt
                    x, y = self._scale_coordinates_fixed((x_orig, y_orig))
                    response = getattr(feature, 'response', 50)
                    
                    # Scale circle size based on feature response
                    radius = max(2, min(8, int(response / 10)))
                    
                    # Color based on feature strength
                    intensity = min(255, int(response * 5))
                    color = (0, intensity, 255 - intensity)  # Blue to green gradient
                    
                elif len(feature) >= 2:
                    # Point coordinates - scale coordinates
                    x, y = self._scale_coordinates_fixed((feature[0], feature[1]))
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
    
    def _apply_agricultural_overlay_fixed(self, frame: np.ndarray) -> np.ndarray:
        """FIXED: Apply agricultural features overlay with proper structured data handling"""
        try:
            # FIXED: Draw crop rows with structured CropRowDetection objects
            crop_rows = self.agricultural_features.get('crop_rows', [])
            for i, crop_row in enumerate(crop_rows):
                try:
                    # Handle structured CropRowDetection objects (FIXED)
                    if hasattr(crop_row, 'start_point') and hasattr(crop_row, 'end_point'):
                        start = crop_row.start_point
                        end = crop_row.end_point
                        confidence = getattr(crop_row, 'confidence', 1.0)
                        
                        # FIXED: Scale coordinates to display size
                        start_scaled = self._scale_coordinates_fixed(start)
                        end_scaled = self._scale_coordinates_fixed(end)
                        
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
                        
                    # Handle legacy array format (fallback)
                    elif hasattr(crop_row, '__len__') and len(crop_row) >= 4:
                        if hasattr(crop_row[0], '__len__') and len(crop_row[0]) >= 4:
                            # OpenCV HoughLinesP format: [[x1, y1, x2, y2]]
                            x1, y1, x2, y2 = crop_row[0]
                            start_scaled = self._scale_coordinates_fixed((x1, y1))
                            end_scaled = self._scale_coordinates_fixed((x2, y2))
                            
                            color = (0, 200, 0)  # Default green
                            cv2.line(frame, start_scaled, end_scaled, color, 3)
                            
                except Exception as crop_error:
                    print(f"Individual crop row rendering error: {crop_error}")
                    continue
            
            # FIXED: Draw ground plane indicator with structured GroundPlaneDetection
            ground_plane = self.agricultural_features.get('ground_plane')
            if ground_plane is not None:
                try:
                    # Handle structured GroundPlaneDetection object
                    if hasattr(ground_plane, 'confidence'):
                        confidence = ground_plane.confidence
                        distance = getattr(ground_plane, 'distance_to_camera', 0.0)
                        
                        # Draw ground plane indicator at bottom of frame
                        plane_text = f"Ground Plane: {confidence:.2f} (d:{distance:.1f}m)"
                        cv2.putText(frame, plane_text, (10, self.display_height - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        
                        # Draw ground plane horizon line if visible
                        if confidence > 0.5:
                            horizon_y = int(self.display_height * 0.6)  # Approximate horizon
                            cv2.line(frame, (0, horizon_y), (self.display_width, horizon_y), 
                                    (255, 255, 0), 2, cv2.LINE_TYPE_AA)
                    
                    # Handle legacy format (fallback)
                    else:
                        confidence = 0.5  # Default
                        plane_text = f"Ground Plane: {confidence:.2f}"
                        cv2.putText(frame, plane_text, (10, self.display_height - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                        
                except Exception as ground_error:
                    print(f"Ground plane rendering error: {ground_error}")
            
            # FIXED: Draw scene analysis information with proper data access
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
            print(f"❌ Agricultural overlay error: {e}")
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
    
    def _scale_coordinates_fixed(self, coords) -> Tuple[int, int]:
        """FIXED: Scale coordinates to display size using dynamic source dimensions"""
        try:
            if len(coords) >= 2:
                # FIXED: Use actual source frame dimensions instead of hardcoded values
                scale_x = self.display_width / self.source_frame_width
                scale_y = self.display_height / self.source_frame_height
                
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
        """FIXED: Convert frame to QPixmap and display with proper format handling"""
        try:
            # FIXED: Ensure proper color space conversion for Qt
            if len(frame.shape) == 3:
                # Convert BGR to RGB for Qt display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Grayscale to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
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
            print(f"❌ Frame display error: {e}")
            self.camera_label.setText(f"Display Error\n{str(e)}")
    
    def _update_info_display(self):
        """FIXED: Update information display with proper data access"""
        try:
            info_parts = []
            
            # Frame count and source resolution
            info_parts.append(f"Frame: {self.frame_count}")
            info_parts.append(f"Res: {self.source_frame_width}x{self.source_frame_height}")
            
            # Feature count
            if self.feature_count_history:
                avg_features = np.mean(self.feature_count_history[-10:])
                info_parts.append(f"Features: {avg_features:.0f}")
            
            # Agricultural score
            if self.agricultural_score_history:
                avg_agri_score = np.mean(self.agricultural_score_history[-10:])
                info_parts.append(f"Agri: {avg_agri_score:.2f}")
            
            # Crop rows count
            if self.agricultural_features:
                crop_count = len(self.agricultural_features.get('crop_rows', []))
                if crop_count > 0:
                    info_parts.append(f"Crops: {crop_count}")
            
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
        print(f"🎥 Features display: {'ON' if self.show_features else 'OFF'}")
    
    def toggle_agricultural(self):
        """Toggle agricultural overlay display"""
        self.show_agricultural = self.agricultural_checkbox.isChecked()
        print(f"🌾 Agricultural display: {'ON' if self.show_agricultural else 'OFF'}")
    
    def toggle_depth_overlay(self):
        """Toggle depth overlay display"""
        self.show_depth_overlay = self.depth_checkbox.isChecked()
        print(f"📏 Depth overlay: {'ON' if self.show_depth_overlay else 'OFF'}")
    
    def toggle_grid(self):
        """Toggle grid overlay display"""
        self.show_grid = self.grid_checkbox.isChecked()
        print(f"📐 Grid overlay: {'ON' if self.show_grid else 'OFF'}")
    
    def on_mouse_click(self, event):
        """FIXED: Handle mouse click events with proper coordinate conversion"""
        try:
            x = event.pos().x()
            y = event.pos().y()
            
            # FIXED: Scale coordinates to actual image coordinates
            label_size = self.camera_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                # Calculate scaling factors from label to display
                scale_x = self.display_width / label_size.width()
                scale_y = self.display_height / label_size.height()
                
                # Convert to display coordinates
                display_x = int(x * scale_x)
                display_y = int(y * scale_y)
                
                # Convert to source image coordinates
                source_scale_x = self.source_frame_width / self.display_width
                source_scale_y = self.source_frame_height / self.display_height
                
                source_x = int(display_x * source_scale_x)
                source_y = int(display_y * source_scale_y)
                
                # Emit signal with source coordinates
                self.point_clicked.emit(source_x, source_y)
                
                print(f"🎥 Camera click: label({x},{y}) -> display({display_x},{display_y}) -> source({source_x},{source_y})")
            
        except Exception as e:
            print(f"❌ Mouse click handling error: {e}")
    
    def on_mouse_move(self, event):
        """FIXED: Handle mouse move events with proper coordinate conversion"""
        try:
            if self.mouse_tracking:
                x = event.pos().x()
                y = event.pos().y()
                
                # FIXED: Scale to display coordinates for overlay
                label_size = self.camera_label.size()
                if label_size.width() > 0 and label_size.height() > 0:
                    scale_x = self.display_width / label_size.width()
                    scale_y = self.display_height / label_size.height()
                    
                    display_x = int(x * scale_x)
                    display_y = int(y * scale_y)
                    
                    self.current_mouse_pos = (display_x, display_y)
            
        except Exception as e:
            print(f"Mouse move handling error: {e}")
    
    def set_display_size(self, width: int, height: int):
        """FIXED: Set display size and update camera label"""
        self.display_width = width
        self.display_height = height
        self.aspect_ratio = width / height
        self.camera_label.setMinimumSize(width, height)
        print(f"🎥 Display size updated: {width}x{height}")
    
    def set_source_frame_size(self, width: int, height: int):
        """NEW: Explicitly set source frame dimensions for coordinate scaling"""
        self.source_frame_width = width
        self.source_frame_height = height
        print(f"🎥 Source frame size set: {width}x{height}")
    
    def set_target_fps(self, fps: int):
        """Set target FPS for performance optimization"""
        self.target_fps = max(1, min(fps, 60))
        print(f"🎥 Target FPS: {self.target_fps}")
    
    def clear_display(self):
        """Clear camera display and reset state"""
        self.camera_label.clear()
        self.camera_label.setText("Camera Feed\nNo signal")
        self.current_frame = None
        self.current_depth = None
        self.current_features = []
        self.agricultural_features = {}
        self.frame_count = 0
        
        # Reset frame dimensions
        self.source_frame_width = 640
        self.source_frame_height = 480
        
        print("🎥 Camera display cleared")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'frame_count': self.frame_count,
            'avg_features': np.mean(self.feature_count_history) if self.feature_count_history else 0,
            'avg_agricultural_score': np.mean(self.agricultural_score_history) if self.agricultural_score_history else 0,
            'target_fps': self.target_fps,
            'display_size': (self.display_width, self.display_height),
            'source_size': (self.source_frame_width, self.source_frame_height),
            'coordinate_scaling': (self.display_width / self.source_frame_width, self.display_height / self.source_frame_height),
            'last_frame_format': self.last_frame_format
        }

# Compatibility class for existing code
class CameraWidget(EnhancedCameraWidget):
    """
    FIXED: Compatibility wrapper for existing code
    Maintains the original interface while using enhanced implementation
    """
    
    def __init__(self, width: int = 640, height: int = 480):
        super().__init__(width, height)
        print(f"🎥 Legacy CameraWidget wrapper initialized (FIXED)")
    
    def update_frame(self, frame: np.ndarray):
        """FIXED: Update with single frame (compatibility method)"""
        super().update_frame(frame)
    
    def set_show_features(self, show: bool):
        """Set feature display (compatibility method)"""
        self.show_features = show
        self.features_checkbox.setChecked(show)
        print(f"🎥 Legacy feature display: {'ON' if show else 'OFF'}")
    
    def set_feature_overlay_data(self, features: List, matches: List = None):
        """Set feature overlay data (compatibility method)"""
        self.current_features = features
        if matches:
            self.current_matches = matches

# Test function
def test_enhanced_camera_widget():
    """Test enhanced camera widget with sample data (FIXED)"""
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    widget = EnhancedCameraWidget()
    widget.show()
    
    # Create test frame with specific dimensions
    test_width, test_height = 848, 480  # Match typical agricultural config
    test_frame = np.random.randint(0, 255, (test_height, test_width, 3), dtype=np.uint8)
    
    # Set source frame size explicitly
    widget.set_source_frame_size(test_width, test_height)
    
    # Create test features with proper coordinates
    test_features = []
    for i in range(50):
        feature = type('Feature', (), {})()
        feature.pt = (np.random.randint(0, test_width), np.random.randint(0, test_height))
        feature.response = np.random.uniform(10, 100)
        test_features.append(feature)
    
    # Create test agricultural features (FIXED FORMAT)
    from dataclasses import dataclass
    
    @dataclass
    class TestCropRow:
        start_point: tuple
        end_point: tuple
        confidence: float
        
    @dataclass
    class TestGroundPlane:
        confidence: float
        distance_to_camera: float
    
    test_crop_rows = [
        TestCropRow((100, 300), (700, 320), 0.8),
        TestCropRow((120, 350), (720, 370), 0.9),
    ]
    
    test_ground_plane = TestGroundPlane(0.7, 2.5)
    
    test_agricultural = {
        'scene_type': 'crop_rows',
        'agricultural_score': 0.8,
        'scene_complexity': 0.6,
        'lighting_quality': 0.7,
        'crop_rows': test_crop_rows,
        'ground_plane': test_ground_plane
    }
    
    # Update widget
    widget.update_frame(test_frame, features=test_features, 
                       agricultural_features=test_agricultural)
    
    print("✅ Test camera widget displayed (FIXED VERSION)")
    sys.exit(app.exec())

if __name__ == "__main__":
    test_enhanced_camera_widget()