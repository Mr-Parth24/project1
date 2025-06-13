"""
Stable Live Feed Visualizer - Fixed Threading and Memory Issues
Author: Mr-Parth24
Date: 2025-06-13
Time: 21:15:16 UTC
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Optional, Tuple, Any
import logging
import math
from collections import deque

class StableLiveFeedVisualizer:
    """Fixed live camera feed with stable threading and memory management"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Thread management
        self.is_active = False
        self.display_thread = None
        self.stop_event = threading.Event()
        self.window_lock = threading.Lock()
        
        # Window management
        self.live_window = "Live Camera Feed - Enhanced VO System"
        self.debug_window = "Debug View - Feature Analysis"
        self.windows_created = False
        
        # Frame processing
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue to prevent lag
        self.processing_lock = threading.Lock()
        
        # Visualization settings
        self.show_3d_markers = True
        self.show_features = True
        self.show_motion_vectors = True
        self.show_trajectory_overlay = True
        self.show_statistics = True
        
        # Performance tracking
        self.fps_counter = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        # Trajectory data
        self.trajectory_points = deque(maxlen=100)
        self.motion_vectors = deque(maxlen=10)
        
        # 3D visualization parameters
        self.axis_length = 0.08  # 8cm axes
        
        # Colors
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }
        
        self.logger.info("Stable live feed visualizer initialized")
    
    def start_visualization(self) -> bool:
        """Start live feed visualization with proper error handling"""
        try:
            if self.is_active:
                self.logger.warning("Visualization already active")
                return True
            
            self.logger.info("Starting live feed visualization...")
            
            # Initialize OpenCV windows on main thread
            with self.window_lock:
                try:
                    cv2.namedWindow(self.live_window, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
                    cv2.namedWindow(self.debug_window, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO)
                    
                    # Position windows
                    cv2.moveWindow(self.live_window, 100, 100)
                    cv2.moveWindow(self.debug_window, 750, 100)
                    
                    self.windows_created = True
                    self.logger.info("OpenCV windows created successfully")
                    
                except cv2.error as e:
                    self.logger.error(f"Failed to create OpenCV windows: {e}")
                    return False
            
            # Start display thread
            self.is_active = True
            self.stop_event.clear()
            
            self.display_thread = threading.Thread(
                target=self._stable_display_worker, 
                daemon=True,
                name="LiveFeedDisplayThread"
            )
            self.display_thread.start()
            
            self.logger.info("Live feed visualization started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start visualization: {e}")
            self._cleanup_windows()
            return False
    
    def _stable_display_worker(self):
        """Stable display worker with comprehensive error handling"""
        self.logger.info("Display worker thread started")
        
        frame_count = 0
        last_heartbeat = time.time()
        
        try:
            while self.is_active and not self.stop_event.is_set():
                try:
                    # Get frame with timeout
                    try:
                        display_data = self.frame_queue.get(timeout=1.0)
                    except queue.Empty:
                        # Check if we should continue waiting
                        current_time = time.time()
                        if current_time - last_heartbeat > 5.0:
                            self.logger.debug("No frames received for 5 seconds")
                            last_heartbeat = current_time
                        continue
                    
                    # Verify windows still exist
                    if not self.windows_created:
                        self.logger.warning("Windows not created, skipping frame")
                        continue
                    
                    # Display frames safely
                    try:
                        with self.window_lock:
                            if self.windows_created:
                                cv2.imshow(self.live_window, display_data['live_frame'])
                                cv2.imshow(self.debug_window, display_data['debug_frame'])
                        
                        # Handle keyboard input
                        key = cv2.waitKey(1) & 0xFF
                        if key != 255:  # Key pressed
                            self._handle_keyboard_input(key)
                            if key == 27:  # ESC
                                self.logger.info("ESC pressed, stopping visualization")
                                break
                        
                        frame_count += 1
                        last_heartbeat = time.time()
                        
                    except cv2.error as e:
                        self.logger.error(f"OpenCV display error: {e}")
                        break
                    except Exception as e:
                        self.logger.error(f"Frame display error: {e}")
                        continue
                
                except Exception as e:
                    self.logger.error(f"Display worker iteration error: {e}")
                    time.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(f"Critical display worker error: {e}")
        finally:
            self.logger.info(f"Display worker stopping (processed {frame_count} frames)")
            self._cleanup_windows()
    
    def update_display(self, color_frame: np.ndarray, depth_frame: np.ndarray,
                      tracking_result: Dict, camera_matrix: np.ndarray = None):
        """Thread-safe display update with error handling"""
        
        if not self.is_active or not self.windows_created:
            return
        
        try:
            with self.processing_lock:
                # Calculate FPS
                current_time = time.time()
                if hasattr(self, '_last_update_time'):
                    frame_time = current_time - self._last_update_time
                    if frame_time > 0:
                        fps = 1.0 / frame_time
                        self.fps_counter.append(fps)
                self._last_update_time = current_time
                
                # Create display frames
                live_frame = self._create_enhanced_live_display(
                    color_frame, depth_frame, tracking_result, camera_matrix
                )
                debug_frame = self._create_debug_display(
                    color_frame, tracking_result
                )
                
                # Update trajectory data
                self._update_trajectory_data(tracking_result)
                
                # Prepare display data
                display_data = {
                    'live_frame': live_frame,
                    'debug_frame': debug_frame,
                    'timestamp': current_time,
                    'tracking_result': tracking_result.copy()
                }
                
                # Add to queue (non-blocking)
                try:
                    # Clear old frames to prevent lag
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    
                    self.frame_queue.put_nowait(display_data)
                    
                except queue.Full:
                    # Queue management failed, skip this frame
                    pass
                
        except Exception as e:
            self.logger.error(f"Display update error: {e}")
    
    def _create_enhanced_live_display(self, color_frame: np.ndarray, depth_frame: np.ndarray,
                                    tracking_result: Dict, camera_matrix: np.ndarray = None) -> np.ndarray:
        """Create enhanced live display with all overlays"""
        
        try:
            display_frame = color_frame.copy()
            
            # Draw features
            if self.show_features and 'keypoints' in tracking_result:
                display_frame = self._draw_enhanced_features(display_frame, tracking_result)
            
            # Draw 3D coordinate system
            if self.show_3d_markers and camera_matrix is not None:
                display_frame = self._draw_3d_coordinate_system(display_frame, camera_matrix)
            
            # Draw motion vectors
            if self.show_motion_vectors:
                display_frame = self._draw_motion_vectors(display_frame, tracking_result)
            
            # Draw trajectory overlay
            if self.show_trajectory_overlay:
                display_frame = self._draw_trajectory_overlay(display_frame)
            
            # Draw comprehensive statistics
            if self.show_statistics:
                display_frame = self._draw_comprehensive_statistics(display_frame, tracking_result)
            
            # Draw validation status
            display_frame = self._draw_validation_status(display_frame, tracking_result)
            
            return display_frame
            
        except Exception as e:
            self.logger.error(f"Enhanced display creation error: {e}")
            return color_frame
    
    def _draw_enhanced_features(self, frame: np.ndarray, tracking_result: Dict) -> np.ndarray:
        """Draw enhanced feature visualization"""
        
        try:
            keypoints = tracking_result.get('keypoints', [])
            if not keypoints:
                return frame
            
            # Draw features with quality-based coloring
            for i, kp in enumerate(keypoints[:100]):  # Limit for performance
                x, y = int(kp.pt[0]), int(kp.pt[1])
                
                # Color based on response strength
                response_norm = min(kp.response / 100.0, 1.0)
                
                if response_norm > 0.7:
                    color = self.colors['green']  # Strong features
                elif response_norm > 0.4:
                    color = self.colors['yellow']  # Medium features
                else:
                    color = self.colors['blue']   # Weak features
                
                # Draw feature point
                cv2.circle(frame, (x, y), 3, color, 1)
                
                # Draw orientation for strong features
                if kp.angle >= 0 and response_norm > 0.5:
                    length = int(8 + response_norm * 10)
                    end_x = int(x + length * math.cos(math.radians(kp.angle)))
                    end_y = int(y + length * math.sin(math.radians(kp.angle)))
                    cv2.line(frame, (x, y), (end_x, end_y), color, 1)
                
                # Add index for very strong features
                if response_norm > 0.8 and i < 20:
                    cv2.putText(frame, str(i), (x+5, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Feature drawing error: {e}")
            return frame
    
    def _draw_3d_coordinate_system(self, frame: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
        """Draw 3D coordinate system overlay"""
        
        try:
            # Define 3D axis points
            axis_points_3d = np.array([
                [0, 0, 0],                      # Origin
                [self.axis_length, 0, 0],       # X-axis (Red)
                [0, self.axis_length, 0],       # Y-axis (Green)  
                [0, 0, self.axis_length],       # Z-axis (Blue)
                [self.axis_length/2, self.axis_length/2, 0],  # XY plane indicator
                [self.axis_length/2, 0, self.axis_length/2],  # XZ plane indicator
                [0, self.axis_length/2, self.axis_length/2]   # YZ plane indicator
            ], dtype=np.float32)
            
            # Project to 2D
            rvec = np.zeros((3, 1))
            tvec = np.array([[0], [0], [0.3]])  # Place 30cm in front of camera
            
            projected_points, _ = cv2.projectPoints(
                axis_points_3d, rvec, tvec, camera_matrix, np.zeros((4, 1))
            )
            
            # Extract points
            points_2d = [tuple(map(int, pt[0])) for pt in projected_points]
            origin, x_end, y_end, z_end, xy_pt, xz_pt, yz_pt = points_2d
            
            # Check if origin is visible
            h, w = frame.shape[:2]
            if 0 <= origin[0] < w and 0 <= origin[1] < h:
                
                # Draw main axes with varying thickness
                cv2.line(frame, origin, x_end, self.colors['red'], 4)      # X: Red
                cv2.line(frame, origin, y_end, self.colors['green'], 4)    # Y: Green
                cv2.line(frame, origin, z_end, self.colors['blue'], 4)     # Z: Blue
                
                # Draw plane indicators (lighter lines)
                cv2.line(frame, origin, xy_pt, self.colors['yellow'], 2)   # XY plane
                cv2.line(frame, origin, xz_pt, self.colors['magenta'], 2)  # XZ plane
                cv2.line(frame, origin, yz_pt, self.colors['cyan'], 2)     # YZ plane
                
                # Draw axis labels with background
                labels = [
                    (x_end, 'X', self.colors['red']),
                    (y_end, 'Y', self.colors['green']),
                    (z_end, 'Z', self.colors['blue'])
                ]
                
                for pos, label, color in labels:
                    if 0 <= pos[0] < w and 0 <= pos[1] < h:
                        # Background rectangle
                        cv2.rectangle(frame, (pos[0]-8, pos[1]-15), (pos[0]+15, pos[1]+5), 
                                     self.colors['black'], -1)
                        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw origin marker
                cv2.circle(frame, origin, 6, self.colors['white'], -1)
                cv2.circle(frame, origin, 6, self.colors['black'], 2)
                cv2.putText(frame, 'O', (origin[0]-5, origin[1]+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['black'], 1)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"3D coordinate system drawing error: {e}")
            return frame
    
    def _draw_motion_vectors(self, frame: np.ndarray, tracking_result: Dict) -> np.ndarray:
        """Draw motion vectors showing camera movement"""
        
        try:
            # Get motion information
            current_pos = tracking_result.get('current_position', [0, 0, 0])
            motion_magnitude = tracking_result.get('distance_moved', 0.0)
            direction_angle = tracking_result.get('direction_angle', 0.0)
            
            if motion_magnitude < 0.001:  # No significant motion
                return frame
            
            # Calculate display position
            center_x, center_y = frame.shape[1] // 2, 50
            
            # Scale motion for visualization
            scale = min(2000 * motion_magnitude, 100)  # Adaptive scaling
            
            # Calculate end point based on direction
            angle_rad = math.radians(direction_angle - 90)  # Adjust for screen coordinates
            end_x = int(center_x + scale * math.cos(angle_rad))
            end_y = int(center_y + scale * math.sin(angle_rad))
            
            # Draw motion vector
            cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                           self.colors['yellow'], 3, tipLength=0.3)
            
            # Draw motion circle (magnitude indicator)
            radius = max(int(scale), 5)
            cv2.circle(frame, (center_x, center_y), radius, self.colors['yellow'], 2)
            
            # Add motion text
            motion_text = f"Motion: {motion_magnitude:.3f}m"
            direction_text = f"Dir: {direction_angle:.0f}°"
            
            cv2.putText(frame, motion_text, (center_x - 50, center_y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['yellow'], 2)
            cv2.putText(frame, direction_text, (center_x - 50, center_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['yellow'], 2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Motion vector drawing error: {e}")
            return frame
    
    def _draw_trajectory_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw trajectory mini-map overlay"""
        
        try:
            if len(self.trajectory_points) < 2:
                return frame
            
            # Mini-map parameters
            map_size = 120
            map_x = frame.shape[1] - map_size - 15
            map_y = 15
            
            # Create semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (map_x-5, map_y-5), 
                         (map_x + map_size + 5, map_y + map_size + 5),
                         (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Draw border
            cv2.rectangle(frame, (map_x-5, map_y-5), 
                         (map_x + map_size + 5, map_y + map_size + 5),
                         self.colors['white'], 2)
            
            # Convert trajectory to numpy array
            traj_array = np.array(list(self.trajectory_points))
            if len(traj_array) == 0:
                return frame
            
            x_coords = traj_array[:, 0]
            z_coords = traj_array[:, 2]  # Use Z for forward/backward
            
            # Scale to fit map
            if len(x_coords) > 1:
                x_range = max(np.max(x_coords) - np.min(x_coords), 0.1)
                z_range = max(np.max(z_coords) - np.min(z_coords), 0.1)
                
                x_min, z_min = np.min(x_coords), np.min(z_coords)
                margin = 10
                
                # Draw trajectory with color progression
                prev_point = None
                for i, (x, z) in enumerate(zip(x_coords, z_coords)):
                    # Normalize coordinates
                    norm_x = (x - x_min) / x_range
                    norm_z = (z - z_min) / z_range
                    
                    map_point = (
                        int(map_x + margin + norm_x * (map_size - 2 * margin)),
                        int(map_y + margin + norm_z * (map_size - 2 * margin))
                    )
                    
                    if prev_point is not None:
                        # Color progression from blue to red
                        progress = i / len(x_coords)
                        color = (
                            int(255 * (1 - progress)),  # Blue decreases
                            0,
                            int(255 * progress)         # Red increases
                        )
                        cv2.line(frame, prev_point, map_point, color, 2)
                    
                    prev_point = map_point
                
                # Draw current position
                if prev_point:
                    cv2.circle(frame, prev_point, 4, self.colors['green'], -1)
                    cv2.circle(frame, prev_point, 4, self.colors['white'], 1)
            
            # Add scale reference
            scale_text = f"Scale: {max(x_range, z_range):.2f}m"
            cv2.putText(frame, scale_text, (map_x, map_y + map_size + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['white'], 1)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Trajectory overlay drawing error: {e}")
            return frame
    
    def _draw_comprehensive_statistics(self, frame: np.ndarray, tracking_result: Dict) -> np.ndarray:
        """Draw comprehensive statistics overlay"""
        
        try:
            # Calculate current FPS
            current_fps = self._calculate_display_fps()
            
            # Statistics background
            stats_height = 200
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (380, 10 + stats_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Border
            cv2.rectangle(frame, (10, 10), (380, 10 + stats_height), self.colors['white'], 2)
            
            # Title
            cv2.putText(frame, "ENHANCED VISUAL ODOMETRY", (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['green'], 2)
            
            # Statistics data
            stats_data = [
                f"Display FPS: {current_fps:.1f}",
                f"Features: {tracking_result.get('num_features', 0)}",
                f"Matches: {tracking_result.get('num_matches', 0)}",
                f"Total Distance: {tracking_result.get('total_distance', 0.0):.3f}m",
                f"From Start: {tracking_result.get('displacement_from_start', 0.0):.3f}m",
                f"Speed: {tracking_result.get('current_speed', 0.0):.3f}m/s",
                f"Direction: {tracking_result.get('direction_angle', 0.0):.0f}°",
                f"Quality: {tracking_result.get('quality_score', 0.0):.2f}",
                f"Confidence: {tracking_result.get('tracking_confidence', 0.0):.2f}"
            ]
            
            # Draw statistics
            y_offset = 55
            for i, stat in enumerate(stats_data):
                color = self.colors['white']
                
                # Color coding for important stats
                if "Quality:" in stat:
                    value = float(stat.split(": ")[1])
                    if value > 0.7:
                        color = self.colors['green']
                    elif value > 0.4:
                        color = self.colors['yellow']
                    else:
                        color = self.colors['red']
                
                cv2.putText(frame, stat, (20, y_offset + i * 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Statistics drawing error: {e}")
            return frame
    
    def _draw_validation_status(self, frame: np.ndarray, tracking_result: Dict) -> np.ndarray:
        """Draw motion validation status"""
        
        try:
            # Validation status
            validation_info = tracking_result.get('validation_debug', {})
            motion_valid = tracking_result.get('motion_valid', False)
            
            # Status position
            status_x, status_y = frame.shape[1] - 200, frame.shape[0] - 80
            
            # Background
            overlay = frame.copy()
            cv2.rectangle(overlay, (status_x - 10, status_y - 30), 
                         (frame.shape[1] - 10, frame.shape[0] - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Status indicator
            status_color = self.colors['green'] if motion_valid else self.colors['red']
            status_text = "MOTION VALID" if motion_valid else "STATIONARY"
            
            cv2.putText(frame, status_text, (status_x, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Validation details
            if validation_info:
                motion_mag = validation_info.get('motion_magnitude', 0.0)
                cv2.putText(frame, f"Mag: {motion_mag:.4f}m", (status_x, status_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['white'], 1)
                
                result = validation_info.get('result', 'unknown')
                cv2.putText(frame, f"Status: {result}", (status_x, status_y + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['white'], 1)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Validation status drawing error: {e}")
            return frame
    
    def _create_debug_display(self, color_frame: np.ndarray, tracking_result: Dict) -> np.ndarray:
        """Create debug view with detailed analysis"""
        
        try:
            debug_frame = color_frame.copy()
            
            # Draw all features with detailed information
            keypoints = tracking_result.get('keypoints', [])
            
            for i, kp in enumerate(keypoints):
                x, y = int(kp.pt[0]), int(kp.pt[1])
                
                # Feature strength visualization
                response = kp.response
                size = max(int(kp.size / 4), 2)
                
                # Color based on response
                if response > 80:
                    color = self.colors['red']    # Very strong
                elif response > 40:
                    color = self.colors['yellow'] # Strong
                elif response > 20:
                    color = self.colors['green']  # Medium
                else:
                    color = self.colors['blue']   # Weak
                
                # Draw feature with size indication
                cv2.circle(debug_frame, (x, y), size, color, 2)
                cv2.circle(debug_frame, (x, y), 1, color, -1)
                
                # Draw orientation
                if kp.angle >= 0:
                    length = size + 5
                    end_x = int(x + length * math.cos(math.radians(kp.angle)))
                    end_y = int(y + length * math.sin(math.radians(kp.angle)))
                    cv2.line(debug_frame, (x, y), (end_x, end_y), color, 1)
                
                # Add feature info for strong features
                if response > 60 and i < 30:
                    cv2.putText(debug_frame, f"{i}:{response:.0f}", (x+5, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Add debug information overlay
            debug_info = [
                f"Feature Analysis:",
                f"Total Features: {len(keypoints)}",
                f"Strong (>60): {sum(1 for kp in keypoints if kp.response > 60)}",
                f"Medium (20-60): {sum(1 for kp in keypoints if 20 <= kp.response <= 60)}",
                f"Weak (<20): {sum(1 for kp in keypoints if kp.response < 20)}",
                "",
                f"Tracking Status:",
                f"Matches: {tracking_result.get('num_matches', 0)}",
                f"Quality: {tracking_result.get('quality_score', 0.0):.3f}",
                f"Motion Valid: {tracking_result.get('motion_valid', False)}"
            ]
            
            # Draw debug info
            y_offset = 25
            for info in debug_info:
                cv2.putText(debug_frame, info, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['white'], 1)
                y_offset += 20
            
            return debug_frame
            
        except Exception as e:
            self.logger.error(f"Debug display creation error: {e}")
            return color_frame
    
    def _update_trajectory_data(self, tracking_result: Dict):
        """Update trajectory data for visualization"""
        
        try:
            current_pos = tracking_result.get('current_position', [0, 0, 0])
            if current_pos != [0, 0, 0]:  # Valid position
                self.trajectory_points.append(current_pos)
            
            # Update motion vectors
            motion_vector = [
                tracking_result.get('x_displacement', 0.0),
                tracking_result.get('y_displacement', 0.0),
                tracking_result.get('z_displacement', 0.0)
            ]
            self.motion_vectors.append(motion_vector)
            
        except Exception as e:
            self.logger.error(f"Trajectory data update error: {e}")
    
    def _calculate_display_fps(self) -> float:
        """Calculate display FPS"""
        
        if len(self.fps_counter) < 2:
            return 0.0
        
        return sum(self.fps_counter) / len(self.fps_counter)
    
    def _handle_keyboard_input(self, key: int):
        """Handle keyboard input for visualization controls"""
        
        try:
            if key == ord('f'):  # Toggle features
                self.show_features = not self.show_features
                self.logger.info(f"Features display: {'ON' if self.show_features else 'OFF'}")
                
            elif key == ord('3'):  # Toggle 3D markers
                self.show_3d_markers = not self.show_3d_markers
                self.logger.info(f"3D markers: {'ON' if self.show_3d_markers else 'OFF'}")
                
            elif key == ord('m'):  # Toggle motion vectors
                self.show_motion_vectors = not self.show_motion_vectors
                self.logger.info(f"Motion vectors: {'ON' if self.show_motion_vectors else 'OFF'}")
                
            elif key == ord('t'):  # Toggle trajectory
                self.show_trajectory_overlay = not self.show_trajectory_overlay
                self.logger.info(f"Trajectory overlay: {'ON' if self.show_trajectory_overlay else 'OFF'}")
                
            elif key == ord('s'):  # Toggle statistics
                self.show_statistics = not self.show_statistics
                self.logger.info(f"Statistics: {'ON' if self.show_statistics else 'OFF'}")
                
            elif key == ord('r'):  # Reset trajectory
                self.trajectory_points.clear()
                self.motion_vectors.clear()
                self.logger.info("Trajectory data reset")
                
            elif key == ord('h'):  # Help
                self._show_help()
                
        except Exception as e:
            self.logger.error(f"Keyboard input handling error: {e}")
    
    def _show_help(self):
        """Show keyboard shortcuts help"""
        help_text = """
        KEYBOARD SHORTCUTS:
        F - Toggle feature points
        3 - Toggle 3D coordinate system
        M - Toggle motion vectors
        T - Toggle trajectory overlay
        S - Toggle statistics
        R - Reset trajectory data
        H - Show this help
        ESC - Close live feed
        """
        self.logger.info(help_text)
    
    def _cleanup_windows(self):
        """Safely cleanup OpenCV windows"""
        
        try:
            with self.window_lock:
                if self.windows_created:
                    cv2.destroyWindow(self.live_window)
                    cv2.destroyWindow(self.debug_window)
                    cv2.waitKey(1)  # Process window destruction
                    self.windows_created = False
                    self.logger.info("OpenCV windows cleaned up")
        except Exception as e:
            self.logger.error(f"Window cleanup error: {e}")
    
    def stop_visualization(self):
        """Stop live feed visualization safely"""
        
        try:
            self.logger.info("Stopping live feed visualization...")
            
            # Signal stop
            self.is_active = False
            self.stop_event.set()
            
            # Wait for display thread
            if self.display_thread and self.display_thread.is_alive():
                self.display_thread.join(timeout=3.0)
                if self.display_thread.is_alive():
                    self.logger.warning("Display thread did not stop gracefully")
            
            # Cleanup windows
            self._cleanup_windows()
            
            # Clear queues
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            self.logger.info("Live feed visualization stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping visualization: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get visualizer status information"""
        
        return {
            'is_active': self.is_active,
            'windows_created': self.windows_created,
            'display_fps': self._calculate_display_fps(),
            'queue_size': self.frame_queue.qsize(),
            'trajectory_points': len(self.trajectory_points),
            'settings': {
                'show_3d_markers': self.show_3d_markers,
                'show_features': self.show_features,
                'show_motion_vectors': self.show_motion_vectors,
                'show_trajectory_overlay': self.show_trajectory_overlay,
                'show_statistics': self.show_statistics
            }
        }