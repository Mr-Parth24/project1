"""
Live Feed Visualizer with 3D Markers and Real-time Overlays
Author: Mr-Parth24
Date: 2025-06-13
Time: 20:52:08 UTC
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import Dict, List, Optional, Tuple
import logging
import math

class LiveFeedVisualizer:
    """Live camera feed with 3D markers and enhanced overlays"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Display windows
        self.live_window = "Live Camera Feed - Visual Odometry"
        self.debug_window = "Debug View - Features & Tracking"
        
        # Visualization state
        self.is_active = False
        self.show_3d_markers = True
        self.show_features = True
        self.show_motion_vectors = True
        self.show_trajectory_overlay = True
        
        # Display queues
        self.display_queue = queue.Queue(maxsize=5)
        self.display_thread = None
        
        # Trajectory for overlay
        self.trajectory_points = []
        self.max_trajectory_points = 100
        
        # 3D coordinate system
        self.axis_length = 0.1  # 10cm axes
        
        # Statistics overlay
        self.stats_overlay = {
            'fps': 0.0,
            'features': 0,
            'matches': 0,
            'distance': 0.0,
            'confidence': 0.0,
            'tracking_status': 'INIT'
        }
        
        self.logger.info("Live feed visualizer initialized")
    
    def start_visualization(self):
        """Start live feed visualization"""
        try:
            self.is_active = True
            
            # Create windows
            cv2.namedWindow(self.live_window, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(self.debug_window, cv2.WINDOW_AUTOSIZE)
            
            # Start display thread
            self.display_thread = threading.Thread(target=self._display_worker, daemon=True)
            self.display_thread.start()
            
            self.logger.info("Live feed visualization started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start visualization: {e}")
            return False
    
    def update_display(self, color_frame: np.ndarray, depth_frame: np.ndarray,
                      tracking_result: Dict, camera_matrix: np.ndarray = None):
        """Update live display with tracking information"""
        try:
            if not self.is_active:
                return
            
            # Create display frame
            display_frame = color_frame.copy()
            debug_frame = color_frame.copy()
            
            # Update statistics
            self._update_stats(tracking_result)
            
            # Draw 3D coordinate system
            if self.show_3d_markers and camera_matrix is not None:
                display_frame = self._draw_3d_coordinate_system(
                    display_frame, camera_matrix, tracking_result
                )
            
            # Draw features
            if self.show_features and 'keypoints' in tracking_result:
                display_frame, debug_frame = self._draw_features(
                    display_frame, debug_frame, tracking_result['keypoints']
                )
            
            # Draw motion vectors
            if self.show_motion_vectors:
                display_frame = self._draw_motion_vectors(display_frame, tracking_result)
            
            # Draw trajectory overlay
            if self.show_trajectory_overlay:
                display_frame = self._draw_trajectory_overlay(display_frame, tracking_result)
            
            # Draw statistics overlay
            display_frame = self._draw_statistics_overlay(display_frame)
            debug_frame = self._draw_debug_overlay(debug_frame, tracking_result)
            
            # Draw distance and direction indicators
            display_frame = self._draw_distance_indicators(display_frame, tracking_result)
            
            # Add to display queue
            display_data = {
                'live_frame': display_frame,
                'debug_frame': debug_frame,
                'timestamp': time.time()
            }
            
            try:
                self.display_queue.put_nowait(display_data)
            except queue.Full:
                try:
                    self.display_queue.get_nowait()
                    self.display_queue.put_nowait(display_data)
                except queue.Empty:
                    pass
            
        except Exception as e:
            self.logger.error(f"Display update error: {e}")
    
    def _display_worker(self):
        """Worker thread for displaying frames"""
        while self.is_active:
            try:
                display_data = self.display_queue.get(timeout=1.0)
                
                # Show frames
                cv2.imshow(self.live_window, display_data['live_frame'])
                cv2.imshow(self.debug_window, display_data['debug_frame'])
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                self._handle_keyboard_input(key)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Display worker error: {e}")
                time.sleep(0.1)
    
    def _draw_3d_coordinate_system(self, frame: np.ndarray, camera_matrix: np.ndarray,
                                  tracking_result: Dict) -> np.ndarray:
        """Draw 3D coordinate system at current position"""
        try:
            # Define 3D axis points
            axis_points_3d = np.array([
                [0, 0, 0],              # Origin
                [self.axis_length, 0, 0],  # X-axis (Red)
                [0, self.axis_length, 0],  # Y-axis (Green)
                [0, 0, self.axis_length]   # Z-axis (Blue)
            ], dtype=np.float32)
            
            # Use identity pose for display (camera-relative coordinates)
            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))
            
            # Project 3D points to 2D
            projected_points, _ = cv2.projectPoints(
                axis_points_3d, rvec, tvec, camera_matrix, np.zeros((4, 1))
            )
            
            # Convert to integer coordinates
            origin = tuple(map(int, projected_points[0][0]))
            x_end = tuple(map(int, projected_points[1][0]))
            y_end = tuple(map(int, projected_points[2][0]))
            z_end = tuple(map(int, projected_points[3][0]))
            
            # Check if points are within frame bounds
            h, w = frame.shape[:2]
            if (0 <= origin[0] < w and 0 <= origin[1] < h):
                # Draw axes with different colors
                cv2.line(frame, origin, x_end, (0, 0, 255), 3)  # X-axis: Red
                cv2.line(frame, origin, y_end, (0, 255, 0), 3)  # Y-axis: Green
                cv2.line(frame, origin, z_end, (255, 0, 0), 3)  # Z-axis: Blue
                
                # Draw axis labels
                cv2.putText(frame, 'X', x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, 'Y', y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, 'Z', z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Draw origin marker
                cv2.circle(frame, origin, 5, (255, 255, 255), -1)
                cv2.circle(frame, origin, 5, (0, 0, 0), 2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"3D coordinate system drawing error: {e}")
            return frame
    
    def _draw_features(self, live_frame: np.ndarray, debug_frame: np.ndarray,
                      keypoints: List) -> Tuple[np.ndarray, np.ndarray]:
        """Draw feature points on frames"""
        try:
            if not keypoints:
                return live_frame, debug_frame
            
            # Draw features on live frame (subset for clarity)
            for i, kp in enumerate(keypoints[:50]):  # Limit for performance
                x, y = int(kp.pt[0]), int(kp.pt[1])
                
                # Different colors based on feature strength
                response_norm = min(kp.response / 100.0, 1.0)
                color = (
                    int(255 * (1 - response_norm)),  # Blue
                    int(255 * response_norm),        # Green
                    0                                # Red
                )
                
                cv2.circle(live_frame, (x, y), 3, color, 1)
                
                # Add index for strong features
                if kp.response > 50:
                    cv2.putText(live_frame, str(i), (x+5, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # Draw all features on debug frame
            for i, kp in enumerate(keypoints):
                x, y = int(kp.pt[0]), int(kp.pt[1])
                
                # Color coding by response strength
                if kp.response > 80:
                    color = (0, 255, 0)     # Green for strong features
                elif kp.response > 40:
                    color = (0, 255, 255)   # Yellow for medium features
                else:
                    color = (255, 0, 0)     # Blue for weak features
                
                cv2.circle(debug_frame, (x, y), 2, color, -1)
                
                # Draw orientation if available
                if kp.angle >= 0:
                    end_x = int(x + 15 * math.cos(math.radians(kp.angle)))
                    end_y = int(y + 15 * math.sin(math.radians(kp.angle)))
                    cv2.line(debug_frame, (x, y), (end_x, end_y), color, 1)
            
            return live_frame, debug_frame
            
        except Exception as e:
            self.logger.error(f"Feature drawing error: {e}")
            return live_frame, debug_frame
    
    def _draw_motion_vectors(self, frame: np.ndarray, tracking_result: Dict) -> np.ndarray:
        """Draw motion vectors showing camera movement"""
        try:
            # Get current position and previous position
            current_pos = tracking_result.get('current_position', [0, 0, 0])
            
            if len(self.trajectory_points) > 1:
                prev_pos = self.trajectory_points[-1]
                
                # Calculate motion vector in 2D (top-down projection)
                motion_x = current_pos[0] - prev_pos[0]
                motion_z = current_pos[2] - prev_pos[2]  # Use Z as forward direction
                
                # Scale and position the vector
                center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                scale = 1000  # Scale factor for visualization
                
                end_x = int(center_x + motion_x * scale)
                end_y = int(center_y + motion_z * scale)
                
                # Draw motion vector
                if abs(motion_x) > 0.001 or abs(motion_z) > 0.001:
                    cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                                   (0, 255, 255), 3, tipLength=0.3)
                    
                    # Draw motion magnitude
                    motion_magnitude = math.sqrt(motion_x**2 + motion_z**2)
                    cv2.putText(frame, f"Motion: {motion_magnitude:.3f}m",
                               (center_x - 80, center_y - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Motion vector drawing error: {e}")
            return frame
    
    def _draw_trajectory_overlay(self, frame: np.ndarray, tracking_result: Dict) -> np.ndarray:
        """Draw mini trajectory overlay on the frame"""
        try:
            # Update trajectory points
            current_pos = tracking_result.get('current_position', [0, 0, 0])
            self.trajectory_points.append(current_pos)
            
            # Limit trajectory points
            if len(self.trajectory_points) > self.max_trajectory_points:
                self.trajectory_points.pop(0)
            
            if len(self.trajectory_points) < 2:
                return frame
            
            # Draw mini-map in top-right corner
            map_size = 150
            map_x = frame.shape[1] - map_size - 10
            map_y = 10
            
            # Create semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (map_x, map_y), (map_x + map_size, map_y + map_size),
                         (50, 50, 50), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Draw trajectory
            traj_array = np.array(self.trajectory_points)
            x_coords = traj_array[:, 0]
            z_coords = traj_array[:, 2]  # Use Z for forward direction
            
            if len(x_coords) > 1:
                # Scale to fit map
                x_range = max(np.max(x_coords) - np.min(x_coords), 0.1)
                z_range = max(np.max(z_coords) - np.min(z_coords), 0.1)
                
                x_min, z_min = np.min(x_coords), np.min(z_coords)
                margin = 10
                
                prev_point = None
                for i, (x, z) in enumerate(zip(x_coords, z_coords)):
                    # Normalize to map coordinates
                    norm_x = (x - x_min) / x_range
                    norm_z = (z - z_min) / z_range
                    
                    map_point = (
                        int(map_x + margin + norm_x * (map_size - 2 * margin)),
                        int(map_y + margin + norm_z * (map_size - 2 * margin))
                    )
                    
                    if prev_point is not None:
                        # Color gradient from blue to red
                        progress = i / len(x_coords)
                        color = (int(255 * progress), 0, int(255 * (1 - progress)))
                        cv2.line(frame, prev_point, map_point, color, 2)
                    
                    prev_point = map_point
                
                # Draw current position
                if prev_point:
                    cv2.circle(frame, prev_point, 4, (0, 255, 0), -1)
                    cv2.circle(frame, prev_point, 4, (255, 255, 255), 1)
            
            # Draw compass
            compass_center = (map_x + map_size - 25, map_y + 25)
            cv2.circle(frame, compass_center, 15, (255, 255, 255), 1)
            cv2.putText(frame, 'N', (compass_center[0] - 5, compass_center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Trajectory overlay drawing error: {e}")
            return frame
    
    def _draw_statistics_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw real-time statistics overlay"""
        try:
            # Create semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 200), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Draw border
            cv2.rectangle(frame, (10, 10), (350, 200), (255, 255, 255), 2)
            
            # Statistics text
            y_offset = 35
            line_height = 25
            
            stats_text = [
                f"FPS: {self.stats_overlay['fps']:.1f}",
                f"Features: {self.stats_overlay['features']}",
                f"Matches: {self.stats_overlay['matches']}",
                f"Distance: {self.stats_overlay['distance']:.3f}m",
                f"Confidence: {self.stats_overlay['confidence']:.2f}",
                f"Status: {self.stats_overlay['tracking_status']}",
                f"Time: {time.strftime('%H:%M:%S')}"
            ]
            
            for i, text in enumerate(stats_text):
                color = (255, 255, 255)
                if "Status:" in text:
                    if "EXCELLENT" in text:
                        color = (0, 255, 0)
                    elif "GOOD" in text:
                        color = (0, 255, 255)
                    elif "FAIR" in text:
                        color = (0, 165, 255)
                    elif "POOR" in text:
                        color = (0, 0, 255)
                
                cv2.putText(frame, text, (20, y_offset + i * line_height),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Statistics overlay drawing error: {e}")
            return frame
    
    def _draw_debug_overlay(self, frame: np.ndarray, tracking_result: Dict) -> np.ndarray:
        """Draw debug information overlay"""
        try:
            # Debug information
            debug_info = [
                f"Keypoints: {len(tracking_result.get('keypoints', []))}",
                f"Quality: {tracking_result.get('quality_score', 0.0):.3f}",
                f"Feature Quality: {tracking_result.get('feature_quality', 0.0):.3f}",
                f"X: {tracking_result.get('x_displacement', 0.0):.3f}m",
                f"Y: {tracking_result.get('y_displacement', 0.0):.3f}m",
                f"Z: {tracking_result.get('z_displacement', 0.0):.3f}m",
                f"Speed: {tracking_result.get('current_speed', 0.0):.3f}m/s",
                f"Direction: {tracking_result.get('direction_angle', 0.0):.1f}°"
            ]
            
            # Draw debug overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, frame.shape[0] - 220), (300, frame.shape[0] - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            y_start = frame.shape[0] - 200
            for i, info in enumerate(debug_info):
                cv2.putText(frame, info, (20, y_start + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Debug overlay drawing error: {e}")
            return frame
    
    def _draw_distance_indicators(self, frame: np.ndarray, tracking_result: Dict) -> np.ndarray:
        """Draw distance and direction indicators"""
        try:
            # Distance from start indicator
            displacement = tracking_result.get('displacement_from_start', 0.0)
            total_distance = tracking_result.get('total_distance', 0.0)
            direction = tracking_result.get('direction_angle', 0.0)
            
            # Draw distance bar
            bar_width = 200
            bar_height = 20
            bar_x = frame.shape[1] - bar_width - 20
            bar_y = frame.shape[0] - 60
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (255, 255, 255), 1)
            
            # Distance fill (scale up to 10 meters)
            fill_ratio = min(displacement / 10.0, 1.0)
            fill_width = int(fill_ratio * (bar_width - 4))
            
            if fill_width > 0:
                color = (0, int(255 * fill_ratio), int(255 * (1 - fill_ratio)))
                cv2.rectangle(frame, (bar_x + 2, bar_y + 2),
                             (bar_x + 2 + fill_width, bar_y + bar_height - 2), color, -1)
            
            # Distance text
            cv2.putText(frame, f"From Start: {displacement:.2f}m",
                       (bar_x, bar_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Total: {total_distance:.2f}m",
                       (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Direction indicator (compass-like)
            compass_center = (frame.shape[1] - 50, frame.shape[0] - 100)
            compass_radius = 25
            
            # Draw compass circle
            cv2.circle(frame, compass_center, compass_radius, (255, 255, 255), 2)
            
            # Draw direction arrow
            direction_rad = math.radians(direction - 90)  # Adjust for North being up
            arrow_end = (
                int(compass_center[0] + compass_radius * 0.8 * math.cos(direction_rad)),
                int(compass_center[1] + compass_radius * 0.8 * math.sin(direction_rad))
            )
            
            cv2.arrowedLine(frame, compass_center, arrow_end, (0, 255, 0), 2, tipLength=0.4)
            
            # Draw N, S, E, W markers
            cv2.putText(frame, 'N', (compass_center[0] - 5, compass_center[1] - compass_radius - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Direction text
            direction_text = self._get_direction_text(direction)
            cv2.putText(frame, direction_text, (compass_center[0] - 20, compass_center[1] + compass_radius + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Distance indicators drawing error: {e}")
            return frame
    
    def _get_direction_text(self, angle: float) -> str:
        """Convert angle to compass direction text"""
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        index = int((angle + 22.5) / 45) % 8
        return directions[index]
    
    def _update_stats(self, tracking_result: Dict):
        """Update statistics for overlay"""
        self.stats_overlay.update({
            'features': tracking_result.get('num_features', 0),
            'matches': tracking_result.get('num_matches', 0),
            'distance': tracking_result.get('total_distance', 0.0),
            'confidence': tracking_result.get('tracking_confidence', 0.0),
            'tracking_status': tracking_result.get('tracking_status', 'UNKNOWN')
        })
        
        # Calculate FPS (simple moving average)
        current_time = time.time()
        if hasattr(self, '_last_frame_time'):
            frame_time = current_time - self._last_frame_time
            if frame_time > 0:
                current_fps = 1.0 / frame_time
                self.stats_overlay['fps'] = 0.8 * self.stats_overlay['fps'] + 0.2 * current_fps
        
        self._last_frame_time = current_time
    
    def _handle_keyboard_input(self, key: int):
        """Handle keyboard input for visualization controls"""
        if key == ord('f'):  # Toggle features
            self.show_features = not self.show_features
            self.logger.info(f"Features display: {'ON' if self.show_features else 'OFF'}")
        elif key == ord('3'):  # Toggle 3D markers
            self.show_3d_markers = not self.show_3d_markers
            self.logger.info(f"3D markers: {'ON' if self.show_3d_markers else 'OFF'}")
        elif key == ord('m'):  # Toggle motion vectors
            self.show_motion_vectors = not self.show_motion_vectors
            self.logger.info(f"Motion vectors: {'ON' if self.show_motion_vectors else 'OFF'}")
        elif key == ord('t'):  # Toggle trajectory overlay
            self.show_trajectory_overlay = not self.show_trajectory_overlay
            self.logger.info(f"Trajectory overlay: {'ON' if self.show_trajectory_overlay else 'OFF'}")
        elif key == ord('r'):  # Reset trajectory
            self.trajectory_points.clear()
            self.logger.info("Trajectory overlay reset")
    
    def stop_visualization(self):
        """Stop live feed visualization"""
        try:
            self.is_active = False
            
            if self.display_thread and self.display_thread.is_alive():
                self.display_thread.join(timeout=2.0)
            
            cv2.destroyAllWindows()
            
            self.logger.info("Live feed visualization stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping visualization: {e}")