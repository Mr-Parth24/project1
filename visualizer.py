"""
Advanced Real-time Visualization System
Author: Mr-Parth24
Date: 2025-06-13
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import threading
import queue
import time
from typing import Dict, List, Any, Optional, Tuple
import logging

from feature_processor import FeatureData, MatchResult
from enhanced_pose_estimator import PoseResult
from loop_detector import LoopDetectionResult

class Visualizer:
    """Advanced real-time visualization system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Visualization configuration
        self.enable_3d_plot = config.get('visualization', {}).get('enable_3d_plot', True)
        self.enable_feature_display = config.get('visualization', {}).get('enable_feature_display', True)
        self.enable_match_display = config.get('visualization', {}).get('enable_match_display', True)
        self.update_frequency = config.get('visualization', {}).get('update_frequency', 5)
        self.plot_window_size = config.get('visualization', {}).get('plot_window_size', 500)
        
        # Display windows
        self.window_names = {
            'main': 'Visual Odometry - Main View',
            'features': 'Feature Detection',
            'matches': 'Feature Matches',
            'trajectory': '3D Trajectory'
        }
        
        # 3D plotting
        self.fig = None
        self.ax_3d = None
        self.trajectory_line = None
        self.keyframe_points = None
        self.loop_closure_lines = None
        
        # Data for visualization
        self.trajectory_points = []
        self.keyframe_positions = []
        self.loop_closures = []
        self.current_pose = np.eye(4)
        
        # Frame buffers
        self.current_frame = None
        self.previous_frame = None
        self.feature_frame = None
        self.match_frame = None
        
        # Statistics overlay
        self.stats = {
            'frame_count': 0,
            'fps': 0.0,
            'features': 0,
            'matches': 0,
            'distance': 0.0,
            'pose_confidence': 0.0,
            'loop_detected': False,
            'method': 'Unknown'
        }
        
        # Threading for 3D plot updates
        self.plot_update_queue = queue.Queue(maxsize=10)
        self.plot_thread = None
        self.plot_active = False
        
        # Initialize visualization
        self._initialize_windows()
        if self.enable_3d_plot:
            self._initialize_3d_plot()
        
        self.logger.info("Visualizer initialized")
    
    def _initialize_windows(self):
        """Initialize OpenCV windows"""
        try:
            # Create windows
            cv2.namedWindow(self.window_names['main'], cv2.WINDOW_AUTOSIZE)
            
            if self.enable_feature_display:
                cv2.namedWindow(self.window_names['features'], cv2.WINDOW_AUTOSIZE)
            
            if self.enable_match_display:
                cv2.namedWindow(self.window_names['matches'], cv2.WINDOW_AUTOSIZE)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize windows: {e}")
    
    def _initialize_3d_plot(self):
        """Initialize 3D matplotlib plot"""
        try:
            plt.ion()  # Interactive mode
            self.fig = plt.figure(figsize=(10, 8))
            self.ax_3d = self.fig.add_subplot(111, projection='3d')
            
            self.ax_3d.set_xlabel('X (meters)')
            self.ax_3d.set_ylabel('Y (meters)')
            self.ax_3d.set_zlabel('Z (meters)')
            self.ax_3d.set_title('Real-time 3D Trajectory')
            
            # Initialize empty plots
            self.trajectory_line, = self.ax_3d.plot([], [], [], 'b-', linewidth=2, label='Trajectory')
            self.keyframe_points, = self.ax_3d.plot([], [], [], 'ro', markersize=8, label='Keyframes')
            
            self.ax_3d.legend()
            
            # Start plot update thread
            self.plot_active = True
            self.plot_thread = threading.Thread(target=self._plot_update_worker, daemon=True)
            self.plot_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize 3D plot: {e}")
            self.enable_3d_plot = False
    
    def _plot_update_worker(self):
        """Worker thread for updating 3D plot"""
        while self.plot_active:
            try:
                # Get update data from queue
                update_data = self.plot_update_queue.get(timeout=1.0)
                self._update_3d_plot(update_data)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Plot update error: {e}")
    
    def update(self, color_frame: np.ndarray, features: FeatureData,
               pose_result: PoseResult, loop_result: LoopDetectionResult):
        """Update all visualizations"""
        try:
            # Update current frame
            self.current_frame = color_frame.copy()
            
            # Update statistics
            self._update_stats(features, pose_result, loop_result)
            
            # Update trajectory data
            if pose_result.success:
                position = self.current_pose[:3, 3]
                self.trajectory_points.append(position.copy())
            
            # Create main display
            main_display = self._create_main_display(color_frame, features, pose_result, loop_result)
            
            # Show main window
            cv2.imshow(self.window_names['main'], main_display)
            
            # Update feature display
            if self.enable_feature_display:
                feature_display = self._create_feature_display(color_frame, features)
                cv2.imshow(self.window_names['features'], feature_display)
            
            # Update 3D plot (async)
            if self.enable_3d_plot and self.stats['frame_count'] % self.update_frequency == 0:
                plot_data = {
                    'trajectory': self.trajectory_points.copy(),
                    'keyframes': self.keyframe_positions.copy(),
                    'current_position': position if pose_result.success else None,
                    'loop_detected': loop_result.detected
                }
                
                try:
                    self.plot_update_queue.put_nowait(plot_data)
                except queue.Full:
                    pass  # Skip update if queue is full
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            return self._handle_keyboard_input(key)
            
        except Exception as e:
            self.logger.error(f"Visualization update failed: {e}")
            return False
    
    def _create_main_display(self, color_frame: np.ndarray, features: FeatureData,
                            pose_result: PoseResult, loop_result: LoopDetectionResult) -> np.ndarray:
        """Create main display with overlays"""
        display = color_frame.copy()
        
        # Draw features
        if features.keypoints:
            for kp in features.keypoints[:50]:  # Limit display for performance
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(display, (x, y), 3, (0, 255, 0), 1)
                cv2.circle(display, (x, y), int(kp.size/2), (0, 255, 0), 1)
        
        # Draw status overlay
        self._draw_status_overlay(display, pose_result, loop_result)
        
        # Draw trajectory visualization (mini-map)
        self._draw_mini_trajectory(display)
        
        # Draw pose confidence indicator
        self._draw_confidence_indicator(display, pose_result.confidence)
        
        return display
    
    def _create_feature_display(self, color_frame: np.ndarray, features: FeatureData) -> np.ndarray:
        """Create feature detection display"""
        display = color_frame.copy()
        
        if features.keypoints:
            # Draw all features with different colors based on response
            for kp in features.keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                
                # Color based on feature response (strength)
                response_norm = min(kp.response / 100.0, 1.0)
                color = (
                    int(255 * (1 - response_norm)),  # Blue component
                    int(255 * response_norm),        # Green component
                    0                                # Red component
                )
                
                cv2.circle(display, (x, y), 2, color, -1)
                
                # Draw orientation
                if kp.angle >= 0:
                    end_x = int(x + 10 * np.cos(np.radians(kp.angle)))
                    end_y = int(y + 10 * np.sin(np.radians(kp.angle)))
                    cv2.line(display, (x, y), (end_x, end_y), color, 1)
        
        # Add feature statistics
        cv2.putText(display, f"Features: {len(features.keypoints) if features.keypoints else 0}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display, f"Quality: {features.quality_score:.3f}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return display
    
    def _draw_status_overlay(self, display: np.ndarray, pose_result: PoseResult,
                           loop_result: LoopDetectionResult):
        """Draw status information overlay"""
        # Background for text
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        # Status text
        y_offset = 30
        text_color = (255, 255, 255)
        
        cv2.putText(display, f"Frame: {self.stats['frame_count']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y_offset += 25
        
        cv2.putText(display, f"FPS: {self.stats['fps']:.1f}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y_offset += 25
        
        cv2.putText(display, f"Features: {self.stats['features']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y_offset += 25
        
        cv2.putText(display, f"Matches: {self.stats['matches']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y_offset += 25
        
        cv2.putText(display, f"Distance: {self.stats['distance']:.2f}m", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y_offset += 25
        
        cv2.putText(display, f"Method: {self.stats['method']}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        y_offset += 25
        
        # Loop closure indicator
        if loop_result.detected:
            cv2.putText(display, "LOOP CLOSURE DETECTED!", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def _draw_mini_trajectory(self, display: np.ndarray):
        """Draw mini trajectory map"""
        if len(self.trajectory_points) < 2:
            return
        
        # Mini-map parameters
        map_size = 150
        map_x = display.shape[1] - map_size - 10
        map_y = 10
        
        # Create mini-map background
        cv2.rectangle(display, (map_x, map_y), (map_x + map_size, map_y + map_size), 
                     (50, 50, 50), -1)
        cv2.rectangle(display, (map_x, map_y), (map_x + map_size, map_y + map_size), 
                     (255, 255, 255), 2)
        
        # Get trajectory bounds
        traj_array = np.array(self.trajectory_points)
        if len(traj_array) > 0:
            x_coords = traj_array[:, 0]
            z_coords = traj_array[:, 2]  # Use Z as Y in top-down view
            
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            z_min, z_max = np.min(z_coords), np.max(z_coords)
            
            # Avoid division by zero
            x_range = max(x_max - x_min, 0.1)
            z_range = max(z_max - z_min, 0.1)
            
            # Draw trajectory
            prev_point = None
            for point in self.trajectory_points[-100:]:  # Last 100 points
                # Normalize to mini-map coordinates
                x_norm = (point[0] - x_min) / x_range
                z_norm = (point[2] - z_min) / z_range
                
                map_point = (
                    int(map_x + x_norm * (map_size - 20) + 10),
                    int(map_y + z_norm * (map_size - 20) + 10)
                )
                
                if prev_point is not None:
                    cv2.line(display, prev_point, map_point, (0, 255, 255), 1)
                
                prev_point = map_point
            
            # Draw current position
            if prev_point:
                cv2.circle(display, prev_point, 3, (0, 0, 255), -1)
    
    def _draw_confidence_indicator(self, display: np.ndarray, confidence: float):
        """Draw pose confidence indicator"""
        # Confidence bar
        bar_width = 200
        bar_height = 20
        bar_x = display.shape[1] - bar_width - 10
        bar_y = display.shape[0] - bar_height - 10
        
        # Background
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (50, 50, 50), -1)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 1)
        
        # Confidence fill
        fill_width = int(confidence * (bar_width - 4))
        if fill_width > 0:
            # Color based on confidence level
            if confidence > 0.7:
                color = (0, 255, 0)  # Green
            elif confidence > 0.4:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            cv2.rectangle(display, (bar_x + 2, bar_y + 2), 
                         (bar_x + 2 + fill_width, bar_y + bar_height - 2), color, -1)
        
        # Confidence text
        cv2.putText(display, f"Confidence: {confidence:.2f}", 
                   (bar_x, bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _update_3d_plot(self, plot_data: Dict[str, Any]):
        """Update 3D trajectory plot"""
        try:
            if not self.enable_3d_plot or self.ax_3d is None:
                return
            
            trajectory = plot_data.get('trajectory', [])
            keyframes = plot_data.get('keyframes', [])
            current_pos = plot_data.get('current_position')
            
            if len(trajectory) < 2:
                return
            
            # Convert to numpy array
            traj_array = np.array(trajectory)
            
            # Update trajectory line
            self.trajectory_line.set_data_3d(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2])
            
            # Update keyframe points
            if keyframes:
                kf_array = np.array(keyframes)
                self.keyframe_points.set_data_3d(kf_array[:, 0], kf_array[:, 1], kf_array[:, 2])
            
            # Set axis limits
            self._set_axis_limits(traj_array)
            
            # Add current position marker
            if current_pos is not None:
                # Clear previous current position markers
                for artist in self.ax_3d.collections:
                    if hasattr(artist, '_current_pos_marker'):
                        artist.remove()
                
                scatter = self.ax_3d.scatter(current_pos[0], current_pos[1], current_pos[2], 
                                           c='red', s=100, marker='o', alpha=0.8)
                scatter._current_pos_marker = True
            
            # Update title with loop closure status
            if plot_data.get('loop_detected', False):
                self.ax_3d.set_title('3D Trajectory - LOOP CLOSURE DETECTED!', color='red')
            else:
                self.ax_3d.set_title('Real-time 3D Trajectory', color='black')
            
            # Refresh plot
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            self.logger.error(f"3D plot update failed: {e}")
    
    def _set_axis_limits(self, trajectory: np.ndarray):
        """Set appropriate axis limits for 3D plot"""
        try:
            if len(trajectory) == 0:
                return
            
            # Calculate bounds
            x_min, x_max = np.min(trajectory[:, 0]), np.max(trajectory[:, 0])
            y_min, y_max = np.min(trajectory[:, 1]), np.max(trajectory[:, 1])
            z_min, z_max = np.min(trajectory[:, 2]), np.max(trajectory[:, 2])
            
            # Add margins
            x_margin = max(0.1, (x_max - x_min) * 0.1)
            y_margin = max(0.1, (y_max - y_min) * 0.1)
            z_margin = max(0.1, (z_max - z_min) * 0.1)
            
            self.ax_3d.set_xlim(x_min - x_margin, x_max + x_margin)
            self.ax_3d.set_ylim(y_min - y_margin, y_max + y_margin)
            self.ax_3d.set_zlim(z_min - z_margin, z_max + z_margin)
            
            # Equal aspect ratio
            max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
            if max_range > 0:
                center_x = (x_max + x_min) / 2
                center_y = (y_max + y_min) / 2
                center_z = (z_max + z_min) / 2
                
                self.ax_3d.set_xlim(center_x - max_range/2, center_x + max_range/2)
                self.ax_3d.set_ylim(center_y - max_range/2, center_y + max_range/2)
                self.ax_3d.set_zlim(center_z - max_range/2, center_z + max_range/2)
            
        except Exception as e:
            self.logger.warning(f"Failed to set axis limits: {e}")
    
    def _update_stats(self, features: FeatureData, pose_result: PoseResult,
                     loop_result: LoopDetectionResult):
        """Update visualization statistics"""
        self.stats['frame_count'] += 1
        self.stats['features'] = len(features.keypoints) if features.keypoints else 0
        self.stats['matches'] = pose_result.num_matches
        self.stats['distance'] = pose_result.total_distance
        self.stats['pose_confidence'] = pose_result.confidence
        self.stats['loop_detected'] = loop_result.detected
        self.stats['method'] = pose_result.method_used.value
        
        # Calculate FPS (simple moving average)
        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            frame_time = current_time - self.last_frame_time
            if frame_time > 0:
                current_fps = 1.0 / frame_time
                self.stats['fps'] = 0.9 * self.stats['fps'] + 0.1 * current_fps
        
        self.last_frame_time = current_time
    
    def _handle_keyboard_input(self, key: int) -> bool:
        """Handle keyboard input"""
        if key == ord('q') or key == 27:  # 'q' or ESC
            return False
        elif key == ord('s'):  # Save screenshot
            self._save_screenshot()
        elif key == ord('r'):  # Reset view
            self._reset_view()
        elif key == ord('f'):  # Toggle feature display
            self.enable_feature_display = not self.enable_feature_display
        elif key == ord('3'):  # Toggle 3D plot
            self.enable_3d_plot = not self.enable_3d_plot
        
        return True
    
    def _save_screenshot(self):
        """Save current visualization as screenshot"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            if self.current_frame is not None:
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, self.current_frame)
                self.logger.info(f"Screenshot saved: {filename}")
            
            if self.enable_3d_plot and self.fig is not None:
                filename = f"trajectory_3d_{timestamp}.png"
                self.fig.savefig(filename, dpi=150, bbox_inches='tight')
                self.logger.info(f"3D plot saved: {filename}")
                
        except Exception as e:
            self.logger.error(f"Failed to save screenshot: {e}")
    
    def _reset_view(self):
        """Reset visualization view"""
        try:
            if self.enable_3d_plot and self.ax_3d is not None:
                self.ax_3d.view_init(elev=20, azim=45)
                self.fig.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Failed to reset view: {e}")
    
    def add_keyframe_position(self, position: np.ndarray):
        """Add keyframe position for visualization"""
        self.keyframe_positions.append(position.copy())
    
    def add_loop_closure(self, pos1: np.ndarray, pos2: np.ndarray):
        """Add loop closure for visualization"""
        self.loop_closures.append((pos1.copy(), pos2.copy()))
    
    def cleanup(self):
        """Cleanup visualization resources"""
        try:
            # Stop plot thread
            self.plot_active = False
            if self.plot_thread and self.plot_thread.is_alive():
                self.plot_thread.join(timeout=2.0)
            
            # Close OpenCV windows
            cv2.destroyAllWindows()
            
            # Close matplotlib
            if self.fig is not None:
                plt.close(self.fig)
            
            self.logger.info("Visualizer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Visualizer cleanup error: {e}")