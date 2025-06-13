"""
Real Camera Integration for Visual Odometry System
Author: Mr-Parth24
Date: 2025-06-13
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import time
import queue
from typing import Optional, Tuple
import logging

class RealCameraManager:
    """Real Intel RealSense camera integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Camera components
        self.pipeline = None
        self.config = None
        self.align = None
        self.is_running = False
        
        # Frame data
        self.frame_queue = queue.Queue(maxsize=5)
        self.capture_thread = None
        
        # Camera settings
        self.width = 640
        self.height = 480
        self.fps = 30
        
        # Statistics
        self.frame_count = 0
        self.last_frame_time = time.time()
        
    def initialize(self) -> bool:
        """Initialize RealSense camera"""
        try:
            self.logger.info("Initializing Intel RealSense D435i...")
            
            # Check for connected devices
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                self.logger.error("No RealSense devices found!")
                return False
            
            # Get device info
            device = devices[0]
            device_name = device.get_info(rs.camera_info.name)
            serial_number = device.get_info(rs.camera_info.serial_number)
            
            self.logger.info(f"Found device: {device_name} (Serial: {serial_number})")
            
            # Configure pipeline
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Enable streams
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            # Create alignment object
            self.align = rs.align(rs.stream.color)
            
            # Get camera intrinsics
            color_profile = profile.get_stream(rs.stream.color)
            self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            self.logger.info(f"Camera initialized successfully!")
            self.logger.info(f"Resolution: {self.width}x{self.height} @ {self.fps}fps")
            self.logger.info(f"Intrinsics: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            return False
    
    def start_capture(self) -> bool:
        """Start frame capture thread"""
        if self.is_running:
            return True
            
        try:
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            self.logger.info("Camera capture started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start capture: {e}")
            self.is_running = False
            return False
    
    def _capture_loop(self):
        """Main capture loop"""
        while self.is_running:
            try:
                # Get frames
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                
                # Align depth to color
                aligned_frames = self.align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Add to queue (non-blocking)
                frame_data = {
                    'color': color_image,
                    'depth': depth_image,
                    'timestamp': time.time(),
                    'frame_id': self.frame_count
                }
                
                try:
                    self.frame_queue.put_nowait(frame_data)
                    self.frame_count += 1
                except queue.Full:
                    # Remove old frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Empty:
                        pass
                
            except Exception as e:
                if self.is_running:  # Only log if we're supposed to be running
                    self.logger.error(f"Capture error: {e}")
                    time.sleep(0.1)
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get latest frame"""
        try:
            frame_data = self.frame_queue.get(timeout=0.1)
            return frame_data['color'], frame_data['depth']
        except queue.Empty:
            return None
    
    def get_fps(self) -> float:
        """Calculate current FPS"""
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        
        if time_diff > 0:
            fps = 1.0 / time_diff
            self.last_frame_time = current_time
            return fps
        
        return 0.0
    
    def stop_capture(self):
        """Stop capture thread"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.pipeline:
            self.pipeline.stop()
        
        self.logger.info("Camera capture stopped")
    
    def get_camera_info(self) -> dict:
        """Get camera information"""
        if not hasattr(self, 'intrinsics'):
            return {}
        
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'fx': self.intrinsics.fx,
            'fy': self.intrinsics.fy,
            'cx': self.intrinsics.ppx,
            'cy': self.intrinsics.ppy,
            'frame_count': self.frame_count
        }

class SimpleFeatureTracker:
    """Simple feature detection and tracking"""
    
    def __init__(self):
        self.detector = cv2.ORB_create(nfeatures=500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_frame = None
        
        # Trajectory tracking
        self.trajectory = [[0, 0, 0]]
        self.total_distance = 0.0
        
    def process_frame(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> dict:
        """Process frame and extract features"""
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        result = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'num_features': len(keypoints) if keypoints else 0,
            'num_matches': 0,
            'distance_moved': 0.0,
            'total_distance': self.total_distance,
            'trajectory': self.trajectory.copy()
        }
        
        # Match with previous frame
        if (self.prev_descriptors is not None and descriptors is not None and 
            len(self.prev_descriptors) > 0 and len(descriptors) > 0):
            
            try:
                matches = self.matcher.match(self.prev_descriptors, descriptors)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Filter good matches
                good_matches = matches[:min(50, len(matches))]
                result['num_matches'] = len(good_matches)
                
                # Simple motion estimation (placeholder)
                if len(good_matches) > 10:
                    # Calculate approximate motion
                    motion = self._estimate_motion(good_matches)
                    
                    # Update trajectory
                    new_position = [
                        self.trajectory[-1][0] + motion[0],
                        self.trajectory[-1][1] + motion[1],
                        self.trajectory[-1][2] + motion[2]
                    ]
                    
                    distance_moved = np.linalg.norm(motion)
                    self.total_distance += distance_moved
                    
                    self.trajectory.append(new_position)
                    
                    result['distance_moved'] = distance_moved
                    result['total_distance'] = self.total_distance
                    result['trajectory'] = self.trajectory.copy()
                
            except Exception as e:
                print(f"Matching error: {e}")
        
        # Update previous frame data
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_frame = gray.copy()
        
        return result
    
    def _estimate_motion(self, matches) -> np.ndarray:
        """Simple motion estimation (placeholder)"""
        # This is a very simplified motion estimation
        # In the full system, this would use proper pose estimation
        
        if not matches or len(matches) < 5:
            return np.array([0.0, 0.0, 0.0])
        
        # Calculate average displacement
        total_dx = 0
        total_dy = 0
        
        for match in matches[:10]:  # Use best 10 matches
            pt1 = self.prev_keypoints[match.queryIdx].pt
            pt2 = self.prev_keypoints[match.trainIdx].pt if hasattr(self, 'current_keypoints') else pt1
            
            total_dx += pt2[0] - pt1[0]
            total_dy += pt2[1] - pt1[1]
        
        avg_dx = total_dx / len(matches[:10])
        avg_dy = total_dy / len(matches[:10])
        
        # Convert pixel motion to rough world coordinates (very approximate)
        scale_factor = 0.001  # Rough conversion factor
        
        return np.array([avg_dx * scale_factor, avg_dy * scale_factor, 0.02])