"""
Enhanced Real Camera Integration with Live Feed Support
Author: Mr-Parth24
Date: 2025-06-13
Time: 20:47:06 UTC
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import time
import queue
from typing import Optional, Tuple, Dict, Any
import logging
from collections import deque

class EnhancedRealCameraManager:
    """Enhanced Intel RealSense camera with live feed capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Camera components
        self.pipeline = None
        self.config = None
        self.align = None
        self.is_running = False
        
        # Enhanced frame processing
        self.frame_queue = queue.Queue(maxsize=10)
        self.display_queue = queue.Queue(maxsize=5)  # For live display
        self.capture_thread = None
        
        # Camera settings
        self.width = 640
        self.height = 480
        self.fps = 30
        
        # Enhanced intrinsics
        self.intrinsics = None
        self.depth_intrinsics = None
        
        # Performance tracking
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps_counter = deque(maxlen=30)
        
        # Camera matrix for 3D projections
        self.camera_matrix = None
        self.dist_coeffs = None
        
    def initialize(self) -> bool:
        """Enhanced camera initialization with better error handling"""
        try:
            self.logger.info("Initializing Enhanced Intel RealSense D435i...")
            
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
            firmware_version = device.get_info(rs.camera_info.firmware_version)
            
            self.logger.info(f"Found device: {device_name}")
            self.logger.info(f"Serial: {serial_number}")
            self.logger.info(f"Firmware: {firmware_version}")
            
            # Configure pipeline with enhanced settings
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Enable streams with higher quality
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            # Create alignment object
            self.align = rs.align(rs.stream.color)
            
            # Extract enhanced intrinsics
            self._extract_enhanced_intrinsics(profile)
            
            # Configure advanced camera settings
            self._configure_advanced_settings(device)
            
            self.logger.info("Enhanced camera initialized successfully!")
            self.logger.info(f"Resolution: {self.width}x{self.height} @ {self.fps}fps")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Enhanced camera initialization failed: {e}")
            return False
    
    def _extract_enhanced_intrinsics(self, profile):
        """Extract enhanced camera intrinsic parameters"""
        try:
            # Color intrinsics
            color_profile = profile.get_stream(rs.stream.color)
            color_intr = color_profile.as_video_stream_profile().get_intrinsics()
            
            # Depth intrinsics
            depth_profile = profile.get_stream(rs.stream.depth)
            depth_intr = depth_profile.as_video_stream_profile().get_intrinsics()
            
            self.intrinsics = color_intr
            self.depth_intrinsics = depth_intr
            
            # Create camera matrix for OpenCV operations
            self.camera_matrix = np.array([
                [color_intr.fx, 0, color_intr.ppx],
                [0, color_intr.fy, color_intr.ppy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            self.dist_coeffs = np.array(color_intr.coeffs, dtype=np.float32)
            
            self.logger.info(f"Enhanced intrinsics extracted:")
            self.logger.info(f"  Color: fx={color_intr.fx:.2f}, fy={color_intr.fy:.2f}")
            self.logger.info(f"  Principal: cx={color_intr.ppx:.2f}, cy={color_intr.ppy:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to extract intrinsics: {e}")
    
    def _configure_advanced_settings(self, device):
        """Configure advanced camera settings for better tracking"""
        try:
            # Color sensor settings
            color_sensor = device.first_color_sensor()
            
            # Auto exposure
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            
            # Auto white balance
            if color_sensor.supports(rs.option.enable_auto_white_balance):
                color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
            
            # Depth sensor settings
            depth_sensor = device.first_depth_sensor()
            
            # Depth preset for accuracy
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 3)  # High accuracy
            
            # Depth units
            if depth_sensor.supports(rs.option.depth_units):
                depth_sensor.set_option(rs.option.depth_units, 0.001)  # 1mm precision
            
            self.logger.info("Advanced camera settings configured")
            
        except Exception as e:
            self.logger.warning(f"Could not configure advanced settings: {e}")
    
    def start_capture(self) -> bool:
        """Start enhanced frame capture with live feed support"""
        if self.is_running:
            return True
            
        try:
            self.is_running = True
            self.capture_thread = threading.Thread(target=self._enhanced_capture_loop, daemon=True)
            self.capture_thread.start()
            
            self.logger.info("Enhanced camera capture started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start enhanced capture: {e}")
            self.is_running = False
            return False
    
    def _enhanced_capture_loop(self):
        """Enhanced capture loop with live feed processing"""
        while self.is_running:
            try:
                # Get frames with timeout
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
                
                # Calculate FPS
                current_time = time.time()
                if self.last_frame_time > 0:
                    frame_time = current_time - self.last_frame_time
                    if frame_time > 0:
                        fps = 1.0 / frame_time
                        self.fps_counter.append(fps)
                
                self.last_frame_time = current_time
                
                # Enhanced frame data
                frame_data = {
                    'color': color_image,
                    'depth': depth_image,
                    'timestamp': current_time,
                    'frame_id': self.frame_count,
                    'color_frame_obj': color_frame,  # For 3D projections
                    'depth_frame_obj': depth_frame
                }
                
                # Add to processing queue
                try:
                    self.frame_queue.put_nowait(frame_data)
                except queue.Full:
                    # Remove old frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Empty:
                        pass
                
                # Add to display queue for live feed
                display_data = {
                    'color': color_image.copy(),
                    'depth': depth_image.copy(),
                    'timestamp': current_time,
                    'frame_id': self.frame_count
                }
                
                try:
                    self.display_queue.put_nowait(display_data)
                except queue.Full:
                    try:
                        self.display_queue.get_nowait()
                        self.display_queue.put_nowait(display_data)
                    except queue.Empty:
                        pass
                
                self.frame_count += 1
                
            except Exception as e:
                if self.is_running:
                    self.logger.error(f"Enhanced capture error: {e}")
                    time.sleep(0.1)
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get latest frame for processing"""
        try:
            frame_data = self.frame_queue.get(timeout=0.1)
            return frame_data['color'], frame_data['depth']
        except queue.Empty:
            return None
    
    def get_display_frame(self) -> Optional[Dict[str, Any]]:
        """Get latest frame for live display"""
        try:
            return self.display_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_3d_point(self, u: int, v: int, depth_value: float) -> Optional[np.ndarray]:
        """Convert 2D pixel to 3D point using depth"""
        try:
            if self.intrinsics is None or depth_value <= 0:
                return None
            
            # Convert depth from mm to meters
            depth_m = depth_value / 1000.0
            
            # Convert to 3D coordinates
            x = (u - self.intrinsics.ppx) * depth_m / self.intrinsics.fx
            y = (v - self.intrinsics.ppy) * depth_m / self.intrinsics.fy
            z = depth_m
            
            return np.array([x, y, z])
            
        except Exception as e:
            self.logger.error(f"3D point conversion failed: {e}")
            return None
    
    def project_3d_to_2d(self, point_3d: np.ndarray) -> Optional[Tuple[int, int]]:
        """Project 3D point to 2D pixel coordinates"""
        try:
            if self.intrinsics is None:
                return None
            
            x, y, z = point_3d
            if z <= 0:
                return None
            
            u = int((x * self.intrinsics.fx / z) + self.intrinsics.ppx)
            v = int((y * self.intrinsics.fy / z) + self.intrinsics.ppy)
            
            return (u, v)
            
        except Exception as e:
            self.logger.error(f"3D to 2D projection failed: {e}")
            return None
    
    def get_enhanced_fps(self) -> float:
        """Get enhanced FPS calculation"""
        if len(self.fps_counter) == 0:
            return 0.0
        return sum(self.fps_counter) / len(self.fps_counter)
    
    def get_enhanced_camera_info(self) -> Dict[str, Any]:
        """Get enhanced camera information"""
        info = {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'frame_count': self.frame_count,
            'current_fps': self.get_enhanced_fps()
        }
        
        if self.intrinsics:
            info.update({
                'fx': self.intrinsics.fx,
                'fy': self.intrinsics.fy,
                'cx': self.intrinsics.ppx,
                'cy': self.intrinsics.ppy,
                'distortion': self.intrinsics.coeffs
            })
        
        if self.camera_matrix is not None:
            info['camera_matrix'] = self.camera_matrix.tolist()
            
        return info
    
    def stop_capture(self):
        """Stop enhanced capture"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=3.0)
        
        if self.pipeline:
            self.pipeline.stop()
        
        self.logger.info("Enhanced camera capture stopped")
    
    def cleanup(self):
        """Enhanced cleanup"""
        self.stop_capture()
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.display_queue.empty():
            try:
                self.display_queue.get_nowait()
            except queue.Empty:
                break
        
        self.logger.info("Enhanced camera cleanup completed")