"""
RealSense D435i Camera Manager
Handles camera initialization, frame capture, and stream management
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import time
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

@dataclass
class FrameData:
    """Container for frame data"""
    color_image: np.ndarray
    depth_image: np.ndarray
    aligned_depth_image: np.ndarray
    infrared_image: np.ndarray
    timestamp: float
    frame_number: int
    imu_data: Optional[Dict[str, Any]] = None

class CameraManager:
    """RealSense D435i Camera Manager"""
    
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.logger = logging.getLogger(__name__)
        
        # Camera parameters
        self.width = width
        self.height = height
        self.fps = fps
        
        # RealSense pipeline and configuration
        self.pipeline = None
        self.config = None
        self.align = None
        self.colorizer = rs.colorizer()
        
        # Stream states
        self.is_streaming = False
        self.frame_data = None
        self.frame_lock = threading.Lock()
        
        # Frame statistics
        self.frame_count = 0
        self.start_time = None
        
        # Initialize camera
        self.initialize_camera()
    
    def initialize_camera(self) -> bool:
        """Initialize RealSense camera"""
        try:
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure streams
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps)
            self.config.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps)
            
            # Enable IMU streams if available
            try:
                self.config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
                self.config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
                self.logger.info("IMU streams enabled")
            except Exception as e:
                self.logger.warning(f"Could not enable IMU streams: {e}")
            
            # Create alignment object (align depth to color)
            self.align = rs.align(rs.stream.color)
            
            self.logger.info(f"Camera initialized: {self.width}x{self.height} at {self.fps} FPS")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def start_streaming(self) -> bool:
        """Start camera streaming"""
        if self.is_streaming:
            self.logger.warning("Camera is already streaming")
            return True
        
        try:
            # Start pipeline
            profile = self.pipeline.start(self.config)
            
            # Get device and enable advanced mode if available
            device = profile.get_device()
            self.logger.info(f"Connected to device: {device.get_info(rs.camera_info.name)}")
            
            # Configure depth sensor settings for better accuracy
            depth_sensor = device.first_depth_sensor()
            if depth_sensor.supports(rs.option.depth_units):
                depth_sensor.set_option(rs.option.depth_units, 0.001)  # 1mm precision
            
            # Set visual presets for better outdoor performance
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, rs.l500_visual_preset.max_range)
            
            # Enable laser emitter for better depth accuracy
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1)
            
            self.is_streaming = True
            self.start_time = time.time()
            self.frame_count = 0
            
            self.logger.info("Camera streaming started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            return False
    
    def stop_streaming(self):
        """Stop camera streaming"""
        if not self.is_streaming:
            return
        
        try:
            self.pipeline.stop()
            self.is_streaming = False
            self.logger.info("Camera streaming stopped")
        except Exception as e:
            self.logger.error(f"Error stopping camera: {e}")
    
    def get_frame(self) -> Optional[FrameData]:
        """Get the latest frame data"""
        if not self.is_streaming:
            return None
        
        try:
            # Wait for frames with timeout
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            
            # Align depth to color
            aligned_frames = self.align.process(frames)
            
            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            infrared_frame = frames.get_infrared_frame(1)
            
            if not color_frame or not depth_frame:
                return None
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            aligned_depth_image = np.asanyarray(aligned_frames.get_depth_frame().get_data())
            infrared_image = np.asanyarray(infrared_frame.get_data()) if infrared_frame else None
            
            # Get IMU data if available
            imu_data = self._get_imu_data(frames)
            
            # Create frame data
            frame_data = FrameData(
                color_image=color_image,
                depth_image=depth_image,
                aligned_depth_image=aligned_depth_image,
                infrared_image=infrared_image,
                timestamp=time.time(),
                frame_number=self.frame_count,
                imu_data=imu_data
            )
            
            # Update frame statistics
            self.frame_count += 1
            
            # Thread-safe update
            with self.frame_lock:
                self.frame_data = frame_data
            
            return frame_data
            
        except Exception as e:
            self.logger.error(f"Error getting frame: {e}")
            return None
    
    def _get_imu_data(self, frames) -> Optional[Dict[str, Any]]:
        """Extract IMU data from frames"""
        try:
            imu_data = {}
            
            # Get accelerometer data
            accel_frame = frames.first_or_default(rs.stream.accel)
            if accel_frame:
                accel_data = accel_frame.as_motion_frame().get_motion_data()
                imu_data['accel'] = np.array([accel_data.x, accel_data.y, accel_data.z])
                imu_data['accel_timestamp'] = accel_frame.get_timestamp()
            
            # Get gyroscope data
            gyro_frame = frames.first_or_default(rs.stream.gyro)
            if gyro_frame:
                gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                imu_data['gyro'] = np.array([gyro_data.x, gyro_data.y, gyro_data.z])
                imu_data['gyro_timestamp'] = gyro_frame.get_timestamp()
            
            return imu_data if imu_data else None
            
        except Exception as e:
            # IMU data is optional, don't log as error
            return None
    
    def get_camera_intrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get camera intrinsic parameters"""
        if not self.is_streaming:
            return None, None
        
        try:
            profile = self.pipeline.get_active_profile()
            
            # Get color stream intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            # Get depth stream intrinsics
            depth_stream = profile.get_stream(rs.stream.depth)
            depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            # Convert to camera matrix format
            color_matrix = np.array([
                [color_intrinsics.fx, 0, color_intrinsics.ppx],
                [0, color_intrinsics.fy, color_intrinsics.ppy],
                [0, 0, 1]
            ])
            
            depth_matrix = np.array([
                [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                [0, 0, 1]
            ])
            
            return color_matrix, depth_matrix
            
        except Exception as e:
            self.logger.error(f"Error getting camera intrinsics: {e}")
            return None, None
    
    def get_depth_scale(self) -> float:
        """Get depth scale factor"""
        if not self.is_streaming:
            return 0.001  # Default 1mm
        
        try:
            profile = self.pipeline.get_active_profile()
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            return depth_sensor.get_depth_scale()
        except Exception as e:
            self.logger.error(f"Error getting depth scale: {e}")
            return 0.001
    
    def get_fps(self) -> float:
        """Calculate actual FPS"""
        if not self.start_time or self.frame_count == 0:
            return 0.0
        
        elapsed_time = time.time() - self.start_time
        return self.frame_count / elapsed_time
    
    def __del__(self):
        """Cleanup resources"""
        self.stop_streaming()