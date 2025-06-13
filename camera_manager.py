"""
Advanced Camera Manager for Intel RealSense D435i
Handles camera initialization, configuration, and frame capture
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import queue
import time
from typing import Optional, Tuple, Dict, Any
import logging
from dataclasses import dataclass

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: np.ndarray

@dataclass
class FrameData:
    color: np.ndarray
    depth: np.ndarray
    timestamp: float
    frame_number: int

class CameraManager:
    """Advanced camera management with error handling and optimization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Camera components
        self.pipeline = None
        self.config_rs = None
        self.profile = None
        self.align = None
        self.intrinsics = None
        
        # Frame processing
        self.frame_queue = queue.Queue(maxsize=5)
        self.capture_thread = None
        self.capturing = False
        
        # Settings
        self.width = config.get('camera', {}).get('width', 640)
        self.height = config.get('camera', {}).get('height', 480)
        self.fps = config.get('camera', {}).get('fps', 30)
        self.enable_auto_exposure = config.get('camera', {}).get('auto_exposure', True)
        
        # Statistics
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_frame_time = 0
        
    def initialize(self) -> bool:
        """Initialize camera with comprehensive error handling"""
        try:
            self.logger.info("Initializing Intel RealSense D435i camera...")
            
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config_rs = rs.config()
            
            # Configure streams
            self.config_rs.enable_stream(
                rs.stream.depth, 
                self.width, 
                self.height, 
                rs.format.z16, 
                self.fps
            )
            self.config_rs.enable_stream(
                rs.stream.color, 
                self.width, 
                self.height, 
                rs.format.bgr8, 
                self.fps
            )
            
            # Start pipeline
            self.profile = self.pipeline.start(self.config_rs)
            
            # Create alignment object
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            # Get camera intrinsics
            self._extract_intrinsics()
            
            # Configure camera settings
            self._configure_camera_settings()
            
            # Start capture thread
            self.capturing = True
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            self.logger.info("Camera initialized successfully")
            self.logger.info(f"Resolution: {self.width}x{self.height} @ {self.fps}fps")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def _extract_intrinsics(self):
        """Extract camera intrinsic parameters"""
        color_profile = self.profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        
        self.intrinsics = CameraIntrinsics(
            fx=intr.fx,
            fy=intr.fy,
            cx=intr.ppx,
            cy=intr.ppy,
            width=intr.width,
            height=intr.height,
            distortion=np.array(intr.coeffs)
        )
        
        self.logger.info(f"Camera intrinsics: fx={intr.fx:.2f}, fy={intr.fy:.2f}")
    
    def _configure_camera_settings(self):
        """Configure advanced camera settings"""
        try:
            device = self.profile.get_device()
            
            # Configure color sensor
            color_sensor = device.first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 
                                      1 if self.enable_auto_exposure else 0)
            
            # Configure depth sensor
            depth_sensor = device.first_depth_sensor()
            if depth_sensor.supports(rs.option.visual_preset):
                depth_sensor.set_option(rs.option.visual_preset, 
                                      rs.l500_visual_preset.short_range)
            
            self.logger.info("Camera settings configured")
            
        except Exception as e:
            self.logger.warning(f"Could not configure camera settings: {e}")
    
    def _capture_loop(self):
        """Continuous frame capture loop"""
        while self.capturing:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                
                # Align frames
                aligned_frames = self.align.process(frames)
                
                # Get color and depth frames
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue
                
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Create frame data
                frame_data = FrameData(
                    color=color_image,
                    depth=depth_image,
                    timestamp=time.time(),
                    frame_number=self.frame_count
                )
                
                # Add to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame_data)
                    self.frame_count += 1
                except queue.Full:
                    self.dropped_frames += 1
                    # Remove oldest frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame_data)
                    except queue.Empty:
                        pass
                
            except Exception as e:
                self.logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
    
    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get latest camera frames"""
        try:
            frame_data = self.frame_queue.get(timeout=1.0)
            return frame_data.color, frame_data.depth
        except queue.Empty:
            return None
    
    def get_intrinsics(self) -> CameraIntrinsics:
        """Get camera intrinsic parameters"""
        return self.intrinsics
    
    def update_calibration(self, calibration_data: Dict[str, Any]):
        """Update camera calibration parameters"""
        if 'intrinsics' in calibration_data:
            intr = calibration_data['intrinsics']
            self.intrinsics = CameraIntrinsics(
                fx=intr['fx'],
                fy=intr['fy'],
                cx=intr['cx'],
                cy=intr['cy'],
                width=intr['width'],
                height=intr['height'],
                distortion=np.array(intr['distortion'])
            )
            self.logger.info("Camera calibration updated")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get camera statistics"""
        return {
            'total_frames': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'drop_rate': self.dropped_frames / max(1, self.frame_count),
            'queue_size': self.frame_queue.qsize()
        }
    
    def cleanup(self):
        """Cleanup camera resources"""
        self.capturing = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        if self.pipeline:
            self.pipeline.stop()
        
        self.logger.info("Camera cleanup completed")