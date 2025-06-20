"""
Intel RealSense D435i Camera Manager
Handles camera initialization, streaming, and frame processing
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import time
from typing import Optional, Tuple, Dict, Any
import yaml
import os

class CameraManager:
    def __init__(self, config_path: str = "config/camera_config.yaml"):
        """Initialize the camera manager with configuration"""
        self.config = self._load_config(config_path)
        self.pipeline = None
        self.config_rs = None
        self.is_streaming = False
        self.latest_frames = None
        self.frame_lock = threading.Lock()
        
        # Frame processing
        self.color_frame = None
        self.depth_frame = None
        self.aligned_depth_frame = None
        
        # Initialize filters
        self._init_filters()
        
        print("Camera Manager initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load camera configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                print(f"Configuration loaded from {config_path}")
                return config
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            print("Using default configuration")
            return self._get_default_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default camera configuration"""
        return {
            'camera': {
                'color_width': 640,
                'color_height': 480,
                'depth_width': 640,
                'depth_height': 480,
                'fps': 30,
                'enable_depth': True,
                'enable_color': True,
                'enable_imu': False,
                'auto_exposure': True,
                'exposure': 100,
                'gain': 50
            }
        }
    
    def _init_filters(self):
        """Initialize RealSense post-processing filters"""
        try:
            # Decimation filter - reduces resolution to improve performance
            self.decimation = rs.decimation_filter()
            
            # Spatial filter - edge-preserving spatial smoothing
            self.spatial = rs.spatial_filter()
            
            # Temporal filter - reduces noise using temporal data
            self.temporal = rs.temporal_filter()
            
            # Hole filling filter - fills holes in depth data
            self.hole_filling = rs.hole_filling_filter()
            
            # Alignment object to align depth to color
            self.align = rs.align(rs.stream.color)
            
            print("Depth filters initialized")
        except Exception as e:
            print(f"Error initializing filters: {e}")
    
    def initialize_camera(self) -> bool:
        """Initialize the RealSense camera"""
        try:
            # Create pipeline and config
            self.pipeline = rs.pipeline()
            self.config_rs = rs.config()
            
            # Configure streams
            cam_config = self.config['camera']
            
            if cam_config['enable_color']:
                self.config_rs.enable_stream(
                    rs.stream.color,
                    cam_config['color_width'],
                    cam_config['color_height'],
                    rs.format.bgr8,
                    cam_config['fps']
                )
                print(f"Color stream configured: {cam_config['color_width']}x{cam_config['color_height']} @ {cam_config['fps']}fps")
            
            if cam_config['enable_depth']:
                self.config_rs.enable_stream(
                    rs.stream.depth,
                    cam_config['depth_width'],
                    cam_config['depth_height'],
                    rs.format.z16,
                    cam_config['fps']
                )
                print(f"Depth stream configured: {cam_config['depth_width']}x{cam_config['depth_height']} @ {cam_config['fps']}fps")
            
            # Start pipeline
            profile = self.pipeline.start(self.config_rs)
            
            # Get device and set options
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            color_sensor = device.first_color_sensor()
            
            # Configure auto exposure
            if cam_config['auto_exposure']:
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                print("Auto exposure enabled")
            else:
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                color_sensor.set_option(rs.option.exposure, cam_config['exposure'])
                color_sensor.set_option(rs.option.gain, cam_config['gain'])
                print(f"Manual exposure set: {cam_config['exposure']}, gain: {cam_config['gain']}")
            
            print("Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize camera: {e}")
            return False
    
    def start_streaming(self) -> bool:
        """Start camera streaming"""
        if not self.pipeline:
            print("Camera not initialized. Call initialize_camera() first.")
            return False
        
        try:
            self.is_streaming = True
            print("Camera streaming started")
            return True
        except Exception as e:
            print(f"Failed to start streaming: {e}")
            return False
    
    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Get the latest color and depth frames"""
        if not self.is_streaming:
            return None
        
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # Align depth to color
            aligned_frames = self.align.process(frames)
            
            # Get color and aligned depth frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None
            
            # Apply filters to depth frame
            depth_frame = self._apply_filters(depth_frame)
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Store frames for other methods
            with self.frame_lock:
                self.color_frame = color_image.copy()
                self.depth_frame = depth_image.copy()
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"Error getting frames: {e}")
            return None
    
    def _apply_filters(self, depth_frame):
        """Apply post-processing filters to depth frame"""
        try:
            if self.config['camera'].get('depth_filters', {}).get('decimation', False):
                depth_frame = self.decimation.process(depth_frame)
            
            if self.config['camera'].get('depth_filters', {}).get('spatial', False):
                depth_frame = self.spatial.process(depth_frame)
            
            if self.config['camera'].get('depth_filters', {}).get('temporal', False):
                depth_frame = self.temporal.process(depth_frame)
            
            if self.config['camera'].get('depth_filters', {}).get('hole_filling', False):
                depth_frame = self.hole_filling.process(depth_frame)
            
            return depth_frame
        except Exception as e:
            print(f"Error applying filters: {e}")
            return depth_frame
    
    def get_color_frame(self) -> Optional[np.ndarray]:
        """Get the latest color frame"""
        with self.frame_lock:
            return self.color_frame.copy() if self.color_frame is not None else None
    
    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Get the latest depth frame"""
        with self.frame_lock:
            return self.depth_frame.copy() if self.depth_frame is not None else None
    
    def stop_streaming(self):
        """Stop camera streaming"""
        self.is_streaming = False
        if self.pipeline:
            try:
                self.pipeline.stop()
                print("Camera streaming stopped")
            except Exception as e:
                print(f"Error stopping pipeline: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_streaming()
        print("Camera Manager destroyed")

# Test function
def test_camera():
    """Test the camera manager"""
    print("Testing Camera Manager...")
    
    camera = CameraManager()
    
    if not camera.initialize_camera():
        print("Failed to initialize camera")
        return
    
    if not camera.start_streaming():
        print("Failed to start streaming")
        return
    
    print("Press 'q' to quit")
    
    try:
        while True:
            frames = camera.get_frames()
            if frames is not None:
                color_image, depth_image = frames
                
                # Create depth colormap for visualization
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # Display images
                cv2.imshow('Color', color_image)
                cv2.imshow('Depth', depth_colormap)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    
    finally:
        camera.stop_streaming()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()