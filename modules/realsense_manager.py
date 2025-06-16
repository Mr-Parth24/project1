"""
RealSense Camera Manager
Handles Intel RealSense D435i camera operations
"""

import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseManager:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.width = width
        self.height = height
        self.fps = fps
        
        # Camera intrinsics (will be updated when camera starts)
        self.intrinsics = None
        
    def start(self):
        """Start the RealSense camera"""
        try:
            # Configure streams
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # Start streaming
            profile = self.pipeline.start(self.config)
            
            # Get camera intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            # Create align object to align depth to color
            self.align = rs.align(rs.stream.color)
            
            print(f"✅ RealSense camera started ({self.width}x{self.height} @ {self.fps}fps)")
            print(f"📷 Camera intrinsics: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start RealSense camera: {e}")
            print("💡 Make sure your Intel RealSense D435i is connected")
            return False
            
    def get_frames(self):
        """Get aligned color and depth frames"""
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames()
            
            # Align depth to color
            aligned_frames = self.align.process(frames)
            
            # Get frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                return None, None
                
            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            return color_image, depth_image
            
        except Exception as e:
            print(f"❌ Error getting frames: {e}")
            return None, None
            
    def get_intrinsics(self):
        """Get camera intrinsics"""
        if self.intrinsics:
            return {
                'fx': self.intrinsics.fx,
                'fy': self.intrinsics.fy,
                'cx': self.intrinsics.ppx,
                'cy': self.intrinsics.ppy,
                'width': self.intrinsics.width,
                'height': self.intrinsics.height
            }
        return None
        
    def stop(self):
        """Stop the camera"""
        try:
            self.pipeline.stop()
            print("📷 RealSense camera stopped")
        except:
            pass