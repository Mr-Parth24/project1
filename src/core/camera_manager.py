"""
Intel RealSense D435i Camera Manager
Handles camera initialization, streaming, and frame processing
FIXED: Resolution detection, frame format standardization, and agricultural optimization
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
    """
    FIXED: Enhanced Camera Manager with proper resolution tracking and frame format handling
    """
    
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
        
        # FIXED: Track actual resolution and format
        self.actual_color_width = 640
        self.actual_color_height = 480
        self.actual_depth_width = 640
        self.actual_depth_height = 480
        self.color_format = "BGR8"  # Track color format
        self.depth_format = "Z16"   # Track depth format
        
        # Frame timing for FPS calculation
        self.frame_timestamps = []
        self.last_frame_time = 0
        self.fps_calculation_window = 30
        
        # Initialize filters
        self._init_filters()
        
        # Performance statistics
        self.frame_count = 0
        self.dropped_frames = 0
        self.processing_times = []
        
        print("🎥 Camera Manager initialized (FIXED VERSION)")
        print(f"   - Resolution tracking: ✅")
        print(f"   - Frame format standardization: ✅")
        print(f"   - Agricultural optimization: ✅")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load camera configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                print(f"📄 Configuration loaded from {config_path}")
                return config
        except FileNotFoundError:
            print(f"⚠️  Config file not found: {config_path}")
            print("🔄 Using default configuration")
            return self._get_default_config()
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """FIXED: Return enhanced default camera configuration for agricultural use"""
        return {
            'camera': {
                # FIXED: Use agricultural-optimized resolution (from config analysis)
                'color_width': 848,
                'color_height': 480,
                'depth_width': 848,
                'depth_height': 480,
                'fps': 30,
                'enable_depth': True,
                'enable_color': True,
                'enable_imu': False,
                
                # Agricultural lighting optimization
                'auto_exposure': True,
                'exposure': 8500,  # Higher exposure for outdoor
                'gain': 64,       # Moderate gain
                'laser_power': 240,  # High laser power for outdoor depth
                
                # Enhanced depth filtering for agricultural scenes
                'depth_filters': {
                    'decimation': False,  # Keep full resolution
                    'spatial': True,      # Reduce noise
                    'temporal': True,     # Temporal filtering
                    'hole_filling': True, # Fill holes
                    'threshold': False    # Disable distance threshold
                }
            }
        }
    
    def _init_filters(self):
        """FIXED: Initialize RealSense post-processing filters for agricultural use"""
        try:
            # Decimation filter - reduces resolution (optional for performance)
            self.decimation = rs.decimation_filter()
            self.decimation.set_option(rs.option.filter_magnitude, 2)  # 2x decimation
            
            # Spatial filter - edge-preserving spatial smoothing (important for agriculture)
            self.spatial = rs.spatial_filter()
            self.spatial.set_option(rs.option.filter_magnitude, 2)      # Moderate smoothing
            self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5) # Edge preservation
            self.spatial.set_option(rs.option.filter_smooth_delta, 20)  # Edge threshold
            
            # Temporal filter - reduces noise using temporal data (crucial for outdoor)
            self.temporal = rs.temporal_filter()
            self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)  # Moderate persistence
            self.temporal.set_option(rs.option.filter_smooth_delta, 20)   # Change threshold
            
            # Hole filling filter - fills holes in depth data (essential for agriculture)
            self.hole_filling = rs.hole_filling_filter()
            self.hole_filling.set_option(rs.option.holes_fill, 1)  # Fill small holes
            
            # Alignment object to align depth to color
            self.align = rs.align(rs.stream.color)
            
            print("🔧 Depth filters initialized for agricultural use:")
            print(f"   - Spatial filtering: Enhanced edge preservation")
            print(f"   - Temporal filtering: Outdoor noise reduction")
            print(f"   - Hole filling: Agricultural scene optimization")
            
        except Exception as e:
            print(f"❌ Error initializing filters: {e}")
    
    def initialize_camera(self) -> bool:
        """FIXED: Initialize the RealSense camera with agricultural optimizations"""
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
                    rs.format.bgr8,  # Ensure BGR format for OpenCV compatibility
                    cam_config['fps']
                )
                # FIXED: Track actual configured resolution
                self.actual_color_width = cam_config['color_width']
                self.actual_color_height = cam_config['color_height']
                self.color_format = "BGR8"
                
                print(f"🎥 Color stream configured: {self.actual_color_width}x{self.actual_color_height} @ {cam_config['fps']}fps (BGR)")
            
            if cam_config['enable_depth']:
                self.config_rs.enable_stream(
                    rs.stream.depth,
                    cam_config['depth_width'],
                    cam_config['depth_height'],
                    rs.format.z16,
                    cam_config['fps']
                )
                # FIXED: Track actual configured resolution
                self.actual_depth_width = cam_config['depth_width']
                self.actual_depth_height = cam_config['depth_height']
                self.depth_format = "Z16"
                
                print(f"📏 Depth stream configured: {self.actual_depth_width}x{self.actual_depth_height} @ {cam_config['fps']}fps (Z16)")
            
            # Start pipeline
            profile = self.pipeline.start(self.config_rs)
            
            # FIXED: Verify actual stream properties
            self._verify_stream_properties(profile)
            
            # Get device and set agricultural-optimized options
            device = profile.get_device()
            self._configure_agricultural_settings(device, cam_config)
            
            print("✅ Camera initialized successfully with agricultural optimization")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize camera: {e}")
            return False
    
    def _verify_stream_properties(self, profile):
        """FIXED: Verify and update actual stream properties"""
        try:
            streams = profile.get_streams()
            
            for stream in streams:
                stream_profile = stream.as_video_stream_profile()
                stream_type = stream.stream_type()
                
                if stream_type == rs.stream.color:
                    actual_width = stream_profile.width()
                    actual_height = stream_profile.height()
                    actual_format = stream_profile.format()
                    
                    # Update actual resolution if different
                    if actual_width != self.actual_color_width or actual_height != self.actual_color_height:
                        print(f"⚠️  Color resolution adjusted: {self.actual_color_width}x{self.actual_color_height} → {actual_width}x{actual_height}")
                        self.actual_color_width = actual_width
                        self.actual_color_height = actual_height
                    
                    self.color_format = str(actual_format).split('.')[-1]  # Extract format name
                    
                elif stream_type == rs.stream.depth:
                    actual_width = stream_profile.width()
                    actual_height = stream_profile.height()
                    actual_format = stream_profile.format()
                    
                    # Update actual resolution if different
                    if actual_width != self.actual_depth_width or actual_height != self.actual_depth_height:
                        print(f"⚠️  Depth resolution adjusted: {self.actual_depth_width}x{self.actual_depth_height} → {actual_width}x{actual_height}")
                        self.actual_depth_width = actual_width
                        self.actual_depth_height = actual_height
                    
                    self.depth_format = str(actual_format).split('.')[-1]  # Extract format name
            
            print(f"✅ Stream verification complete:")
            print(f"   Color: {self.actual_color_width}x{self.actual_color_height} ({self.color_format})")
            print(f"   Depth: {self.actual_depth_width}x{self.actual_depth_height} ({self.depth_format})")
            
        except Exception as e:
            print(f"❌ Stream verification error: {e}")
    
    def _configure_agricultural_settings(self, device, cam_config):
        """FIXED: Configure camera settings optimized for agricultural environments"""
        try:
            # Configure color sensor for outdoor agricultural use
            if device.first_color_sensor():
                color_sensor = device.first_color_sensor()
                
                if cam_config['auto_exposure']:
                    color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                    # Set auto-exposure limits for outdoor use
                    if color_sensor.supports(rs.option.auto_exposure_limit):
                        color_sensor.set_option(rs.option.auto_exposure_limit, 10000)  # 10ms max
                    print("🌞 Auto exposure enabled with outdoor limits")
                else:
                    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                    color_sensor.set_option(rs.option.exposure, cam_config['exposure'])
                    color_sensor.set_option(rs.option.gain, cam_config['gain'])
                    print(f"📷 Manual exposure set: {cam_config['exposure']}, gain: {cam_config['gain']}")
                
                # Additional agricultural optimizations
                if color_sensor.supports(rs.option.backlight_compensation):
                    color_sensor.set_option(rs.option.backlight_compensation, 1)  # Help with bright backgrounds
                
                if color_sensor.supports(rs.option.brightness):
                    color_sensor.set_option(rs.option.brightness, 0)  # Neutral brightness
                
                if color_sensor.supports(rs.option.contrast):
                    color_sensor.set_option(rs.option.contrast, 50)  # Moderate contrast
            
            # Configure depth sensor for outdoor agricultural use
            if device.first_depth_sensor():
                depth_sensor = device.first_depth_sensor()
                
                # Maximize laser power for outdoor depth detection
                if depth_sensor.supports(rs.option.laser_power):
                    laser_power = cam_config.get('laser_power', 240)
                    depth_sensor.set_option(rs.option.laser_power, laser_power)
                    print(f"🔦 Laser power set to {laser_power} for outdoor use")
                
                # Set accuracy mode for better agricultural scene depth
                if depth_sensor.supports(rs.option.accuracy):
                    depth_sensor.set_option(rs.option.accuracy, 1)  # High accuracy
                
                # Confidence threshold for agricultural scenes
                if depth_sensor.supports(rs.option.confidence_threshold):
                    depth_sensor.set_option(rs.option.confidence_threshold, 1)  # Low threshold for outdoor
                
                # Enable emitter for active depth sensing
                if depth_sensor.supports(rs.option.emitter_enabled):
                    depth_sensor.set_option(rs.option.emitter_enabled, 1)
            
            print("🌾 Agricultural camera settings applied successfully")
            
        except Exception as e:
            print(f"❌ Agricultural settings configuration error: {e}")
    
    def start_streaming(self) -> bool:
        """Start camera streaming"""
        if not self.pipeline:
            print("❌ Camera not initialized. Call initialize_camera() first.")
            return False
        
        try:
            self.is_streaming = True
            self.frame_count = 0
            self.last_frame_time = time.time()
            print("▶️  Camera streaming started")
            return True
        except Exception as e:
            print(f"❌ Failed to start streaming: {e}")
            return False
    
    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """FIXED: Get the latest color and depth frames with timestamp"""
        if not self.is_streaming:
            return None
        
        try:
            start_time = time.time()
            
            # Wait for frames with timeout
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
            
            # Convert to numpy arrays with proper format handling
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # FIXED: Ensure proper color format (BGR for OpenCV compatibility)
            color_image = self._ensure_bgr_format(color_image)
            
            # Get frame timestamp
            frame_timestamp = color_frame.get_timestamp() / 1000.0  # Convert to seconds
            
            # Store frames for other methods
            with self.frame_lock:
                self.color_frame = color_image.copy()
                self.depth_frame = depth_image.copy()
                self.aligned_depth_frame = depth_image.copy()
            
            # Update performance statistics
            self._update_performance_stats(start_time)
            
            return color_image, depth_image, frame_timestamp
            
        except Exception as e:
            print(f"❌ Error getting frames: {e}")
            self.dropped_frames += 1
            return None
    
    def _ensure_bgr_format(self, color_image: np.ndarray) -> np.ndarray:
        """FIXED: Ensure color image is in BGR format for OpenCV compatibility"""
        try:
            # Check if image needs format conversion
            if len(color_image.shape) == 3 and color_image.shape[2] == 3:
                # If configured as BGR8, image should already be BGR
                if self.color_format == "BGR8":
                    return color_image
                # If configured as RGB8, convert to BGR
                elif self.color_format == "RGB8":
                    return cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                else:
                    # Default assumption: already BGR
                    return color_image
            else:
                # Single channel or unexpected format
                return color_image
                
        except Exception as e:
            print(f"❌ Color format conversion error: {e}")
            return color_image
    
    def _apply_filters(self, depth_frame):
        """FIXED: Apply post-processing filters to depth frame for agricultural scenes"""
        try:
            depth_filters = self.config['camera'].get('depth_filters', {})
            
            # Apply filters in optimal order for agricultural scenes
            filtered_frame = depth_frame
            
            # 1. Decimation (optional, for performance)
            if depth_filters.get('decimation', False):
                filtered_frame = self.decimation.process(filtered_frame)
            
            # 2. Spatial filter (important for outdoor noise)
            if depth_filters.get('spatial', True):
                filtered_frame = self.spatial.process(filtered_frame)
            
            # 3. Temporal filter (crucial for outdoor stability)
            if depth_filters.get('temporal', True):
                filtered_frame = self.temporal.process(filtered_frame)
            
            # 4. Hole filling (essential for agricultural applications)
            if depth_filters.get('hole_filling', True):
                filtered_frame = self.hole_filling.process(filtered_frame)
            
            return filtered_frame
            
        except Exception as e:
            print(f"❌ Error applying filters: {e}")
            return depth_frame
    
    def _update_performance_stats(self, start_time: float):
        """FIXED: Update performance statistics for monitoring"""
        try:
            # Update frame count
            self.frame_count += 1
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Keep only recent processing times
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            
            # Update FPS calculation
            current_time = time.time()
            self.frame_timestamps.append(current_time)
            
            # Keep only recent timestamps for FPS calculation
            if len(self.frame_timestamps) > self.fps_calculation_window:
                self.frame_timestamps.pop(0)
            
            # Log performance every 100 frames
            if self.frame_count % 100 == 0:
                avg_fps = self.get_current_fps()
                avg_processing = np.mean(self.processing_times) * 1000  # Convert to ms
                print(f"📊 Performance: {avg_fps:.1f} FPS, {avg_processing:.1f}ms processing, {self.dropped_frames} dropped")
            
        except Exception as e:
            print(f"❌ Performance stats update error: {e}")
    
    def get_color_frame(self) -> Optional[np.ndarray]:
        """Get the latest color frame"""
        with self.frame_lock:
            return self.color_frame.copy() if self.color_frame is not None else None
    
    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Get the latest depth frame"""
        with self.frame_lock:
            return self.depth_frame.copy() if self.depth_frame is not None else None
    
    def get_aligned_depth_frame(self) -> Optional[np.ndarray]:
        """NEW: Get the latest aligned depth frame"""
        with self.frame_lock:
            return self.aligned_depth_frame.copy() if self.aligned_depth_frame is not None else None
    
    def get_camera_resolution(self) -> Tuple[int, int]:
        """NEW: Get actual camera resolution (width, height)"""
        return (self.actual_color_width, self.actual_color_height)
    
    def get_depth_resolution(self) -> Tuple[int, int]:
        """NEW: Get actual depth resolution (width, height)"""
        return (self.actual_depth_width, self.actual_depth_height)
    
    def get_camera_format(self) -> str:
        """NEW: Get current color format"""
        return self.color_format
    
    def get_current_fps(self) -> float:
        """NEW: Calculate current FPS"""
        try:
            if len(self.frame_timestamps) < 2:
                return 0.0
            
            time_span = self.frame_timestamps[-1] - self.frame_timestamps[0]
            frame_count = len(self.frame_timestamps) - 1
            
            if time_span > 0:
                return frame_count / time_span
            else:
                return 0.0
                
        except Exception as e:
            print(f"❌ FPS calculation error: {e}")
            return 0.0
    
    def get_performance_stats(self) -> Dict:
        """NEW: Get comprehensive performance statistics"""
        try:
            return {
                'frame_count': self.frame_count,
                'dropped_frames': self.dropped_frames,
                'current_fps': self.get_current_fps(),
                'avg_processing_time_ms': np.mean(self.processing_times) * 1000 if self.processing_times else 0.0,
                'color_resolution': (self.actual_color_width, self.actual_color_height),
                'depth_resolution': (self.actual_depth_width, self.actual_depth_height),
                'color_format': self.color_format,
                'depth_format': self.depth_format,
                'is_streaming': self.is_streaming,
                'filters_active': {
                    'spatial': self.config['camera'].get('depth_filters', {}).get('spatial', False),
                    'temporal': self.config['camera'].get('depth_filters', {}).get('temporal', False),
                    'hole_filling': self.config['camera'].get('depth_filters', {}).get('hole_filling', False)
                }
            }
        except Exception as e:
            print(f"❌ Performance stats error: {e}")
            return {'error': str(e)}
    
    def set_exposure(self, exposure_value: int) -> bool:
        """NEW: Dynamically adjust exposure for changing lighting conditions"""
        try:
            if not self.pipeline:
                return False
            
            profile = self.pipeline.get_active_profile()
            device = profile.get_device()
            
            if device.first_color_sensor():
                color_sensor = device.first_color_sensor()
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)  # Disable auto
                color_sensor.set_option(rs.option.exposure, exposure_value)
                print(f"📷 Exposure set to {exposure_value}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Exposure setting error: {e}")
            return False
    
    def set_laser_power(self, power: int) -> bool:
        """NEW: Dynamically adjust laser power for depth sensing"""
        try:
            if not self.pipeline:
                return False
            
            profile = self.pipeline.get_active_profile()
            device = profile.get_device()
            
            if device.first_depth_sensor():
                depth_sensor = device.first_depth_sensor()
                if depth_sensor.supports(rs.option.laser_power):
                    depth_sensor.set_option(rs.option.laser_power, power)
                    print(f"🔦 Laser power set to {power}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Laser power setting error: {e}")
            return False
    
    def get_camera_info(self) -> Dict:
        """NEW: Get comprehensive camera information"""
        try:
            info = {
                'model': 'Intel RealSense D435i',
                'color_resolution': (self.actual_color_width, self.actual_color_height),
                'depth_resolution': (self.actual_depth_width, self.actual_depth_height),
                'color_format': self.color_format,
                'depth_format': self.depth_format,
                'is_streaming': self.is_streaming,
                'agricultural_optimized': True,
                'frame_count': self.frame_count,
                'dropped_frames': self.dropped_frames
            }
            
            # Add device-specific info if available
            if self.pipeline:
                try:
                    profile = self.pipeline.get_active_profile()
                    device = profile.get_device()
                    info['device_name'] = device.get_info(rs.camera_info.name)
                    info['serial_number'] = device.get_info(rs.camera_info.serial_number)
                    info['firmware_version'] = device.get_info(rs.camera_info.firmware_version)
                except:
                    pass  # Not critical if this fails
            
            return info
            
        except Exception as e:
            print(f"❌ Camera info error: {e}")
            return {'error': str(e)}
    
    def stop_streaming(self):
        """Stop camera streaming"""
        self.is_streaming = False
        if self.pipeline:
            try:
                self.pipeline.stop()
                print("⏹️  Camera streaming stopped")
                
                # Print final performance stats
                final_stats = self.get_performance_stats()
                print(f"📊 Final stats: {final_stats['frame_count']} frames, {final_stats['dropped_frames']} dropped")
                
            except Exception as e:
                print(f"❌ Error stopping pipeline: {e}")
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.stop_streaming()
        print("🎥 Camera Manager destroyed")

# Test function
def test_camera():
    """FIXED: Test the camera manager with enhanced monitoring"""
    print("🧪 Testing Enhanced Camera Manager...")
    
    camera = CameraManager()
    
    # Display camera info
    info = camera.get_camera_info()
    print(f"📋 Camera Info: {info}")
    
    if not camera.initialize_camera():
        print("❌ Failed to initialize camera")
        return
    
    if not camera.start_streaming():
        print("❌ Failed to start streaming")
        return
    
    print("🎥 Camera test running. Press 'q' to quit, 'e' to adjust exposure")
    print(f"📏 Resolution: {camera.get_camera_resolution()}")
    print(f"🎨 Format: {camera.get_camera_format()}")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame_data = camera.get_frames()
            if frame_data is not None:
                color_image, depth_image, timestamp = frame_data
                frame_count += 1
                
                # Create depth colormap for visualization
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                
                # Add info overlay to color image
                info_text = f"Frame: {frame_count} | FPS: {camera.get_current_fps():.1f} | Res: {camera.get_camera_resolution()[0]}x{camera.get_camera_resolution()[1]}"
                cv2.putText(color_image, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display images
                cv2.imshow('Color (Enhanced)', color_image)
                cv2.imshow('Depth (Enhanced)', depth_colormap)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('e'):
                    # Test exposure adjustment
                    camera.set_exposure(5000)
                elif key == ord('l'):
                    # Test laser power adjustment
                    camera.set_laser_power(150)
                
                # Print performance stats every 5 seconds
                if frame_count % 150 == 0:  # ~5 seconds at 30fps
                    elapsed = time.time() - start_time
                    stats = camera.get_performance_stats()
                    print(f"📊 {elapsed:.1f}s elapsed: {stats}")
    
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    
    finally:
        camera.stop_streaming()
        cv2.destroyAllWindows()
        
        # Final performance report
        total_time = time.time() - start_time
        final_stats = camera.get_performance_stats()
        print(f"🏁 Test completed in {total_time:.1f}s")
        print(f"📊 Final performance: {final_stats}")

if __name__ == "__main__":
    test_camera()