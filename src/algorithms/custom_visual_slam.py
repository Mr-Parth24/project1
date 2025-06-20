"""
Custom Visual SLAM Implementation
Combines feature detection, visual odometry, and mapping
Optimized for Intel RealSense D435i without IMU dependency
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from collections import deque
import threading

# Import core components with relative imports
from ..core.camera_manager import CameraManager
from ..core.feature_detector import FeatureDetector
from ..core.visual_odometry import VisualOdometry
from ..utils.config_manager import get_config_manager
from ..utils.data_logger import get_data_logger

class CustomVisualSLAM:
    """
    Custom Visual SLAM implementation for agricultural applications
    Combines feature detection, visual odometry, and mapping without IMU
    """
    
    def __init__(self):
        """Initialize Custom Visual SLAM system"""
        print("Initializing Custom Visual SLAM...")
        
        # Configuration
        self.config_manager = get_config_manager()
        self.data_logger = get_data_logger()
        
        # Core components
        self.camera_manager = None
        self.visual_odometry = None
        self.feature_detector = None
        
        # SLAM state
        self.is_initialized = False
        self.is_running = False
        self.current_frame = None
        self.frame_count = 0
        
        # Trajectory and mapping
        self.trajectory = []
        self.keyframes = []
        self.map_points = []
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.fps_counter = 0
        self.last_fps_time = time.time()
        
        # Threading
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        print("✅ Custom Visual SLAM initialized")
    
    def initialize(self) -> bool:
        """Initialize SLAM components"""
        try:
            print("Initializing SLAM components...")
            
            # Initialize camera manager
            self.camera_manager = CameraManager()
            if not self.camera_manager.initialize_camera():
                print("❌ Camera initialization failed")
                return False
            
            # Initialize feature detector
            max_features = self.config_manager.get_slam_parameter('max_features', 1000)
            self.feature_detector = FeatureDetector(max_features=max_features)
            
            # Initialize visual odometry
            camera_matrix = self.config_manager.get_camera_matrix()
            dist_coeffs = self.config_manager.get_distortion_coefficients()
            self.visual_odometry = VisualOdometry(camera_matrix, dist_coeffs)
            
            self.is_initialized = True
            print("✅ SLAM components initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ SLAM initialization failed: {e}")
            return False
    
    def start_slam(self) -> bool:
        """Start SLAM processing"""
        try:
            if not self.is_initialized:
                if not self.initialize():
                    return False
            
            # Start camera streaming
            if not self.camera_manager.start_streaming():
                print("❌ Failed to start camera streaming")
                return False
            
            self.is_running = True
            self.stop_event.clear()
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            print("✅ Custom Visual SLAM started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start SLAM: {e}")
            return False
    
    def stop_slam(self):
        """Stop SLAM processing"""
        try:
            self.is_running = False
            self.stop_event.set()
            
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
            
            if self.camera_manager:
                self.camera_manager.stop_streaming()
            
            print("✅ Custom Visual SLAM stopped")
            
        except Exception as e:
            print(f"❌ SLAM stop error: {e}")
    
    def _processing_loop(self):
        """Main SLAM processing loop"""
        try:
            print("SLAM processing loop started")
            
            while self.is_running and not self.stop_event.is_set():
                start_time = time.time()
                
                # Get camera frames
                frames = self.camera_manager.get_frames()
                if frames is None:
                    continue
                
                color_frame, depth_frame = frames
                self.current_frame = color_frame
                self.frame_count += 1
                
                # Process frame
                self._process_frame(color_frame, depth_frame)
                
                # Update performance metrics
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # Calculate FPS
                self._update_fps()
                
                # Sleep to control frame rate
                target_fps = 30
                sleep_time = max(0, (1.0 / target_fps) - processing_time)
                time.sleep(sleep_time)
            
            print("SLAM processing loop ended")
            
        except Exception as e:
            print(f"❌ SLAM processing error: {e}")
    
    def _process_frame(self, color_frame: np.ndarray, depth_frame: np.ndarray):
        """Process a single frame"""
        try:
            # Visual odometry processing
            vo_result = self.visual_odometry.process_frame(color_frame, depth_frame)
            
            # Update trajectory
            if vo_result['pose_estimated']:
                position = vo_result['position']
                self.trajectory.append(position.copy())
                
                # Create keyframe if needed
                if self._should_create_keyframe(vo_result):
                    self._create_keyframe(color_frame, depth_frame, vo_result)
            
        except Exception as e:
            print(f"Frame processing error: {e}")
    
    def _should_create_keyframe(self, vo_result: Dict) -> bool:
        """Determine if a keyframe should be created"""
        try:
            # Simple keyframe creation criteria
            if len(self.keyframes) == 0:
                return True
            
            # Distance-based keyframe creation
            if len(self.trajectory) >= 2:
                last_pos = self.trajectory[-2] if len(self.trajectory) > 1 else np.zeros(3)
                current_pos = self.trajectory[-1]
                distance = np.linalg.norm(current_pos - last_pos)
                
                keyframe_threshold = self.config_manager.get_slam_parameter('keyframe_distance_threshold', 0.3)
                return distance > keyframe_threshold
            
            return False
            
        except Exception as e:
            print(f"Keyframe decision error: {e}")
            return False
    
    def _create_keyframe(self, color_frame: np.ndarray, depth_frame: np.ndarray, vo_result: Dict):
        """Create a new keyframe"""
        try:
            keyframe = {
                'id': len(self.keyframes),
                'timestamp': time.time(),
                'frame_count': self.frame_count,
                'color_frame': color_frame.copy(),
                'depth_frame': depth_frame.copy() if depth_frame is not None else None,
                'position': vo_result['position'].copy(),
                'rotation': vo_result['rotation'].copy(),
                'features': vo_result.get('num_features', 0)
            }
            
            self.keyframes.append(keyframe)
            
        except Exception as e:
            print(f"Keyframe creation error: {e}")
    
    def _update_fps(self):
        """Update FPS calculation"""
        try:
            self.fps_counter += 1
            current_time = time.time()
            
            if current_time - self.last_fps_time >= 1.0:  # Update every second
                self.current_fps = self.fps_counter
                self.fps_counter = 0
                self.last_fps_time = current_time
            
        except Exception as e:
            print(f"FPS update error: {e}")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current camera frame"""
        return self.current_frame
    
    def get_trajectory(self) -> List[np.ndarray]:
        """Get the current trajectory"""
        return self.trajectory.copy()
    
    def get_keyframes(self) -> List[Dict]:
        """Get all keyframes"""
        return self.keyframes.copy()
    
    def get_distance_traveled(self) -> float:
        """Calculate total distance traveled"""
        if len(self.trajectory) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(self.trajectory)):
            distance = np.linalg.norm(self.trajectory[i] - self.trajectory[i-1])
            total_distance += distance
        
        return total_distance
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        try:
            stats = {
                'is_running': self.is_running,
                'frame_count': self.frame_count,
                'trajectory_points': len(self.trajectory),
                'keyframes': len(self.keyframes),
                'distance_traveled': self.get_distance_traveled(),
                'current_fps': getattr(self, 'current_fps', 0),
                'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0
            }
            
            # Add visual odometry stats if available
            if self.visual_odometry:
                vo_stats = self.visual_odometry.get_performance_stats()
                stats.update(vo_stats)
            
            return stats
            
        except Exception as e:
            print(f"Performance stats error: {e}")
            return {}
    
    def reset(self):
        """Reset SLAM state"""
        try:
            # Stop if running
            if self.is_running:
                self.stop_slam()
            
            # Clear data
            self.trajectory.clear()
            self.keyframes.clear()
            self.map_points.clear()
            self.frame_count = 0
            
            # Reset components
            if self.visual_odometry:
                self.visual_odometry.reset()
            
            print("✅ Custom Visual SLAM reset")
            
        except Exception as e:
            print(f"❌ SLAM reset error: {e}")
    
    def save_session(self, filepath: str = None) -> str:
        """Save current SLAM session"""
        try:
            if not self.trajectory:
                print("⚠️  No trajectory data to save")
                return ""
            
            # Prepare session data
            session_data = {
                'trajectory': np.array(self.trajectory),
                'keyframes_count': len(self.keyframes),
                'distance_traveled': self.get_distance_traveled(),
                'frame_count': self.frame_count,
                'performance_stats': self.get_performance_stats()
            }
            
            # Save using data logger
            saved_path = self.data_logger.save_trajectory(
                session_data['trajectory'],
                metadata=session_data
            )
            
            if saved_path:
                print(f"✅ Session saved: {saved_path}")
            else:
                print("❌ Failed to save session")
            
            return saved_path
            
        except Exception as e:
            print(f"❌ Session save error: {e}")
            return ""

# Test function
def test_custom_visual_slam():
    """Test Custom Visual SLAM"""
    slam = CustomVisualSLAM()
    
    if slam.initialize():
        print("✅ Custom Visual SLAM test passed")
        slam.reset()
    else:
        print("❌ Custom Visual SLAM test failed")

if __name__ == "__main__":
    test_custom_visual_slam()