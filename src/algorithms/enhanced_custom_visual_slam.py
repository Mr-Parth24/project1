"""
Streamlined Enhanced Custom Visual SLAM
Optimized integration with agricultural core components
Focus on reliability and real-time performance
"""

import numpy as np
import cv2
import time
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from src.core.agri_slam_core import AgriSLAMCore
from ..core.precision_distance_tracker import PrecisionDistanceTracker
from ..core.camera_manager import CameraManager
from ..utils.data_logger import get_data_logger

@dataclass
class StreamlinedSLAMResults:
    """Streamlined SLAM results for better performance"""
    timestamp: float
    pose_estimated: bool
    position: np.ndarray
    rotation: np.ndarray
    total_distance: float
    precision_distance: float
    tracking_quality: float
    num_features: int
    num_keyframes: int
    slam_mode: str
    processing_time: float
    agricultural_context: Dict

class StreamlinedEnhancedSLAM:
    """
    Streamlined Enhanced SLAM for Agricultural Applications
    Optimized for real-time performance and reliability
    """
    
    def __init__(self, camera_manager: CameraManager):
        """Initialize streamlined enhanced SLAM"""
        self.camera_manager = camera_manager
        self.data_logger = get_data_logger()
        
        # Core components - simplified initialization
        self.slam_core = AgriSLAMCore(camera_manager)
        self.distance_tracker = PrecisionDistanceTracker()
        
        # Streamlined state tracking
        self.current_results = None
        self.is_initialized = False
        self.processing_active = False
        
        # Performance optimization
        self.frame_skip_counter = 0
        self.frame_skip_threshold = 1  # Process every frame initially
        self.performance_mode = "BALANCED"  # FAST, BALANCED, ACCURATE
        
        # Thread safety - simplified
        self.results_lock = threading.Lock()
        
        # Agricultural tracking - essential only
        self.agricultural_stats = {
            'crop_rows_detected': False,
            'ground_plane_valid': False,
            'field_coverage_area': 0.0,
            'total_trajectory_points': 0
        }
        
        # Performance monitoring - lightweight
        self.perf_monitor = {
            'avg_fps': 0.0,
            'processing_times': deque(maxlen=30),
            'tracking_success_count': 0,
            'total_frame_count': 0
        }
        
        print("Streamlined Enhanced SLAM initialized:")
        print(f"  - Performance mode: {self.performance_mode}")
        print(f"  - Frame processing: optimized")
        print(f"  - Agricultural features: essential only")
    
    def set_performance_mode(self, mode: str):
        """
        Set performance mode for different use cases
        
        Args:
            mode: 'FAST', 'BALANCED', or 'ACCURATE'
        """
        self.performance_mode = mode
        
        if mode == "FAST":
            self.frame_skip_threshold = 2  # Process every 2nd frame
            print("Performance mode: FAST (30+ FPS target)")
        elif mode == "BALANCED":
            self.frame_skip_threshold = 1  # Process every frame
            print("Performance mode: BALANCED (20-30 FPS target)")
        elif mode == "ACCURATE":
            self.frame_skip_threshold = 1  # Process every frame with full validation
            print("Performance mode: ACCURATE (15+ FPS target)")
    
    def process_frame_streamlined(self, color_frame: np.ndarray, 
                                depth_frame: np.ndarray, 
                                timestamp: float = None) -> StreamlinedSLAMResults:
        """
        Streamlined frame processing for optimal performance
        
        Args:
            color_frame: RGB color image
            depth_frame: Depth image  
            timestamp: Frame timestamp
            
        Returns:
            StreamlinedSLAMResults object
        """
        if timestamp is None:
            timestamp = time.time()
        
        processing_start = time.time()
        
        # Frame skipping for performance
        self.frame_skip_counter += 1
        if self.frame_skip_counter < self.frame_skip_threshold:
            # Return previous results with updated timestamp
            if self.current_results:
                return StreamlinedSLAMResults(
                    timestamp=timestamp,
                    pose_estimated=False,
                    position=self.current_results.position.copy(),
                    rotation=self.current_results.rotation.copy(),
                    total_distance=self.current_results.total_distance,
                    precision_distance=self.current_results.precision_distance,
                    tracking_quality=self.current_results.tracking_quality,
                    num_features=self.current_results.num_features,
                    num_keyframes=self.current_results.num_keyframes,
                    slam_mode="SKIPPED",
                    processing_time=0.001,
                    agricultural_context=self.current_results.agricultural_context.copy()
                )
        
        self.frame_skip_counter = 0
        self.perf_monitor['total_frame_count'] += 1
        
        try:
            with self.results_lock:
                # Process with SLAM core
                slam_results = self.slam_core.process_frame(color_frame, depth_frame, timestamp)
                
                # Process distance with validation
                distance_results = self._process_distance_streamlined(slam_results, timestamp)
                
                # Update agricultural context
                agricultural_context = self._update_agricultural_context(slam_results)
                
                # Create streamlined results
                results = StreamlinedSLAMResults(
                    timestamp=timestamp,
                    pose_estimated=slam_results['pose_estimated'],
                    position=slam_results['position'].copy(),
                    rotation=slam_results['rotation'].copy(),
                    total_distance=slam_results['total_distance'],
                    precision_distance=distance_results['precision_distance'],
                    tracking_quality=slam_results.get('tracking_quality', 0.0),
                    num_features=slam_results['num_features'],
                    num_keyframes=slam_results['num_keyframes'],
                    slam_mode=slam_results['slam_mode'],
                    processing_time=time.time() - processing_start,
                    agricultural_context=agricultural_context
                )
                
                # Update performance monitoring
                self._update_performance_monitor(results)
                
                # Store current results
                self.current_results = results
                
                # Update initialization status
                if not self.is_initialized and slam_results['slam_mode'] == "TRACKING":
                    self.is_initialized = True
                    print("🌾 Streamlined SLAM initialized and tracking")
                
                return results
                
        except Exception as e:
            print(f"Streamlined SLAM processing error: {e}")
            # Return safe default results
            return self._get_default_results(timestamp, processing_start)
    
    def _process_distance_streamlined(self, slam_results: Dict, timestamp: float) -> Dict:
        """Streamlined distance processing with essential validation only"""
        try:
            if not slam_results['pose_estimated']:
                return {'precision_distance': self.distance_tracker.get_total_distance()}
            
            # Get movement data
            trajectory = self.slam_core.get_trajectory_3d()
            if len(trajectory) >= 2:
                translation = trajectory[-1] - trajectory[-2]
                rotation = slam_results['rotation']
                
                # Process with distance tracker (simplified validation)
                distance_results = self.distance_tracker.process_movement(
                    translation=translation,
                    rotation=rotation,
                    timestamp=timestamp
                )
                
                return {
                    'precision_distance': distance_results['cumulative_distance'],
                    'distance_confidence': distance_results.get('confidence', 0.5),
                    'validation_passed': distance_results['validation_passed']
                }
            
            return {'precision_distance': self.distance_tracker.get_total_distance()}
            
        except Exception as e:
            print(f"Distance processing error: {e}")
            return {'precision_distance': self.distance_tracker.get_total_distance()}
    
    def _update_agricultural_context(self, slam_results: Dict) -> Dict:
        """Update agricultural context with essential information only"""
        try:
            scene_info = slam_results.get('scene_info', {})
            
            # Update essential agricultural stats
            if scene_info.get('crop_rows_detected', False):
                self.agricultural_stats['crop_rows_detected'] = True
            
            if scene_info.get('ground_plane') is not None:
                self.agricultural_stats['ground_plane_valid'] = True
            
            # Update trajectory point count
            trajectory = self.slam_core.get_trajectory_3d()
            self.agricultural_stats['total_trajectory_points'] = len(trajectory)
            
            # Calculate approximate field coverage (simplified)
            if len(trajectory) > 10:
                # Bounding box area calculation
                positions = trajectory[-100:]  # Recent 100 points
                x_coords = positions[:, 0]
                z_coords = positions[:, 2]
                
                x_range = np.max(x_coords) - np.min(x_coords)
                z_range = np.max(z_coords) - np.min(z_coords)
                
                self.agricultural_stats['field_coverage_area'] = x_range * z_range
            
            return {
                'scene_type': scene_info.get('scene_type', 'field'),
                'crop_rows_detected': self.agricultural_stats['crop_rows_detected'],
                'ground_plane_valid': self.agricultural_stats['ground_plane_valid'],
                'field_coverage_m2': self.agricultural_stats['field_coverage_area'],
                'trajectory_points': self.agricultural_stats['total_trajectory_points']
            }
            
        except Exception as e:
            print(f"Agricultural context update error: {e}")
            return {'scene_type': 'field', 'crop_rows_detected': False}
    
    def _update_performance_monitor(self, results: StreamlinedSLAMResults):
        """Update lightweight performance monitoring"""
        try:
            # Track processing times
            self.perf_monitor['processing_times'].append(results.processing_time)
            
            # Update FPS calculation
            if len(self.perf_monitor['processing_times']) >= 10:
                avg_time = np.mean(list(self.perf_monitor['processing_times']))
                self.perf_monitor['avg_fps'] = 1.0 / max(avg_time, 0.001)
            
            # Track success rate
            if results.pose_estimated:
                self.perf_monitor['tracking_success_count'] += 1
            
            # Adaptive performance adjustment
            if self.perf_monitor['avg_fps'] < 15.0 and self.performance_mode == "BALANCED":
                self.frame_skip_threshold = min(3, self.frame_skip_threshold + 1)
                print(f"⚡ Adaptive performance: increased frame skip to {self.frame_skip_threshold}")
            elif self.perf_monitor['avg_fps'] > 25.0 and self.frame_skip_threshold > 1:
                self.frame_skip_threshold = max(1, self.frame_skip_threshold - 1)
                print(f"⚡ Adaptive performance: reduced frame skip to {self.frame_skip_threshold}")
                
        except Exception as e:
            print(f"Performance monitor update error: {e}")
    
    def _get_default_results(self, timestamp: float, processing_start: float) -> StreamlinedSLAMResults:
        """Get safe default results in case of errors"""
        return StreamlinedSLAMResults(
            timestamp=timestamp,
            pose_estimated=False,
            position=np.array([0.0, 0.0, 0.0]),
            rotation=np.eye(3),
            total_distance=0.0,
            precision_distance=self.distance_tracker.get_total_distance(),
            tracking_quality=0.0,
            num_features=0,
            num_keyframes=0,
            slam_mode="ERROR",
            processing_time=time.time() - processing_start,
            agricultural_context={'scene_type': 'unknown', 'crop_rows_detected': False}
        )
    
    def get_current_trajectory_3d(self) -> np.ndarray:
        """Get current 3D trajectory"""
        return self.slam_core.get_trajectory_3d()
    
    def get_distance_comparison(self) -> Dict:
        """Get comparison between SLAM and precision distance measurements"""
        slam_distance = self.slam_core.get_total_distance()
        precision_distance = self.distance_tracker.get_total_distance()
        
        return {
            'slam_distance': slam_distance,
            'precision_distance': precision_distance,
            'difference': abs(slam_distance - precision_distance),
            'difference_percentage': abs(slam_distance - precision_distance) / max(slam_distance, 0.001) * 100,
            'precision_accuracy': self.distance_tracker.get_performance_stats().get('estimated_accuracy', 0.05)
        }
    
    def get_agricultural_summary(self) -> Dict:
        """Get agricultural mapping summary"""
        with self.results_lock:
            trajectory = self.get_current_trajectory_3d()
            distance_comp = self.get_distance_comparison()
            
            return {
                'session_summary': {
                    'total_trajectory_points': len(trajectory),
                    'slam_distance_m': distance_comp['slam_distance'],
                    'precision_distance_m': distance_comp['precision_distance'],
                    'accuracy_cm': distance_comp['precision_accuracy'] * 100,
                    'field_coverage_m2': self.agricultural_stats['field_coverage_area']
                },
                'agricultural_features': {
                    'crop_rows_detected': self.agricultural_stats['crop_rows_detected'],
                    'ground_plane_estimated': self.agricultural_stats['ground_plane_valid'],
                    'scene_understanding': 'active'
                },
                'performance': {
                    'avg_fps': self.perf_monitor['avg_fps'],
                    'tracking_success_rate': (self.perf_monitor['tracking_success_count'] / 
                                            max(self.perf_monitor['total_frame_count'], 1)) * 100,
                    'performance_mode': self.performance_mode,
                    'frame_skip_threshold': self.frame_skip_threshold
                }
            }
    
    def get_realtime_stats(self) -> Dict:
        """Get real-time statistics for display"""
        if not self.current_results:
            return {'status': 'not_started'}
        
        return {
            'status': 'active',
            'slam_mode': self.current_results.slam_mode,
            'tracking_quality': self.current_results.tracking_quality,
            'position': {
                'x': float(self.current_results.position[0]),
                'y': float(self.current_results.position[1]), 
                'z': float(self.current_results.position[2])
            },
            'distances': {
                'total': self.current_results.total_distance,
                'precision': self.current_results.precision_distance
            },
            'features': self.current_results.num_features,
            'keyframes': self.current_results.num_keyframes,
            'performance': {
                'fps': self.perf_monitor['avg_fps'],
                'processing_ms': self.current_results.processing_time * 1000
            },
            'agricultural': self.current_results.agricultural_context
        }
    
    def save_session_streamlined(self, session_name: str = None) -> str:
        """Save streamlined session data"""
        try:
            if not self.current_results:
                print("No session data to save")
                return ""
            
            # Get trajectory and summary
            trajectory = self.get_current_trajectory_3d()
            summary = self.get_agricultural_summary()
            
            # Prepare streamlined session data
            session_data = {
                'trajectory_3d': trajectory,
                'session_summary': summary['session_summary'],
                'agricultural_features': summary['agricultural_features'],
                'performance_summary': summary['performance'],
                'session_metadata': {
                    'session_name': session_name or f"agricultural_slam_{int(time.time())}",
                    'timestamp': time.time(),
                    'slam_version': 'streamlined_v2.0',
                    'total_frames_processed': self.perf_monitor['total_frame_count'],
                    'performance_mode': self.performance_mode
                }
            }
            
            # Save using data logger
            filepath = self.data_logger.save_trajectory(
                trajectory,
                metadata=session_data
            )
            
            print(f"🌾 Streamlined session saved: {filepath}")
            print(f"   SLAM Distance: {summary['session_summary']['slam_distance_m']:.3f}m")
            print(f"   Precision Distance: {summary['session_summary']['precision_distance_m']:.3f}m")
            print(f"   Accuracy: ±{summary['session_summary']['accuracy_cm']:.1f}cm")
            print(f"   Field Coverage: {summary['session_summary']['field_coverage_m2']:.1f}m²")
            
            return filepath
            
        except Exception as e:
            print(f"Error saving streamlined session: {e}")
            return ""
    
    def reset_streamlined(self):
        """Reset streamlined SLAM system"""
        with self.results_lock:
            # Reset core components
            self.slam_core.reset()
            self.distance_tracker.reset()
            
            # Reset state
            self.current_results = None
            self.is_initialized = False
            self.processing_active = False
            
            # Reset performance monitoring
            self.frame_skip_counter = 0
            self.perf_monitor = {
                'avg_fps': 0.0,
                'processing_times': deque(maxlen=30),
                'tracking_success_count': 0,
                'total_frame_count': 0
            }
            
            # Reset agricultural stats
            self.agricultural_stats = {
                'crop_rows_detected': False,
                'ground_plane_valid': False,
                'field_coverage_area': 0.0,
                'total_trajectory_points': 0
            }
            
            print("🌾 Streamlined Enhanced SLAM reset successfully")

# Compatibility wrapper for existing code
class EnhancedCustomVisualSLAM(StreamlinedEnhancedSLAM):
    """
    Compatibility wrapper that maintains the existing interface
    while using the streamlined implementation
    """
    
    def __init__(self, camera_manager: CameraManager, camera_matrix: np.ndarray = None):
        """Initialize with compatibility for existing interface"""
        super().__init__(camera_manager)
        self.camera_matrix = camera_matrix
        if camera_matrix is not None:
            print(f"Camera matrix provided: {camera_matrix.shape}")
    
    def process_frame(self, color_frame: np.ndarray, depth_frame: np.ndarray, 
                     timestamp: float = None) -> Dict:
        """
        Process frame with compatibility for existing interface
        Returns dictionary format expected by existing code
        """
        # Use streamlined processing
        results = self.process_frame_streamlined(color_frame, depth_frame, timestamp)
        
        # Convert to dictionary format for compatibility
        return {
            'timestamp': results.timestamp,
            'pose_estimated': results.pose_estimated,
            'position': results.position,
            'rotation': results.rotation,
            'total_distance': results.total_distance,
            'precision_distance': results.precision_distance,
            'tracking_quality': results.tracking_quality,
            'num_features': results.num_features,
            'num_matches': 0,  # Not tracked in streamlined version
            'num_keyframes': results.num_keyframes,
            'num_map_points': 0,  # Not tracked in streamlined version
            'slam_mode': results.slam_mode,
            'processing_time': results.processing_time,
            'agricultural_scene': results.agricultural_context,
            'debug_info': f"{results.slam_mode} - Features: {results.num_features}"
        }
    
    def get_current_pose(self) -> np.ndarray:
        """Get current pose matrix (compatibility)"""
        return self.slam_core.get_current_pose()
    
    def get_trajectory(self) -> np.ndarray:
        """Get trajectory (compatibility)"""
        return self.get_current_trajectory_3d()
    
    def get_map_points(self) -> List[np.ndarray]:
        """Get map points (compatibility)"""
        map_data = self.slam_core.get_map_data()
        return [mp.position for mp in map_data['map_points']]
    
    def get_statistics(self) -> Dict:
        """Get statistics (compatibility)"""
        summary = self.get_agricultural_summary()
        return {
            'initialized': self.is_initialized,
            'tracking': self.current_results.slam_mode == "TRACKING" if self.current_results else False,
            'frame_count': self.perf_monitor['total_frame_count'],
            'num_keyframes': self.current_results.num_keyframes if self.current_results else 0,
            'num_map_points': 0,  # Not tracked in streamlined version
            'trajectory_length': summary['session_summary']['total_trajectory_points'],
            'distance_traveled': summary['session_summary']['slam_distance_m'],
            'avg_processing_time': np.mean(self.perf_monitor['processing_times']) if self.perf_monitor['processing_times'] else 0.0,
            'avg_features': self.current_results.num_features if self.current_results else 0,
            'avg_matches': 0  # Not tracked in streamlined version
        }
    
    def reset(self):
        """Reset (compatibility)"""
        self.reset_streamlined()

# Test function
def test_streamlined_slam():
    """Test streamlined enhanced SLAM"""
    from ..core.camera_manager import CameraManager
    
    print("Testing Streamlined Enhanced SLAM...")
    
    camera_manager = CameraManager()
    slam = StreamlinedEnhancedSLAM(camera_manager)
    
    # Set performance mode
    slam.set_performance_mode("BALANCED")
    
    # Process test frames
    for i in range(20):
        color_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth_frame = np.random.randint(500, 3000, (480, 640), dtype=np.uint16)
        
        results = slam.process_frame_streamlined(color_frame, depth_frame)
        
        if i % 5 == 0:
            print(f"Frame {i}: {results.slam_mode}, "
                  f"Distance: {results.total_distance:.3f}m, "
                  f"FPS: {slam.perf_monitor['avg_fps']:.1f}")
    
    # Print final summary
    summary = slam.get_agricultural_summary()
    print(f"\nFinal Summary:")
    print(f"  Total frames: {summary['performance']['tracking_success_rate']:.1f}%")
    print(f"  Average FPS: {summary['performance']['avg_fps']:.1f}")
    print(f"  Field coverage: {summary['session_summary']['field_coverage_m2']:.1f}m²")

if __name__ == "__main__":
    test_streamlined_slam()