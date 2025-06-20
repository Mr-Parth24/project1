"""
Enhanced Custom Visual SLAM with 2024 Research Improvements
Addresses pose estimation failures and tracking issues
"""

import numpy as np
import cv2
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from src.core.enhanced_visual_odometry import EnhancedVisualOdometry
from src.core.feature_detector import FeatureDetector
from src.utils.config_manager import get_config_manager

@dataclass
class Keyframe:
    """Enhanced keyframe with better tracking"""
    id: int
    pose: np.ndarray
    features: np.ndarray
    descriptors: np.ndarray
    points_3d: np.ndarray
    timestamp: float

class EnhancedCustomVisualSLAM:
    """
    Enhanced Agricultural Visual SLAM System with Latest Research
    - Fixes pose estimation failures through robust multi-method approach
    - Addresses distance measurement inconsistencies
    - Implements adaptive parameter tuning
    - Includes proper error handling and validation
    """
    
    def __init__(self):
        """Initialize enhanced SLAM system"""
        self.config_manager = get_config_manager()
        
        # Use enhanced visual odometry
        self.visual_odometry = EnhancedVisualOdometry()
        
        # Enhanced feature detection
        max_features = self.config_manager.get_slam_parameter('max_features', 1000)
        self.feature_detector = FeatureDetector(max_features=max_features)
        
        # SLAM state with better tracking
        self.is_initialized = False
        self.keyframes: List[Keyframe] = []
        self.current_keyframe_id = 0
        self.map_points = []
        
        # Enhanced keyframe creation thresholds
        self.keyframe_distance_threshold = 0.15  # 15cm
        self.keyframe_angle_threshold = 0.1      # ~6 degrees
        
        # Performance and error tracking
        self.processing_times = []
        self.slam_status = "INITIALIZING"
        self.last_pose = np.eye(4)
        self.frame_count = 0
        self.error_count = 0
        self.last_successful_frame = 0
        
        # Distance tracking fix
        self.cumulative_distance = 0.0
        self.last_valid_position = np.array([0.0, 0.0, 0.0])
        
        # Agricultural-specific settings
        self.field_mode = self.config_manager.get_slam_parameter('field_mode', True)
        self.robust_tracking = True
        
        # Error recovery mechanisms
        self.max_consecutive_failures = 30
        self.recovery_mode = False
        
        print("✅ Enhanced Custom Visual SLAM initialized")
        print(f"   Thresholds: {self.keyframe_distance_threshold:.2f}m, {np.degrees(self.keyframe_angle_threshold):.1f}°")
    
    def validate_frame_data(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> bool:
        """Validate input frame data to prevent parsing errors"""
        try:
            # Check if frames are valid
            if color_frame is None or depth_frame is None:
                return False
            
            # Check frame shapes
            if len(color_frame.shape) not in [2, 3] or len(depth_frame.shape) != 2:
                return False
            
            # Check frame dimensions
            if color_frame.shape[:2] != depth_frame.shape[:2]:
                print(f"⚠️ Frame dimension mismatch: color {color_frame.shape}, depth {depth_frame.shape}")
                return False
            
            # Check for reasonable frame size
            if color_frame.shape[0] < 100 or color_frame.shape[1] < 100:
                return False
            
            # Check data types
            if color_frame.dtype != np.uint8 or depth_frame.dtype != np.uint16:
                print(f"⚠️ Invalid frame data types: color {color_frame.dtype}, depth {depth_frame.dtype}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Frame validation error: {e}")
            return False
    
    def process_frame(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> Dict:
        """
        Enhanced frame processing with robust error handling
        """
        start_time = time.time()
        self.frame_count += 1
        
        # Initialize default results
        results = {
            'pose_estimated': False,
            'position': self.last_valid_position.copy(),
            'rotation': np.eye(3),
            'num_features': 0,
            'num_matches': 0,
            'inliers': 0,
            'translation_magnitude': 0.0,
            'keyframes_count': len(self.keyframes),
            'map_points_count': len(self.map_points),
            'slam_status': self.slam_status,
            'processing_time': 0.0,
            'debug_info': 'Processing...',
            'distance_traveled': self.cumulative_distance
        }
        
        try:
            # Validate input frames
            if not self.validate_frame_data(color_frame, depth_frame):
                results['debug_info'] = 'Invalid frame data'
                self.error_count += 1
                return results
            
            # Enhanced visual odometry processing
            vo_results = self.visual_odometry.process_frame(color_frame, depth_frame)
            
            # Update results with VO data
            results.update({
                'pose_estimated': vo_results['pose_estimated'],
                'position': vo_results['position'].copy(),
                'rotation': vo_results['rotation'].copy(),
                'num_features': vo_results['num_features'],
                'num_matches': vo_results['num_matches'],
                'inliers': vo_results['inliers'],
                'translation_magnitude': vo_results['translation_magnitude'],
                'debug_info': vo_results['debug_info']
            })
            
            # Handle SLAM state based on VO results
            if not self.is_initialized:
                if vo_results['pose_estimated']:
                    # Enhanced initialization
                    self.initialize_slam(color_frame, depth_frame, vo_results)
                    results['slam_status'] = "INITIALIZED"
                    results['debug_info'] = f"Enhanced SLAM initialized: {vo_results['debug_info']}"
                    print(f"✅ Enhanced SLAM initialized successfully!")
                else:
                    self.slam_status = "INITIALIZING"
                    results['debug_info'] = f"Waiting for initialization: {vo_results['debug_info']}"
            else:
                # Process tracking
                if vo_results['pose_estimated']:
                    self.handle_successful_tracking(color_frame, depth_frame, vo_results, results)
                else:
                    self.handle_tracking_failure(vo_results, results)
            
            # Update distance tracking
            self.update_distance_tracking(vo_results)
            results['distance_traveled'] = self.cumulative_distance
            
            # Reset error count on success
            if vo_results['pose_estimated']:
                self.error_count = 0
                self.last_successful_frame = self.frame_count
                
        except Exception as e:
            print(f"❌ SLAM processing error: {e}")
            results['debug_info'] = f'Processing error: {str(e)}'
            self.error_count += 1
            
            # Error recovery
            if self.error_count > self.max_consecutive_failures:
                self.initiate_recovery_mode()
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        results['processing_time'] = processing_time
        results['keyframes_count'] = len(self.keyframes)
        
        return results
    
    def initialize_slam(self, color_frame: np.ndarray, depth_frame: np.ndarray, vo_results: Dict):
        """Enhanced SLAM initialization"""
        self.is_initialized = True
        self.slam_status = "TRACKING"
        self.last_pose = np.eye(4)
        self.last_pose[:3, :3] = vo_results['rotation']
        self.last_pose[:3, 3] = vo_results['position']
        
        # Create first keyframe with enhanced data
        feature_results = self.feature_detector.process_frame(color_frame)
        if feature_results['keypoints'] is not None:
            self.create_enhanced_keyframe(
                color_frame, depth_frame, self.last_pose,
                feature_results['keypoints'],
                feature_results['descriptors']
            )
        
        self.last_valid_position = vo_results['position'].copy()
    
    def handle_successful_tracking(self, color_frame: np.ndarray, depth_frame: np.ndarray, 
                                 vo_results: Dict, results: Dict):
        """Handle successful tracking frame"""
        # Update pose
        current_pose = np.eye(4)
        current_pose[:3, :3] = vo_results['rotation']
        current_pose[:3, 3] = vo_results['position']
        self.last_pose = current_pose
        
        # Check if we should create a new keyframe
        if self.should_create_keyframe(current_pose):
            feature_results = self.feature_detector.process_frame(color_frame)
            if feature_results['keypoints'] is not None:
                self.create_enhanced_keyframe(
                    color_frame, depth_frame, current_pose,
                    feature_results['keypoints'],
                    feature_results['descriptors']
                )
                results['debug_info'] += f" | New keyframe {self.current_keyframe_id-1}"
        
        self.slam_status = "TRACKING"
        results['slam_status'] = "TRACKING"
        results['debug_info'] = f"Enhanced tracking: {vo_results['debug_info']}"
        
        # Update valid position
        self.last_valid_position = vo_results['position'].copy()
        
        # Log successful tracking occasionally
        if self.frame_count % 30 == 0:
            print(f"✅ Enhanced SLAM tracking: {vo_results['translation_magnitude']:.3f}m movement, {len(self.keyframes)} keyframes")
    
    def handle_tracking_failure(self, vo_results: Dict, results: Dict):
        """Handle tracking failure with recovery mechanisms"""
        self.slam_status = "LOST"
        results['slam_status'] = "LOST"
        results['debug_info'] = f"Tracking lost: {vo_results['debug_info']}"
        
        # Log tracking issues
        if self.frame_count % 30 == 0:
            print(f"⚠️ Enhanced SLAM tracking issue: {vo_results['debug_info']}")
        
        # Check if we need recovery
        frames_since_success = self.frame_count - self.last_successful_frame
        if frames_since_success > self.max_consecutive_failures:
            self.initiate_recovery_mode()
    
    def update_distance_tracking(self, vo_results: Dict):
        """Update cumulative distance with validation"""
        if vo_results['pose_estimated'] and vo_results['translation_magnitude'] > 0:
            # Validate movement magnitude
            if vo_results['translation_magnitude'] < 1.0:  # Reasonable movement
                self.cumulative_distance += vo_results['translation_magnitude']
            else:
                print(f"⚠️ Unreasonable movement detected: {vo_results['translation_magnitude']:.3f}m")
    
    def initiate_recovery_mode(self):
        """Initiate recovery mode for persistent failures"""
        print("🔄 Initiating SLAM recovery mode...")
        self.recovery_mode = True
        self.visual_odometry.reset()
        self.error_count = 0
        
        # Reset adaptive parameters to more sensitive values
        if hasattr(self.visual_odometry, 'adaptive_params'):
            self.visual_odometry.adaptive_params['min_features'] = 50
            self.visual_odometry.adaptive_params['detection_threshold'] = 10
            self.visual_odometry.adaptive_params['match_threshold'] = 0.8
    
    def create_enhanced_keyframe(self, color_frame: np.ndarray, depth_frame: np.ndarray, 
                               pose: np.ndarray, keypoints, descriptors):
        """Create enhanced keyframe with better data"""
        try:
            # Extract keypoint coordinates
            if keypoints:
                features_2d = np.array([kp.pt for kp in keypoints], dtype=np.float32)
                
                # Generate 3D points
                points_3d, valid_indices = self.visual_odometry.enhanced_depth_to_3d(features_2d, depth_frame)
                
                if len(points_3d) > 0:
                    # Create keyframe
                    keyframe = Keyframe(
                        id=self.current_keyframe_id,
                        pose=pose.copy(),
                        features=features_2d[valid_indices],
                        descriptors=descriptors[valid_indices] if descriptors is not None else np.array([]),
                        points_3d=points_3d,
                        timestamp=time.time()
                    )
                    
                    self.keyframes.append(keyframe)
                    self.current_keyframe_id += 1
                    
                    print(f"✅ Created enhanced keyframe {keyframe.id} with {len(points_3d)} 3D points")
                    
        except Exception as e:
            print(f"Error creating keyframe: {e}")
    
    def should_create_keyframe(self, current_pose: np.ndarray) -> bool:
        """Enhanced keyframe creation decision"""
        if len(self.keyframes) == 0:
            return True
        
        last_keyframe = self.keyframes[-1]
        
        # Calculate translation and rotation differences
        translation_diff = np.linalg.norm(current_pose[:3, 3] - last_keyframe.pose[:3, 3])
        
        # Calculate rotation difference
        R_diff = current_pose[:3, :3] @ last_keyframe.pose[:3, :3].T
        angle_diff = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        
        # Enhanced decision criteria
        should_create = (translation_diff > self.keyframe_distance_threshold or 
                        angle_diff > self.keyframe_angle_threshold or
                        len(self.keyframes) < 2)  # Always create second keyframe
        
        return should_create
    
    def get_current_pose(self) -> np.ndarray:
        """Get current camera pose"""
        return self.last_pose.copy()
    
    def get_trajectory(self) -> np.ndarray:
        """Get trajectory from enhanced visual odometry"""
        return self.visual_odometry.get_trajectory()
    
    def get_distance_traveled(self) -> float:
        """Get total distance traveled with enhanced accuracy"""
        return self.cumulative_distance
    
    def get_keyframes(self) -> List[Keyframe]:
        """Get all keyframes"""
        return self.keyframes.copy()
    
    def get_performance_stats(self) -> Dict:
        """Get enhanced SLAM performance statistics"""
        vo_stats = self.visual_odometry.get_performance_stats()
        
        return {
            'slam_status': self.slam_status,
            'is_initialized': self.is_initialized,
            'keyframes_count': len(self.keyframes),
            'map_points_count': len(self.map_points),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0,
            'total_frames': self.frame_count,
            'distance_traveled': self.cumulative_distance,
            'trajectory_length': len(self.get_trajectory()),
            'error_count': self.error_count,
            'recovery_mode': self.recovery_mode,
            # Include enhanced VO stats
            'vo_avg_processing_time': vo_stats['avg_processing_time'],
            'vo_avg_tracks': vo_stats['avg_tracks'],
            'vo_pose_estimation_active': vo_stats['pose_estimation_active'],
            'vo_adaptive_params': vo_stats.get('current_adaptive_params', {})
        }
    
    def reset(self):
        """Reset enhanced SLAM system"""
        self.is_initialized = False
        self.keyframes = []
        self.current_keyframe_id = 0
        self.map_points = []
        self.slam_status = "INITIALIZING"
        self.last_pose = np.eye(4)
        self.frame_count = 0
        self.error_count = 0
        self.last_successful_frame = 0
        self.cumulative_distance = 0.0
        self.last_valid_position = np.array([0.0, 0.0, 0.0])
        self.recovery_mode = False
        self.processing_times = []
        
        # Reset visual odometry
        self.visual_odometry.reset()
        
        print("✅ Enhanced SLAM system reset") 