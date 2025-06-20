"""
Custom Visual SLAM Implementation
Lightweight SLAM optimized for agricultural environments
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import time

from src.core.feature_detector import FeatureDetector
from src.core.visual_odometry import VisualOdometry
from src.utils.coordinate_transform import CoordinateTransform

class CustomVisualSLAM:
    """
    Custom Visual SLAM implementation
    Combines feature detection, visual odometry, and mapping
    """
    
    def __init__(self, camera_matrix: np.ndarray = None):
        """
        Initialize Custom Visual SLAM
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
        """
        # Core components
        self.feature_detector = FeatureDetector(max_features=1000)
        
        if camera_matrix is None:
            camera_matrix = self._get_default_camera_matrix()
        
        self.visual_odometry = VisualOdometry(camera_matrix)
        self.coordinate_transform = CoordinateTransform()
        self.coordinate_transform.set_camera_matrix(camera_matrix)
        
        # SLAM state
        self.is_initialized = False
        self.is_tracking = False
        self.frame_count = 0
        
        # Map data
        self.map_points = []  # 3D map points
        self.keyframes = []   # Keyframe data
        self.trajectory = []  # Camera trajectory
        
        # Performance tracking
        self.processing_times = []
        self.feature_counts = []
        self.match_counts = []
        
        # Agricultural-specific parameters
        self.min_features_for_tracking = 30  # Reduced from 50
        self.max_features_for_keyframe = 200
        self.keyframe_distance_threshold = 0.3  # Reduced from 0.5 meters
        self.keyframe_angle_threshold = 0.15    # Reduced from 0.2 radians
        
        print("Custom Visual SLAM initialized")
    
    def _get_default_camera_matrix(self) -> np.ndarray:
        """Get default camera matrix for D435i"""
        return np.array([
            [615.0, 0, 320.0],
            [0, 615.0, 240.0],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def initialize(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> bool:
        """
        Initialize SLAM with first frame
        
        Args:
            color_frame: First color frame
            depth_frame: First depth frame
            
        Returns:
            True if initialization successful
        """
        try:
            # Detect features in first frame
            feature_results = self.feature_detector.process_frame(color_frame)
            
            if feature_results['num_features'] < self.min_features_for_tracking:
                print(f"Insufficient features for initialization: {feature_results['num_features']}")
                return False
            
            # Process with visual odometry
            vo_results = self.visual_odometry.process_frame(color_frame, depth_frame)
            
            # Create first keyframe
            keyframe = {
                'id': 0,
                'color_frame': color_frame.copy(),
                'depth_frame': depth_frame.copy(),
                'keypoints': feature_results['keypoints'],
                'descriptors': feature_results['descriptors'],
                'pose': np.eye(4),
                'map_points': []
            }
            
            self.keyframes.append(keyframe)
            self.trajectory.append(np.array([0.0, 0.0, 0.0]))
            
            # Initialize 3D map points
            self._initialize_map_points(color_frame, depth_frame, feature_results['keypoints'])
            
            self.is_initialized = True
            self.is_tracking = True
            self.frame_count = 1
            
            print("SLAM initialized successfully")
            return True
            
        except Exception as e:
            print(f"SLAM initialization failed: {e}")
            return False
    
    def _initialize_map_points(self, color_frame: np.ndarray, depth_frame: np.ndarray, keypoints: List):
        """Initialize 3D map points from first frame"""
        if not keypoints:
            return
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            # Check bounds
            if (0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]):
                depth = depth_frame[y, x]
                
                if depth > 0:  # Valid depth
                    # Convert to 3D point
                    point_3d = self.coordinate_transform.pixel_to_camera(
                        np.array([x, y]), depth / 1000.0
                    )
                    
                    # Add to map
                    map_point = {
                        'id': len(self.map_points),
                        'position': point_3d,
                        'observations': [{'keyframe_id': 0, 'keypoint_idx': len(self.map_points)}],
                        'descriptor': None  # Could store descriptor for matching
                    }
                    
                    self.map_points.append(map_point)
    
    def process_frame(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> Dict:
        """
        Process a new frame through the SLAM pipeline
        
        Args:
            color_frame: Current color frame
            depth_frame: Current depth frame
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        if not self.is_initialized:
            # Try to initialize
            success = self.initialize(color_frame, depth_frame)
            return {
                'initialized': success,
                'tracking': success,
                'position': np.array([0.0, 0.0, 0.0]),
                'rotation': np.eye(3),
                'num_features': 0,
                'num_matches': 0,
                'num_map_points': len(self.map_points),
                'processing_time': time.time() - start_time
            }
        
        # Feature detection
        feature_results = self.feature_detector.process_frame(color_frame)
        
        # Visual odometry
        vo_results = self.visual_odometry.process_frame(color_frame, depth_frame)
          # Check if tracking is successful
        tracking_ok = (feature_results['num_features'] >= self.min_features_for_tracking and 
                      vo_results.get('pose_estimated', False))
        
        # Additional check: only track if there's actual movement
        translation_magnitude = vo_results.get('translation_magnitude', 0.0)
        has_significant_movement = translation_magnitude > 0.05  # 5cm threshold
        
        if not tracking_ok:
            if feature_results['num_features'] < self.min_features_for_tracking:
                if self.frame_count % 30 == 0:  # Print occasionally
                    print(f"SLAM: Insufficient features: {feature_results['num_features']} < {self.min_features_for_tracking}")
            if not vo_results.get('pose_estimated', False):
                if self.frame_count % 30 == 0:  # Print occasionally
                    print(f"SLAM: Pose not estimated. Debug: {vo_results.get('debug_info', 'No debug info')}")
            self.is_tracking = False
        elif not has_significant_movement:
            # Features detected and pose estimated, but no significant movement
            if self.frame_count % 60 == 0:  # Print occasionally
                print(f"SLAM: Stationary - Features: {feature_results['num_features']}, Movement: {translation_magnitude:.4f}m (below 0.05m threshold)")
            self.is_tracking = True  # Keep tracking state but don't update trajectory
            # Set pose_estimated to False to prevent trajectory updates
            vo_results['pose_estimated'] = False
        else:
            self.is_tracking = True
            if self.frame_count % 30 == 0:  # Print occasionally
                print(f"SLAM: Tracking OK - Features: {feature_results['num_features']}, Movement: {translation_magnitude:.3f}m")
          # Decide if we need a new keyframe
        need_keyframe = self._need_new_keyframe(vo_results) and has_significant_movement
        
        if need_keyframe and self.is_tracking and vo_results.get('pose_estimated', False):
            self._create_keyframe(color_frame, depth_frame, feature_results, vo_results)
          # Update trajectory only if pose was estimated with significant movement
        if vo_results['pose_estimated'] and has_significant_movement:
            self.trajectory.append(vo_results['position'].copy())
            if self.frame_count % 30 == 0:  # Print occasionally
                pos = vo_results['position']
                dist = self.visual_odometry.get_distance_traveled()
                print(f"SLAM: Position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), Distance: {dist:.3f}m")
        elif vo_results['pose_estimated'] and not has_significant_movement:
            if self.frame_count % 60 == 0:  # Print occasionally  
                print(f"SLAM: Pose estimated but movement too small: {translation_magnitude:.4f}m")
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.feature_counts.append(feature_results['num_features'])
        self.match_counts.append(vo_results['num_matches'])
        
        # Keep only recent history
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
            self.feature_counts.pop(0)
            self.match_counts.pop(0)
        
        self.frame_count += 1
        
        # Prepare results
        results = {
            'initialized': self.is_initialized,
            'tracking': self.is_tracking,
            'position': vo_results['position'],
            'rotation': vo_results['rotation'],
            'num_features': feature_results['num_features'],
            'num_matches': vo_results['num_matches'],
            'num_map_points': len(self.map_points),
            'num_keyframes': len(self.keyframes),
            'frame_count': self.frame_count,
            'processing_time': processing_time,
            'keyframe_created': need_keyframe,
            'distance_traveled': self.visual_odometry.get_distance_traveled(),
            'debug_info': vo_results.get('debug_info', ''),
            'pose_estimated': vo_results.get('pose_estimated', False)
        }
        
        return results
    
    def _need_new_keyframe(self, vo_results: Dict) -> bool:
        """
        Determine if a new keyframe is needed
        
        Args:
            vo_results: Visual odometry results
            
        Returns:
            True if new keyframe needed
        """
        if not self.is_tracking or not vo_results['pose_estimated']:
            return False
        
        if len(self.keyframes) == 0:
            return True
        
        # Get last keyframe pose
        last_keyframe = self.keyframes[-1]
        last_pose = last_keyframe['pose']
        
        # Current pose
        current_pose = np.eye(4)
        current_pose[:3, :3] = vo_results['rotation']
        current_pose[:3, 3] = vo_results['position']
        
        # Calculate distance and angle from last keyframe
        trans_dist, rot_angle = self.coordinate_transform.get_pose_distance(
            current_pose, last_pose
        )
        
        # Check thresholds
        need_keyframe = (
            trans_dist > self.keyframe_distance_threshold or
            rot_angle > self.keyframe_angle_threshold or
            vo_results['num_matches'] < self.min_features_for_tracking
        )
        
        return need_keyframe
    
    def _create_keyframe(self, color_frame: np.ndarray, depth_frame: np.ndarray, 
                        feature_results: Dict, vo_results: Dict):
        """
        Create a new keyframe
        
        Args:
            color_frame: Color frame
            depth_frame: Depth frame
            feature_results: Feature detection results
            vo_results: Visual odometry results
        """
        try:
            # Create pose matrix
            pose = np.eye(4)
            if vo_results['pose_estimated']:
                pose[:3, :3] = vo_results['rotation']
                pose[:3, 3] = vo_results['position']
            
            # Create keyframe
            keyframe = {
                'id': len(self.keyframes),
                'color_frame': color_frame.copy(),
                'depth_frame': depth_frame.copy(),
                'keypoints': feature_results['keypoints'],
                'descriptors': feature_results['descriptors'],
                'pose': pose,
                'map_points': []
            }
            
            self.keyframes.append(keyframe)
            
            # Add new map points from this keyframe
            self._add_map_points_from_keyframe(keyframe)
            
            print(f"Created keyframe {keyframe['id']} with {len(feature_results['keypoints'])} features")
            
        except Exception as e:
            print(f"Error creating keyframe: {e}")
    
    def _add_map_points_from_keyframe(self, keyframe: Dict):
        """Add new map points from keyframe"""
        if not keyframe['keypoints']:
            return
        
        depth_frame = keyframe['depth_frame']
        pose = keyframe['pose']
        keypoints = keyframe['keypoints']
        descriptors = keyframe['descriptors']
        
        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            # Check bounds
            if (0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]):
                depth = depth_frame[y, x]
                
                if depth > 0:  # Valid depth
                    # Convert to 3D point in camera frame
                    point_camera = self.coordinate_transform.pixel_to_camera(
                        np.array([x, y]), depth / 1000.0
                    )
                    
                    # Transform to world frame
                    point_world = self.coordinate_transform.camera_to_world(point_camera)
                    
                    # Get descriptor safely
                    descriptor = None
                    if descriptors is not None and i < len(descriptors):
                        try:
                            descriptor = descriptors[i].copy() if hasattr(descriptors[i], 'copy') else descriptors[i]
                        except Exception as e:
                            print(f"Descriptor access error: {e}")
                            descriptor = None
                    
                    # Add to map
                    map_point = {
                        'id': len(self.map_points),
                        'position': point_world,
                        'observations': [{'keyframe_id': keyframe['id'], 'keypoint_idx': i}],
                        'descriptor': descriptor
                    }
                    
                    self.map_points.append(map_point)
                    keyframe['map_points'].append(map_point['id'])
    
    def get_current_pose(self) -> np.ndarray:
        """Get current camera pose"""
        return self.visual_odometry.get_current_position()
    
    def get_trajectory(self) -> np.ndarray:
        """Get camera trajectory"""
        return np.array(self.trajectory) if self.trajectory else np.array([[0, 0, 0]])
    
    def get_map_points(self) -> List[np.ndarray]:
        """Get 3D map points"""
        return [mp['position'] for mp in self.map_points]
    
    def get_statistics(self) -> Dict:
        """Get SLAM statistics"""
        vo_stats = self.visual_odometry.get_performance_stats()
        
        stats = {
            'initialized': self.is_initialized,
            'tracking': self.is_tracking,
            'frame_count': self.frame_count,
            'num_keyframes': len(self.keyframes),
            'num_map_points': len(self.map_points),
            'trajectory_length': len(self.trajectory),
            'distance_traveled': vo_stats.get('distance_traveled', 0.0),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0,
            'avg_features': np.mean(self.feature_counts) if self.feature_counts else 0.0,
            'avg_matches': np.mean(self.match_counts) if self.match_counts else 0.0,
            'vo_stats': vo_stats
        }
        
        return stats
    
    def reset(self):
        """Reset SLAM system"""
        self.is_initialized = False
        self.is_tracking = False
        self.frame_count = 0
        
        self.map_points = []
        self.keyframes = []
        self.trajectory = []
        
        self.processing_times = []
        self.feature_counts = []
        self.match_counts = []
        
        self.feature_detector.reset()
        self.visual_odometry.reset()
        
        print("SLAM system reset")
    
    def draw_features_overlay(self, frame: np.ndarray, feature_results: Dict) -> np.ndarray:
        """
        Draw feature detection overlay on frame
        
        Args:
            frame: Input frame
            feature_results: Feature detection results
            
        Returns:
            Frame with features overlaid
        """
        if not feature_results['keypoints']:
            return frame
        
        # Draw features
        overlay = self.feature_detector.draw_features(
            frame, feature_results['keypoints'], color=(0, 255, 0)
        )
        
        # Add status text
        status_text = f"Features: {feature_results['num_features']}"
        if feature_results['matches']:
            status_text += f" | Matches: {len(feature_results['matches'])}"
        
        cv2.putText(overlay, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add tracking status
        if self.is_tracking:
            status_color = (0, 255, 0)  # Green
            status_msg = "TRACKING"
        else:
            status_color = (0, 0, 255)  # Red
            status_msg = "LOST"
        
        cv2.putText(overlay, status_msg, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return overlay

# Test function
def test_custom_slam():
    """Test the custom SLAM implementation"""
    slam = CustomVisualSLAM()
    
    # Create test frames
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.ones((480, 640), dtype=np.uint16) * 1000
    
    # Add some features
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.circle(frame, (400, 300), 50, (128, 128, 128), -1)
    
    # Process frame
    results = slam.process_frame(frame, depth)
    
    print("SLAM Test Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    
    # Print statistics
    stats = slam.get_statistics()
    print("\nSLAM Statistics:")
    for key, value in stats.items():
        if key != 'vo_stats':
            print(f"  {key}: {value}")

if __name__ == "__main__":
    test_custom_slam()