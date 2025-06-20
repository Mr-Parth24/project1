"""
Visual Odometry for RealSense D435i
Estimates camera pose using visual features without IMU
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import time
from src.core.feature_detector import FeatureDetector
from src.utils.config_manager import get_config_manager

class VisualOdometry:
    """
    Visual Odometry implementation for stereo camera
    Estimates 6DOF camera pose from visual features
    """
    
    def __init__(self, camera_matrix: np.ndarray = None, dist_coeffs: np.ndarray = None):
        """
        Initialize Visual Odometry
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        # Load configuration
        self.config_manager = get_config_manager()
        
        # Use provided camera matrix or load from config
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
        else:
            self.camera_matrix = self.config_manager.get_camera_matrix()
            
        if dist_coeffs is not None:
            self.dist_coeffs = dist_coeffs
        else:
            self.dist_coeffs = self.config_manager.get_distortion_coefficients()
        
        # Feature detector
        max_features = self.config_manager.get_slam_parameter('max_features', 1000)
        self.feature_detector = FeatureDetector(max_features=max_features)
        
        # Pose tracking
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.trajectory = [np.array([0.0, 0.0, 0.0])]  # List of positions
        self.rotations = [np.eye(3)]  # List of rotation matrices
        
        # Previous frame data
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points_3d = None
          # RANSAC parameters from config
        self.ransac_threshold = self.config_manager.get_slam_parameter('ransac_threshold', 5.0)
        self.ransac_iterations = 1000
        self.min_matches = self.config_manager.get_slam_parameter('min_matches', 6)
        self.max_translation = self.config_manager.get_slam_parameter('max_translation_per_frame', 10.0)
          # Movement filtering parameters - Much more aggressive
        self.min_translation_threshold = self.config_manager.get_slam_parameter('min_translation_threshold', 0.15)  # Increased
        self.min_rotation_threshold = self.config_manager.get_slam_parameter('min_rotation_threshold', 0.1)
        self.stationary_threshold = self.config_manager.get_slam_parameter('stationary_threshold', 0.1)
        self.min_inliers_required = self.config_manager.get_slam_parameter('min_inliers_required', 20)  # Minimum inliers
        
        # Stationary detection - more robust
        self.is_stationary = True
        self.stationary_count = 0
        self.last_significant_position = np.array([0.0, 0.0, 0.0])
        self.movement_history = []  # Track recent movements
        self.consecutive_small_movements = 0  # Track tiny movements in a row
        self.max_consecutive_small_movements = 10  # Allow max 10 tiny movements before forcing stationary
        
        # Performance tracking
        self.processing_times = []
        self.match_counts = []
        self.pose_estimated = False
        
        print(f"Visual Odometry initialized with config: max_trans={self.max_translation}m, threshold={self.ransac_threshold}")
    
    
    def get_default_camera_matrix(self) -> np.ndarray:
        """Get default camera matrix for D435i"""
        # More accurate D435i intrinsics for 640x480
        fx = 607.4  # Focal length x
        fy = 607.4  # Focal length y  
        cx = 319.5  # Principal point x (center)
        cy = 239.5  # Principal point y (center)
        
        matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        print(f"Using camera matrix: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        return matrix
    
    def depth_to_3d_points(self, points_2d: np.ndarray, depth_frame: np.ndarray) -> np.ndarray:
        """
        Convert 2D points to 3D using depth information
        
        Args:
            points_2d: 2D points in image coordinates
            depth_frame: Depth image
            
        Returns:
            3D points in camera coordinates
        """
        if points_2d.size == 0:
            return np.array([])
        
        points_3d = []
        valid_count = 0
        total_count = len(points_2d.reshape(-1, 2))
        
        for point in points_2d.reshape(-1, 2):
            x, y = int(point[0]), int(point[1])
            
            # Check bounds
            if (0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]):
                depth = depth_frame[y, x]
                
                if depth > 100:  # Valid depth (>10cm, depth is in mm)
                    # Convert to meters (RealSense depth is in millimeters)
                    z = depth / 1000.0
                    
                    # Back-project to 3D
                    x_3d = (x - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                    y_3d = (y - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
                    
                    # Sanity check for reasonable 3D points
                    if 0.1 <= z <= 10.0 and abs(x_3d) <= 10.0 and abs(y_3d) <= 10.0:
                        points_3d.append([x_3d, y_3d, z])
                        valid_count += 1
        
        result = np.array(points_3d, dtype=np.float32)
        
        # Debug output occasionally
        if len(self.processing_times) % 60 == 0:  # Every ~2 seconds
            print(f"3D Points: {valid_count}/{total_count} valid, depth range: {np.min(result[:, 2]) if len(result) > 0 else 0:.2f}-{np.max(result[:, 2]) if len(result) > 0 else 0:.2f}m")
        
        return result
    
    def estimate_pose_pnp(self, points_3d: np.ndarray, points_2d: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Estimate camera pose using PnP algorithm
        
        Args:
            points_3d: 3D points from previous frame
            points_2d: Corresponding 2D points in current frame
            
        Returns:
            Tuple of (success, rotation_vector, translation_vector)
        """
        if len(points_3d) < self.min_matches or len(points_2d) < self.min_matches:
            return False, None, None
        
        try:
            # Ensure correct data types and shapes
            points_3d = np.array(points_3d, dtype=np.float32)
            points_2d = np.array(points_2d, dtype=np.float32)
            
            # Reshape to correct format for OpenCV
            if points_3d.ndim == 2:
                points_3d = points_3d.reshape(-1, 1, 3)
            if points_2d.ndim == 2:
                points_2d = points_2d.reshape(-1, 1, 2)
            
            # Debug: Print array info occasionally
            if len(self.processing_times) % 60 == 0:  # Every ~2 seconds
                print(f"PnP Debug: 3D shape: {points_3d.shape}, 2D shape: {points_2d.shape}")
                print(f"PnP Debug: 3D range: {np.min(points_3d, axis=0).flatten()} to {np.max(points_3d, axis=0).flatten()}")
            
            # Try multiple PnP methods
            methods = [
                (cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"),
                (cv2.SOLVEPNP_EPNP, "EPNP"),
                (cv2.SOLVEPNP_P3P, "P3P") if len(points_3d) >= 4 else None,
            ]
            
            for method_info in methods:
                if method_info is None:
                    continue
                    
                method, method_name = method_info
                
                try:
                    # First try without RANSAC for debugging
                    success, rvec, tvec = cv2.solvePnP(
                        points_3d,
                        points_2d,
                        self.camera_matrix,
                        self.dist_coeffs,                        flags=method
                    )
                    
                    if success and rvec is not None and tvec is not None:
                        # Check if solution is reasonable
                        translation_norm = np.linalg.norm(tvec)
                        rotation_norm = np.linalg.norm(rvec)
                        
                        # CRITICAL: Filter out tiny movements (noise)
                        if translation_norm < self.min_translation_threshold and rotation_norm < self.min_rotation_threshold:
                            if len(self.processing_times) % 60 == 0:  # Print occasionally
                                print(f"PnP {method_name}: Movement too small - Trans: {translation_norm:.4f}m, Rot: {rotation_norm:.4f}rad (FILTERED)")
                            continue  # Skip this small movement
                        
                        print(f"PnP {method_name}: Success! Trans: {translation_norm:.3f}m, Rot: {rotation_norm:.3f}rad")
                        
                        # Accept reasonable solutions
                        if translation_norm < 5.0 and rotation_norm < 3.14:  # Max 5m translation, 180° rotation
                            return True, rvec, tvec
                        else:
                            print(f"PnP {method_name}: Solution unreasonable - Trans: {translation_norm:.3f}m, Rot: {rotation_norm:.3f}rad")
                    else:
                        if len(self.processing_times) % 60 == 0:  # Reduce verbosity
                            print(f"PnP {method_name}: Failed")
                        
                except Exception as e:
                    if len(self.processing_times) % 60 == 0:  # Reduce exception spam
                        print(f"PnP {method_name}: Exception - {e}")
                    continue
            
            # If all methods failed, try RANSAC as last resort
            try:
                if len(self.processing_times) % 60 == 0:  # Only print occasionally
                    print("PnP: Trying RANSAC...")
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    points_3d,
                    points_2d,
                    self.camera_matrix,
                    self.dist_coeffs,
                    iterationsCount=500,  # Reduced iterations
                    reprojectionError=5.0,  # More lenient error
                    confidence=0.90  # Reduced confidence
                )
                
                if success and inliers is not None and len(inliers) >= max(4, len(points_3d) // 4):
                    translation_norm = np.linalg.norm(tvec)
                    rotation_norm = np.linalg.norm(rvec)
                    if len(self.processing_times) % 60 == 0:
                        print(f"PnP RANSAC: Success! Inliers: {len(inliers)}/{len(points_3d)}, Trans: {translation_norm:.3f}m")
                    
                    if translation_norm < 5.0 and rotation_norm < 3.14:
                        return True, rvec, tvec
                    else:
                        if len(self.processing_times) % 60 == 0:
                            print(f"PnP RANSAC: Solution unreasonable")
                else:
                    if len(self.processing_times) % 60 == 0:
                        print(f"PnP RANSAC: Failed - Inliers: {len(inliers) if inliers is not None else 0}")
                    
            except Exception as e:
                if len(self.processing_times) % 60 == 0:
                    print(f"PnP RANSAC: Exception - {e}")
            
            return False, None, None
                
        except Exception as e:
            print(f"PnP estimation error: {e}")
            return False, None, None
    
    def process_frame(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> dict:
        """
        Process a frame pair for visual odometry
        
        Args:
            color_frame: RGB/BGR image
            depth_frame: Depth image
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        # Initialize default camera matrix if not provided
        if self.camera_matrix is None:
            self.camera_matrix = self.get_default_camera_matrix()
        
        # Detect features
        feature_results = self.feature_detector.process_frame(color_frame)
        keypoints = feature_results['keypoints']
        descriptors = feature_results['descriptors']
        matches = feature_results['matches']
        
        results = {
            'pose_estimated': False,
            'position': self.trajectory[-1].copy(),
            'rotation': self.rotations[-1].copy(),
            'num_features': len(keypoints) if keypoints else 0,
            'num_matches': len(matches),
            'processing_time': 0.0,
            'inliers': 0,
            'translation_magnitude': 0.0,
            'debug_info': ''
        }
        
        debug_info = f"Features: {len(keypoints) if keypoints else 0}, "
        
        # Skip pose estimation for first frame
        if self.prev_frame is None or len(matches) < self.min_matches:
            self.prev_frame = color_frame.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            
            # Store 3D points from current frame
            if keypoints and depth_frame is not None:
                points_2d = np.array([kp.pt for kp in keypoints], dtype=np.float32)
                self.prev_points_3d = self.depth_to_3d_points(points_2d, depth_frame)
                debug_info += f"3D points: {len(self.prev_points_3d)}, "
            
            debug_info += "First frame"
            results['debug_info'] = debug_info
            results['processing_time'] = time.time() - start_time
            return results
        
        debug_info += f"Matches: {len(matches)}, "
        
        # Get matched points
        if len(matches) >= self.min_matches:
            # Extract matched points
            prev_matched_points = np.array([self.prev_keypoints[m.queryIdx].pt for m in matches], dtype=np.float32)
            curr_matched_points = np.array([keypoints[m.trainIdx].pt for m in matches], dtype=np.float32)
            
            # Get 3D points for previous matched features
            if self.prev_points_3d is not None and len(self.prev_points_3d) > 0:
                # Find 3D points corresponding to matched features
                matched_3d_points = []
                matched_2d_points = []
                
                for i, match in enumerate(matches):
                    prev_idx = match.queryIdx
                    if prev_idx < len(self.prev_points_3d):
                        point_3d = self.prev_points_3d[prev_idx]
                        if len(point_3d) == 3 and point_3d[2] > 0.1:  # Valid 3D point (min 10cm depth)
                            matched_3d_points.append(point_3d)
                            matched_2d_points.append(curr_matched_points[i])
                
                debug_info += f"Valid 3D-2D pairs: {len(matched_3d_points)}, "
                
                if len(matched_3d_points) >= max(self.min_matches, self.min_inliers_required):
                    matched_3d_points = np.array(matched_3d_points, dtype=np.float32)
                    matched_2d_points = np.array(matched_2d_points, dtype=np.float32)
                    
                    # Estimate pose using PnP
                    success, rvec, tvec = self.estimate_pose_pnp(matched_3d_points, matched_2d_points)
                    
                    if success and rvec is not None and tvec is not None:
                        # Convert rotation vector to matrix
                        R, _ = cv2.Rodrigues(rvec)
                        
                        # Calculate movement metrics
                        translation_magnitude = np.linalg.norm(tvec)
                        rotation_magnitude = np.linalg.norm(rvec)
                        
                        # ENHANCED MOVEMENT FILTERING
                        
                        # Step 1: Reject unreasonable movements first
                        if translation_magnitude > 2.0:  # Reject huge jumps (>2m per frame)
                            debug_info += f"Rejecting unreasonable translation: {translation_magnitude:.3f}m"
                            self.consecutive_small_movements += 1
                            if self.consecutive_small_movements > self.max_consecutive_small_movements:
                                self.is_stationary = True
                            return results
                        
                        # Step 2: Check if movement is significant enough
                        is_significant_movement = (
                            translation_magnitude >= self.min_translation_threshold or 
                            rotation_magnitude >= self.min_rotation_threshold
                        )
                        
                        # Step 3: Compare with position history for additional validation
                        current_position = self.trajectory[-1] if self.trajectory else np.zeros(3)
                        new_position = current_position + tvec.reshape(3)
                        distance_from_last_significant = np.linalg.norm(
                            new_position - self.last_significant_position
                        )
                        
                        # Step 4: Track movement history for averaging
                        self.movement_history.append(translation_magnitude)
                        if len(self.movement_history) > 10:
                            self.movement_history.pop(0)
                        
                        # Step 5: Calculate average recent movement
                        avg_recent_movement = np.mean(self.movement_history)
                        
                        # Step 6: Enhanced stationary detection
                        if (not is_significant_movement or 
                            distance_from_last_significant < self.stationary_threshold or
                            avg_recent_movement < 0.02):  # Very small average movement
                            
                            self.consecutive_small_movements += 1
                            self.stationary_count += 1
                            
                            # Force stationary if too many small movements
                            if self.consecutive_small_movements > self.max_consecutive_small_movements:
                                self.is_stationary = True
                                debug_info += f"Forced stationary: {self.consecutive_small_movements} consecutive small movements"
                            else:
                                debug_info += f"Movement too small - trans: {translation_magnitude:.4f}m, rot: {rotation_magnitude:.4f}rad, dist_from_last: {distance_from_last_significant:.4f}m, avg: {avg_recent_movement:.4f}m"
                            
                            # Don't update trajectory for small movements
                            results['pose_estimated'] = False
                            return results
                        
                        # Step 7: Only accept if we have significant movement
                        if translation_magnitude >= self.min_translation_threshold and len(matched_3d_points) >= self.min_inliers_required:
                            
                            # Reset consecutive counter
                            self.consecutive_small_movements = 0
                            self.is_stationary = False
                            self.stationary_count = 0
                            
                            # Store trajectory
                            self.trajectory.append(new_position.copy())
                            self.rotations.append(R.copy())
                            self.last_significant_position = new_position.copy()
                            
                            # Update current pose matrix
                            self.current_pose[:3, :3] = R
                            self.current_pose[:3, 3] = new_position
                            
                            # Update results
                            results.update({
                                'pose_estimated': True,
                                'position': new_position,
                                'rotation': R,
                                'inliers': len(matched_3d_points),
                                'translation_magnitude': translation_magnitude
                            })
                            
                            self.pose_estimated = True
                            debug_info += f"Significant movement - trans: {translation_magnitude:.4f}m, rot: {rotation_magnitude:.4f}rad, inliers: {len(matched_3d_points)}"
                        else:
                            debug_info += f"Insufficient movement or inliers: trans={translation_magnitude:.4f}m, inliers={len(matched_3d_points)}"
                    else:
                        debug_info += "PnP failed or returned invalid results"
                else:
                    debug_info += f"Insufficient 3D-2D pairs: {len(matched_3d_points)} < {max(self.min_matches, self.min_inliers_required)}"
            else:
                debug_info += "No previous 3D points"
        else:
            debug_info += f"Insufficient matches: {len(matches)}<{self.min_matches}"
        
        # Update previous frame data
        self.prev_frame = color_frame.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        # Store 3D points from current frame
        if keypoints and depth_frame is not None:
            points_2d = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            self.prev_points_3d = self.depth_to_3d_points(points_2d, depth_frame)
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.match_counts.append(len(matches))
        
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
            self.match_counts.pop(0)
        
        results['processing_time'] = processing_time
        results['debug_info'] = debug_info
        
        # Print debug info occasionally
        if len(self.processing_times) % 30 == 0:  # Every ~1 second
            print(f"VO Debug: {debug_info}")
        
        return results
    
    def get_current_position(self) -> np.ndarray:
        """Get current camera position"""
        return self.trajectory[-1].copy()
    
    def get_trajectory(self) -> np.ndarray:
        """Get full trajectory as numpy array"""
        return np.array(self.trajectory)
    
    def get_distance_traveled(self) -> float:
        """Calculate total distance traveled"""
        if len(self.trajectory) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(self.trajectory)):
            distance = np.linalg.norm(self.trajectory[i] - self.trajectory[i-1])
            total_distance += distance
        
        return total_distance
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        stats = {
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0,
            'avg_matches': np.mean(self.match_counts) if self.match_counts else 0.0,
            'total_frames': len(self.processing_times),
            'trajectory_length': len(self.trajectory),
            'distance_traveled': self.get_distance_traveled(),
            'pose_estimation_active': self.pose_estimated,
            'is_stationary': self.is_stationary,
            'stationary_count': self.stationary_count,
            'recent_movement_avg': np.mean(self.movement_history) if self.movement_history else 0.0
        }
        
        return stats
    
    def reset(self):
        """Reset visual odometry state"""
        self.current_pose = np.eye(4)
        self.trajectory = [np.array([0.0, 0.0, 0.0])]
        self.rotations = [np.eye(3)]
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points_3d = None
        self.processing_times = []
        self.match_counts = []
        self.pose_estimated = False
        
        # Reset stationary detection
        self.is_stationary = True
        self.stationary_count = 0
        self.last_significant_position = np.array([0.0, 0.0, 0.0])
        self.movement_history = []
        
        self.feature_detector.reset()
        print("Visual Odometry reset with stationary detection")

# Test function
def test_visual_odometry():
    """Test visual odometry with sample data"""
    # Create test camera matrix
    camera_matrix = np.array([
        [615.0, 0, 320.0],
        [0, 615.0, 240.0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    vo = VisualOdometry(camera_matrix)
    
    # Create test frames
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    depth1 = np.ones((480, 640), dtype=np.uint16) * 1000  # 1 meter depth
    
    # Add some features
    cv2.rectangle(frame1, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.circle(frame1, (400, 300), 50, (128, 128, 128), -1)
    
    results = vo.process_frame(frame1, depth1)
    print(f"Test frame processed: {results['num_features']} features detected")
    
    # Print statistics
    stats = vo.get_performance_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test_visual_odometry()