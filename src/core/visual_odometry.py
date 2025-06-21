"""
Visual Odometry for RealSense D435i
Estimates camera pose using visual features without IMU
FIXED: Enhanced for precise distance tracking with relaxed validation
Date: 2025-06-21 01:31:43 UTC
User: Mr-Parth24
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import time
from .feature_detector import FeatureDetector
from ..utils.config_manager import get_config_manager

class VisualOdometry:
    """
    FIXED: Visual Odometry implementation for accurate distance tracking
    Optimized for precise distance measurement and path tracking without IMU
    """
    
    def __init__(self, camera_matrix: np.ndarray = None, dist_coeffs: np.ndarray = None):
        """
        Initialize FIXED Visual Odometry
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        # Load configuration
        self.config_manager = get_config_manager()
        
        # ✅ FIXED: Enhanced camera parameters for D435i
        if camera_matrix is not None:
            self.camera_matrix = camera_matrix
        else:
            self.camera_matrix = self.get_optimized_camera_matrix()
            
        if dist_coeffs is not None:
            self.dist_coeffs = dist_coeffs
        else:
            self.dist_coeffs = self.config_manager.get_distortion_coefficients()
        
        # ✅ FIXED: Enhanced feature detector with relaxed parameters
        max_features = self.config_manager.get_slam_parameter('max_features', 1500)
        self.feature_detector = FeatureDetector(max_features=max_features)
        
        # Pose tracking with enhanced precision
        self.current_pose = np.eye(4, dtype=np.float64)  # Use double precision
        self.trajectory = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
        self.rotations = [np.eye(3, dtype=np.float64)]
        
        # Previous frame data
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points_3d = None
        
        # ✅ FIXED: Much more lenient RANSAC parameters
        self.ransac_threshold = self.config_manager.get_slam_parameter('ransac_threshold', 3.0)  # INCREASED from 2.0
        self.ransac_iterations = 5000  # INCREASED from 2000
        self.min_matches = self.config_manager.get_slam_parameter('min_matches', 4)  # REDUCED from 8
        self.max_translation = self.config_manager.get_slam_parameter('max_translation_per_frame', 5.0)  # INCREASED
        
        # ✅ FIXED: Very permissive movement filtering for continuous tracking
        self.min_translation_threshold = 0.001  # 1mm minimum (was 1cm)
        self.min_rotation_threshold = 0.01     # ~0.6 degrees (was 3 degrees)
        self.stationary_threshold = 0.002      # 2mm stationary threshold
        self.min_inliers_required = 4          # REDUCED from 12
        
        # Enhanced distance tracking
        self.precise_distance = 0.0
        self.distance_validation_window = []
        self.scale_estimates = []
        self.baseline_scale_factor = 1.0  # Stereo baseline scale correction
        
        # ✅ FIXED: Very permissive movement validation
        self.movement_history = []
        self.valid_movements = []
        self.consecutive_small_movements = 0
        self.max_consecutive_small_movements = 10  # INCREASED tolerance
        
        # Performance tracking
        self.processing_times = []
        self.match_counts = []
        self.pose_estimated = False
        
        # ✅ ADDED: Debug counters
        self.pose_attempts = 0
        self.pose_successes = 0
        self.movement_rejections = 0
        self.movement_acceptances = 0
        
        print(f"✅ FIXED Visual Odometry initialized:")
        print(f"  - Precision tracking: 1mm minimum movement (was 1cm)")
        print(f"  - RANSAC iterations: {self.ransac_iterations} (increased)")
        print(f"  - Min inliers: {self.min_inliers_required} (reduced from 12)")
        print(f"  - Movement validation: VERY PERMISSIVE")
    
    def get_optimized_camera_matrix(self) -> np.ndarray:
        """Get optimized camera matrix for D435i with enhanced precision"""
        # Optimized intrinsics based on factory calibration
        fx = 615.8  # Slightly adjusted for better accuracy
        fy = 615.8  # Matched focal lengths
        cx = 319.5  # Principal point
        cy = 239.5  # Principal point
        
        matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)  # Double precision
        
        return matrix
    
    def enhanced_depth_to_3d_points(self, points_2d: np.ndarray, depth_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ✅ FIXED: Much more lenient 3D point generation for real-world use
        """
        if points_2d.size == 0:
            return np.array([]), np.array([])
        
        points_3d = []
        valid_indices = []
        total_count = len(points_2d.reshape(-1, 2))
        
        for i, point in enumerate(points_2d.reshape(-1, 2)):
            x, y = int(point[0]), int(point[1])
            
            # ✅ FIXED: Much wider bounds checking (was 5 pixel margin, now 3)
            if (3 <= x < depth_frame.shape[1] - 3 and 3 <= y < depth_frame.shape[0] - 3):
                # ✅ FIXED: More robust depth sampling with smaller patch
                depth_patch = depth_frame[y-1:y+2, x-1:x+2]
                valid_depths = depth_patch[depth_patch > 50]  # REDUCED from 100 to 50
                
                # ✅ FIXED: Reduced minimum valid depths requirement
                if len(valid_depths) >= 2:  # REDUCED from 3 to 2
                    # Use median depth for robustness
                    depth = np.median(valid_depths)
                    
                    # ✅ FIXED: Much wider depth range for indoor/outdoor use
                    if 50 <= depth <= 12000:  # EXPANDED from 200-8000 to 50-12000 (5cm to 12m)
                        z = depth / 1000.0  # Convert to meters
                        
                        # Back-project to 3D with double precision
                        x_3d = (x - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                        y_3d = (y - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
                        
                        # ✅ FIXED: Much more lenient sanity check for real-world scenes
                        if abs(x_3d) <= 15.0 and abs(y_3d) <= 15.0 and 0.05 <= z <= 12.0:  # EXPANDED limits
                            points_3d.append([x_3d, y_3d, z])
                            valid_indices.append(i)
        
        result = np.array(points_3d, dtype=np.float64)
        valid_indices_array = np.array(valid_indices)
        
        # ✅ ADDED: Debug output every 50 frames
        if len(self.processing_times) % 50 == 0 and total_count > 0:
            valid_count = len(result)
            print(f"🎯 3D Points: {valid_count}/{total_count} valid ({100*valid_count/total_count:.1f}%) - range: 5cm-12m")
        
        return result, valid_indices_array
    
    def enhanced_pose_estimation(self, points_3d: np.ndarray, points_2d: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, int]:
        """
        ✅ FIXED: Much more robust pose estimation for real-world use
        """
        self.pose_attempts += 1
        
        # ✅ FIXED: Reduced minimum points requirement dramatically
        if len(points_3d) < 4 or len(points_2d) < 4:  # REDUCED from 8 to 4
            print(f"❌ Insufficient points: 3D={len(points_3d)}, 2D={len(points_2d)} (need 4+)")
            return False, None, None, 0
        
        try:
            # Ensure correct data types and shapes
            points_3d = np.array(points_3d, dtype=np.float64)
            points_2d = np.array(points_2d, dtype=np.float64)
            
            # Reshape for OpenCV
            if points_3d.ndim == 2:
                points_3d = points_3d.reshape(-1, 1, 3)
            if points_2d.ndim == 2:
                points_2d = points_2d.reshape(-1, 1, 2)
            
            # ✅ FIXED: Enhanced solver cascade with better order
            solvers = [
                (cv2.SOLVEPNP_EPNP, "EPNP"),                # Fast and robust
                (cv2.SOLVEPNP_SQPNP, "SQPNP"),              # Most accurate
                (cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"),      # Refinement
                (cv2.SOLVEPNP_P3P, "P3P")                   # Minimal case
            ]
            
            best_solution = None
            max_inliers = 0
            
            for method, name in solvers:
                try:
                    # ✅ FIXED: Much more aggressive RANSAC
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        points_3d.astype(np.float32),
                        points_2d.astype(np.float32),
                        self.camera_matrix.astype(np.float32),
                        self.dist_coeffs.astype(np.float32),
                        iterationsCount=self.ransac_iterations,  # 5000 iterations
                        reprojectionError=self.ransac_threshold,  # 3.0 pixels (relaxed)
                        confidence=0.99,
                        flags=method
                    )
                    
                    if success and inliers is not None:
                        num_inliers = len(inliers)
                        translation_norm = np.linalg.norm(tvec)
                        rotation_norm = np.linalg.norm(rvec)
                        
                        # ✅ FIXED: Much more lenient validation criteria
                        if (num_inliers >= self.min_inliers_required and  # 4 inliers minimum
                            0.001 <= translation_norm <= self.max_translation and  # 1mm to 5m
                            rotation_norm <= 2.0 and    # ~114 degrees max (was 57)
                            num_inliers > max_inliers):
                            
                            max_inliers = num_inliers
                            best_solution = (True, rvec, tvec, num_inliers)
                            
                            print(f"✅ {name}: {num_inliers}/{len(points_3d)} inliers, "
                                  f"trans: {translation_norm:.4f}m, rot: {rotation_norm:.3f}rad")
                        else:
                            print(f"⚠️  {name}: {num_inliers} inliers rejected - "
                                  f"trans: {translation_norm:.4f}m, rot: {rotation_norm:.3f}rad")
                        
                except Exception as e:
                    print(f"❌ Solver {name} failed: {e}")
                    continue
            
            if best_solution:
                self.pose_successes += 1
                success_rate = (self.pose_successes / self.pose_attempts) * 100
                print(f"🎯 Pose estimation SUCCESS (rate: {success_rate:.1f}%)")
                return best_solution
            else:
                print(f"❌ All pose estimation methods failed")
                return False, None, None, 0
                
        except Exception as e:
            print(f"❌ Enhanced pose estimation error: {e}")
            return False, None, None, 0
    
    def validate_movement(self, tvec: np.ndarray, rvec: np.ndarray) -> bool:
        """
        ✅ FIXED: Much more permissive movement validation for continuous tracking
        """
        translation_magnitude = np.linalg.norm(tvec)
        rotation_magnitude = np.linalg.norm(rvec)
        
        # ✅ FIXED: Very permissive movement thresholds
        min_movement = 0.001  # 1mm minimum (was 1cm)
        max_movement = 5.0    # 5m maximum per frame (was 3m)
        max_rotation = 2.0    # ~114 degrees maximum (was ~57)
        
        # Basic sanity check only
        is_reasonable = (min_movement <= translation_magnitude <= max_movement and 
                        rotation_magnitude <= max_rotation)
        
        if not is_reasonable:
            self.movement_rejections += 1
            print(f"❌ Movement rejected: trans={translation_magnitude:.4f}m "
                  f"(range: {min_movement:.3f}-{max_movement:.1f}), "
                  f"rot={rotation_magnitude:.4f}rad (max: {max_rotation:.1f})")
            return False
        
        # ✅ SIMPLIFIED: Remove complex consistency checks that block valid movement
        # Just track movement history for statistics
        self.movement_history.append(translation_magnitude)
        if len(self.movement_history) > 50:  # Keep recent history
            self.movement_history.pop(0)
        
        # Accept the movement
        self.valid_movements.append(translation_magnitude)
        if len(self.valid_movements) > 100:
            self.valid_movements.pop(0)
        
        self.movement_acceptances += 1
        acceptance_rate = (self.movement_acceptances / (self.movement_acceptances + self.movement_rejections)) * 100
        
        print(f"✅ Movement accepted: {translation_magnitude:.4f}m, "
              f"rot: {rotation_magnitude:.4f}rad (acceptance rate: {acceptance_rate:.1f}%)")
        
        return True
    
    def process_frame(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> dict:
        """
        ✅ FIXED: Enhanced frame processing for reliable distance tracking
        """
        start_time = time.time()
        
        # Detect features using enhanced robust detection
        feature_results = self.feature_detector.process_frame(color_frame)
        keypoints = feature_results['keypoints']
        descriptors = feature_results['descriptors']
        matches = feature_results['matches']
        
        # Initialize results
        results = {
            'pose_estimated': False,
            'position': self.trajectory[-1].copy(),
            'rotation': self.rotations[-1].copy(),
            'num_features': len(keypoints) if keypoints else 0,
            'num_matches': len(matches),
            'processing_time': 0.0,
            'inliers': 0,
            'translation_magnitude': 0.0,
            'distance_traveled': self.precise_distance,
            'debug_info': '',
            'pose_attempts': self.pose_attempts,
            'pose_successes': self.pose_successes,
            'movement_acceptances': self.movement_acceptances,
            'movement_rejections': self.movement_rejections
        }
        
        # First frame initialization
        if self.prev_frame is None or len(matches) < self.min_matches:
            self.prev_frame = color_frame.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            
            if keypoints and depth_frame is not None:
                points_2d = np.array([kp.pt for kp in keypoints], dtype=np.float64)
                self.prev_points_3d, _ = self.enhanced_depth_to_3d_points(points_2d, depth_frame)
                results['debug_info'] = f"INITIALIZED with {len(self.prev_points_3d)} 3D points from {len(keypoints)} features"
                print(f"🚀 SLAM INITIALIZED: {len(self.prev_points_3d)} 3D points from {len(keypoints)} features")
            else:
                results['debug_info'] = "WAITING for features or depth data"
            
            results['processing_time'] = time.time() - start_time
            return results
        
        # ✅ FIXED: Process matches for pose estimation with reduced requirements
        if len(matches) >= self.min_matches and self.prev_points_3d is not None:
            # Extract matched points
            matched_3d_points = []
            matched_2d_points = []
            
            for match in matches:
                prev_idx = match.queryIdx
                if prev_idx < len(self.prev_points_3d):
                    point_3d = self.prev_points_3d[prev_idx]
                    if len(point_3d) == 3 and point_3d[2] > 0.05:  # Valid depth (5cm minimum)
                        matched_3d_points.append(point_3d)
                        matched_2d_points.append(keypoints[match.trainIdx].pt)
            
            # ✅ FIXED: Reduced minimum requirement for matched points
            if len(matched_3d_points) >= self.min_inliers_required:  # 4 minimum (was 12)
                matched_3d_points = np.array(matched_3d_points, dtype=np.float64)
                matched_2d_points = np.array(matched_2d_points, dtype=np.float64)
                
                # Enhanced pose estimation
                success, rvec, tvec, num_inliers = self.enhanced_pose_estimation(
                    matched_3d_points, matched_2d_points
                )
                
                if success and self.validate_movement(tvec, rvec):
                    # Update pose with enhanced precision
                    R, _ = cv2.Rodrigues(rvec)
                    current_position = self.trajectory[-1] + tvec.ravel()
                    
                    # Enhanced distance calculation
                    movement_distance = np.linalg.norm(tvec)
                    self.precise_distance += movement_distance
                    
                    # Update trajectory
                    self.trajectory.append(current_position.copy())
                    self.rotations.append(R.copy())
                    
                    # Update pose matrix
                    self.current_pose[:3, :3] = R
                    self.current_pose[:3, 3] = current_position
                    
                    results.update({
                        'pose_estimated': True,
                        'position': current_position,
                        'rotation': R,
                        'num_matches': len(matches),
                        'inliers': num_inliers,
                        'translation_magnitude': movement_distance,
                        'distance_traveled': self.precise_distance,
                        'debug_info': f"TRACKING: {len(matches)} matches → {num_inliers} inliers → "
                                    f"movement: {movement_distance:.4f}m → total: {self.precise_distance:.3f}m"
                    })
                    
                    self.pose_estimated = True
                else:
                    results['debug_info'] = f"POSE/MOVEMENT validation failed - matches: {len(matches)}, " \
                                          f"3D-2D pairs: {len(matched_3d_points)}"
            else:
                results['debug_info'] = f"INSUFFICIENT 3D-2D pairs: {len(matched_3d_points)} < {self.min_inliers_required}"
        else:
            results['debug_info'] = f"INSUFFICIENT matches: {len(matches)} < {self.min_matches}"
        
        # Update for next frame
        self.prev_frame = color_frame.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        if keypoints and depth_frame is not None:
            points_2d = np.array([kp.pt for kp in keypoints], dtype=np.float64)
            self.prev_points_3d, _ = self.enhanced_depth_to_3d_points(points_2d, depth_frame)
        
        # Performance tracking
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.match_counts.append(len(matches))
        
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
            self.match_counts.pop(0)
        
        results['processing_time'] = processing_time
        return results
    
    def get_debug_info(self) -> dict:
        """✅ NEW: Get detailed debug information for troubleshooting"""
        return {
            'features_detected': len(self.prev_keypoints) if self.prev_keypoints else 0,
            'points_3d_generated': len(self.prev_points_3d) if self.prev_points_3d is not None else 0,
            'pose_estimation_attempts': self.pose_attempts,
            'pose_estimation_successes': self.pose_successes,
            'pose_success_rate': (self.pose_successes / max(self.pose_attempts, 1)) * 100,
            'movement_validations_passed': self.movement_acceptances,
            'movement_validations_rejected': self.movement_rejections,
            'movement_acceptance_rate': (self.movement_acceptances / max(self.movement_acceptances + self.movement_rejections, 1)) * 100,
            'last_translation_magnitude': self.valid_movements[-1] if self.valid_movements else 0.0,
            'current_thresholds': {
                'min_translation': self.min_translation_threshold,
                'min_inliers': self.min_inliers_required,
                'ransac_threshold': self.ransac_threshold,
                'min_matches': self.min_matches
            }
        }
    
    def get_current_position(self) -> np.ndarray:
        """Get current camera position"""
        return self.trajectory[-1].copy()
    
    def get_trajectory(self) -> np.ndarray:
        """Get full trajectory"""
        return np.array(self.trajectory)
    
    def get_distance_traveled(self) -> float:
        """Get precise distance traveled"""
        return self.precise_distance
    
    def get_performance_stats(self) -> dict:
        """✅ ENHANCED: Get comprehensive performance statistics"""
        debug_info = self.get_debug_info()
        
        stats = {
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0,
            'avg_matches': np.mean(self.match_counts) if self.match_counts else 0.0,
            'total_frames': len(self.processing_times),
            'trajectory_length': len(self.trajectory),
            'precise_distance_traveled': self.precise_distance,
            'pose_estimation_active': self.pose_estimated,
            'valid_movements_count': len(self.valid_movements),
            'movement_accuracy': np.std(self.valid_movements) if self.valid_movements else 0.0,
            'debug_info': debug_info
        }
        return stats
    
    def reset(self):
        """✅ ENHANCED: Reset visual odometry state with debug counters"""
        self.current_pose = np.eye(4, dtype=np.float64)
        self.trajectory = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
        self.rotations = [np.eye(3, dtype=np.float64)]
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points_3d = None
        self.processing_times = []
        self.match_counts = []
        self.pose_estimated = False
        
        # Reset distance tracking
        self.precise_distance = 0.0
        self.movement_history = []
        self.valid_movements = []
        self.consecutive_small_movements = 0
        
        # Reset debug counters
        self.pose_attempts = 0
        self.pose_successes = 0
        self.movement_rejections = 0
        self.movement_acceptances = 0
        
        self.feature_detector.reset()
        print("✅ FIXED Visual Odometry reset - precision tracking reinitialized")
        print(f"  - Movement validation: VERY PERMISSIVE (1mm minimum)")
        print(f"  - Pose estimation: RELAXED (4+ inliers)")
        print(f"  - 3D point generation: EXPANDED range (5cm-12m)")