"""
Precise Pose Estimator with Enhanced Distance and Direction Tracking
Author: Mr-Parth24
Date: 2025-06-13
Time: 20:47:06 UTC
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
from dataclasses import dataclass
from collections import deque
import math

@dataclass
class PrecisePoseResult:
    success: bool
    position_3d: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    transformation_matrix: np.ndarray
    total_distance: float
    displacement_from_start: float
    current_speed: float
    direction_angle: float
    confidence: float
    num_matches: int
    num_inliers: int
    reprojection_error: float
    method_used: str

class PrecisePoseEstimator:
    """Precise pose estimation for accurate distance and direction tracking"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Pose tracking state
        self.current_pose = np.eye(4)
        self.start_pose = np.eye(4)
        self.pose_history = deque(maxlen=100)
        
        # Distance tracking
        self.total_distance = 0.0
        self.distance_segments = []
        
        # Direction and speed tracking
        self.velocity_history = deque(maxlen=20)
        self.position_history = deque(maxlen=50)
        
        # Estimation parameters
        self.min_matches = 15
        self.ransac_threshold = 2.0
        self.ransac_confidence = 0.99
        
        # Motion validation
        self.max_translation_per_frame = 0.5  # meters
        self.max_rotation_per_frame = 0.3  # radians
        
        # Scale estimation
        self.scale_estimates = deque(maxlen=15)
        self.current_scale = 1.0
        
        self.logger.info("Precise pose estimator initialized")
    
    def estimate_precise_pose(self, keypoints_prev, descriptors_prev, keypoints_curr, 
                            descriptors_curr, depth_frame_prev, camera_matrix) -> PrecisePoseResult:
        """Estimate precise pose with enhanced accuracy"""
        try:
            # Match features
            matches = self._match_features_robust(descriptors_prev, descriptors_curr)
            
            if len(matches) < self.min_matches:
                return self._create_failed_result("Insufficient matches")
            
            # Extract 2D-3D correspondences
            points_3d, points_2d, valid_matches = self._extract_correspondences(
                keypoints_prev, keypoints_curr, matches, depth_frame_prev, camera_matrix
            )
            
            if len(points_3d) < 8:
                return self._create_failed_result("Insufficient 3D points")
            
            # Solve PnP with RANSAC
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                np.array(points_3d, dtype=np.float32),
                np.array(points_2d, dtype=np.float32),
                camera_matrix,
                np.zeros((4, 1)),
                confidence=self.ransac_confidence,
                reprojectionError=self.ransac_threshold,
                flags=cv2.SOLVEPNP_EPNP
            )
            
            if not success or inliers is None or len(inliers) < 6:
                return self._create_failed_result("PnP estimation failed")
            
            # Convert to rotation matrix and translation
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.flatten()
            
            # Validate motion
            if not self._validate_pose_change(R, t):
                return self._create_failed_result("Invalid pose change")
            
            # Calculate reprojection error
            reprojection_error = self._calculate_reprojection_error(
                points_3d, points_2d, rvec, tvec, camera_matrix, inliers
            )
            
            # Update pose
            transformation = np.eye(4)
            transformation[:3, :3] = R
            transformation[:3, 3] = t
            
            self.current_pose = self.current_pose @ transformation
            
            # Calculate metrics
            current_position = self.current_pose[:3, 3]
            distance_moved = np.linalg.norm(t)
            self.total_distance += distance_moved
            
            # Update history
            self.pose_history.append(self.current_pose.copy())
            self.position_history.append(current_position.copy())
            self.velocity_history.append(t)
            
            # Calculate derived metrics
            displacement_from_start = self._calculate_displacement_from_start()
            current_speed = self._calculate_current_speed()
            direction_angle = self._calculate_direction_angle()
            confidence = self._calculate_confidence(len(inliers), len(matches), reprojection_error)
            
            return PrecisePoseResult(
                success=True,
                position_3d=current_position,
                rotation_matrix=R,
                translation_vector=t,
                transformation_matrix=transformation,
                total_distance=self.total_distance,
                displacement_from_start=displacement_from_start,
                current_speed=current_speed,
                direction_angle=direction_angle,
                confidence=confidence,
                num_matches=len(matches),
                num_inliers=len(inliers),
                reprojection_error=reprojection_error,
                method_used="PnP_RANSAC"
            )
            
        except Exception as e:
            self.logger.error(f"Precise pose estimation error: {e}")
            return self._create_failed_result(str(e))
    
    def _match_features_robust(self, desc1, desc2):
        """Robust feature matching with outlier rejection"""
        try:
            # Use BFMatcher with cross-check
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(desc1, desc2)
            
            # Sort by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Filter by distance threshold
            if matches:
                distances = [m.distance for m in matches]
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                threshold = mean_dist + 1.5 * std_dist
                
                filtered_matches = [m for m in matches if m.distance <= threshold]
                return filtered_matches[:100]  # Limit number of matches
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Feature matching error: {e}")
            return []
    
    def _extract_correspondences(self, kp_prev, kp_curr, matches, depth_frame, camera_matrix):
        """Extract 2D-3D correspondences from matches"""
        points_3d = []
        points_2d = []
        valid_matches = []
        
        try:
            for match in matches:
                # Previous frame keypoint (3D)
                kp_prev_pt = kp_prev[match.queryIdx].pt
                u_prev, v_prev = int(kp_prev_pt[0]), int(kp_prev_pt[1])
                
                # Current frame keypoint (2D)
                kp_curr_pt = kp_curr[match.trainIdx].pt
                
                # Get depth value
                if (0 <= u_prev < depth_frame.shape[1] and 
                    0 <= v_prev < depth_frame.shape[0]):
                    
                    depth = depth_frame[v_prev, u_prev] / 1000.0  # Convert to meters
                    
                    if 0.1 < depth < 10.0:  # Valid depth range
                        # Convert to 3D point
                        x = (u_prev - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
                        y = (v_prev - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
                        z = depth
                        
                        points_3d.append([x, y, z])
                        points_2d.append([kp_curr_pt[0], kp_curr_pt[1]])
                        valid_matches.append(match)
            
            return points_3d, points_2d, valid_matches
            
        except Exception as e:
            self.logger.error(f"Correspondence extraction error: {e}")
            return [], [], []
    
    def _validate_pose_change(self, R, t):
        """Validate pose change for reasonableness"""
        try:
            # Check translation magnitude
            translation_norm = np.linalg.norm(t)
            if translation_norm > self.max_translation_per_frame:
                self.logger.warning(f"Large translation: {translation_norm:.3f}m")
                return False
            
            # Check rotation magnitude
            rotation_angle = np.linalg.norm(cv2.Rodrigues(R)[0])
            if rotation_angle > self.max_rotation_per_frame:
                self.logger.warning(f"Large rotation: {rotation_angle:.3f}rad")
                return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Pose validation error: {e}")
            return True
    
    def _calculate_reprojection_error(self, points_3d, points_2d, rvec, tvec, 
                                    camera_matrix, inliers):
        """Calculate reprojection error"""
        try:
            # Project 3D points to 2D
            projected_points, _ = cv2.projectPoints(
                np.array([points_3d[i] for i in inliers.flatten()]),
                rvec, tvec, camera_matrix, np.zeros((4, 1))
            )
            
            # Calculate errors
            errors = []
            for i, inlier_idx in enumerate(inliers.flatten()):
                observed = np.array(points_2d[inlier_idx])
                projected = projected_points[i][0]
                error = np.linalg.norm(observed - projected)
                errors.append(error)
            
            return np.mean(errors) if errors else 0.0
            
        except Exception as e:
            self.logger.warning(f"Reprojection error calculation failed: {e}")
            return 0.0
    
    def _calculate_displacement_from_start(self):
        """Calculate straight-line displacement from start"""
        current_position = self.current_pose[:3, 3]
        start_position = self.start_pose[:3, 3]
        return np.linalg.norm(current_position - start_position)
    
    def _calculate_current_speed(self):
        """Calculate current speed in m/s"""
        try:
            if len(self.velocity_history) < 2:
                return 0.0
            
            # Use recent velocities
            recent_velocities = list(self.velocity_history)[-5:]
            recent_distances = [np.linalg.norm(v) for v in recent_velocities]
            
            # Assume 30 FPS
            fps = 30.0
            avg_distance = np.mean(recent_distances)
            speed = avg_distance * fps
            
            return speed
            
        except Exception as e:
            self.logger.warning(f"Speed calculation error: {e}")
            return 0.0
    
    def _calculate_direction_angle(self):
        """Calculate direction angle from start position"""
        try:
            current_position = self.current_pose[:3, 3]
            start_position = self.start_pose[:3, 3]
            displacement = current_position - start_position
            
            # Calculate angle in XZ plane (top-down view)
            angle_rad = math.atan2(displacement[2], displacement[0])
            angle_deg = math.degrees(angle_rad)
            
            # Normalize to 0-360
            if angle_deg < 0:
                angle_deg += 360
                
            return angle_deg
            
        except Exception as e:
            self.logger.warning(f"Direction calculation error: {e}")
            return 0.0
    
    def _calculate_confidence(self, num_inliers, num_matches, reprojection_error):
        """Calculate pose estimation confidence"""
        try:
            # Inlier ratio
            inlier_ratio = num_inliers / max(1, num_matches)
            
            # Reprojection error score
            error_score = 1.0 / (1.0 + reprojection_error)
            
            # Match count score
            match_score = min(num_matches / 50.0, 1.0)
            
            # Combined confidence
            confidence = 0.5 * inlier_ratio + 0.3 * error_score + 0.2 * match_score
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation error: {e}")
            return 0.0
    
    def _create_failed_result(self, error_msg):
        """Create failed pose estimation result"""
        return PrecisePoseResult(
            success=False,
            position_3d=np.zeros(3),
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros(3),
            transformation_matrix=np.eye(4),
            total_distance=self.total_distance,
            displacement_from_start=self._calculate_displacement_from_start(),
            current_speed=0.0,
            direction_angle=0.0,
            confidence=0.0,
            num_matches=0,
            num_inliers=0,
            reprojection_error=0.0,
            method_used="FAILED",
        )
    
    def reset(self):
        """Reset pose estimator"""
        self.current_pose = np.eye(4)
        self.start_pose = np.eye(4)
        self.total_distance = 0.0
        
        self.pose_history.clear()
        self.position_history.clear()
        self.velocity_history.clear()
        self.distance_segments.clear()
        self.scale_estimates.clear()
        
        self.current_scale = 1.0
        
        self.logger.info("Precise pose estimator reset")