"""
Advanced Pose Estimation with Multiple Methods and Error Recovery
Author: Mr-Parth24
Date: 2025-06-13
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import logging
from dataclasses import dataclass
from enum import Enum
from scipy.spatial.transform import Rotation
from collections import deque

from feature_processor import FeatureData, MatchResult

class PoseEstimationMethod(Enum):
    PNP_RANSAC = "pnp_ransac"
    ESSENTIAL_MATRIX = "essential_matrix"
    FUNDAMENTAL_MATRIX = "fundamental_matrix"
    ICP = "icp"

@dataclass
class PoseResult:
    success: bool
    position: np.ndarray
    rotation: np.ndarray
    transformation_matrix: np.ndarray
    num_matches: int
    num_inliers: int
    reprojection_error: float
    confidence: float
    total_distance: float
    method_used: PoseEstimationMethod
    error_message: Optional[str] = None

class PoseEstimator:
    """Advanced pose estimation with multiple fallback methods"""
    
    def __init__(self, intrinsics, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Camera parameters
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.cx
        self.cy = intrinsics.cy
        
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.dist_coeffs = intrinsics.distortion
        
        # Pose estimation parameters
        self.min_triangulation_points = config.get('pose', {}).get('min_triangulation_points', 8)
        self.ransac_threshold = config.get('pose', {}).get('ransac_threshold', 1.0)
        self.ransac_confidence = config.get('pose', {}).get('ransac_confidence', 0.99)
        self.max_reprojection_error = config.get('pose', {}).get('max_reprojection_error', 2.0)
        
        # Keyframe management
        self.keyframe_distance_threshold = config.get('pose', {}).get('keyframe_distance_threshold', 0.1)
        self.keyframe_angle_threshold = config.get('pose', {}).get('keyframe_angle_threshold', 0.1)
        
        # State variables
        self.current_pose = np.eye(4)
        self.trajectory = []
        self.keyframes = []
        self.total_distance = 0.0
        
        # Previous frame data
        self.prev_features = None
        self.prev_depth = None
        self.prev_pose = np.eye(4)
        
        # Motion model for prediction
        self.velocity_buffer = deque(maxlen=5)
        self.angular_velocity_buffer = deque(maxlen=5)
        
        # Scale estimation
        self.scale_estimates = deque(maxlen=10)
        self.current_scale = 1.0
        
        # Error recovery
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        self.logger.info("Pose estimator initialized")
    
    def estimate_pose(self, features: FeatureData, depth_image: np.ndarray, 
                     frame_number: int) -> PoseResult:
        """Estimate pose using multiple methods with fallback"""
        try:
            if self.prev_features is None:
                # First frame - initialize
                return self._initialize_first_frame(features, depth_image, frame_number)
            
            # Match features with previous frame
            from feature_processor import FeatureProcessor
            feature_processor = FeatureProcessor(self.config)
            match_result = feature_processor.match_features(self.prev_features, features)
            
            if match_result is None:
                return self._handle_matching_failure(frame_number)
            
            # Try different pose estimation methods
            pose_result = None
            
            # Method 1: PnP with 3D points from depth
            if match_result.match_quality > 0.5:
                pose_result = self._estimate_pose_pnp(match_result, depth_image)
                if pose_result.success:
                    pose_result.method_used = PoseEstimationMethod.PNP_RANSAC
            
            # Method 2: Essential matrix (fallback)
            if pose_result is None or not pose_result.success:
                pose_result = self._estimate_pose_essential_matrix(match_result)
                if pose_result.success:
                    pose_result.method_used = PoseEstimationMethod.ESSENTIAL_MATRIX
            
            # Method 3: Fundamental matrix (last resort)
            if pose_result is None or not pose_result.success:
                pose_result = self._estimate_pose_fundamental_matrix(match_result)
                if pose_result.success:
                    pose_result.method_used = PoseEstimationMethod.FUNDAMENTAL_MATRIX
            
            if pose_result is None or not pose_result.success:
                return self._handle_pose_estimation_failure(frame_number)
            
            # Validate and filter pose
            if not self._validate_pose(pose_result):
                return self._handle_invalid_pose(frame_number)
            
            # Update pose and trajectory
            self._update_pose(pose_result, features, depth_image, frame_number)
            
            # Reset failure counter
            self.consecutive_failures = 0
            
            return pose_result
            
        except Exception as e:
            self.logger.error(f"Error in pose estimation: {e}")
            return self._create_failure_result(str(e))
    
    def _initialize_first_frame(self, features: FeatureData, depth_image: np.ndarray, 
                               frame_number: int) -> PoseResult:
        """Initialize with first frame"""
        self.prev_features = features
        self.prev_depth = depth_image
        self.trajectory.append([0, 0, 0])
        
        # Add as keyframe
        self._add_keyframe(features, depth_image, self.current_pose, frame_number)
        
        return PoseResult(
            success=True,
            position=np.array([0, 0, 0]),
            rotation=np.eye(3),
            transformation_matrix=np.eye(4),
            num_matches=0,
            num_inliers=0,
            reprojection_error=0.0,
            confidence=1.0,
            total_distance=0.0,
            method_used=PoseEstimationMethod.PNP_RANSAC
        )
    
    def _estimate_pose_pnp(self, match_result: MatchResult, depth_image: np.ndarray) -> PoseResult:
        """Estimate pose using PnP with 3D points from depth"""
        try:
            # Get 3D points from previous frame depth
            points_3d = []
            points_2d = []
            
            for match in match_result.inlier_matches:
                # Previous frame 2D point
                pt_prev = match_result.keypoints1[match.queryIdx].pt
                u_prev, v_prev = int(pt_prev[0]), int(pt_prev[1])
                
                # Current frame 2D point
                pt_curr = match_result.keypoints2[match.trainIdx].pt
                
                # Get depth from previous frame
                if (0 <= u_prev < depth_image.shape[1] and 0 <= v_prev < depth_image.shape[0]):
                    depth = self.prev_depth[v_prev, u_prev] / 1000.0  # Convert to meters
                    
                    if 0.1 < depth < 10.0:  # Valid depth range
                        # Convert to 3D point in camera coordinates
                        x = (u_prev - self.cx) * depth / self.fx
                        y = (v_prev - self.cy) * depth / self.fy
                        z = depth
                        
                        points_3d.append([x, y, z])
                        points_2d.append(pt_curr)
            
            if len(points_3d) < self.min_triangulation_points:
                return PoseResult(success=False, position=np.zeros(3), rotation=np.eye(3),
                                transformation_matrix=np.eye(4), num_matches=0, num_inliers=0,
                                reprojection_error=0.0, confidence=0.0, total_distance=0.0,
                                method_used=PoseEstimationMethod.PNP_RANSAC,
                                error_message="Insufficient 3D points")
            
            points_3d = np.array(points_3d, dtype=np.float32)
            points_2d = np.array(points_2d, dtype=np.float32)
            
            # Solve PnP with RANSAC
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d,
                points_2d,
                self.camera_matrix,
                self.dist_coeffs,
                confidence=self.ransac_confidence,
                reprojectionError=self.ransac_threshold,
                flags=cv2.SOLVEPNP_EPNP
            )
            
            if not success or inliers is None or len(inliers) < self.min_triangulation_points:
                return PoseResult(success=False, position=np.zeros(3), rotation=np.eye(3),
                                transformation_matrix=np.eye(4), num_matches=len(points_3d), 
                                num_inliers=0, reprojection_error=0.0, confidence=0.0, 
                                total_distance=0.0, method_used=PoseEstimationMethod.PNP_RANSAC,
                                error_message="PnP RANSAC failed")
            
            # Convert to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            
            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()
            
            # Calculate reprojection error
            projected_points, _ = cv2.projectPoints(
                points_3d[inliers.flatten()], rvec, tvec, 
                self.camera_matrix, self.dist_coeffs
            )
            
            reprojection_error = np.mean([
                np.linalg.norm(points_2d[inliers[i][0]] - projected_points[i][0])
                for i in range(len(inliers))
            ])
            
            # Calculate confidence based on inlier ratio and reprojection error
            inlier_ratio = len(inliers) / len(points_3d)
            error_confidence = 1.0 / (1.0 + reprojection_error)
            confidence = 0.7 * inlier_ratio + 0.3 * error_confidence
            
            return PoseResult(
                success=True,
                position=tvec.flatten(),
                rotation=R,
                transformation_matrix=T,
                num_matches=len(points_3d),
                num_inliers=len(inliers),
                reprojection_error=reprojection_error,
                confidence=confidence,
                total_distance=self.total_distance,
                method_used=PoseEstimationMethod.PNP_RANSAC
            )
            
        except Exception as e:
            self.logger.error(f"PnP estimation failed: {e}")
            return PoseResult(success=False, position=np.zeros(3), rotation=np.eye(3),
                            transformation_matrix=np.eye(4), num_matches=0, num_inliers=0,
                            reprojection_error=0.0, confidence=0.0, total_distance=0.0,
                            method_used=PoseEstimationMethod.PNP_RANSAC, error_message=str(e))
    
    def _estimate_pose_essential_matrix(self, match_result: MatchResult) -> PoseResult:
        """Estimate pose using essential matrix"""
        try:
            if len(match_result.inlier_matches) < 8:
                return PoseResult(success=False, position=np.zeros(3), rotation=np.eye(3),
                                transformation_matrix=np.eye(4), num_matches=0, num_inliers=0,
                                reprojection_error=0.0, confidence=0.0, total_distance=0.0,
                                method_used=PoseEstimationMethod.ESSENTIAL_MATRIX,
                                error_message="Insufficient matches for essential matrix")
            
            # Extract matched points
            pts1 = np.array([match_result.keypoints1[m.queryIdx].pt for m in match_result.inlier_matches])
            pts2 = np.array([match_result.keypoints2[m.trainIdx].pt for m in match_result.inlier_matches])
            
            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(
                pts1, pts2, self.camera_matrix,
                method=cv2.RANSAC,
                prob=self.ransac_confidence,
                threshold=self.ransac_threshold
            )
            
            if E is None or mask is None:
                return PoseResult(success=False, position=np.zeros(3), rotation=np.eye(3),
                                transformation_matrix=np.eye(4), num_matches=len(pts1), 
                                num_inliers=0, reprojection_error=0.0, confidence=0.0,
                                total_distance=0.0, method_used=PoseEstimationMethod.ESSENTIAL_MATRIX,
                                error_message="Essential matrix estimation failed")
            
            # Recover pose from essential matrix
            num_inliers, R, t, mask_pose = cv2.recoverPose(
                E, pts1, pts2, self.camera_matrix, mask=mask
            )
            
            if num_inliers < self.min_triangulation_points:
                return PoseResult(success=False, position=np.zeros(3), rotation=np.eye(3),
                                transformation_matrix=np.eye(4), num_matches=len(pts1),
                                num_inliers=num_inliers, reprojection_error=0.0, confidence=0.0,
                                total_distance=0.0, method_used=PoseEstimationMethod.ESSENTIAL_MATRIX,
                                error_message="Pose recovery failed")
            
            # Scale estimation using depth information
            scale = self._estimate_scale(pts1, pts2, R, t, mask_pose)
            t_scaled = t * scale
            
            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t_scaled.flatten()
            
            # Calculate confidence
            inlier_ratio = num_inliers / len(pts1)
            confidence = min(inlier_ratio * 1.2, 1.0)  # Boost essential matrix confidence
            
            return PoseResult(
                success=True,
                position=t_scaled.flatten(),
                rotation=R,
                transformation_matrix=T,
                num_matches=len(pts1),
                num_inliers=num_inliers,
                reprojection_error=1.0,  # Placeholder
                confidence=confidence,
                total_distance=self.total_distance,
                method_used=PoseEstimationMethod.ESSENTIAL_MATRIX
            )
            
        except Exception as e:
            self.logger.error(f"Essential matrix estimation failed: {e}")
            return PoseResult(success=False, position=np.zeros(3), rotation=np.eye(3),
                            transformation_matrix=np.eye(4), num_matches=0, num_inliers=0,
                            reprojection_error=0.0, confidence=0.0, total_distance=0.0,
                            method_used=PoseEstimationMethod.ESSENTIAL_MATRIX, error_message=str(e))
    
    def _estimate_scale(self, pts1: np.ndarray, pts2: np.ndarray, R: np.ndarray, 
                       t: np.ndarray, mask: np.ndarray) -> float:
        """Estimate scale using depth information"""
        try:
            if self.prev_depth is None:
                return self.current_scale
            
            scales = []
            inlier_indices = np.where(mask.ravel())[0]
            
            for i in inlier_indices[:10]:  # Use first 10 inliers
                pt1 = pts1[i]
                u1, v1 = int(pt1[0]), int(pt1[1])
                
                if (0 <= u1 < self.prev_depth.shape[1] and 0 <= v1 < self.prev_depth.shape[0]):
                    depth = self.prev_depth[v1, u1] / 1000.0
                    
                    if 0.1 < depth < 10.0:
                        # Calculate expected depth change
                        # This is a simplified scale estimation
                        expected_depth_change = np.linalg.norm(t)
                        if expected_depth_change > 0:
                            scale_estimate = depth / (depth + expected_depth_change)
                            scales.append(scale_estimate)
            
            if scales:
                estimated_scale = np.median(scales)
                self.scale_estimates.append(estimated_scale)
                self.current_scale = np.median(list(self.scale_estimates))
                return self.current_scale
            else:
                return self.current_scale
                
        except Exception as e:
            self.logger.warning(f"Scale estimation failed: {e}")
            return self.current_scale
    
    def _estimate_pose_fundamental_matrix(self, match_result: MatchResult) -> PoseResult:
        """Estimate pose using fundamental matrix (least accurate)"""
        try:
            if len(match_result.inlier_matches) < 8:
                return PoseResult(success=False, position=np.zeros(3), rotation=np.eye(3),
                                transformation_matrix=np.eye(4), num_matches=0, num_inliers=0,
                                reprojection_error=0.0, confidence=0.0, total_distance=0.0,
                                method_used=PoseEstimationMethod.FUNDAMENTAL_MATRIX,
                                error_message="Insufficient matches")
            
            # This is a simplified implementation
            # In practice, fundamental matrix alone cannot provide accurate metric pose
            # Use motion model prediction
            predicted_motion = self._predict_motion()
            
            return PoseResult(
                success=True,
                position=predicted_motion[:3, 3],
                rotation=predicted_motion[:3, :3],
                transformation_matrix=predicted_motion,
                num_matches=len(match_result.inlier_matches),
                num_inliers=len(match_result.inlier_matches) // 2,
                reprojection_error=2.0,
                confidence=0.3,  # Low confidence for fundamental matrix
                total_distance=self.total_distance,
                method_used=PoseEstimationMethod.FUNDAMENTAL_MATRIX
            )
            
        except Exception as e:
            self.logger.error(f"Fundamental matrix estimation failed: {e}")
            return PoseResult(success=False, position=np.zeros(3), rotation=np.eye(3),
                            transformation_matrix=np.eye(4), num_matches=0, num_inliers=0,
                            reprojection_error=0.0, confidence=0.0, total_distance=0.0,
                            method_used=PoseEstimationMethod.FUNDAMENTAL_MATRIX, error_message=str(e))
    
    def _predict_motion(self) -> np.ndarray:
        """Predict motion using velocity model"""
        if len(self.velocity_buffer) < 2:
            return np.eye(4)
        
        # Simple constant velocity model
        avg_velocity = np.mean(list(self.velocity_buffer), axis=0)
        avg_angular_velocity = np.mean(list(self.angular_velocity_buffer), axis=0)
        
        # Create predicted transformation
        T_pred = np.eye(4)
        T_pred[:3, 3] = avg_velocity
        
        # Add rotation (simplified)
        if np.linalg.norm(avg_angular_velocity) > 0:
            angle = np.linalg.norm(avg_angular_velocity)
            axis = avg_angular_velocity / angle
            R_pred = Rotation.from_rotvec(axis * angle).as_matrix()
            T_pred[:3, :3] = R_pred
        
        return T_pred
    
    def _validate_pose(self, pose_result: PoseResult) -> bool:
        """Validate pose estimate"""
        # Check for reasonable translation
        translation_norm = np.linalg.norm(pose_result.position)
        if translation_norm > 1.0:  # More than 1 meter per frame is suspicious
            self.logger.warning(f"Large translation detected: {translation_norm:.3f}m")
            return False
        
        # Check reprojection error
        if pose_result.reprojection_error > self.max_reprojection_error:
            self.logger.warning(f"High reprojection error: {pose_result.reprojection_error:.3f}")
            return False
        
        # Check confidence
        if pose_result.confidence < 0.2:
            self.logger.warning(f"Low confidence: {pose_result.confidence:.3f}")
            return False
        
        return True
    
    def _update_pose(self, pose_result: PoseResult, features: FeatureData, 
                    depth_image: np.ndarray, frame_number: int):
        """Update pose and trajectory"""
        # Update current pose
        self.current_pose = self.current_pose @ pose_result.transformation_matrix
        
        # Update trajectory
        current_position = self.current_pose[:3, 3]
        self.trajectory.append(current_position.tolist())
        
        # Update total distance
        if len(self.trajectory) > 1:
            prev_pos = np.array(self.trajectory[-2])
            curr_pos = np.array(self.trajectory[-1])
            distance_increment = np.linalg.norm(curr_pos - prev_pos)
            self.total_distance += distance_increment
        
        pose_result.total_distance = self.total_distance
        
        # Update motion model
        velocity = pose_result.position
        self.velocity_buffer.append(velocity)
        
        # Extract angular velocity (simplified)
        r = Rotation.from_matrix(pose_result.rotation)
        angular_velocity = r.as_rotvec()
        self.angular_velocity_buffer.append(angular_velocity)
        
        # Check if we need a new keyframe
        if self._should_create_keyframe(pose_result):
            self._add_keyframe(features, depth_image, self.current_pose, frame_number)
        
        # Update previous frame data
        self.prev_features = features
        self.prev_depth = depth_image
        self.prev_pose = self.current_pose.copy()
    
    def _should_create_keyframe(self, pose_result: PoseResult) -> bool:
        """Determine if a new keyframe should be created"""
        if not self.keyframes:
            return True
        
        last_keyframe_pose = self.keyframes[-1]['pose']
        
        # Check translation distance
        translation_dist = np.linalg.norm(
            self.current_pose[:3, 3] - last_keyframe_pose[:3, 3]
        )
        
        if translation_dist > self.keyframe_distance_threshold:
            return True
        
        # Check rotation angle
        R_rel = self.current_pose[:3, :3] @ last_keyframe_pose[:3, :3].T
        angle = np.abs(Rotation.from_matrix(R_rel).as_rotvec())
        angle_norm = np.linalg.norm(angle)
        
        if angle_norm > self.keyframe_angle_threshold:
            return True
        
        return False
    
    def _add_keyframe(self, features: FeatureData, depth_image: np.ndarray, 
                     pose: np.ndarray, frame_number: int):
        """Add a new keyframe"""
        keyframe = {
            'frame_number': frame_number,
            'pose': pose.copy(),
            'features': features,
            'depth': depth_image.copy(),
            'timestamp': frame_number  # Use frame number as timestamp
        }
        
        self.keyframes.append(keyframe)
        self.logger.info(f"Added keyframe {len(self.keyframes)} at frame {frame_number}")
        
        # Limit number of keyframes
        max_keyframes = 50
        if len(self.keyframes) > max_keyframes:
            self.keyframes.pop(0)
    
    def _handle_matching_failure(self, frame_number: int) -> PoseResult:
        """Handle feature matching failure"""
        self.consecutive_failures += 1
        self.logger.warning(f"Feature matching failed at frame {frame_number}")
        
        if self.consecutive_failures < self.max_consecutive_failures:
            # Use motion prediction
            predicted_motion = self._predict_motion()
            return PoseResult(
                success=True,
                position=predicted_motion[:3, 3],
                rotation=predicted_motion[:3, :3],
                transformation_matrix=predicted_motion,
                num_matches=0,
                num_inliers=0,
                reprojection_error=5.0,
                confidence=0.1,
                total_distance=self.total_distance,
                method_used=PoseEstimationMethod.PNP_RANSAC,
                error_message="Using motion prediction"
            )
        else:
            return self._create_failure_result("Too many consecutive matching failures")
    
    def _handle_pose_estimation_failure(self, frame_number: int) -> PoseResult:
        """Handle pose estimation failure"""
        self.consecutive_failures += 1
        self.logger.warning(f"Pose estimation failed at frame {frame_number}")
        return self._create_failure_result("All pose estimation methods failed")
    
    def _handle_invalid_pose(self, frame_number: int) -> PoseResult:
        """Handle invalid pose"""
        self.consecutive_failures += 1
        self.logger.warning(f"Invalid pose at frame {frame_number}")
        return self._create_failure_result("Pose validation failed")
    
    def _create_failure_result(self, error_message: str) -> PoseResult:
        """Create a failure result"""
        return PoseResult(
            success=False,
            position=np.zeros(3),
            rotation=np.eye(3),
            transformation_matrix=np.eye(4),
            num_matches=0,
            num_inliers=0,
            reprojection_error=0.0,
            confidence=0.0,
            total_distance=self.total_distance,
            method_used=PoseEstimationMethod.PNP_RANSAC,
            error_message=error_message
        )
    
    def get_current_pose(self) -> np.ndarray:
        """Get current pose matrix"""
        return self.current_pose.copy()
    
    def get_trajectory(self) -> List[List[float]]:
        """Get full trajectory"""
        return self.trajectory.copy()
    
    def get_total_distance(self) -> float:
        """Get total distance traveled"""
        return self.total_distance
    
    def reset(self):
        """Reset pose estimator"""
        self.current_pose = np.eye(4)
        self.trajectory = []
        self.keyframes = []
        self.total_distance = 0.0
        self.prev_features = None
        self.prev_depth = None
        self.prev_pose = np.eye(4)
        self.velocity_buffer.clear()
        self.angular_velocity_buffer.clear()
        self.scale_estimates.clear()
        self.current_scale = 1.0
        self.consecutive_failures = 0
        
        self.logger.info("Pose estimator reset")