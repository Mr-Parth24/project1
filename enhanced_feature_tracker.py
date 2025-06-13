"""
Advanced Feature Tracker with 3D Visualization Support
Author: Mr-Parth24
Date: 2025-06-13
Time: 20:47:06 UTC
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from collections import deque
import math

class AdvancedFeatureTracker:
    """Advanced feature tracking with 3D markers and precise measurements"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Enhanced feature detectors
        self.orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8)
        self.sift = cv2.SIFT_create(nfeatures=800)
        self.fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
        
        # Enhanced matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.flann_matcher = cv2.FlannBasedMatcher()
        
        # Tracking state
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_frame = None
        self.prev_depth = None
        
        # Enhanced trajectory tracking
        self.trajectory_3d = [[0, 0, 0]]
        self.trajectory_distances = [0.0]
        self.total_distance = 0.0
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_rotation = np.eye(3)
        
        # Direction tracking
        self.start_position = np.array([0.0, 0.0, 0.0])
        self.movement_history = deque(maxlen=20)
        self.speed_history = deque(maxlen=10)
        
        # Quality assessment
        self.quality_history = deque(maxlen=30)
        self.tracking_confidence = 0.0
        
        # Motion filtering and validation
        self.motion_filter = deque(maxlen=15)
        self.max_motion_per_frame = 0.3  # meters
        self.min_motion_threshold = 0.001  # meters
        
        # Feature quality tracking
        self.feature_quality_history = deque(maxlen=20)
        
        self.logger.info("Advanced feature tracker initialized")
    
    def process_frame_enhanced(self, color_frame: np.ndarray, depth_frame: np.ndarray, 
                             camera_matrix: np.ndarray = None) -> Dict:
        """Enhanced frame processing with 3D tracking"""
        try:
            gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            
            # Enhanced preprocessing
            gray = self._enhanced_preprocessing(gray)
            
            # Multi-method feature detection
            keypoints, descriptors = self._detect_features_multi_method(gray)
            
            # Initialize result structure
            result = {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'num_features': len(keypoints) if keypoints else 0,
                'num_matches': 0,
                'distance_moved': 0.0,
                'total_distance': self.total_distance,
                'trajectory': self.trajectory_3d.copy(),
                'current_position': self.current_position.tolist(),
                'displacement_from_start': self._calculate_displacement_from_start(),
                'current_speed': self._calculate_current_speed(),
                'direction_angle': self._calculate_direction_angle(),
                'quality_score': 0.0,
                'tracking_confidence': self.tracking_confidence,
                'tracking_status': 'INITIALIZING',
                'feature_quality': self._assess_feature_quality(keypoints, gray),
                'x_displacement': self.current_position[0],
                'y_displacement': self.current_position[1], 
                'z_displacement': self.current_position[2]
            }
            
            # Enhanced matching and motion estimation
            if (self.prev_descriptors is not None and descriptors is not None and 
                len(self.prev_descriptors) > 0 and len(descriptors) > 0):
                
                try:
                    # Enhanced feature matching
                    matches = self._enhanced_feature_matching(self.prev_descriptors, descriptors)
                    good_matches = self._filter_matches_advanced(matches, keypoints)
                    
                    result['num_matches'] = len(good_matches)
                    
                    if len(good_matches) > 20:  # Sufficient matches for reliable tracking
                        # Enhanced 3D motion estimation
                        motion_3d, rotation_3d, quality = self._estimate_3d_motion_enhanced(
                            good_matches, keypoints, depth_frame, camera_matrix
                        )
                        
                        # Validate and filter motion
                        if self._validate_motion_advanced(motion_3d):
                            # Update position and rotation
                            self.current_position += motion_3d
                            self.current_rotation = rotation_3d @ self.current_rotation
                            
                            # Calculate distance moved
                            distance_moved = np.linalg.norm(motion_3d)
                            self.total_distance += distance_moved
                            
                            # Update trajectory
                            self.trajectory_3d.append(self.current_position.copy().tolist())
                            self.trajectory_distances.append(self.total_distance)
                            
                            # Update movement history for speed calculation
                            self.movement_history.append({
                                'motion': motion_3d,
                                'timestamp': len(self.trajectory_3d),
                                'distance': distance_moved
                            })
                            
                            # Update quality metrics
                            self.quality_history.append(quality)
                            self.tracking_confidence = self._calculate_tracking_confidence()
                            
                            # Update result
                            result.update({
                                'distance_moved': distance_moved,
                                'total_distance': self.total_distance,
                                'trajectory': self.trajectory_3d.copy(),
                                'current_position': self.current_position.tolist(),
                                'displacement_from_start': self._calculate_displacement_from_start(),
                                'current_speed': self._calculate_current_speed(),
                                'direction_angle': self._calculate_direction_angle(),
                                'quality_score': quality,
                                'tracking_confidence': self.tracking_confidence,
                                'tracking_status': self._get_tracking_status(quality),
                                'x_displacement': self.current_position[0],
                                'y_displacement': self.current_position[1],
                                'z_displacement': self.current_position[2]
                            })
                    
                except Exception as e:
                    self.logger.warning(f"Enhanced matching error: {e}")
            
            # Update previous frame data
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_frame = gray.copy()
            self.prev_depth = depth_frame.copy()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced frame processing error: {e}")
            return self._get_default_result()
    
    def _enhanced_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing for better feature detection"""
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(enhanced, 5, 50, 50)
        
        # Slight sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(filtered, -1, kernel)
        
        return sharpened
    
    def _detect_features_multi_method(self, gray: np.ndarray) -> Tuple[List, np.ndarray]:
        """Multi-method feature detection for robustness"""
        try:
            # Primary: ORB features
            kp_orb, desc_orb = self.orb.detectAndCompute(gray, None)
            
            if desc_orb is not None and len(kp_orb) > 200:
                return kp_orb, desc_orb
            
            # Fallback: SIFT features
            kp_sift, desc_sift = self.sift.detectAndCompute(gray, None)
            
            if desc_sift is not None and len(kp_sift) > 100:
                # Convert SIFT descriptors for consistency
                desc_sift_uint8 = (desc_sift * 255).astype(np.uint8)
                return kp_sift, desc_sift_uint8
            
            # Last resort: FAST + ORB descriptors
            kp_fast = self.fast.detect(gray, None)
            if len(kp_fast) > 50:
                kp_fast, desc_fast = self.orb.compute(gray, kp_fast)
                if desc_fast is not None:
                    return kp_fast, desc_fast
            
            return kp_orb, desc_orb
            
        except Exception as e:
            self.logger.error(f"Multi-method feature detection error: {e}")
            return [], None
    
    def _enhanced_feature_matching(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Enhanced feature matching with multiple algorithms"""
        try:
            # Primary matching with cross-check
            matches = self.matcher.match(desc1, desc2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Additional ratio test for better quality
            if len(matches) > 100:
                # Use kNN matching for ratio test
                try:
                    knn_matches = self.matcher.knnMatch(desc1, desc2, k=2)
                    ratio_matches = []
                    for match_pair in knn_matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.75 * n.distance:
                                ratio_matches.append(m)
                    
                    if len(ratio_matches) > 50:
                        return ratio_matches
                except:
                    pass
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Enhanced matching error: {e}")
            return []
    
    def _filter_matches_advanced(self, matches: List, keypoints: List) -> List:
        """Advanced match filtering with outlier removal"""
        if not matches or len(matches) < 10:
            return matches
        
        try:
            # Distance-based filtering
            distances = [m.distance for m in matches]
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # Keep matches within 2 standard deviations
            distance_threshold = mean_dist + 2.0 * std_dist
            filtered_matches = [m for m in matches if m.distance <= distance_threshold]
            
            # Spatial distribution filtering
            if len(filtered_matches) > 30:
                # Remove clusters of matches (potential false positives)
                spatial_filtered = self._filter_spatial_clusters(filtered_matches, keypoints)
                if len(spatial_filtered) > 20:
                    filtered_matches = spatial_filtered
            
            # Return top matches
            return filtered_matches[:100]  # Limit for performance
            
        except Exception as e:
            self.logger.warning(f"Advanced match filtering error: {e}")
            return matches[:50]
    
    def _filter_spatial_clusters(self, matches: List, keypoints: List) -> List:
        """Filter out spatially clustered matches"""
        try:
            if not matches or not keypoints:
                return matches
            
            filtered_matches = []
            min_distance = 10.0  # Minimum pixel distance between matches
            
            for match in matches:
                kp = keypoints[match.trainIdx]
                is_valid = True
                
                for existing_match in filtered_matches:
                    existing_kp = keypoints[existing_match.trainIdx]
                    distance = np.linalg.norm(
                        np.array(kp.pt) - np.array(existing_kp.pt)
                    )
                    
                    if distance < min_distance:
                        is_valid = False
                        break
                
                if is_valid:
                    filtered_matches.append(match)
                    
                if len(filtered_matches) >= 80:  # Limit number of matches
                    break
            
            return filtered_matches
            
        except Exception as e:
            self.logger.warning(f"Spatial cluster filtering error: {e}")
            return matches
    
    def _estimate_3d_motion_enhanced(self, matches: List, keypoints: List, 
                                   depth_frame: np.ndarray, camera_matrix: np.ndarray) -> Tuple:
        """Enhanced 3D motion estimation with depth information"""
        try:
            if len(matches) < 8:
                return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
            
            # Extract 2D-3D correspondences
            points_2d_prev = []
            points_2d_curr = []
            points_3d_prev = []
            
            for match in matches:
                # Previous frame keypoint
                kp_prev = self.prev_keypoints[match.queryIdx]
                u_prev, v_prev = int(kp_prev.pt[0]), int(kp_prev.pt[1])
                
                # Current frame keypoint
                kp_curr = keypoints[match.trainIdx]
                u_curr, v_curr = int(kp_curr.pt[0]), int(kp_curr.pt[1])
                
                # Get depth from previous frame
                if (0 <= u_prev < self.prev_depth.shape[1] and 
                    0 <= v_prev < self.prev_depth.shape[0]):
                    
                    depth = self.prev_depth[v_prev, u_prev] / 1000.0  # Convert to meters
                    
                    if 0.1 < depth < 8.0:  # Valid depth range
                        # Calculate 3D point in previous frame
                        if camera_matrix is not None:
                            x = (u_prev - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
                            y = (v_prev - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
                            z = depth
                        else:
                            # Use default camera parameters if not provided
                            x = (u_prev - 320) * depth / 600
                            y = (v_prev - 240) * depth / 600
                            z = depth
                        
                        points_3d_prev.append([x, y, z])
                        points_2d_prev.append([u_prev, v_prev])
                        points_2d_curr.append([u_curr, v_curr])
            
            if len(points_3d_prev) < 8:
                return self._estimate_2d_motion_enhanced(matches, keypoints)
            
            # Convert to numpy arrays
            points_3d_prev = np.array(points_3d_prev, dtype=np.float32)
            points_2d_curr = np.array(points_2d_curr, dtype=np.float32)
            
            # Solve PnP to find camera pose
            if camera_matrix is not None:
                camera_mat = camera_matrix
            else:
                camera_mat = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
            
            dist_coeffs = np.zeros((4, 1))
            
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d_prev, points_2d_curr, camera_mat, dist_coeffs,
                confidence=0.99, reprojectionError=2.0
            )
            
            if success and inliers is not None and len(inliers) > 5:
                # Convert rotation vector to matrix
                R, _ = cv2.Rodrigues(rvec)
                t = tvec.flatten()
                
                # Calculate quality based on inlier ratio
                inlier_ratio = len(inliers) / len(points_3d_prev)
                quality = min(inlier_ratio * 1.2, 1.0)
                
                return t, R, quality
            else:
                return self._estimate_2d_motion_enhanced(matches, keypoints)
                
        except Exception as e:
            self.logger.warning(f"3D motion estimation error: {e}")
            return self._estimate_2d_motion_enhanced(matches, keypoints)
    
    def _estimate_2d_motion_enhanced(self, matches: List, keypoints: List) -> Tuple:
        """Enhanced 2D motion estimation as fallback"""
        try:
            if len(matches) < 5:
                return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
            
            # Calculate optical flow-based motion
            displacements = []
            for match in matches[:20]:  # Use best 20 matches
                pt_prev = self.prev_keypoints[match.queryIdx].pt
                pt_curr = keypoints[match.trainIdx].pt
                
                dx = pt_curr[0] - pt_prev[0]
                dy = pt_curr[1] - pt_prev[1]
                
                displacements.append([dx, dy])
            
            if displacements:
                # Use median to reject outliers
                displacements = np.array(displacements)
                median_displacement = np.median(displacements, axis=0)
                
                # Convert pixel motion to approximate world coordinates
                scale_factor = 0.001  # Approximate scale
                motion_2d = median_displacement * scale_factor
                
                # Add small forward motion assumption
                motion_3d = np.array([motion_2d[0], motion_2d[1], 0.01])
                
                # Calculate quality based on consistency
                consistency = 1.0 / (1.0 + np.std(displacements, axis=0).mean())
                quality = min(consistency * 0.8, 1.0)  # Cap at 0.8 for 2D estimation
                
                return motion_3d, np.eye(3), quality
            
            return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
            
        except Exception as e:
            self.logger.warning(f"2D motion estimation error: {e}")
            return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
    
    def _validate_motion_advanced(self, motion: np.ndarray) -> bool:
        """Advanced motion validation with multiple checks"""
        try:
            motion_magnitude = np.linalg.norm(motion)
            
            # Check for excessive motion
            if motion_magnitude > self.max_motion_per_frame:
                return False
            
            # Check for too small motion (noise)
            if motion_magnitude < self.min_motion_threshold:
                return False
            
            # Add to motion filter for consistency checking
            self.motion_filter.append(motion_magnitude)
            
            # Check consistency with recent motion
            if len(self.motion_filter) > 5:
                recent_motions = list(self.motion_filter)[-5:]
                median_motion = np.median(recent_motions)
                
                # Reject if motion is too different from recent pattern
                if motion_magnitude > median_motion * 4:
                    return False
            
            # Check for oscillation (back-and-forth motion)
            if len(self.movement_history) > 3:
                recent_motions = [m['motion'] for m in list(self.movement_history)[-3:]]
                if self._detect_oscillation(recent_motions):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Motion validation error: {e}")
            return True
    
    def _detect_oscillation(self, recent_motions: List) -> bool:
        """Detect oscillating motion patterns"""
        try:
            if len(recent_motions) < 3:
                return False
            
            # Check if consecutive motions are in opposite directions
            for i in range(len(recent_motions) - 1):
                dot_product = np.dot(recent_motions[i], recent_motions[i + 1])
                if dot_product < -0.5:  # Opposite directions
                    return True
            
            return False
            
        except Exception as e:
            return False
    
    def _calculate_displacement_from_start(self) -> float:
        """Calculate straight-line distance from starting position"""
        return np.linalg.norm(self.current_position - self.start_position)
    
    def _calculate_current_speed(self) -> float:
        """Calculate current speed in m/s"""
        try:
            if len(self.movement_history) < 2:
                return 0.0
            
            # Use last few movements to calculate speed
            recent_movements = list(self.movement_history)[-5:]
            total_distance = sum(m['distance'] for m in recent_movements)
            time_span = len(recent_movements)  # Approximate time span in frames
            
            # Assume 30 FPS for time calculation
            fps = 30.0
            time_seconds = time_span / fps
            
            speed = total_distance / time_seconds if time_seconds > 0 else 0.0
            
            # Add to speed history for smoothing
            self.speed_history.append(speed)
            
            # Return smoothed speed
            return sum(self.speed_history) / len(self.speed_history)
            
        except Exception as e:
            self.logger.warning(f"Speed calculation error: {e}")
            return 0.0
    
    def _calculate_direction_angle(self) -> float:
        """Calculate direction angle relative to start position"""
        try:
            displacement = self.current_position - self.start_position
            
            # Calculate angle in XZ plane (top-down view)
            angle_rad = math.atan2(displacement[2], displacement[0])
            angle_deg = math.degrees(angle_rad)
            
            # Normalize to 0-360 degrees
            if angle_deg < 0:
                angle_deg += 360
            
            return angle_deg
            
        except Exception as e:
            self.logger.warning(f"Direction calculation error: {e}")
            return 0.0
    
    def _assess_feature_quality(self, keypoints: List, image: np.ndarray) -> float:
        """Assess quality of detected features"""
        try:
            if not keypoints:
                return 0.0
            
            # Distribution score
            coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
            h, w = image.shape
            
            # Check spread across image quadrants
            quadrant_counts = [0, 0, 0, 0]
            for x, y in coords:
                if x < w/2 and y < h/2:
                    quadrant_counts[0] += 1
                elif x >= w/2 and y < h/2:
                    quadrant_counts[1] += 1
                elif x < w/2 and y >= h/2:
                    quadrant_counts[2] += 1
                else:
                    quadrant_counts[3] += 1
            
            distribution_score = min(quadrant_counts) / max(quadrant_counts) if max(quadrant_counts) > 0 else 0
            
            # Response strength score
            responses = [kp.response for kp in keypoints]
            avg_response = np.mean(responses) if responses else 0
            response_score = min(avg_response / 50.0, 1.0)
            
            # Combined score
            quality = 0.7 * distribution_score + 0.3 * response_score
            
            # Add to quality history
            self.feature_quality_history.append(quality)
            
            return quality
            
        except Exception as e:
            self.logger.warning(f"Feature quality assessment error: {e}")
            return 0.0
    
    def _calculate_tracking_confidence(self) -> float:
        """Calculate overall tracking confidence"""
        try:
            if not self.quality_history:
                return 0.0
            
            # Recent quality average
            recent_quality = sum(list(self.quality_history)[-10:]) / min(10, len(self.quality_history))
            
            # Motion consistency score
            motion_consistency = 1.0
            if len(self.motion_filter) > 5:
                motion_std = np.std(list(self.motion_filter)[-10:])
                motion_consistency = 1.0 / (1.0 + motion_std * 10)
            
            # Feature quality consistency
            feature_consistency = 1.0
            if len(self.feature_quality_history) > 5:
                feature_std = np.std(list(self.feature_quality_history)[-10:])
                feature_consistency = 1.0 / (1.0 + feature_std * 5)
            
            # Combined confidence
            confidence = 0.5 * recent_quality + 0.3 * motion_consistency + 0.2 * feature_consistency
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation error: {e}")
            return 0.0
    
    def _get_tracking_status(self, quality: float) -> str:
        """Get tracking status based on quality"""
        if quality > 0.8:
            return "EXCELLENT"
        elif quality > 0.6:
            return "GOOD"
        elif quality > 0.4:
            return "FAIR"
        elif quality > 0.2:
            return "POOR"
        else:
            return "LOST"
    
    def _get_default_result(self) -> Dict:
        """Get default result structure"""
        return {
            'keypoints': [],
            'descriptors': None,
            'num_features': 0,
            'num_matches': 0,
            'distance_moved': 0.0,
            'total_distance': self.total_distance,
            'trajectory': self.trajectory_3d.copy(),
            'current_position': self.current_position.tolist(),
            'displacement_from_start': self._calculate_displacement_from_start(),
            'current_speed': 0.0,
            'direction_angle': 0.0,
            'quality_score': 0.0,
            'tracking_confidence': 0.0,
            'tracking_status': 'ERROR',
            'feature_quality': 0.0,
            'x_displacement': self.current_position[0],
            'y_displacement': self.current_position[1],
            'z_displacement': self.current_position[2]
        }
    
    def reset_trajectory(self):
        """Reset trajectory tracking to origin"""
        self.trajectory_3d = [[0, 0, 0]]
        self.trajectory_distances = [0.0]
        self.total_distance = 0.0
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_rotation = np.eye(3)
        self.start_position = np.array([0.0, 0.0, 0.0])
        
        # Clear history
        self.movement_history.clear()
        self.speed_history.clear()
        self.quality_history.clear()
        self.motion_filter.clear()
        self.feature_quality_history.clear()
        
        self.tracking_confidence = 0.0
        
        self.logger.info("Advanced feature tracker trajectory reset")