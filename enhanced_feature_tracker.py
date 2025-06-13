"""
Enhanced Feature Tracker with Advanced Motion Validation
Author: Mr-Parth24
Date: 2025-06-13
Time: 21:15:16 UTC
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import deque
import math

from advanced_motion_validator import AdvancedMotionValidator

class EnhancedFeatureTracker:
    """Enhanced feature tracking with motion validation and drift prevention"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Enhanced feature detectors
        self.orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        self.sift = cv2.SIFT_create(nfeatures=1000)
        self.fast = cv2.FastFeatureDetector_create(threshold=10, nonmaxSuppression=True)
        
        # Enhanced matchers
        self.orb_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.sift_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Motion validation
        self.motion_validator = AdvancedMotionValidator()
        
        # Tracking state
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_frame = None
        self.prev_depth = None
        
        # Enhanced trajectory tracking
        self.trajectory_3d = [[0, 0, 0]]
        self.total_distance = 0.0
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.start_position = np.array([0.0, 0.0, 0.0])
        
        # Motion tracking
        self.movement_history = deque(maxlen=30)
        self.speed_history = deque(maxlen=15)
        self.direction_history = deque(maxlen=20)
        
        # Quality tracking
        self.quality_history = deque(maxlen=25)
        self.confidence_history = deque(maxlen=20)
        self.feature_quality_history = deque(maxlen=15)
        
        # Motion filtering
        self.position_filter = deque(maxlen=10)
        self.velocity_filter = deque(maxlen=8)
        
        # State management
        self.is_initialized = False
        self.tracking_lost = False
        self.consecutive_failures = 0
        self.max_failures = 15
        
        # Performance tracking
        self.frame_count = 0
        self.successful_frames = 0
        
        self.logger.info("Enhanced feature tracker initialized with motion validation")
    
    def process_frame_enhanced(self, color_frame: np.ndarray, depth_frame: np.ndarray, 
                             camera_matrix: np.ndarray = None) -> Dict:
        """Enhanced frame processing with comprehensive validation"""
        
        try:
            self.frame_count += 1
            
            # Preprocess frame
            gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            enhanced_gray = self._enhanced_preprocessing(gray)
            
            # Detect features with multiple methods
            keypoints, descriptors = self._detect_features_robust(enhanced_gray)
            
            # Initialize result structure
            result = self._initialize_result_structure()
            result.update({
                'keypoints': keypoints,
                'descriptors': descriptors,
                'num_features': len(keypoints) if keypoints else 0,
                'feature_quality': self._assess_feature_quality(keypoints, enhanced_gray),
                'frame_count': self.frame_count
            })
            
            # First frame initialization
            if not self.is_initialized:
                return self._initialize_first_frame(result, keypoints, descriptors, enhanced_gray, depth_frame)
            
            # Feature matching and validation
            if self._has_sufficient_features(keypoints, descriptors):
                motion_result = self._process_motion_estimation(
                    keypoints, descriptors, depth_frame, camera_matrix, enhanced_gray
                )
                result.update(motion_result)
            else:
                self._handle_insufficient_features()
                result.update({
                    'tracking_status': 'INSUFFICIENT_FEATURES',
                    'motion_valid': False,
                    'validation_reason': 'Too few features detected'
                })
            
            # Update state for next frame
            self._update_frame_state(keypoints, descriptors, enhanced_gray, depth_frame)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Enhanced frame processing error: {e}")
            return self._get_error_result(str(e))
    
    def _enhanced_preprocessing(self, gray: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing for optimal feature detection"""
        
        try:
            # 1. Adaptive histogram equalization
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # 2. Bilateral filtering for noise reduction while preserving edges
            filtered = cv2.bilateralFilter(enhanced, 5, 50, 50)
            
            # 3. Gentle sharpening to enhance feature distinctiveness
            kernel = np.array([[-0.5, -1, -0.5], 
                              [-1, 7, -1], 
                              [-0.5, -1, -0.5]]) / 3.0
            sharpened = cv2.filter2D(filtered, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            self.logger.error(f"Preprocessing error: {e}")
            return gray
    
    def _detect_features_robust(self, gray: np.ndarray) -> Tuple[List, np.ndarray]:
        """Robust multi-method feature detection with quality assessment"""
        
        try:
            # Primary: ORB detection
            kp_orb, desc_orb = self.orb.detectAndCompute(gray, None)
            
            if desc_orb is not None and len(kp_orb) > 300:
                # Good ORB detection
                quality_score = self._calculate_detection_quality(kp_orb, gray)
                if quality_score > 0.4:
                    return kp_orb, desc_orb
            
            # Fallback 1: SIFT detection (more robust but slower)
            self.logger.debug("Using SIFT fallback for feature detection")
            kp_sift, desc_sift = self.sift.detectAndCompute(gray, None)
            
            if desc_sift is not None and len(kp_sift) > 150:
                # Convert SIFT descriptors to uint8 for consistency
                desc_sift_uint8 = (desc_sift * 255).astype(np.uint8)
                quality_score = self._calculate_detection_quality(kp_sift, gray)
                if quality_score > 0.3:
                    return kp_sift, desc_sift_uint8
            
            # Fallback 2: FAST + ORB descriptors
            self.logger.debug("Using FAST+ORB fallback for feature detection")
            kp_fast = self.fast.detect(gray, None)
            
            if len(kp_fast) > 100:
                kp_fast, desc_fast = self.orb.compute(gray, kp_fast)
                if desc_fast is not None:
                    return kp_fast, desc_fast
            
            # Return ORB results even if suboptimal
            return kp_orb if kp_orb else [], desc_orb
            
        except Exception as e:
            self.logger.error(f"Feature detection error: {e}")
            return [], None
    
    def _calculate_detection_quality(self, keypoints: List, gray: np.ndarray) -> float:
        """Calculate quality score for detected features"""
        
        try:
            if not keypoints:
                return 0.0
            
            # 1. Feature distribution score
            coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
            h, w = gray.shape
            
            # Check distribution across image quadrants
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
            
            # Distribution uniformity (higher is better)
            min_count = min(quadrant_counts)
            max_count = max(quadrant_counts)
            distribution_score = min_count / max(max_count, 1)
            
            # 2. Feature strength score
            responses = [kp.response for kp in keypoints]
            avg_response = np.mean(responses) if responses else 0
            response_score = min(avg_response / 100.0, 1.0)
            
            # 3. Feature density score
            density = len(keypoints) / (h * w) * 100000  # Normalize
            density_score = min(density / 50.0, 1.0)
            
            # Combined quality score
            quality = 0.4 * distribution_score + 0.4 * response_score + 0.2 * density_score
            
            return quality
            
        except Exception as e:
            self.logger.error(f"Quality calculation error: {e}")
            return 0.0
    
    def _initialize_result_structure(self) -> Dict:
        """Initialize comprehensive result structure"""
        
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
            'current_speed': self._calculate_current_speed(),
            'direction_angle': self._calculate_direction_angle(),
            'quality_score': 0.0,
            'feature_quality': 0.0,
            'tracking_confidence': 0.0,
            'tracking_status': 'INITIALIZING',
            'motion_valid': False,
            'validation_reason': 'Not validated',
            'validation_debug': {},

            'x_displacement': self.current_position[0],
            'y_displacement': self.current_position[1],
            'z_displacement': self.current_position[2],
            'is_initialized': self.is_initialized,
            'tracking_lost': self.tracking_lost,
            'consecutive_failures': self.consecutive_failures,
            'frame_count': self.frame_count,
            'success_rate': self.successful_frames / max(self.frame_count, 1)
        }
    
    def _initialize_first_frame(self, result: Dict, keypoints: List, descriptors: np.ndarray,
                               gray: np.ndarray, depth_frame: np.ndarray) -> Dict:
        """Initialize tracking with first frame"""
        
        try:
            self.is_initialized = True
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_frame = gray.copy()
            self.prev_depth = depth_frame.copy()
            
            # Initialize position tracking
            self.current_position = np.array([0.0, 0.0, 0.0])
            self.start_position = np.array([0.0, 0.0, 0.0])
            self.trajectory_3d = [[0, 0, 0]]
            self.total_distance = 0.0
            
            result.update({
                'tracking_status': 'INITIALIZED',
                'motion_valid': True,
                'validation_reason': 'First frame initialization',
                'is_initialized': True,
                'quality_score': 1.0,
                'tracking_confidence': 1.0
            })
            
            self.successful_frames += 1
            self.logger.info(f"Tracking initialized with {len(keypoints)} features")
            
            return result
            
        except Exception as e:
            self.logger.error(f"First frame initialization error: {e}")
            result.update({
                'tracking_status': 'INITIALIZATION_FAILED',
                'motion_valid': False,
                'validation_reason': f'Initialization error: {str(e)}'
            })
            return result
    
    def _has_sufficient_features(self, keypoints: List, descriptors: np.ndarray) -> bool:
        """Check if we have sufficient features for tracking"""
        
        return (keypoints is not None and 
                descriptors is not None and 
                len(keypoints) >= 20 and
                len(descriptors) >= 20)
    
    def _process_motion_estimation(self, keypoints: List, descriptors: np.ndarray,
                                 depth_frame: np.ndarray, camera_matrix: np.ndarray,
                                 gray: np.ndarray) -> Dict:
        """Process motion estimation with comprehensive validation"""
        
        try:
            # Match features with previous frame
            matches = self._match_features_enhanced(self.prev_descriptors, descriptors)
            
            if len(matches) < 15:
                self.consecutive_failures += 1
                return {
                    'num_matches': len(matches),
                    'tracking_status': 'INSUFFICIENT_MATCHES',
                    'motion_valid': False,
                    'validation_reason': f'Only {len(matches)} matches found'
                }
            
            # Filter matches for quality
            good_matches = self._filter_matches_advanced(matches, keypoints)
            
            if len(good_matches) < 10:
                self.consecutive_failures += 1
                return {
                    'num_matches': len(good_matches),
                    'tracking_status': 'POOR_MATCH_QUALITY',
                    'motion_valid': False,
                    'validation_reason': f'Only {len(good_matches)} good matches'
                }
            
            # Estimate 3D motion
            motion_3d, rotation_3d, estimation_quality = self._estimate_3d_motion_robust(
                good_matches, keypoints, depth_frame, camera_matrix
            )
            
            # Validate motion with comprehensive checks
            motion_valid, validation_reason, validation_debug = self.motion_validator.validate_motion(
                motion_3d, self.prev_keypoints, keypoints, good_matches, depth_frame
            )
            
            # Check camera stationary state
            is_stationary, stationary_info = self.motion_validator.is_camera_stationary()
            
            result = {
                'num_matches': len(good_matches),
                'motion_valid': motion_valid,
                'validation_reason': validation_reason,
                'validation_debug': validation_debug,
                'is_stationary': is_stationary,
                'stationary_info': stationary_info,
                'quality_score': estimation_quality
            }
            
            if motion_valid and not is_stationary:
                # Update position and trajectory
                return self._update_position_and_trajectory(
                    motion_3d, rotation_3d, estimation_quality, result
                )
            else:
                # Camera is stationary or motion is invalid
                self._handle_stationary_camera(result, is_stationary, validation_reason)
                return result
                
        except Exception as e:
            self.logger.error(f"Motion estimation error: {e}")
            self.consecutive_failures += 1
            return {
                'tracking_status': 'MOTION_ESTIMATION_ERROR',
                'motion_valid': False,
                'validation_reason': f'Estimation error: {str(e)}',
                'num_matches': 0
            }
    
    def _match_features_enhanced(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Enhanced feature matching with multiple algorithms"""
        
        try:
            # Determine descriptor type and use appropriate matcher
            if desc1.dtype == np.uint8:
                # ORB descriptors
                matches = self.orb_matcher.match(desc1, desc2)
            else:
                # SIFT descriptors - use ratio test
                raw_matches = self.sift_matcher.knnMatch(desc1, desc2, k=2)
                matches = []
                for match_pair in raw_matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.75 * n.distance:  # Lowe's ratio test
                            matches.append(m)
            
            # Sort by distance (best matches first)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Return top matches to avoid overwhelming the system
            return matches[:200]
            
        except Exception as e:
            self.logger.error(f"Feature matching error: {e}")
            return []
    
    def _filter_matches_advanced(self, matches: List, keypoints: List) -> List:
        """Advanced match filtering with multiple criteria"""
        
        try:
            if not matches:
                return []
            
            # 1. Distance-based filtering
            distances = [m.distance for m in matches]
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            # Keep matches within 1.5 standard deviations
            distance_threshold = mean_dist + 1.5 * std_dist
            distance_filtered = [m for m in matches if m.distance <= distance_threshold]
            
            # 2. Spatial distribution filtering
            spatial_filtered = self._filter_spatial_clusters(distance_filtered, keypoints)
            
            # 3. Geometric consistency filtering
            if len(spatial_filtered) > 20:
                geometric_filtered = self._filter_geometric_outliers(spatial_filtered, keypoints)
                if len(geometric_filtered) > 15:
                    return geometric_filtered
            
            return spatial_filtered
            
        except Exception as e:
            self.logger.error(f"Match filtering error: {e}")
            return matches[:50]  # Return top 50 as fallback
    
    def _filter_spatial_clusters(self, matches: List, keypoints: List) -> List:
        """Remove spatially clustered matches that might be false positives"""
        
        try:
            if not matches or not keypoints:
                return matches
            
            filtered_matches = []
            min_distance = 8.0  # Minimum pixel distance between matches
            
            for match in matches:
                try:
                    kp = keypoints[match.trainIdx]
                    is_valid = True
                    
                    # Check distance to existing matches
                    for existing_match in filtered_matches:
                        existing_kp = keypoints[existing_match.trainIdx]
                        pixel_distance = np.linalg.norm(
                            np.array(kp.pt) - np.array(existing_kp.pt)
                        )
                        
                        if pixel_distance < min_distance:
                            is_valid = False
                            break
                    
                    if is_valid:
                        filtered_matches.append(match)
                        
                    # Limit number of matches for performance
                    if len(filtered_matches) >= 100:
                        break
                        
                except (IndexError, AttributeError):
                    continue
            
            return filtered_matches
            
        except Exception as e:
            self.logger.error(f"Spatial clustering filter error: {e}")
            return matches
    
    def _filter_geometric_outliers(self, matches: List, keypoints: List) -> List:
        """Filter matches using geometric consistency (fundamental matrix)"""
        
        try:
            if len(matches) < 8:  # Need at least 8 points for fundamental matrix
                return matches
            
            # Extract matched points
            pts1 = []
            pts2 = []
            
            for match in matches:
                try:
                    pt1 = self.prev_keypoints[match.queryIdx].pt
                    pt2 = keypoints[match.trainIdx].pt
                    pts1.append(pt1)
                    pts2.append(pt2)
                except (IndexError, AttributeError):
                    continue
            
            if len(pts1) < 8:
                return matches
            
            pts1 = np.array(pts1, dtype=np.float32)
            pts2 = np.array(pts2, dtype=np.float32)
            
            # Find fundamental matrix with RANSAC
            F, mask = cv2.findFundamentalMat(
                pts1, pts2,
                cv2.FM_RANSAC,
                ransacReprojThreshold=2.0,
                confidence=0.99
            )
            
            if F is not None and mask is not None:
                # Keep only inlier matches
                inlier_matches = [matches[i] for i, m in enumerate(mask.flatten()) if m == 1]
                
                if len(inlier_matches) > 10:
                    return inlier_matches
            
            return matches
            
        except Exception as e:
            self.logger.error(f"Geometric filtering error: {e}")
            return matches
    
    def _estimate_3d_motion_robust(self, matches: List, keypoints: List,
                                  depth_frame: np.ndarray, camera_matrix: np.ndarray) -> Tuple:
        """Robust 3D motion estimation with multiple fallback methods"""
        
        try:
            # Method 1: PnP with depth information (most accurate)
            if camera_matrix is not None and depth_frame is not None:
                motion, rotation, quality = self._estimate_motion_pnp_ransac(
                    matches, keypoints, depth_frame, camera_matrix
                )
                
                if quality > 0.3:  # Good quality threshold
                    return motion, rotation, quality
            
            # Method 2: Essential matrix estimation (fallback)
            self.logger.debug("Using essential matrix fallback for motion estimation")
            motion, rotation, quality = self._estimate_motion_essential_matrix(
                matches, keypoints, camera_matrix
            )
            
            if quality > 0.2:
                return motion, rotation, quality
            
            # Method 3: 2D motion estimation (last resort)
            self.logger.debug("Using 2D motion fallback")
            motion, rotation, quality = self._estimate_motion_2d_fallback(matches, keypoints)
            
            return motion, rotation, quality
            
        except Exception as e:
            self.logger.error(f"3D motion estimation error: {e}")
            return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
    
    def _estimate_motion_pnp_ransac(self, matches: List, keypoints: List,
                                   depth_frame: np.ndarray, camera_matrix: np.ndarray) -> Tuple:
        """Estimate motion using PnP RANSAC with depth information"""
        
        try:
            # Extract 3D-2D correspondences
            points_3d = []
            points_2d = []
            valid_matches = []
            
            for match in matches:
                try:
                    # Previous frame keypoint with depth
                    kp_prev = self.prev_keypoints[match.queryIdx]
                    u_prev, v_prev = int(kp_prev.pt[0]), int(kp_prev.pt[1])
                    
                    # Current frame keypoint
                    kp_curr = keypoints[match.trainIdx]
                    
                    # Get depth from previous frame
                    if (0 <= u_prev < self.prev_depth.shape[1] and 
                        0 <= v_prev < self.prev_depth.shape[0]):
                        
                        depth = self.prev_depth[v_prev, u_prev] / 1000.0  # Convert to meters
                        
                        if 0.1 < depth < 8.0:  # Valid depth range
                            # Convert to 3D point
                            x = (u_prev - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
                            y = (v_prev - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
                            z = depth
                            
                            points_3d.append([x, y, z])
                            points_2d.append([kp_curr.pt[0], kp_curr.pt[1]])
                            valid_matches.append(match)
                            
                except (IndexError, AttributeError):
                    continue
            
            if len(points_3d) < 8:
                return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
            
            # Convert to numpy arrays
            points_3d = np.array(points_3d, dtype=np.float32)
            points_2d = np.array(points_2d, dtype=np.float32)
            
            # Solve PnP with RANSAC
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d, points_2d, camera_matrix, np.zeros((4, 1)),
                confidence=0.99,
                reprojectionError=2.5,
                flags=cv2.SOLVEPNP_EPNP
            )
            
            if success and inliers is not None and len(inliers) > 5:
                # Convert rotation vector to matrix
                R, _ = cv2.Rodrigues(rvec)
                t = tvec.flatten()
                
                # Calculate quality based on inlier ratio and reprojection error
                inlier_ratio = len(inliers) / len(points_3d)
                
                # Calculate reprojection error for quality assessment
                projected_points, _ = cv2.projectPoints(
                    points_3d[inliers.flatten()], rvec, tvec, camera_matrix, np.zeros((4, 1))
                )
                
                errors = []
                for i, inlier_idx in enumerate(inliers.flatten()):
                    observed = points_2d[inlier_idx]
                    projected = projected_points[i][0]
                    error = np.linalg.norm(observed - projected)
                    errors.append(error)
                
                avg_error = np.mean(errors) if errors else 0.0
                error_quality = 1.0 / (1.0 + avg_error)
                
                # Combined quality score
                quality = 0.7 * inlier_ratio + 0.3 * error_quality
                
                return t, R, quality
            
            return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
            
        except Exception as e:
            self.logger.error(f"PnP RANSAC estimation error: {e}")
            return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
    
    def _estimate_motion_essential_matrix(self, matches: List, keypoints: List,
                                        camera_matrix: np.ndarray) -> Tuple:
        """Estimate motion using essential matrix (scale ambiguous)"""
        
        try:
            if len(matches) < 8 or camera_matrix is None:
                return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
            
            # Extract matched points
            pts1 = []
            pts2 = []
            
            for match in matches:
                try:
                    pt1 = self.prev_keypoints[match.queryIdx].pt
                    pt2 = keypoints[match.trainIdx].pt
                    pts1.append(pt1)
                    pts2.append(pt2)
                except (IndexError, AttributeError):
                    continue
            
            if len(pts1) < 8:
                return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
            
            pts1 = np.array(pts1, dtype=np.float32)
            pts2 = np.array(pts2, dtype=np.float32)
            
            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(
                pts1, pts2, camera_matrix,
                method=cv2.RANSAC,
                prob=0.99,
                threshold=1.0
            )
            
            if E is None or mask is None:
                return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
            
            # Recover pose from essential matrix
            num_inliers, R, t, mask_pose = cv2.recoverPose(
                E, pts1, pts2, camera_matrix, mask=mask
            )
            
            if num_inliers < 8:
                return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
            
            # Scale estimation using previous motion patterns
            scale = self._estimate_scale_from_history(t)
            t_scaled = t.flatten() * scale
            
            # Quality based on inlier ratio
            inlier_ratio = num_inliers / len(pts1)
            quality = min(inlier_ratio * 0.8, 0.8)  # Cap at 0.8 for essential matrix
            
            return t_scaled, R, quality
            
        except Exception as e:
            self.logger.error(f"Essential matrix estimation error: {e}")
            return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
    
    def _estimate_motion_2d_fallback(self, matches: List, keypoints: List) -> Tuple:
        """2D motion estimation as last resort"""
        
        try:
            if len(matches) < 5:
                return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
            
            # Calculate optical flow-like motion
            displacements = []
            
            for match in matches[:20]:  # Use best 20 matches
                try:
                    pt1 = self.prev_keypoints[match.queryIdx].pt
                    pt2 = keypoints[match.trainIdx].pt
                    
                    dx = pt2[0] - pt1[0]
                    dy = pt2[1] - pt1[1]
                    
                    displacements.append([dx, dy])
                except (IndexError, AttributeError):
                    continue
            
            if not displacements:
                return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
            
            # Use median to reject outliers
            displacements = np.array(displacements)
            median_displacement = np.median(displacements, axis=0)
            
            # Convert to approximate world coordinates
            scale_factor = 0.0005  # Conservative scale factor
            motion_2d = median_displacement * scale_factor
            
            # Add small forward motion assumption
            motion_3d = np.array([motion_2d[0], motion_2d[1], 0.005])
            
            # Calculate quality based on consistency
            std_displacement = np.std(displacements, axis=0)
            consistency = 1.0 / (1.0 + np.mean(std_displacement))
            quality = min(consistency * 0.6, 0.6)  # Cap at 0.6 for 2D estimation
            
            return motion_3d, np.eye(3), quality
            
        except Exception as e:
            self.logger.error(f"2D motion estimation error: {e}")
            return np.array([0.0, 0.0, 0.0]), np.eye(3), 0.0
    
    def _estimate_scale_from_history(self, translation_unit_vector: np.ndarray) -> float:
        """Estimate scale using motion history patterns"""
        
        try:
            if len(self.movement_history) < 3:
                return 0.01  # Default scale
            
            # Analyze recent motion magnitudes
            recent_motions = [m['distance'] for m in list(self.movement_history)[-5:]]
            avg_motion = np.mean(recent_motions)
            
            # Use average motion as scale estimate
            scale = max(avg_motion, 0.001)  # Minimum scale
            scale = min(scale, 0.1)         # Maximum scale
            
            return scale
            
        except Exception as e:
            self.logger.error(f"Scale estimation error: {e}")
            return 0.01
    
    def _update_position_and_trajectory(self, motion_3d: np.ndarray, rotation_3d: np.ndarray,
                                      estimation_quality: float, result: Dict) -> Dict:
        """Update position and trajectory with new motion"""
        
        try:
            # Update current position
            self.current_position += motion_3d
            
            # Calculate distance moved
            distance_moved = np.linalg.norm(motion_3d)
            self.total_distance += distance_moved
            
            # Update trajectory
            self.trajectory_3d.append(self.current_position.copy().tolist())
            
            # Update movement history
            movement_data = {
                'motion': motion_3d.copy(),
                'distance': distance_moved,
                'timestamp': self.frame_count,
                'quality': estimation_quality
            }
            self.movement_history.append(movement_data)
            
            # Update quality tracking
            self.quality_history.append(estimation_quality)
            tracking_confidence = self._calculate_tracking_confidence()
            self.confidence_history.append(tracking_confidence)
            
            # Update result with new information
            result.update({
                'distance_moved': distance_moved,
                'total_distance': self.total_distance,
                'trajectory': self.trajectory_3d.copy(),
                'current_position': self.current_position.tolist(),
                'displacement_from_start': self._calculate_displacement_from_start(),
                'current_speed': self._calculate_current_speed(),
                'direction_angle': self._calculate_direction_angle(),
                'quality_score': estimation_quality,
                'tracking_confidence': tracking_confidence,
                'tracking_status': self._get_tracking_status(estimation_quality),
                'x_displacement': self.current_position[0],
                'y_displacement': self.current_position[1],
                'z_displacement': self.current_position[2]
            })
            
            # Reset failure counter on success
            self.consecutive_failures = 0
            self.tracking_lost = False
            self.successful_frames += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Position update error: {e}")
            result.update({
                'tracking_status': 'POSITION_UPDATE_ERROR',
                'motion_valid': False,
                'validation_reason': f'Update error: {str(e)}'
            })
            return result
    
    def _handle_stationary_camera(self, result: Dict, is_stationary: bool, reason: str):
        """Handle stationary camera state"""
        
        try:
            if is_stationary:
                result.update({
                    'tracking_status': 'CAMERA_STATIONARY',
                    'distance_moved': 0.0,
                    'current_speed': 0.0
                })
                # Reset failure counter for stationary state
                self.consecutive_failures = 0
                self.successful_frames += 1
            else:
                result.update({
                    'tracking_status': 'MOTION_REJECTED',
                })
                self.consecutive_failures += 1
            
            # Update quality metrics
            current_quality = result.get('quality_score', 0.0)
            self.quality_history.append(current_quality)
            
            tracking_confidence = self._calculate_tracking_confidence()
            result.update({
                'tracking_confidence': tracking_confidence,
                'quality_score': current_quality
            })
            
        except Exception as e:
            self.logger.error(f"Stationary camera handling error: {e}")
    
    def _handle_insufficient_features(self):
        """Handle insufficient features situation"""
        
        self.consecutive_failures += 1
        
        if self.consecutive_failures > self.max_failures:
            self.tracking_lost = True
            self.logger.warning("Tracking lost due to consecutive failures")
    
    def _update_frame_state(self, keypoints: List, descriptors: np.ndarray,
                           gray: np.ndarray, depth_frame: np.ndarray):
        """Update state for next frame processing"""
        
        try:
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.prev_frame = gray.copy()
            self.prev_depth = depth_frame.copy()
            
        except Exception as e:
            self.logger.error(f"Frame state update error: {e}")
    
    def _calculate_displacement_from_start(self) -> float:
        """Calculate straight-line distance from starting position"""
        
        return float(np.linalg.norm(self.current_position - self.start_position))
    
    def _calculate_current_speed(self) -> float:
        """Calculate current speed in m/s"""
        
        try:
            if len(self.movement_history) < 2:
                return 0.0
            
            # Use recent movements for speed calculation
            recent_movements = list(self.movement_history)[-3:]
            total_distance = sum(m['distance'] for m in recent_movements)
            time_span = len(recent_movements)
            
            # Assume 30 FPS for time calculation
            fps = 30.0
            time_seconds = time_span / fps
            
            speed = total_distance / time_seconds if time_seconds > 0 else 0.0
            
            # Add to speed history for smoothing
            self.speed_history.append(speed)
            
            # Return smoothed speed
            return float(sum(self.speed_history) / len(self.speed_history))
            
        except Exception as e:
            self.logger.error(f"Speed calculation error: {e}")
            return 0.0
    
    def _calculate_direction_angle(self) -> float:
        """Calculate direction angle relative to start position in degrees"""
        
        try:
            displacement = self.current_position - self.start_position
            
            # Calculate angle in XZ plane (top-down view)
            angle_rad = math.atan2(displacement[2], displacement[0])
            angle_deg = math.degrees(angle_rad)
            
            # Normalize to 0-360 degrees
            if angle_deg < 0:
                angle_deg += 360
            
            # Add to direction history for smoothing
            self.direction_history.append(angle_deg)
            
            # Return smoothed direction
            if len(self.direction_history) > 1:
                # Handle angle wrapping for averaging
                angles_rad = [math.radians(a) for a in list(self.direction_history)[-5:]]
                avg_x = np.mean([math.cos(a) for a in angles_rad])
                avg_y = np.mean([math.sin(a) for a in angles_rad])
                avg_angle = math.degrees(math.atan2(avg_y, avg_x))
                
                if avg_angle < 0:
                    avg_angle += 360
                
                return float(avg_angle)
            
            return float(angle_deg)
            
        except Exception as e:
            self.logger.error(f"Direction calculation error: {e}")
            return 0.0
    
    def _calculate_tracking_confidence(self) -> float:
        """Calculate overall tracking confidence"""
        
        try:
            if not self.quality_history:
                return 0.0
            
            # Recent quality average
            recent_quality = sum(list(self.quality_history)[-10:]) / min(10, len(self.quality_history))
            
            # Success rate factor
            success_rate = self.successful_frames / max(self.frame_count, 1)
            
            # Feature quality consistency
            if len(self.feature_quality_history) > 3:
                feature_std = np.std(list(self.feature_quality_history)[-5:])
                feature_consistency = 1.0 / (1.0 + feature_std * 3)
            else:
                feature_consistency = 0.5
            
            # Failure penalty
            failure_penalty = max(0.0, 1.0 - (self.consecutive_failures / self.max_failures))
            
            # Combined confidence
            confidence = (0.4 * recent_quality + 
                         0.3 * success_rate + 
                         0.2 * feature_consistency + 
                         0.1 * failure_penalty)
            
            return float(min(confidence, 1.0))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {e}")
            return 0.0
    
    def _get_tracking_status(self, quality: float) -> str:
        """Get tracking status based on quality and state"""
        
        if self.tracking_lost:
            return "TRACKING_LOST"
        elif self.consecutive_failures > 5:
            return "POOR"
        elif quality > 0.8:
            return "EXCELLENT"
        elif quality > 0.6:
            return "GOOD"
        elif quality > 0.4:
            return "FAIR"
        elif quality > 0.2:
            return "POOR"
        else:
            return "VERY_POOR"
    
    def _assess_feature_quality(self, keypoints: List, image: np.ndarray) -> float:
        """Assess quality of detected features"""
        
        try:
            if not keypoints:
                return 0.0
            
            quality = self._calculate_detection_quality(keypoints, image)
            self.feature_quality_history.append(quality)
            
            return quality
            
        except Exception as e:
            self.logger.error(f"Feature quality assessment error: {e}")
            return 0.0
    
    def _get_error_result(self, error_message: str) -> Dict:
        """Get error result structure"""
        
        result = self._initialize_result_structure()
        result.update({
            'tracking_status': 'ERROR',
            'motion_valid': False,
            'validation_reason': error_message,
            'quality_score': 0.0,
            'tracking_confidence': 0.0
        })
        
        self.consecutive_failures += 1
        
        return result
    
    def reset_trajectory(self):
        """Reset trajectory tracking to origin"""
        
        try:
            self.trajectory_3d = [[0, 0, 0]]
            self.total_distance = 0.0
            self.current_position = np.array([0.0, 0.0, 0.0])
            self.start_position = np.array([0.0, 0.0, 0.0])
            
            # Clear history
            self.movement_history.clear()
            self.speed_history.clear()
            self.direction_history.clear()
            self.quality_history.clear()
            self.confidence_history.clear()
            self.feature_quality_history.clear()
            self.position_filter.clear()
            self.velocity_filter.clear()
            
            # Reset state
            self.consecutive_failures = 0
            self.tracking_lost = False
            self.successful_frames = 0
            
            # Reset motion validator
            self.motion_validator.reset()
            
            self.logger.info("Enhanced feature tracker trajectory reset")
            
        except Exception as e:
            self.logger.error(f"Trajectory reset error: {e}")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information"""
        
        try:
            diagnostics = {
                'tracking_state': {
                    'is_initialized': self.is_initialized,
                    'tracking_lost': self.tracking_lost,
                    'consecutive_failures': self.consecutive_failures,
                    'frame_count': self.frame_count,
                    'successful_frames': self.successful_frames,
                    'success_rate': self.successful_frames / max(self.frame_count, 1)
                },
                'motion_state': {
                    'current_position': self.current_position.tolist(),
                    'total_distance': self.total_distance,
                    'displacement_from_start': self._calculate_displacement_from_start(),
                    'current_speed': self._calculate_current_speed(),
                    'direction_angle': self._calculate_direction_angle()
                },
                'quality_metrics': {
                    'tracking_confidence': self._calculate_tracking_confidence(),
                    'recent_quality': np.mean(list(self.quality_history)[-5:]) if self.quality_history else 0.0,
                    'recent_feature_quality': np.mean(list(self.feature_quality_history)[-3:]) if self.feature_quality_history else 0.0
                },
                'history_lengths': {
                    'movement_history': len(self.movement_history),
                    'quality_history': len(self.quality_history),
                    'speed_history': len(self.speed_history),
                    'direction_history': len(self.direction_history)
                },
                'motion_validator': self.motion_validator.get_diagnostics()
            }
            
            return diagnostics
            
        except Exception as e:
            self.logger.error(f"Diagnostics generation error: {e}")
            return {'error': str(e)}
            