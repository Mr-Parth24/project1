"""
Enhanced Visual Odometry System with Better Tracking and Debugging
Fixed issues with pose estimation and added comprehensive logging
"""

import cv2
import numpy as np
import logging
import time
from typing import Optional, Tuple, List, Dict, Any
from scipy.spatial.transform import Rotation as R
from core.camera_manager import FrameData

class FeatureTracker:
    """Enhanced feature tracking with better matching and validation"""
    
    def __init__(self, detector_type: str = 'ORB', max_features: int = 1000):
        self.logger = logging.getLogger(f"{__name__}.FeatureTracker")
        self.detector_type = detector_type
        self.max_features = max_features
        
        # Initialize feature detector
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()
        
        # Previous frame data
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_image = None
        self.prev_points_3d = None
        
        # Tracking statistics
        self.total_features_detected = 0
        self.total_matches_found = 0
        self.good_matches_ratio = 0.0
        
    def _create_detector(self):
        """Create feature detector based on type"""
        try:
            if self.detector_type == 'ORB':
                detector = cv2.ORB_create(
                    nfeatures=self.max_features,
                    scaleFactor=1.2,
                    nlevels=8,
                    edgeThreshold=31,
                    firstLevel=0,
                    WTA_K=2,
                    scoreType=cv2.ORB_HARRIS_SCORE,
                    patchSize=31,
                    fastThreshold=20
                )
            elif self.detector_type == 'SIFT':
                detector = cv2.SIFT_create(
                    nfeatures=self.max_features,
                    nOctaveLayers=3,
                    contrastThreshold=0.04,
                    edgeThreshold=10,
                    sigma=1.6
                )
            elif self.detector_type == 'AKAZE':
                detector = cv2.AKAZE_create()
            else:
                self.logger.warning(f"Unknown detector type: {self.detector_type}, using ORB")
                detector = cv2.ORB_create(nfeatures=self.max_features)
            
            self.logger.info(f"Created {self.detector_type} detector with {self.max_features} max features")
            return detector
            
        except Exception as e:
            self.logger.error(f"Failed to create detector: {e}")
            return cv2.ORB_create(nfeatures=self.max_features)
    
    def _create_matcher(self):
        """Create feature matcher"""
        if self.detector_type in ['ORB', 'AKAZE']:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        return matcher
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Detect keypoints and compute descriptors"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Enhance image for better feature detection
            gray = cv2.equalizeHist(gray)
            
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            
            self.total_features_detected = len(keypoints)
            self.logger.debug(f"Detected {len(keypoints)} features")
            
            return keypoints, descriptors
            
        except Exception as e:
            self.logger.error(f"Feature detection failed: {e}")
            return [], None
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, 
                      ratio_threshold: float = 0.7) -> List:
        """Match features between two descriptor sets with ratio test"""
        if desc1 is None or desc2 is None:
            return []
        
        try:
            # Use KNN matching for ratio test
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
            
            self.total_matches_found = len(good_matches)
            self.good_matches_ratio = len(good_matches) / max(1, len(matches))
            
            self.logger.debug(f"Found {len(good_matches)} good matches from {len(matches)} total")
            
            return good_matches
            
        except Exception as e:
            self.logger.error(f"Feature matching failed: {e}")
            return []

class PoseEstimator:
    """Enhanced 3D pose estimation with better validation"""
    
    def __init__(self, camera_matrix: np.ndarray, depth_scale: float = 0.001):
        self.logger = logging.getLogger(f"{__name__}.PoseEstimator")
        self.camera_matrix = camera_matrix
        self.depth_scale = depth_scale
        
        # RANSAC parameters
        self.ransac_threshold = 3.0  # Increased for better robustness
        self.ransac_confidence = 0.99
        self.min_inliers = 15  # Reduced for more sensitivity
        
        # Pose estimation statistics
        self.last_inlier_count = 0
        self.pose_estimation_success_rate = 0.0
        self.total_estimations = 0
        self.successful_estimations = 0
        
        self.logger.info(f"PoseEstimator initialized with min_inliers={self.min_inliers}")
        
    def get_3d_points(self, points_2d: np.ndarray, depth_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert 2D points to 3D using depth information"""
        points_3d = []
        valid_indices = []
        
        for i, pt in enumerate(points_2d):
            x, y = int(np.round(pt[0])), int(np.round(pt[1]))
            
            # Check bounds
            if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
                depth = depth_image[y, x] * self.depth_scale
                
                # Valid depth range check
                if 0.1 < depth < 10.0:  # Between 10cm and 10m
                    # Convert to 3D coordinates
                    z = depth
                    x_3d = (x - self.camera_matrix[0, 2]) * z / self.camera_matrix[0, 0]
                    y_3d = (y - self.camera_matrix[1, 2]) * z / self.camera_matrix[1, 1]
                    
                    points_3d.append([x_3d, y_3d, z])
                    valid_indices.append(i)
        
        if len(points_3d) == 0:
            return np.array([]), np.array([])
        
        return np.array(points_3d, dtype=np.float32), np.array(valid_indices)
    
    def estimate_pose_pnp(self, points_2d_prev: np.ndarray, points_3d_prev: np.ndarray,
                         points_2d_curr: np.ndarray, depth_image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate pose using PnP algorithm with better validation"""
        
        self.total_estimations += 1
        
        if len(points_3d_prev) < self.min_inliers:
            self.logger.warning(f"Not enough 3D points: {len(points_3d_prev)} < {self.min_inliers}")
            return None, None, None
        
        try:
            # Use cv2.solvePnPRansac for robust pose estimation
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d_prev,
                points_2d_curr,
                self.camera_matrix,
                None,  # No distortion
                reprojectionError=self.ransac_threshold,
                confidence=self.ransac_confidence,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success and inliers is not None and len(inliers) >= self.min_inliers:
                self.last_inlier_count = len(inliers)
                self.successful_estimations += 1
                self.pose_estimation_success_rate = self.successful_estimations / self.total_estimations
                
                # Convert to transformation matrix
                R_matrix, _ = cv2.Rodrigues(rvec)
                transformation = np.eye(4)
                transformation[:3, :3] = R_matrix
                transformation[:3, 3] = tvec.flatten()
                
                # Validate transformation (check for reasonable movement)
                translation_magnitude = np.linalg.norm(tvec)
                if translation_magnitude > 2.0:  # More than 2m movement per frame is suspicious
                    self.logger.warning(f"Large translation detected: {translation_magnitude:.2f}m")
                    return None, None, None
                
                self.logger.debug(f"PnP successful: {len(inliers)} inliers, translation: {translation_magnitude:.3f}m")
                return rvec, tvec, transformation
            else:
                self.logger.debug("PnP failed: insufficient inliers or estimation failed")
                return None, None, None
                
        except Exception as e:
            self.logger.error(f"PnP estimation error: {e}")
            return None, None, None

class VisualOdometry:
    """Enhanced Visual Odometry with better tracking and debugging"""
    
    def __init__(self, camera_matrix: np.ndarray, depth_scale: float = 0.001, 
                 detector_type: str = 'ORB', max_features: int = 1000):
        self.logger = logging.getLogger(f"{__name__}.VisualOdometry")
        
        # Initialize components
        self.feature_tracker = FeatureTracker(detector_type, max_features)
        self.pose_estimator = PoseEstimator(camera_matrix, depth_scale)
        
        # Camera parameters
        self.camera_matrix = camera_matrix
        self.depth_scale = depth_scale
        
        # Odometry state
        self.current_pose = np.eye(4)
        self.trajectory = [self.current_pose.copy()]
        self.total_distance = 0.0
        
        # Frame tracking
        self.prev_frame_data = None
        self.frame_count = 0
        self.initialization_frames = 0
        self.is_initialized = False
        
        # Performance tracking
        self.processing_times = []
        self.successful_estimates = 0
        
        self.logger.info("VisualOdometry initialized")
        
    def process_frame(self, frame_data: FrameData) -> Dict[str, Any]:
        """Process a new frame and update odometry"""
        
        start_time = time.time()
        
        result = {
            'pose': self.current_pose.copy(),
            'position': self.current_pose[:3, 3].copy(),
            'rotation': self.current_pose[:3, :3].copy(),
            'distance_traveled': self.total_distance,
            'features_tracked': 0,
            'loop_closure': False,
            'tracking_quality': 'Unknown',
            'processing_time': 0.0,
            'pose_updated': False
        }
        
        try:
            # Detect and compute features
            keypoints, descriptors = self.feature_tracker.detect_and_compute(frame_data.color_image)
            
            if descriptors is None or len(keypoints) < 10:
                self.logger.warning("Insufficient features detected")
                result['tracking_quality'] = 'Poor - Few Features'
                return result
            
            # First frame - initialize
            if not self.is_initialized:
                self._initialize_tracking(frame_data, keypoints, descriptors)
                result['tracking_quality'] = 'Initializing'
                return result
            
            # Match features with previous frame
            matches = self.feature_tracker.match_features(
                self.feature_tracker.prev_descriptors,
                descriptors
            )
            
            result['features_tracked'] = len(matches)
            
            if len(matches) < self.pose_estimator.min_inliers:
                self.logger.warning(f"Insufficient matches: {len(matches)} < {self.pose_estimator.min_inliers}")
                result['tracking_quality'] = 'Poor - Few Matches'
                
                # Update for next frame but don't estimate pose
                self._update_previous_frame(frame_data, keypoints, descriptors)
                return result
            
            # Extract matched points
            points_2d_prev = np.array([self.feature_tracker.prev_keypoints[m.queryIdx].pt for m in matches], dtype=np.float32)
            points_2d_curr = np.array([keypoints[m.trainIdx].pt for m in matches], dtype=np.float32)
            
            # Get 3D points from previous frame
            points_3d_prev, valid_indices = self.pose_estimator.get_3d_points(
                points_2d_prev, 
                self.prev_frame_data.aligned_depth_image
            )
            
            if len(points_3d_prev) < self.pose_estimator.min_inliers:
                self.logger.warning(f"Insufficient 3D points: {len(points_3d_prev)}")
                result['tracking_quality'] = 'Poor - Insufficient Depth'
                self._update_previous_frame(frame_data, keypoints, descriptors)
                return result
            
            # Filter corresponding 2D points
            points_2d_curr_valid = points_2d_curr[valid_indices]
            
            # Estimate pose
            rvec, tvec, transformation = self.pose_estimator.estimate_pose_pnp(
                points_2d_prev, points_3d_prev, points_2d_curr_valid, frame_data.aligned_depth_image
            )
            
            if transformation is not None:
                # Update current pose
                self.current_pose = self.current_pose @ transformation
                
                # Calculate distance traveled
                distance_delta = np.linalg.norm(tvec)
                self.total_distance += distance_delta
                
                # Update result
                result['pose'] = self.current_pose.copy()
                result['position'] = self.current_pose[:3, 3].copy()
                result['rotation'] = self.current_pose[:3, :3].copy()
                result['distance_traveled'] = self.total_distance
                result['pose_updated'] = True
                result['tracking_quality'] = self._assess_tracking_quality()
                
                self.successful_estimates += 1
                
                self.logger.debug(f"Pose updated: distance_delta={distance_delta:.4f}m, total={self.total_distance:.4f}m")
                
            else:
                result['tracking_quality'] = 'Failed - PnP Error'
                self.logger.warning("Pose estimation failed")
            
            # Update trajectory
            if result['pose_updated']:
                self.trajectory.append(self.current_pose.copy())
                
                # Limit trajectory size for memory
                if len(self.trajectory) > 10000:
                    self.trajectory = self.trajectory[-5000:]  # Keep last 5000 poses
            
            # Update for next frame
            self._update_previous_frame(frame_data, keypoints, descriptors)
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            result['tracking_quality'] = f'Error: {str(e)[:50]}'
        
        # Update performance metrics
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        self.processing_times.append(processing_time)
        
        # Keep only recent processing times
        if len(self.processing_times) > 100:
            self.processing_times = self.processing_times[-50:]
        
        self.frame_count += 1
        
        return result
    
    def _initialize_tracking(self, frame_data: FrameData, keypoints: List, descriptors: np.ndarray):
        """Initialize tracking with first frame"""
        self.initialization_frames += 1
        
        if self.initialization_frames >= 3:  # Wait for a few frames
            self.is_initialized = True
            self.logger.info("Visual odometry initialized")
        
        self._update_previous_frame(frame_data, keypoints, descriptors)
    
    def _update_previous_frame(self, frame_data: FrameData, keypoints: List, descriptors: np.ndarray):
        """Update previous frame data"""
        self.prev_frame_data = frame_data
        self.feature_tracker.prev_keypoints = keypoints
        self.feature_tracker.prev_descriptors = descriptors
        self.feature_tracker.prev_image = frame_data.color_image
    
    def _assess_tracking_quality(self) -> str:
        """Assess current tracking quality"""
        features = self.feature_tracker.total_features_detected
        matches = self.feature_tracker.total_matches_found
        inliers = self.pose_estimator.last_inlier_count
        
        if inliers >= 50 and matches >= 100:
            return "Excellent"
        elif inliers >= 30 and matches >= 50:
            return "Good"
        elif inliers >= 15 and matches >= 25:
            return "Fair"
        else:
            return "Poor"
    
    def get_trajectory_points(self) -> np.ndarray:
        """Get trajectory as 3D points"""
        if len(self.trajectory) == 0:
            return np.array([])
        
        points = np.array([pose[:3, 3] for pose in self.trajectory])
        return points
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        success_rate = self.successful_estimates / max(1, self.frame_count)
        
        return {
            'frame_count': self.frame_count,
            'successful_estimates': self.successful_estimates,
            'success_rate': success_rate,
            'total_distance': self.total_distance,
            'trajectory_points': len(self.trajectory),
            'avg_processing_time': avg_processing_time,
            'features_detected': self.feature_tracker.total_features_detected,
            'matches_found': self.feature_tracker.total_matches_found,
            'pose_estimation_success_rate': self.pose_estimator.pose_estimation_success_rate,
            'is_initialized': self.is_initialized
        }
    
    def reset(self):
        """Reset the visual odometry system"""
        self.current_pose = np.eye(4)
        self.trajectory = [self.current_pose.copy()]
        self.total_distance = 0.0
        self.prev_frame_data = None
        self.frame_count = 0
        self.initialization_frames = 0
        self.is_initialized = False
        self.successful_estimates = 0
        self.processing_times = []
        
        # Reset component statistics
        self.feature_tracker.total_features_detected = 0
        self.feature_tracker.total_matches_found = 0
        self.pose_estimator.total_estimations = 0
        self.pose_estimator.successful_estimations = 0
        
        self.logger.info("Visual odometry system reset")