"""
Advanced Motion Validation System
Author: Mr-Parth24
Date: 2025-06-13
Time: 21:15:16 UTC
"""

import numpy as np
import cv2
from collections import deque
from typing import List, Tuple, Dict, Any
import logging

class AdvancedMotionValidator:
    """Prevents false motion detection when camera is stationary"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Motion thresholds
        self.static_threshold = 0.0008  # 0.8mm - very sensitive
        self.max_reasonable_speed = 1.5  # 1.5 m/s maximum
        self.noise_threshold = 0.0005   # 0.5mm - below this is noise
        
        # History tracking
        self.motion_history = deque(maxlen=30)
        self.feature_stability_history = deque(maxlen=20)
        self.pixel_motion_history = deque(maxlen=15)
        
        # Feature validation
        self.min_features_for_motion = 25
        self.max_pixel_movement = 80  # pixels
        self.min_pixel_movement = 0.5  # pixels
        
        # Statistical validation
        self.outlier_z_threshold = 2.5
        self.consistency_window = 10
        
        # State tracking
        self.stationary_frames = 0
        self.motion_frames = 0
        self.last_validation_result = False
        
    def validate_motion(self, motion_vector: np.ndarray, 
                       keypoints_prev: List, keypoints_curr: List,
                       matches: List, depth_frame: np.ndarray = None) -> Tuple[bool, str, Dict]:
        """
        Comprehensive motion validation with multiple checks
        
        Returns:
            (is_valid, reason, debug_info)
        """
        
        debug_info = {}
        motion_magnitude = np.linalg.norm(motion_vector)
        debug_info['motion_magnitude'] = motion_magnitude
        
        # 1. Basic magnitude checks
        if motion_magnitude < self.noise_threshold:
            self.stationary_frames += 1
            self.motion_frames = 0
            debug_info['result'] = 'noise_level'
            return False, "Below noise threshold", debug_info
        
        if motion_magnitude > self.max_reasonable_speed / 30.0:  # 30 FPS assumption
            debug_info['result'] = 'speed_exceeded'
            return False, f"Exceeds max speed: {motion_magnitude:.4f}m", debug_info
        
        # 2. Feature-based validation
        feature_valid, feature_reason, feature_debug = self._validate_feature_motion(
            keypoints_prev, keypoints_curr, matches
        )
        debug_info.update(feature_debug)
        
        if not feature_valid:
            self.stationary_frames += 1
            self.motion_frames = 0
            debug_info['result'] = 'feature_invalid'
            return False, f"Feature validation failed: {feature_reason}", debug_info
        
        # 3. Temporal consistency check
        temporal_valid, temporal_reason = self._validate_temporal_consistency(motion_magnitude)
        debug_info['temporal_valid'] = temporal_valid
        
        if not temporal_valid:
            debug_info['result'] = 'temporal_invalid'
            return False, f"Temporal inconsistency: {temporal_reason}", debug_info
        
        # 4. Statistical outlier detection
        statistical_valid, statistical_reason = self._validate_statistical_consistency(motion_magnitude)
        debug_info['statistical_valid'] = statistical_valid
        
        if not statistical_valid:
            debug_info['result'] = 'statistical_outlier'
            return False, f"Statistical outlier: {statistical_reason}", debug_info
        
        # 5. Depth consistency (if available)
        if depth_frame is not None:
            depth_valid, depth_reason = self._validate_depth_consistency(
                keypoints_prev, keypoints_curr, matches, depth_frame, motion_vector
            )
            debug_info['depth_valid'] = depth_valid
            
            if not depth_valid:
                debug_info['result'] = 'depth_invalid'
                return False, f"Depth inconsistency: {depth_reason}", debug_info
        
        # Motion is valid
        self.motion_history.append(motion_magnitude)
        self.motion_frames += 1
        self.stationary_frames = 0
        self.last_validation_result = True
        
        debug_info['result'] = 'valid_motion'
        return True, "Motion validated", debug_info
    
    def _validate_feature_motion(self, keypoints_prev: List, keypoints_curr: List, 
                                matches: List) -> Tuple[bool, str, Dict]:
        """Validate motion based on feature movement patterns"""
        
        debug_info = {}
        
        if len(matches) < self.min_features_for_motion:
            debug_info['num_matches'] = len(matches)
            return False, f"Insufficient matches: {len(matches)}", debug_info
        
        # Analyze pixel movements
        pixel_movements = []
        valid_movements = 0
        
        for match in matches:
            try:
                pt_prev = np.array(keypoints_prev[match.queryIdx].pt)
                pt_curr = np.array(keypoints_curr[match.trainIdx].pt)
                
                pixel_displacement = np.linalg.norm(pt_curr - pt_prev)
                pixel_movements.append(pixel_displacement)
                
                # Count valid movements (not too small, not too large)
                if self.min_pixel_movement < pixel_displacement < self.max_pixel_movement:
                    valid_movements += 1
                    
            except (IndexError, AttributeError):
                continue
        
        if not pixel_movements:
            debug_info['pixel_movements'] = []
            return False, "No valid pixel movements", debug_info
        
        # Statistical analysis of pixel movements
        pixel_movements = np.array(pixel_movements)
        mean_pixel_movement = np.mean(pixel_movements)
        std_pixel_movement = np.std(pixel_movements)
        median_pixel_movement = np.median(pixel_movements)
        
        debug_info.update({
            'mean_pixel_movement': mean_pixel_movement,
            'median_pixel_movement': median_pixel_movement,
            'std_pixel_movement': std_pixel_movement,
            'valid_movements': valid_movements,
            'total_matches': len(matches)
        })
        
        # Store pixel motion history
        self.pixel_motion_history.append(mean_pixel_movement)
        
        # Validate movement characteristics
        valid_movement_ratio = valid_movements / len(matches)
        debug_info['valid_movement_ratio'] = valid_movement_ratio
        
        if valid_movement_ratio < 0.15:  # Less than 15% valid movements
            return False, f"Low valid movement ratio: {valid_movement_ratio:.2f}", debug_info
        
        # Check for suspicious uniformity (all features moving identically - likely noise)
        if std_pixel_movement < 0.5 and mean_pixel_movement < 2.0:
            return False, f"Suspicious uniform movement: std={std_pixel_movement:.2f}", debug_info
        
        # Check for excessive scatter (chaotic movement - likely error)
        if std_pixel_movement > mean_pixel_movement * 2 and mean_pixel_movement > 5.0:
            return False, f"Excessive movement scatter: std={std_pixel_movement:.2f}", debug_info
        
        return True, "Feature motion validated", debug_info
    
    def _validate_temporal_consistency(self, motion_magnitude: float) -> Tuple[bool, str]:
        """Check consistency with recent motion history"""
        
        if len(self.motion_history) < 5:
            return True, "Insufficient history"
        
        # Analyze recent motion pattern
        recent_motions = np.array(list(self.motion_history)[-10:])
        recent_mean = np.mean(recent_motions)
        recent_std = np.std(recent_motions)
        
        # Check for sudden jumps
        if recent_mean > 0:
            motion_ratio = motion_magnitude / recent_mean
            if motion_ratio > 4.0:  # 4x sudden increase
                return False, f"Sudden motion increase: {motion_ratio:.2f}x"
            if motion_ratio < 0.1:  # Sudden decrease
                return False, f"Sudden motion decrease: {motion_ratio:.2f}x"
        
        # Check if motion is within reasonable variance
        if recent_std > 0:
            z_score = abs(motion_magnitude - recent_mean) / recent_std
            if z_score > 3.0:  # More than 3 standard deviations
                return False, f"High z-score: {z_score:.2f}"
        
        return True, "Temporally consistent"
    
    def _validate_statistical_consistency(self, motion_magnitude: float) -> Tuple[bool, str]:
        """Statistical outlier detection using robust methods"""
        
        if len(self.motion_history) < 10:
            return True, "Insufficient statistical data"
        
        history_array = np.array(self.motion_history)
        
        # Use robust statistics (median and MAD)
        median_motion = np.median(history_array)
        mad = np.median(np.abs(history_array - median_motion))
        
        if mad > 0:
            # Modified z-score using MAD
            modified_z_score = 0.6745 * (motion_magnitude - median_motion) / mad
            
            if abs(modified_z_score) > self.outlier_z_threshold:
                return False, f"Statistical outlier: z-score={modified_z_score:.2f}"
        
        return True, "Statistically consistent"
    
    def _validate_depth_consistency(self, keypoints_prev: List, keypoints_curr: List,
                                  matches: List, depth_frame: np.ndarray,
                                  motion_vector: np.ndarray) -> Tuple[bool, str]:
        """Validate motion against depth information"""
        
        if len(matches) < 10:
            return True, "Insufficient matches for depth validation"
        
        # Sample depth values at feature locations
        depth_values = []
        for match in matches[:20]:  # Sample first 20 matches
            try:
                pt_prev = keypoints_prev[match.queryIdx].pt
                u, v = int(pt_prev[0]), int(pt_prev[1])
                
                if 0 <= u < depth_frame.shape[1] and 0 <= v < depth_frame.shape[0]:
                    depth = depth_frame[v, u] / 1000.0  # Convert to meters
                    if 0.1 < depth < 8.0:  # Valid depth range
                        depth_values.append(depth)
            except (IndexError, AttributeError):
                continue
        
        if len(depth_values) < 5:
            return True, "Insufficient depth data"
        
        # Check if motion is reasonable given the depth
        mean_depth = np.mean(depth_values)
        motion_magnitude = np.linalg.norm(motion_vector)
        
        # At closer distances, same motion creates larger pixel displacement
        expected_pixel_motion = motion_magnitude / mean_depth * 600  # Approximate focal length
        
        # Get actual pixel motion
        if len(self.pixel_motion_history) > 0:
            actual_pixel_motion = self.pixel_motion_history[-1]
            
            # Check consistency
            motion_ratio = actual_pixel_motion / max(expected_pixel_motion, 0.1)
            
            if motion_ratio > 5.0 or motion_ratio < 0.2:
                return False, f"Depth-motion inconsistency: ratio={motion_ratio:.2f}"
        
        return True, "Depth consistent"
    
    def is_camera_stationary(self) -> Tuple[bool, Dict]:
        """Determine if camera is likely stationary with confidence metrics"""
        
        stationary_info = {
            'stationary_frames': self.stationary_frames,
            'motion_frames': self.motion_frames,
            'recent_motion_avg': 0.0,
            'confidence': 0.0
        }
        
        if len(self.motion_history) < 5:
            stationary_info['confidence'] = 0.5
            return False, stationary_info
        
        # Analyze recent motion
        recent_motions = np.array(list(self.motion_history)[-10:])
        recent_avg = np.mean(recent_motions)
        stationary_info['recent_motion_avg'] = recent_avg
        
        # Determine stationary state
        is_stationary = (
            recent_avg < self.static_threshold and 
            self.stationary_frames > 5
        )
        
        # Calculate confidence
        if is_stationary:
            confidence = min(self.stationary_frames / 20.0, 1.0)
        else:
            confidence = min(self.motion_frames / 10.0, 1.0)
        
        stationary_info['confidence'] = confidence
        
        return is_stationary, stationary_info
    
    def reset(self):
        """Reset validator state"""
        self.motion_history.clear()
        self.feature_stability_history.clear()
        self.pixel_motion_history.clear()
        self.stationary_frames = 0
        self.motion_frames = 0
        self.last_validation_result = False
        
        self.logger.info("Motion validator reset")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information"""
        
        diagnostics = {
            'motion_history_length': len(self.motion_history),
            'stationary_frames': self.stationary_frames,
            'motion_frames': self.motion_frames,
            'last_validation': self.last_validation_result,
            'thresholds': {
                'static_threshold': self.static_threshold,
                'noise_threshold': self.noise_threshold,
                'max_reasonable_speed': self.max_reasonable_speed
            }
        }
        
        if len(self.motion_history) > 0:
            motion_array = np.array(self.motion_history)
            diagnostics.update({
                'motion_stats': {
                    'mean': np.mean(motion_array),
                    'std': np.std(motion_array),
                    'median': np.median(motion_array),
                    'min': np.min(motion_array),
                    'max': np.max(motion_array)
                }
            })
        
        if len(self.pixel_motion_history) > 0:
            pixel_array = np.array(self.pixel_motion_history)
            diagnostics.update({
                'pixel_motion_stats': {
                    'mean': np.mean(pixel_array),
                    'std': np.std(pixel_array),
                    'median': np.median(pixel_array)
                }
            })
        
        return diagnostics