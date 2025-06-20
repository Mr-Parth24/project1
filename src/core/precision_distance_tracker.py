"""
Precision Distance Tracker for Agricultural SLAM
Implements centimeter-level distance measurement with drift correction
Based on 2025 research for stereo camera scale validation
"""

import numpy as np
import cv2
from typing import Tuple, Dict, List, Optional
from collections import deque
import time
import math
from dataclasses import dataclass

@dataclass
class MovementMeasurement:
    """Single movement measurement with validation data"""
    timestamp: float
    translation: np.ndarray
    rotation: np.ndarray
    distance: float
    confidence: float
    scale_factor: float
    validation_passed: bool

@dataclass
class DistanceValidationMetrics:
    """Metrics for distance measurement validation"""
    mean_error: float
    std_error: float
    drift_rate: float
    scale_consistency: float
    measurement_count: int

class StereoBaselineValidator:
    """
    Validates scale using D435i stereo baseline
    Uses 50mm baseline for absolute scale recovery
    """
    
    def __init__(self, baseline_mm: float = 50.0):
        """
        Initialize stereo baseline validator
        
        Args:
            baseline_mm: Stereo baseline in millimeters (D435i = 50mm)
        """
        self.baseline_m = baseline_mm / 1000.0
        self.focal_length = 615.0  # D435i approximate focal length
        
        # Scale validation parameters
        self.scale_history = deque(maxlen=50)
        self.baseline_measurements = deque(maxlen=20)
        self.scale_confidence_threshold = 0.8
        
        print(f"Stereo baseline validator initialized: {baseline_mm}mm baseline")
    
    def validate_scale_from_disparity(self, left_points: np.ndarray, 
                                    right_points: np.ndarray, 
                                    depth_values: np.ndarray) -> float:
        """
        Validate scale using stereo disparity
        
        Args:
            left_points: Points in left image
            right_points: Corresponding points in right image  
            depth_values: Depth measurements for points
            
        Returns:
            Scale correction factor
        """
        try:
            if len(left_points) != len(right_points) or len(left_points) < 5:
                return 1.0
            
            scale_estimates = []
            
            for i in range(len(left_points)):
                # Calculate disparity
                disparity = abs(left_points[i][0] - right_points[i][0])
                
                if disparity > 1.0:  # Valid disparity
                    # Calculate depth from disparity
                    theoretical_depth = (self.focal_length * self.baseline_m) / disparity
                    measured_depth = depth_values[i] / 1000.0  # Convert mm to m
                    
                    if 0.2 <= measured_depth <= 8.0:  # Valid depth range
                        scale_estimate = theoretical_depth / measured_depth
                        if 0.5 <= scale_estimate <= 2.0:  # Reasonable scale
                            scale_estimates.append(scale_estimate)
            
            if len(scale_estimates) >= 3:
                # Use median for robustness
                scale_factor = np.median(scale_estimates)
                self.scale_history.append(scale_factor)
                return scale_factor
            
            return 1.0
            
        except Exception as e:
            print(f"Scale validation error: {e}")
            return 1.0
    
    def get_scale_confidence(self) -> float:
        """Get confidence in current scale estimates"""
        if len(self.scale_history) < 5:
            return 0.0
        
        recent_scales = list(self.scale_history)[-10:]
        std_dev = np.std(recent_scales)
        
        # Lower standard deviation = higher confidence
        confidence = max(0.0, 1.0 - std_dev / 0.5)
        return confidence

class KalmanFilterDriftCorrector:
    """
    Kalman filter for correcting drift in distance measurements
    Implements constant velocity model with process noise
    """
    
    def __init__(self):
        """Initialize Kalman filter for drift correction"""
        # State: [position_x, position_y, position_z, velocity_x, velocity_y, velocity_z]
        self.state_dim = 6
        self.measurement_dim = 3
        
        # State vector [px, py, pz, vx, vy, vz]
        self.state = np.zeros(self.state_dim)
        
        # State covariance matrix
        self.P = np.eye(self.state_dim) * 1.0
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * 0.01
        
        # Measurement noise covariance  
        self.R = np.eye(self.measurement_dim) * 0.05
        
        # State transition matrix (will be updated with dt)
        self.F = np.eye(self.state_dim)
        
        # Measurement matrix (observe position only)
        self.H = np.zeros((self.measurement_dim, self.state_dim))
        self.H[:3, :3] = np.eye(3)
        
        self.last_timestamp = None
        self.is_initialized = False
        
        print("Kalman drift corrector initialized")
    
    def predict(self, dt: float):
        """Predict step of Kalman filter"""
        # Update state transition matrix with time step
        self.F[:3, 3:6] = np.eye(3) * dt
        
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Update Kalman filter with new measurement
        
        Args:
            measurement: 3D position measurement
            timestamp: Measurement timestamp
            
        Returns:
            Corrected position estimate
        """
        try:
            if not self.is_initialized:
                # Initialize state with first measurement
                self.state[:3] = measurement
                self.state[3:] = 0.0  # Zero velocity
                self.last_timestamp = timestamp
                self.is_initialized = True
                return measurement
            
            # Calculate time step
            dt = timestamp - self.last_timestamp
            if dt <= 0 or dt > 1.0:  # Sanity check
                dt = 0.033  # Default 30fps
            
            # Predict
            self.predict(dt)
            
            # Update
            innovation = measurement - self.H @ self.state
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            
            self.state = self.state + K @ innovation
            self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P
            
            self.last_timestamp = timestamp
            
            return self.state[:3].copy()
            
        except Exception as e:
            print(f"Kalman update error: {e}")
            return measurement
    
    def get_velocity_estimate(self) -> np.ndarray:
        """Get current velocity estimate"""
        return self.state[3:6].copy()
    
    def reset(self):
        """Reset Kalman filter"""
        self.state = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 1.0
        self.last_timestamp = None
        self.is_initialized = False

class StatisticalValidator:
    """
    Statistical validation for movement measurements
    Implements outlier detection and consistency checking
    """
    
    def __init__(self, window_size: int = 20):
        """Initialize statistical validator"""
        self.window_size = window_size
        self.measurements = deque(maxlen=window_size)
        self.outlier_threshold = 3.0  # Standard deviations
        self.consistency_threshold = 0.8
        
    def validate_measurement(self, measurement: float) -> Tuple[bool, float]:
        """
        Validate a new measurement
        
        Args:
            measurement: Distance measurement to validate
            
        Returns:
            Tuple of (is_valid, confidence)
        """
        try:
            # Always accept first few measurements
            if len(self.measurements) < 5:
                self.measurements.append(measurement)
                return True, 1.0
            
            # Calculate statistics
            recent_measurements = list(self.measurements)
            mean_measurement = np.mean(recent_measurements)
            std_measurement = np.std(recent_measurements)
            
            # Outlier detection using Z-score
            if std_measurement > 0:
                z_score = abs(measurement - mean_measurement) / std_measurement
                is_outlier = z_score > self.outlier_threshold
            else:
                is_outlier = False
            
            # Consistency check (measurement should be similar to recent trend)
            recent_trend = np.mean(recent_measurements[-5:])
            consistency = 1.0 - abs(measurement - recent_trend) / max(recent_trend, 0.01)
            consistency = max(0.0, consistency)
            
            # Calculate overall confidence
            confidence = consistency if not is_outlier else 0.0
            
            # Accept measurement if not an outlier
            if not is_outlier:
                self.measurements.append(measurement)
                return True, confidence
            else:
                return False, 0.0
                
        except Exception as e:
            print(f"Statistical validation error: {e}")
            return True, 0.5  # Default to accepting with low confidence

class PrecisionDistanceTracker:
    """
    High-precision distance tracker for agricultural SLAM
    Implements multiple validation layers for centimeter-level accuracy
    """
    
    def __init__(self):
        """Initialize precision distance tracker"""
        # Core components
        self.stereo_validator = StereoBaselineValidator()
        self.drift_corrector = KalmanFilterDriftCorrector()
        self.statistical_validator = StatisticalValidator()
        
        # Distance tracking
        self.total_distance = 0.0
        self.cumulative_position = np.array([0.0, 0.0, 0.0])
        self.last_position = np.array([0.0, 0.0, 0.0])
        
        # Measurement history
        self.measurements = deque(maxlen=1000)
        self.validated_movements = deque(maxlen=500)
        
        # Validation metrics
        self.validation_stats = {
            'total_measurements': 0,
            'accepted_measurements': 0,
            'rejected_outliers': 0,
            'scale_corrections': 0,
            'drift_corrections': 0
        }
        
        # Performance monitoring
        self.accuracy_estimates = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        
        # Agricultural-specific parameters
        self.agricultural_mode = True
        self.vibration_compensation = True
        self.max_reasonable_speed = 5.0  # m/s for agricultural equipment
        
        print("Precision Distance Tracker initialized:")
        print(f"  - Stereo baseline validation: enabled")
        print(f"  - Kalman drift correction: enabled") 
        print(f"  - Statistical validation: enabled")
        print(f"  - Agricultural optimization: enabled")
    
    def process_movement(self, translation: np.ndarray, rotation: np.ndarray, 
                        timestamp: float, additional_data: Dict = None) -> Dict:
        """
        Process a movement measurement with full validation pipeline
        
        Args:
            translation: 3D translation vector
            rotation: 3x3 rotation matrix  
            timestamp: Measurement timestamp
            additional_data: Optional additional validation data
            
        Returns:
            Dictionary with processed movement data
        """
        start_time = time.time()
        
        try:
            # Calculate movement distance
            movement_distance = np.linalg.norm(translation)
            
            # Initialize results
            results = {
                'timestamp': timestamp,
                'raw_distance': movement_distance,
                'validated_distance': 0.0,
                'cumulative_distance': self.total_distance,
                'position': self.cumulative_position.copy(),
                'scale_factor': 1.0,
                'validation_passed': False,
                'confidence': 0.0,
                'validation_stages': {},
                'processing_time': 0.0
            }
            
            # Stage 1: Basic sanity checks
            stage1_passed, stage1_info = self._basic_validation(
                movement_distance, timestamp
            )
            results['validation_stages']['basic'] = {
                'passed': stage1_passed,
                'info': stage1_info
            }
            
            if not stage1_passed:
                results['processing_time'] = time.time() - start_time
                return results
            
            # Stage 2: Statistical validation
            stage2_passed, confidence = self.statistical_validator.validate_measurement(
                movement_distance
            )
            results['validation_stages']['statistical'] = {
                'passed': stage2_passed,
                'confidence': confidence
            }
            
            if not stage2_passed:
                self.validation_stats['rejected_outliers'] += 1
                results['processing_time'] = time.time() - start_time
                return results
            
            # Stage 3: Scale validation (if stereo data available)
            scale_factor = 1.0
            if additional_data and 'stereo_points' in additional_data:
                scale_factor = self._validate_scale(additional_data['stereo_points'])
                self.validation_stats['scale_corrections'] += 1
            
            results['scale_factor'] = scale_factor
            results['validation_stages']['scale'] = {
                'passed': True,
                'scale_factor': scale_factor
            }
            
            # Stage 4: Apply scale correction
            corrected_translation = translation * scale_factor
            corrected_distance = movement_distance * scale_factor
            
            # Stage 5: Drift correction using Kalman filter
            new_position = self.last_position + corrected_translation
            corrected_position = self.drift_corrector.update(new_position, timestamp)
            
            # Calculate actual movement after drift correction
            actual_movement = corrected_position - self.last_position
            actual_distance = np.linalg.norm(actual_movement)
            
            results['validation_stages']['drift_correction'] = {
                'passed': True,
                'original_distance': corrected_distance,
                'corrected_distance': actual_distance
            }
            
            # Stage 6: Final validation
            final_passed = self._final_validation(actual_distance, confidence)
            
            if final_passed:
                # Update tracking state
                self.total_distance += actual_distance
                self.cumulative_position = corrected_position.copy()
                self.last_position = corrected_position.copy()
                
                # Create measurement record
                measurement = MovementMeasurement(
                    timestamp=timestamp,
                    translation=actual_movement,
                    rotation=rotation,
                    distance=actual_distance,
                    confidence=confidence,
                    scale_factor=scale_factor,
                    validation_passed=True
                )
                
                self.measurements.append(measurement)
                self.validated_movements.append(actual_distance)
                self.validation_stats['accepted_measurements'] += 1
                
                # Update results
                results.update({
                    'validated_distance': actual_distance,
                    'cumulative_distance': self.total_distance,
                    'position': self.cumulative_position.copy(),
                    'validation_passed': True,
                    'confidence': confidence
                })
                
                # Estimate accuracy
                accuracy = self._estimate_accuracy()
                self.accuracy_estimates.append(accuracy)
                results['estimated_accuracy'] = accuracy
                
            else:
                results['validation_stages']['final'] = {
                    'passed': False,
                    'reason': 'Failed final validation checks'
                }
            
            self.validation_stats['total_measurements'] += 1
            
            # Performance tracking
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            results['processing_time'] = processing_time
            
            return results
            
        except Exception as e:
            print(f"Movement processing error: {e}")
            results['error'] = str(e)
            return results
    
    def _basic_validation(self, distance: float, timestamp: float) -> Tuple[bool, str]:
        """Basic sanity checks for movement measurements"""
        try:
            # Check distance bounds
            if distance < 0.005:  # 5mm minimum
                return False, f"Distance too small: {distance*1000:.1f}mm"
            
            if distance > 5.0:  # 5m maximum per frame
                return False, f"Distance too large: {distance:.2f}m"
            
            # Check timing
            if hasattr(self, '_last_timestamp'):
                dt = timestamp - self._last_timestamp
                if dt <= 0:
                    return False, "Invalid timestamp"
                
                # Check speed reasonableness
                speed = distance / dt
                if speed > self.max_reasonable_speed:
                    return False, f"Speed too high: {speed:.1f}m/s"
            
            self._last_timestamp = timestamp
            return True, "Basic validation passed"
            
        except Exception as e:
            return False, f"Basic validation error: {e}"
    
    def _validate_scale(self, stereo_data: Dict) -> float:
        """Validate scale using stereo baseline"""
        try:
            if 'left_points' in stereo_data and 'right_points' in stereo_data:
                return self.stereo_validator.validate_scale_from_disparity(
                    stereo_data['left_points'],
                    stereo_data['right_points'],
                    stereo_data.get('depth_values', [])
                )
            return 1.0
        except Exception as e:
            print(f"Scale validation error: {e}")
            return 1.0
    
    def _final_validation(self, distance: float, confidence: float) -> bool:
        """Final validation checks before accepting measurement"""
        try:
            # Require minimum confidence
            if confidence < 0.3:
                return False
            
            # Check against recent movement pattern
            if len(self.validated_movements) >= 5:
                recent_movements = list(self.validated_movements)[-5:]
                median_movement = np.median(recent_movements)
                
                # Reject if too different from recent pattern
                if abs(distance - median_movement) > median_movement * 2.0:
                    return False
            
            return True
            
        except Exception as e:
            print(f"Final validation error: {e}")
            return False
    
    def _estimate_accuracy(self) -> float:
        """Estimate current measurement accuracy in meters"""
        try:
            if len(self.validated_movements) < 10:
                return 0.05  # 5cm default estimate
            
            # Calculate consistency of recent measurements
            recent_movements = list(self.validated_movements)[-20:]
            movement_std = np.std(recent_movements)
            
            # Scale confidence
            scale_confidence = self.stereo_validator.get_scale_confidence()
            
            # Estimate accuracy based on consistency and scale confidence
            base_accuracy = 0.01  # 1cm base accuracy
            consistency_penalty = movement_std * 0.1
            scale_penalty = (1.0 - scale_confidence) * 0.02
            
            estimated_accuracy = base_accuracy + consistency_penalty + scale_penalty
            return min(estimated_accuracy, 0.1)  # Cap at 10cm
            
        except Exception as e:
            print(f"Accuracy estimation error: {e}")
            return 0.05
    
    def get_total_distance(self) -> float:
        """Get total validated distance traveled"""
        return self.total_distance
    
    def get_current_position(self) -> np.ndarray:
        """Get current position"""
        return self.cumulative_position.copy()
    
    def get_validation_metrics(self) -> DistanceValidationMetrics:
        """Get comprehensive validation metrics"""
        try:
            acceptance_rate = 0.0
            if self.validation_stats['total_measurements'] > 0:
                acceptance_rate = (self.validation_stats['accepted_measurements'] / 
                                self.validation_stats['total_measurements'])
            
            # Calculate drift rate
            drift_rate = 0.0
            if len(self.measurements) >= 2:
                start_pos = self.measurements[0].translation
                end_pos = self.measurements[-1].translation  
                total_time = self.measurements[-1].timestamp - self.measurements[0].timestamp
                if total_time > 0:
                    drift_rate = np.linalg.norm(end_pos - start_pos) / total_time
            
            return DistanceValidationMetrics(
                mean_error=np.mean(self.accuracy_estimates) if self.accuracy_estimates else 0.05,
                std_error=np.std(self.accuracy_estimates) if self.accuracy_estimates else 0.02,
                drift_rate=drift_rate,
                scale_consistency=self.stereo_validator.get_scale_confidence(),
                measurement_count=len(self.measurements)
            )
            
        except Exception as e:
            print(f"Metrics calculation error: {e}")
            return DistanceValidationMetrics(0.05, 0.02, 0.0, 0.5, 0)
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        return {
            'total_distance': self.total_distance,
            'total_measurements': self.validation_stats['total_measurements'],
            'accepted_measurements': self.validation_stats['accepted_measurements'],
            'rejection_rate': (self.validation_stats['rejected_outliers'] / 
                             max(self.validation_stats['total_measurements'], 1)),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0,
            'estimated_accuracy': np.mean(self.accuracy_estimates) if self.accuracy_estimates else 0.05,
            'scale_confidence': self.stereo_validator.get_scale_confidence(),
            'drift_correction_active': self.drift_corrector.is_initialized,
            'validation_stats': self.validation_stats.copy()
        }
    
    def reset(self):
        """Reset distance tracker to initial state"""
        # Reset tracking state
        self.total_distance = 0.0
        self.cumulative_position = np.array([0.0, 0.0, 0.0])
        self.last_position = np.array([0.0, 0.0, 0.0])
        
        # Clear history
        self.measurements.clear()
        self.validated_movements.clear()
        self.accuracy_estimates.clear()
        self.processing_times.clear()
        
        # Reset components
        self.drift_corrector.reset()
        self.stereo_validator.scale_history.clear()
        self.statistical_validator.measurements.clear()
        
        # Reset stats
        self.validation_stats = {
            'total_measurements': 0,
            'accepted_measurements': 0,
            'rejected_outliers': 0,
            'scale_corrections': 0,
            'drift_corrections': 0
        }
        
        if hasattr(self, '_last_timestamp'):
            delattr(self, '_last_timestamp')
        
        print("Precision Distance Tracker reset")

# Test function
def test_precision_distance_tracker():
    """Test precision distance tracker with simulated data"""
    tracker = PrecisionDistanceTracker()
    
    print("Testing Precision Distance Tracker...")
    
    # Simulate movement sequence
    position = np.array([0.0, 0.0, 0.0])
    total_simulated = 0.0
    
    for i in range(100):
        # Simulate movement
        movement = np.array([0.1, 0.0, 0.05])  # 10cm forward, 5cm up
        position += movement
        timestamp = time.time() + i * 0.033  # 30fps
        
        # Add some noise
        noisy_movement = movement + np.random.normal(0, 0.002, 3)  # 2mm noise
        
        # Process movement
        results = tracker.process_movement(
            noisy_movement, 
            np.eye(3), 
            timestamp
        )
        
        if results['validation_passed']:
            total_simulated += np.linalg.norm(movement)
    
    print(f"Simulation complete:")
    print(f"  Simulated distance: {total_simulated:.3f}m")
    print(f"  Tracked distance: {tracker.get_total_distance():.3f}m")
    print(f"  Error: {abs(total_simulated - tracker.get_total_distance())*100:.1f}cm")
    
    # Print performance stats
    stats = tracker.get_performance_stats()
    print(f"  Acceptance rate: {(1-stats['rejection_rate'])*100:.1f}%")
    print(f"  Estimated accuracy: {stats['estimated_accuracy']*100:.1f}cm")

if __name__ == "__main__":
    test_precision_distance_tracker()