"""
Robust Filtering Pipeline
Unified filtering system for agricultural SLAM
Combines motion, outlier, and measurement filtering
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from collections import deque
import time

class RobustFilteringPipeline:
    """
    Unified filtering pipeline for agricultural SLAM
    Handles motion filtering, outlier detection, and measurement validation
    """
    
    def __init__(self):
        """Initialize robust filtering pipeline"""
        # Motion filtering
        self.motion_filter = MotionFilter()
        self.outlier_filter = OutlierFilter()
        self.measurement_filter = MeasurementFilter()
        
        # Pipeline configuration
        self.enable_motion_filtering = True
        self.enable_outlier_filtering = True
        self.enable_measurement_filtering = True
        
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        self.filter_stats = {
            'motion_rejections': 0,
            'outlier_rejections': 0,
            'measurement_rejections': 0,
            'total_processed': 0
        }
        
        print("Robust Filtering Pipeline initialized")
    
    def process_slam_data(self, slam_data: Dict) -> Dict:
        """
        Process SLAM data through complete filtering pipeline
        
        Args:
            slam_data: Dictionary with SLAM measurements and features
            
        Returns:
            Filtered SLAM data with validation flags
        """
        start_time = time.time()
        
        try:
            filtered_data = slam_data.copy()
            filtering_results = {
                'motion_filtered': False,
                'outlier_filtered': False,
                'measurement_filtered': False,
                'passed_all_filters': True,
                'rejection_reasons': []
            }
            
            # Stage 1: Motion filtering
            if self.enable_motion_filtering:
                motion_result = self.motion_filter.filter_motion(slam_data)
                if not motion_result['motion_valid']:
                    filtering_results['motion_filtered'] = True
                    filtering_results['passed_all_filters'] = False
                    filtering_results['rejection_reasons'].append('motion_invalid')
                    self.filter_stats['motion_rejections'] += 1
                else:
                    filtered_data.update(motion_result)
            
            # Stage 2: Outlier filtering
            if self.enable_outlier_filtering and filtering_results['passed_all_filters']:
                outlier_result = self.outlier_filter.filter_outliers(filtered_data)
                if outlier_result['outliers_detected']:
                    filtering_results['outlier_filtered'] = True
                    if outlier_result['severe_outliers']:
                        filtering_results['passed_all_filters'] = False
                        filtering_results['rejection_reasons'].append('severe_outliers')
                        self.filter_stats['outlier_rejections'] += 1
                else:
                    filtered_data.update(outlier_result)
            
            # Stage 3: Measurement filtering
            if self.enable_measurement_filtering and filtering_results['passed_all_filters']:
                measurement_result = self.measurement_filter.filter_measurements(filtered_data)
                if not measurement_result['measurements_valid']:
                    filtering_results['measurement_filtered'] = True
                    filtering_results['passed_all_filters'] = False
                    filtering_results['rejection_reasons'].append('measurements_invalid')
                    self.filter_stats['measurement_rejections'] += 1
                else:
                    filtered_data.update(measurement_result)
            
            # Add filtering metadata
            filtered_data['filtering_results'] = filtering_results
            
            # Update statistics
            self.filter_stats['total_processed'] += 1
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return filtered_data
            
        except Exception as e:
            print(f"Filtering pipeline error: {e}")
            error_data = slam_data.copy()
            error_data['filtering_results'] = {
                'passed_all_filters': False,
                'rejection_reasons': ['filtering_error'],
                'error': str(e)
            }
            return error_data
    
    def get_filter_statistics(self) -> Dict:
        """Get filtering pipeline statistics"""
        total = max(self.filter_stats['total_processed'], 1)
        
        return {
            'total_processed': total,
            'motion_rejection_rate': self.filter_stats['motion_rejections'] / total,
            'outlier_rejection_rate': self.filter_stats['outlier_rejections'] / total,
            'measurement_rejection_rate': self.filter_stats['measurement_rejections'] / total,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0,
            'filter_stats': self.filter_stats.copy()
        }

class MotionFilter:
    """Motion filtering for agricultural SLAM"""
    
    def __init__(self):
        self.max_translation_per_frame = 2.0  # meters
        self.max_rotation_per_frame = 1.57    # radians (90 degrees)
        self.min_translation_threshold = 0.01  # 1cm minimum
        self.motion_history = deque(maxlen=10)
    
    def filter_motion(self, slam_data: Dict) -> Dict:
        """Filter motion measurements for validity"""
        try:
            result = slam_data.copy()
            result['motion_valid'] = True
            
            # Check translation
            if 'translation' in slam_data:
                translation = slam_data['translation']
                translation_magnitude = np.linalg.norm(translation)
                
                # Check bounds
                if translation_magnitude > self.max_translation_per_frame:
                    result['motion_valid'] = False
                    result['motion_rejection_reason'] = 'translation_too_large'
                    return result
                
                if translation_magnitude < self.min_translation_threshold:
                    result['motion_valid'] = False
                    result['motion_rejection_reason'] = 'translation_too_small'
                    return result
            
            # Check rotation
            if 'rotation' in slam_data:
                rotation = slam_data['rotation']
                if hasattr(rotation, 'shape') and rotation.shape == (3, 3):
                    # Convert rotation matrix to angle
                    trace = np.trace(rotation)
                    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                    
                    if angle > self.max_rotation_per_frame:
                        result['motion_valid'] = False
                        result['motion_rejection_reason'] = 'rotation_too_large'
                        return result
            
            # Check motion consistency
            if len(self.motion_history) >= 3:
                recent_motions = list(self.motion_history)[-3:]
                current_motion = translation_magnitude if 'translation' in slam_data else 0.0
                
                # Check for sudden motion changes
                motion_std = np.std(recent_motions + [current_motion])
                motion_mean = np.mean(recent_motions)
                
                if motion_std > motion_mean * 2.0 and current_motion > motion_mean * 3.0:
                    result['motion_valid'] = False
                    result['motion_rejection_reason'] = 'motion_inconsistent'
                    return result
            
            # Store motion for history
            if 'translation' in slam_data:
                self.motion_history.append(np.linalg.norm(slam_data['translation']))
            
            return result
            
        except Exception as e:
            print(f"Motion filtering error: {e}")
            result = slam_data.copy()
            result['motion_valid'] = False
            result['motion_rejection_reason'] = 'filtering_error'
            return result

class OutlierFilter:
    """Outlier detection and filtering"""
    
    def __init__(self):
        self.feature_outlier_threshold = 3.0  # Standard deviations
        self.measurement_outlier_threshold = 2.5
        self.max_outlier_ratio = 0.3  # 30% outliers allowed
    
    def filter_outliers(self, slam_data: Dict) -> Dict:
        """Detect and filter outliers in SLAM data"""
        try:
            result = slam_data.copy()
            result['outliers_detected'] = False
            result['severe_outliers'] = False
            
            # Check feature outliers
            if 'features' in slam_data and len(slam_data['features']) > 0:
                outlier_info = self._detect_feature_outliers(slam_data['features'])
                result.update(outlier_info)
            
            # Check measurement outliers
            if 'measurements' in slam_data:
                measurement_outliers = self._detect_measurement_outliers(slam_data['measurements'])
                result['measurement_outliers'] = measurement_outliers
                
                if measurement_outliers.get('outlier_ratio', 0) > self.max_outlier_ratio:
                    result['severe_outliers'] = True
            
            return result
            
        except Exception as e:
            print(f"Outlier filtering error: {e}")
            result = slam_data.copy()
            result['outliers_detected'] = True
            result['severe_outliers'] = True
            return result
    
    def _detect_feature_outliers(self, features: List) -> Dict:
        """Detect outliers in feature data"""
        try:
            if len(features) < 10:
                return {'outliers_detected': False}
            
            # Extract feature responses/qualities
            responses = []
            for feature in features:
                if hasattr(feature, 'response'):
                    responses.append(feature.response)
                elif isinstance(feature, dict) and 'response' in feature:
                    responses.append(feature['response'])
            
            if len(responses) < 10:
                return {'outliers_detected': False}
            
            # Statistical outlier detection
            mean_response = np.mean(responses)
            std_response = np.std(responses)
            
            outlier_count = 0
            for response in responses:
                if abs(response - mean_response) > self.feature_outlier_threshold * std_response:
                    outlier_count += 1
            
            outlier_ratio = outlier_count / len(responses)
            
            return {
                'outliers_detected': outlier_count > 0,
                'feature_outliers': {
                    'count': outlier_count,
                    'ratio': outlier_ratio,
                    'mean_response': mean_response,
                    'std_response': std_response
                }
            }
            
        except Exception as e:
            print(f"Feature outlier detection error: {e}")
            return {'outliers_detected': True, 'error': str(e)}
    
    def _detect_measurement_outliers(self, measurements: Dict) -> Dict:
        """Detect outliers in measurement data"""
        try:
            outlier_info = {'outlier_ratio': 0.0, 'outliers': []}
            
            # Check distance measurements
            if 'distances' in measurements:
                distances = measurements['distances']
                if len(distances) > 5:
                    median_dist = np.median(distances)
                    mad = np.median(np.abs(distances - median_dist))
                    
                    outliers = []
                    for i, dist in enumerate(distances):
                        if abs(dist - median_dist) > self.measurement_outlier_threshold * mad:
                            outliers.append(i)
                    
                    outlier_info['outliers'].extend(outliers)
                    outlier_info['outlier_ratio'] = len(outliers) / len(distances)
            
            return outlier_info
            
        except Exception as e:
            print(f"Measurement outlier detection error: {e}")
            return {'outlier_ratio': 1.0, 'error': str(e)}

class MeasurementFilter:
    """Measurement validation and filtering"""
    
    def __init__(self):
        self.min_feature_count = 30
        self.min_match_count = 15
        self.max_reprojection_error = 3.0
        self.min_baseline_ratio = 0.1
    
    def filter_measurements(self, slam_data: Dict) -> Dict:
        """Validate SLAM measurements"""
        try:
            result = slam_data.copy()
            result['measurements_valid'] = True
            validation_errors = []
            
            # Check feature count
            if 'num_features' in slam_data:
                if slam_data['num_features'] < self.min_feature_count:
                    validation_errors.append('insufficient_features')
            
            # Check match count
            if 'num_matches' in slam_data:
                if slam_data['num_matches'] < self.min_match_count:
                    validation_errors.append('insufficient_matches')
            
            # Check reprojection error
            if 'reprojection_error' in slam_data:
                if slam_data['reprojection_error'] > self.max_reprojection_error:
                    validation_errors.append('high_reprojection_error')
            
            # Set validity based on errors
            if validation_errors:
                result['measurements_valid'] = False
                result['validation_errors'] = validation_errors
            
            return result
            
        except Exception as e:
            print(f"Measurement filtering error: {e}")
            result = slam_data.copy()
            result['measurements_valid'] = False
            result['validation_errors'] = ['filtering_error']
            return result

# Global filtering instance
robust_filter = RobustFilteringPipeline()

def filter_slam_data(slam_data: Dict) -> Dict:
    """Global function for filtering SLAM data"""
    return robust_filter.process_slam_data(slam_data)

def get_filter_statistics() -> Dict:
    """Get global filter statistics"""
    return robust_filter.get_filter_statistics()