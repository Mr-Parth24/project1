"""
Adaptive Thresholding for Agricultural SLAM
Dynamic parameter adjustment based on scene conditions
Optimizes feature detection and tracking for field environments
"""

import numpy as np
import cv2
import time
from typing import Dict, List, Tuple, Optional
from collections import deque

class AdaptiveThresholdManager:
    """
    Manages adaptive thresholding for agricultural SLAM parameters
    Automatically adjusts detection thresholds based on scene conditions
    """
    
    def __init__(self):
        """Initialize adaptive threshold manager"""
        # Feature detection thresholds
        self.orb_threshold = 20
        self.orb_threshold_range = (5, 50)
        self.fast_threshold = 20
        self.fast_threshold_range = (5, 40)
        
        # Matching thresholds
        self.match_threshold = 0.7
        self.match_threshold_range = (0.5, 0.9)
        
        # RANSAC thresholds
        self.ransac_threshold = 3.0
        self.ransac_threshold_range = (1.0, 8.0)
        
        # Agricultural scene parameters
        self.lighting_threshold = 0.3
        self.texture_threshold = 0.4
        self.motion_threshold = 0.2
        
        # Adaptation parameters
        self.adaptation_rate = 0.1
        self.stability_threshold = 0.8
        self.min_adaptation_interval = 1.0  # seconds
        
        # History tracking
        self.feature_count_history = deque(maxlen=30)
        self.match_count_history = deque(maxlen=30)
        self.scene_analysis_history = deque(maxlen=10)
        self.performance_history = deque(maxlen=20)
        
        # Last adaptation time
        self.last_adaptation_time = 0
        
        print("Adaptive Threshold Manager initialized")
    
    def analyze_scene_conditions(self, frame: np.ndarray, 
                                depth_frame: np.ndarray = None) -> Dict:
        """Analyze current scene conditions for threshold adaptation"""
        try:
            # Convert to grayscale for analysis
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            analysis = {
                'timestamp': time.time(),
                'lighting_quality': self._analyze_lighting(gray),
                'texture_density': self._analyze_texture(gray),
                'motion_level': self._estimate_motion_level(),
                'scene_complexity': self._calculate_scene_complexity(gray),
                'depth_quality': self._analyze_depth_quality(depth_frame) if depth_frame is not None else 0.5
            }
            
            # Store in history
            self.scene_analysis_history.append(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"Scene analysis error: {e}")
            return self._get_default_analysis()
    
    def adapt_thresholds(self, scene_analysis: Dict, 
                        performance_metrics: Dict) -> Dict:
        """Adapt thresholds based on scene analysis and performance"""
        try:
            current_time = time.time()
            
            # Check if enough time has passed since last adaptation
            if current_time - self.last_adaptation_time < self.min_adaptation_interval:
                return self._get_current_thresholds()
            
            # Store performance metrics
            self.performance_history.append(performance_metrics)
            
            # Calculate adaptation factors
            adaptation_factors = self._calculate_adaptation_factors(
                scene_analysis, performance_metrics
            )
            
            # Adapt each threshold
            new_thresholds = {}
            
            # ORB threshold adaptation
            new_thresholds['orb_threshold'] = self._adapt_orb_threshold(
                adaptation_factors
            )
            
            # FAST threshold adaptation
            new_thresholds['fast_threshold'] = self._adapt_fast_threshold(
                adaptation_factors
            )
            
            # Match threshold adaptation
            new_thresholds['match_threshold'] = self._adapt_match_threshold(
                adaptation_factors
            )
            
            # RANSAC threshold adaptation
            new_thresholds['ransac_threshold'] = self._adapt_ransac_threshold(
                adaptation_factors
            )
            
            # Apply new thresholds gradually
            self._apply_threshold_changes(new_thresholds)
            
            self.last_adaptation_time = current_time
            
            result = self._get_current_thresholds()
            result['adaptation_factors'] = adaptation_factors
            result['adapted'] = True
            
            return result
            
        except Exception as e:
            print(f"Threshold adaptation error: {e}")
            return self._get_current_thresholds()
    
    def _analyze_lighting(self, gray_frame: np.ndarray) -> float:
        """Analyze lighting quality (0=poor, 1=excellent)"""
        try:
            mean_intensity = np.mean(gray_frame)
            std_intensity = np.std(gray_frame)
            
            # Optimal lighting: mean around 128, good contrast
            intensity_score = 1.0 - abs(mean_intensity - 128) / 128
            contrast_score = min(std_intensity / 64.0, 1.0)
            
            # Check for over/under exposure
            overexposed = np.sum(gray_frame > 240) / gray_frame.size
            underexposed = np.sum(gray_frame < 15) / gray_frame.size
            exposure_score = 1.0 - (overexposed + underexposed)
            
            lighting_quality = (intensity_score + contrast_score + exposure_score) / 3.0
            return np.clip(lighting_quality, 0.0, 1.0)
            
        except Exception as e:
            print(f"Lighting analysis error: {e}")
            return 0.5
    
    def _analyze_texture(self, gray_frame: np.ndarray) -> float:
        """Analyze texture density (0=low texture, 1=high texture)"""
        try:
            # Calculate gradient magnitude
            grad_x = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Calculate texture metrics
            mean_gradient = np.mean(gradient_magnitude)
            std_gradient = np.std(gradient_magnitude)
            
            # Edge density
            edges = cv2.Canny(gray_frame, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Combined texture score
            texture_score = (mean_gradient/100.0 + std_gradient/50.0 + edge_density) / 3.0
            return np.clip(texture_score, 0.0, 1.0)
            
        except Exception as e:
            print(f"Texture analysis error: {e}")
            return 0.5
    
    def _estimate_motion_level(self) -> float:
        """Estimate current motion level from recent performance"""
        try:
            if len(self.performance_history) < 2:
                return 0.0
            
            # Look at recent translation magnitudes
            recent_translations = []
            for perf in list(self.performance_history)[-5:]:
                if 'translation_magnitude' in perf:
                    recent_translations.append(perf['translation_magnitude'])
            
            if not recent_translations:
                return 0.0
            
            # Motion level based on average recent translation
            avg_translation = np.mean(recent_translations)
            motion_level = min(avg_translation / 2.0, 1.0)  # Normalize to 0-1
            
            return motion_level
            
        except Exception as e:
            print(f"Motion estimation error: {e}")
            return 0.0
    
    def _calculate_scene_complexity(self, gray_frame: np.ndarray) -> float:
        """Calculate overall scene complexity"""
        try:
            # Harris corner response
            corners = cv2.cornerHarris(gray_frame, 2, 3, 0.04)
            corner_density = np.sum(corners > 0.01 * corners.max()) / corners.size
            
            # Local binary pattern variance
            radius = 3
            n_points = 8 * radius
            
            # Simplified LBP calculation
            center = gray_frame[radius:-radius, radius:-radius]
            variance = np.var(center)
            
            # Combine metrics
            complexity = (corner_density * 100 + variance / 1000.0) / 2.0
            return np.clip(complexity, 0.0, 1.0)
            
        except Exception as e:
            print(f"Scene complexity calculation error: {e}")
            return 0.5
    
    def _analyze_depth_quality(self, depth_frame: np.ndarray) -> float:
        """Analyze depth frame quality"""
        try:
            # Calculate valid depth ratio
            valid_pixels = np.sum(depth_frame > 0)
            total_pixels = depth_frame.size
            valid_ratio = valid_pixels / total_pixels
            
            # Calculate depth variance (measure of depth diversity)
            valid_depths = depth_frame[depth_frame > 0]
            if len(valid_depths) > 0:
                depth_variance = np.var(valid_depths) / 1000000.0  # Normalize
                depth_variance = min(depth_variance, 1.0)
            else:
                depth_variance = 0.0
            
            # Combined depth quality
            depth_quality = (valid_ratio + depth_variance) / 2.0
            return np.clip(depth_quality, 0.0, 1.0)
            
        except Exception as e:
            print(f"Depth quality analysis error: {e}")
            return 0.5
    
    def _calculate_adaptation_factors(self, scene_analysis: Dict, 
                                    performance_metrics: Dict) -> Dict:
        """Calculate adaptation factors for each parameter"""
        try:
            factors = {}
            
            # Lighting factor - poor lighting needs lower thresholds
            lighting = scene_analysis.get('lighting_quality', 0.5)
            factors['lighting'] = 1.0 - lighting  # Invert: poor lighting = high factor
            
            # Texture factor - low texture needs lower thresholds
            texture = scene_analysis.get('texture_density', 0.5)
            factors['texture'] = 1.0 - texture
            
            # Motion factor - high motion needs higher RANSAC threshold
            motion = scene_analysis.get('motion_level', 0.0)
            factors['motion'] = motion
            
            # Performance factor - poor performance needs threshold reduction
            num_features = performance_metrics.get('num_features', 0)
            target_features = 300
            if num_features < target_features * 0.5:
                factors['performance'] = 1.0  # High adaptation needed
            elif num_features < target_features:
                factors['performance'] = 0.5  # Moderate adaptation
            else:
                factors['performance'] = 0.0  # No adaptation needed
            
            return factors
            
        except Exception as e:
            print(f"Adaptation factor calculation error: {e}")
            return {'lighting': 0.0, 'texture': 0.0, 'motion': 0.0, 'performance': 0.0}
    
    def _adapt_orb_threshold(self, factors: Dict) -> float:
        """Adapt ORB detection threshold"""
        try:
            # Lower threshold for poor lighting/texture, higher for good conditions
            adaptation = -(factors['lighting'] + factors['texture'] + factors['performance']) / 3.0
            adaptation += 0.2 * factors['motion']  # Slightly higher for motion
            
            new_threshold = self.orb_threshold + adaptation * self.adaptation_rate * 20
            return np.clip(new_threshold, *self.orb_threshold_range)
            
        except Exception as e:
            print(f"ORB threshold adaptation error: {e}")
            return self.orb_threshold
    
    def _adapt_fast_threshold(self, factors: Dict) -> float:
        """Adapt FAST detection threshold"""
        try:
            # Similar to ORB but typically lower values
            adaptation = -(factors['lighting'] + factors['texture'] + factors['performance']) / 3.0
            
            new_threshold = self.fast_threshold + adaptation * self.adaptation_rate * 15
            return np.clip(new_threshold, *self.fast_threshold_range)
            
        except Exception as e:
            print(f"FAST threshold adaptation error: {e}")
            return self.fast_threshold
    
    def _adapt_match_threshold(self, factors: Dict) -> float:
        """Adapt feature matching threshold"""
        try:
            # Lower threshold for poor conditions (more lenient matching)
            adaptation = -(factors['lighting'] + factors['texture']) / 2.0
            adaptation += 0.3 * factors['motion']  # Higher threshold for motion
            
            new_threshold = self.match_threshold + adaptation * self.adaptation_rate * 0.2
            return np.clip(new_threshold, *self.match_threshold_range)
            
        except Exception as e:
            print(f"Match threshold adaptation error: {e}")
            return self.match_threshold
    
    def _adapt_ransac_threshold(self, factors: Dict) -> float:
        """Adapt RANSAC threshold"""
        try:
            # Higher threshold for motion and poor conditions
            adaptation = (factors['motion'] + factors['lighting'] * 0.5) / 1.5
            
            new_threshold = self.ransac_threshold + adaptation * self.adaptation_rate * 3.0
            return np.clip(new_threshold, *self.ransac_threshold_range)
            
        except Exception as e:
            print(f"RANSAC threshold adaptation error: {e}")
            return self.ransac_threshold
    
    def _apply_threshold_changes(self, new_thresholds: Dict):
        """Apply threshold changes gradually"""
        try:
            # Gradual adaptation to avoid sudden changes
            rate = self.adaptation_rate
            
            self.orb_threshold = (1 - rate) * self.orb_threshold + rate * new_thresholds['orb_threshold']
            self.fast_threshold = (1 - rate) * self.fast_threshold + rate * new_thresholds['fast_threshold']
            self.match_threshold = (1 - rate) * self.match_threshold + rate * new_thresholds['match_threshold']
            self.ransac_threshold = (1 - rate) * self.ransac_threshold + rate * new_thresholds['ransac_threshold']
            
        except Exception as e:
            print(f"Threshold application error: {e}")
    
    def _get_current_thresholds(self) -> Dict:
        """Get current threshold values"""
        return {
            'orb_threshold': self.orb_threshold,
            'fast_threshold': self.fast_threshold,
            'match_threshold': self.match_threshold,
            'ransac_threshold': self.ransac_threshold,
            'adapted': False
        }
    
    def _get_default_analysis(self) -> Dict:
        """Get default scene analysis"""
        return {
            'timestamp': time.time(),
            'lighting_quality': 0.5,
            'texture_density': 0.5,
                        'motion_level': 0.0,
            'scene_complexity': 0.5,
            'depth_quality': 0.5
        }
    
    def update_performance_feedback(self, performance_data: Dict):
        """Update with performance feedback for learning"""
        try:
            # Store feature counts for trend analysis
            if 'num_features' in performance_data:
                self.feature_count_history.append(performance_data['num_features'])
            
            if 'num_matches' in performance_data:
                self.match_count_history.append(performance_data['num_matches'])
            
            # Analyze trends
            if len(self.feature_count_history) >= 10:
                recent_features = list(self.feature_count_history)[-10:]
                avg_features = np.mean(recent_features)
                
                # Auto-adjust if consistently too few features
                if avg_features < 100 and self.orb_threshold > 10:
                    self.orb_threshold = max(self.orb_threshold - 2, 5)
                    print(f"Auto-reduced ORB threshold to {self.orb_threshold}")
                
                # Auto-adjust if too many features (performance impact)
                elif avg_features > 800 and self.orb_threshold < 40:
                    self.orb_threshold = min(self.orb_threshold + 2, 50)
                    print(f"Auto-increased ORB threshold to {self.orb_threshold}")
        
        except Exception as e:
            print(f"Performance feedback error: {e}")
    
    def get_adaptation_statistics(self) -> Dict:
        """Get adaptation statistics and performance"""
        try:
            return {
                'current_thresholds': self._get_current_thresholds(),
                'adaptation_history': {
                    'feature_count_trend': list(self.feature_count_history)[-20:],
                    'match_count_trend': list(self.match_count_history)[-20:],
                    'avg_features_recent': np.mean(list(self.feature_count_history)[-10:]) if self.feature_count_history else 0,
                    'avg_matches_recent': np.mean(list(self.match_count_history)[-10:]) if self.match_count_history else 0
                },
                'scene_conditions': {
                    'recent_lighting': [s.get('lighting_quality', 0) for s in list(self.scene_analysis_history)[-5:]],
                    'recent_texture': [s.get('texture_density', 0) for s in list(self.scene_analysis_history)[-5:]],
                    'recent_motion': [s.get('motion_level', 0) for s in list(self.scene_analysis_history)[-5:]]
                },
                'adaptation_effectiveness': self._calculate_adaptation_effectiveness()
            }
        
        except Exception as e:
            print(f"Adaptation statistics error: {e}")
            return {}
    
    def _calculate_adaptation_effectiveness(self) -> float:
        """Calculate how effective the adaptations have been"""
        try:
            if len(self.feature_count_history) < 10:
                return 0.5
            
            # Look at feature count stability over time
            recent_features = list(self.feature_count_history)[-10:]
            target_features = 300
            
            # Calculate how close we are to target
            avg_features = np.mean(recent_features)
            target_error = abs(avg_features - target_features) / target_features
            
            # Calculate stability (lower variance = more stable)
            stability = 1.0 / (1.0 + np.var(recent_features) / 10000.0)
            
            # Combined effectiveness
            effectiveness = (1.0 - target_error) * stability
            return np.clip(effectiveness, 0.0, 1.0)
            
        except Exception as e:
            print(f"Adaptation effectiveness calculation error: {e}")
            return 0.5
    
    def reset_adaptation(self):
        """Reset adaptation to default values"""
        try:
            self.orb_threshold = 20
            self.fast_threshold = 20
            self.match_threshold = 0.7
            self.ransac_threshold = 3.0
            
            self.feature_count_history.clear()
            self.match_count_history.clear()
            self.scene_analysis_history.clear()
            self.performance_history.clear()
            
            self.last_adaptation_time = 0
            
            print("Adaptive thresholding reset to defaults")
            
        except Exception as e:
            print(f"Adaptation reset error: {e}")

# Global adaptive threshold manager
adaptive_threshold_manager = AdaptiveThresholdManager()

def get_adaptive_threshold_manager() -> AdaptiveThresholdManager:
    """Get global adaptive threshold manager"""
    return adaptive_threshold_manager

def adapt_detection_thresholds(frame: np.ndarray, depth_frame: np.ndarray = None,
                             performance_metrics: Dict = None) -> Dict:
    """Global function to adapt detection thresholds"""
    manager = get_adaptive_threshold_manager()
    
    # Analyze scene
    scene_analysis = manager.analyze_scene_conditions(frame, depth_frame)
    
    # Adapt thresholds if performance metrics provided
    if performance_metrics:
        return manager.adapt_thresholds(scene_analysis, performance_metrics)
    else:
        return manager._get_current_thresholds()

def update_threshold_performance(performance_data: Dict):
    """Update threshold adaptation with performance feedback"""
    adaptive_threshold_manager.update_performance_feedback(performance_data)