"""
Advanced Feature Detection and Matching
Handles ORB, SIFT, and custom feature detection with quality control
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import logging
from dataclasses import dataclass
from enum import Enum

class FeatureType(Enum):
    ORB = "orb"
    SIFT = "sift"
    SURF = "surf"
    FAST = "fast"

@dataclass
class FeatureData:
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    image_gray: np.ndarray
    quality_score: float

@dataclass
class MatchResult:
    matches: List[cv2.DMatch]
    good_matches: List[cv2.DMatch]
    inlier_matches: List[cv2.DMatch]
    keypoints1: List[cv2.KeyPoint]
    keypoints2: List[cv2.KeyPoint]
    match_quality: float

class FeatureProcessor:
    """Advanced feature detection and matching with quality control"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Feature detection parameters
        self.feature_type = FeatureType(config.get('features', {}).get('type', 'orb'))
        self.max_features = config.get('features', {}).get('max_features', 1000)
        self.quality_threshold = config.get('features', {}).get('quality_threshold', 0.01)
        
        # Matching parameters
        self.match_ratio = config.get('features', {}).get('match_ratio', 0.75)
        self.min_matches = config.get('features', {}).get('min_matches', 20)
        
        # Initialize detectors
        self.detector = self._create_detector()
        self.matcher = self._create_matcher()
        
        # Previous frame data
        self.prev_features = None
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'avg_features': 0,
            'avg_matches': 0,
            'good_frames': 0
        }
        
        self.logger.info(f"Feature processor initialized with {self.feature_type.value}")
    
    def _create_detector(self):
        """Create feature detector based on configuration"""
        if self.feature_type == FeatureType.ORB:
            return cv2.ORB_create(
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
        elif self.feature_type == FeatureType.SIFT:
            return cv2.SIFT_create(
                nfeatures=self.max_features,
                nOctaveLayers=3,
                contrastThreshold=0.04,
                edgeThreshold=10,
                sigma=1.6
            )
        elif self.feature_type == FeatureType.FAST:
            return cv2.FastFeatureDetector_create(
                threshold=10,
                nonmaxSuppression=True,
                type=cv2.FastFeatureDetector_TYPE_9_16
            )
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
    
    def _create_matcher(self):
        """Create feature matcher"""
        if self.feature_type == FeatureType.ORB:
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def process_frame(self, color_image: np.ndarray) -> Optional[FeatureData]:
        """Process frame and extract features"""
        try:
            # Convert to grayscale
            if len(color_image.shape) == 3:
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = color_image.copy()
            
            # Apply preprocessing
            gray = self._preprocess_image(gray)
            
            # Detect features
            if self.feature_type == FeatureType.FAST:
                # FAST detector doesn't compute descriptors
                keypoints = self.detector.detect(gray, None)
                # Use ORB for descriptor computation
                orb = cv2.ORB_create()
                keypoints, descriptors = orb.compute(gray, keypoints)
            else:
                keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            
            if descriptors is None or len(keypoints) < self.min_matches:
                self.logger.warning(f"Insufficient features detected: {len(keypoints) if keypoints else 0}")
                return None
            
            # Quality assessment
            quality_score = self._assess_feature_quality(keypoints, gray)
            
            if quality_score < self.quality_threshold:
                self.logger.warning(f"Low feature quality: {quality_score:.3f}")
                return None
            
            # Update statistics
            self.stats['total_frames'] += 1
            self.stats['avg_features'] = (
                (self.stats['avg_features'] * (self.stats['total_frames'] - 1) + len(keypoints)) 
                / self.stats['total_frames']
            )
            
            return FeatureData(
                keypoints=keypoints,
                descriptors=descriptors,
                image_gray=gray,
                quality_score=quality_score
            )
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return None
    
    def _preprocess_image(self, gray: np.ndarray) -> np.ndarray:
        """Preprocess image for better feature detection"""
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return blurred
    
    def _assess_feature_quality(self, keypoints: List[cv2.KeyPoint], image: np.ndarray) -> float:
        """Assess quality of detected features"""
        if not keypoints:
            return 0.0
        
        # Calculate feature distribution score
        coords = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
        
        # Check distribution across image
        h, w = image.shape
        grid_size = 8
        grid_h, grid_w = h // grid_size, w // grid_size
        
        occupied_cells = set()
        for x, y in coords:
            cell_x = int(x // grid_w)
            cell_y = int(y // grid_h)
            occupied_cells.add((cell_x, cell_y))
        
        distribution_score = len(occupied_cells) / (grid_size * grid_size)
        
        # Calculate response strength score
        responses = [kp.response for kp in keypoints]
        avg_response = np.mean(responses) if responses else 0
        response_score = min(avg_response / 100.0, 1.0)  # Normalize
        
        # Combined quality score
        quality_score = 0.6 * distribution_score + 0.4 * response_score
        
        return quality_score
    
    def match_features(self, features1: FeatureData, features2: FeatureData) -> Optional[MatchResult]:
        """Match features between two frames with quality control"""
        try:
            if features1.descriptors is None or features2.descriptors is None:
                return None
            
            # Perform matching
            if self.feature_type == FeatureType.SIFT:
                # Use Lowe's ratio test for SIFT
                matches = self.matcher.knnMatch(features1.descriptors, features2.descriptors, k=2)
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < self.match_ratio * n.distance:
                            good_matches.append(m)
            else:
                # Use simple matching for ORB
                matches = self.matcher.match(features1.descriptors, features2.descriptors)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Filter by distance threshold
                if matches:
                    max_dist = max(m.distance for m in matches)
                    min_dist = min(m.distance for m in matches)
                    threshold = min_dist + 0.3 * (max_dist - min_dist)
                    good_matches = [m for m in matches if m.distance <= threshold]
                else:
                    good_matches = []
            
            if len(good_matches) < self.min_matches:
                return None
            
            # Apply geometric verification (RANSAC)
            inlier_matches = self._geometric_verification(
                features1.keypoints, features2.keypoints, good_matches
            )
            
            if len(inlier_matches) < self.min_matches:
                return None
            
            # Calculate match quality
            match_quality = self._calculate_match_quality(
                features1.keypoints, features2.keypoints, inlier_matches
            )
            
            # Update statistics
            self.stats['avg_matches'] = (
                (self.stats['avg_matches'] * (self.stats['total_frames'] - 1) + len(inlier_matches))
                / self.stats['total_frames']
            )
            
            if match_quality > 0.5:  # Good quality threshold
                self.stats['good_frames'] += 1
            
            return MatchResult(
                matches=matches if 'matches' in locals() else good_matches,
                good_matches=good_matches,
                inlier_matches=inlier_matches,
                keypoints1=features1.keypoints,
                keypoints2=features2.keypoints,
                match_quality=match_quality
            )
            
        except Exception as e:
            self.logger.error(f"Error matching features: {e}")
            return None
    
    def _geometric_verification(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], 
                              matches: List[cv2.DMatch]) -> List[cv2.DMatch]:
        """Apply geometric verification using RANSAC"""
        if len(matches) < 8:  # Minimum for fundamental matrix
            return matches
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find fundamental matrix with RANSAC
        try:
            _, mask = cv2.findFundamentalMat(
                pts1, pts2, 
                cv2.FM_RANSAC, 
                ransacReprojThreshold=3.0,
                confidence=0.99
            )
            
            if mask is not None:
                inlier_matches = [matches[i] for i, m in enumerate(mask) if m[0] == 1]
                return inlier_matches
            else:
                return matches
                
        except Exception as e:
            self.logger.warning(f"Geometric verification failed: {e}")
            return matches
    
    def _calculate_match_quality(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], 
                               matches: List[cv2.DMatch]) -> float:
        """Calculate overall match quality score"""
        if not matches:
            return 0.0
        
        # Distance score (lower distances are better)
        distances = [m.distance for m in matches]
        distance_score = 1.0 / (1.0 + np.mean(distances))
        
        # Distribution score
        pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
        
        # Calculate spread of matched points
        spread1 = np.std(pts1, axis=0).mean()
        spread2 = np.std(pts2, axis=0).mean()
        spread_score = min((spread1 + spread2) / 200.0, 1.0)  # Normalize
        
        # Combined quality score
        quality = 0.7 * distance_score + 0.3 * spread_score
        
        return quality
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feature processing statistics"""
        success_rate = self.stats['good_frames'] / max(1, self.stats['total_frames'])
        return {
            **self.stats,
            'success_rate': success_rate,
            'feature_type': self.feature_type.value
        }
    
    def reset(self):
        """Reset processor state"""
        self.prev_features = None
        self.stats = {
            'total_frames': 0,
            'avg_features': 0,
            'avg_matches': 0,
            'good_frames': 0
        }
        self.logger.info("Feature processor reset")