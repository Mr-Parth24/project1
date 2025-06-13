"""
Advanced Loop Closure Detection
Implements visual bag-of-words and geometric verification
Author: Mr-Parth24
Date: 2025-06-13
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from collections import defaultdict
import pickle
import os
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from feature_processor import FeatureData

@dataclass
class LoopCandidate:
    keyframe_id: int
    similarity_score: float
    geometric_score: float
    combined_score: float
    transformation: Optional[np.ndarray] = None

@dataclass
class LoopDetectionResult:
    detected: bool
    candidate: Optional[LoopCandidate] = None
    all_candidates: List[LoopCandidate] = None
    closure_transformation: Optional[np.ndarray] = None

class VisualVocabulary:
    """Visual bag-of-words vocabulary for loop detection"""
    
    def __init__(self, vocabulary_size: int = 1000):
        self.vocabulary_size = vocabulary_size
        self.vocabulary = None
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
    
    def train_vocabulary(self, all_descriptors: List[np.ndarray]):
        """Train visual vocabulary using K-means clustering"""
        if not all_descriptors:
            self.logger.error("No descriptors provided for vocabulary training")
            return False
        
        try:
            # Combine all descriptors
            combined_descriptors = np.vstack(all_descriptors)
            self.logger.info(f"Training vocabulary with {len(combined_descriptors)} descriptors")
            
            # Perform K-means clustering
            kmeans = KMeans(
                n_clusters=self.vocabulary_size,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            
            kmeans.fit(combined_descriptors)
            self.vocabulary = kmeans.cluster_centers_
            self.is_trained = True
            
            self.logger.info(f"Vocabulary trained with {self.vocabulary_size} visual words")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to train vocabulary: {e}")
            return False
    
    def get_bow_histogram(self, descriptors: np.ndarray) -> np.ndarray:
        """Convert descriptors to bag-of-words histogram"""
        if not self.is_trained or descriptors is None:
            return np.zeros(self.vocabulary_size)
        
        try:
            # Find closest visual words
            histogram = np.zeros(self.vocabulary_size)
            
            for descriptor in descriptors:
                # Calculate distances to all visual words
                distances = np.linalg.norm(
                    self.vocabulary - descriptor.reshape(1, -1), axis=1
                )
                closest_word = np.argmin(distances)
                histogram[closest_word] += 1
            
            # Normalize histogram
            if np.sum(histogram) > 0:
                histogram = histogram / np.sum(histogram)
            
            return histogram
            
        except Exception as e:
            self.logger.error(f"Failed to compute BoW histogram: {e}")
            return np.zeros(self.vocabulary_size)
    
    def save_vocabulary(self, filepath: str):
        """Save vocabulary to file"""
        try:
            vocab_data = {
                'vocabulary': self.vocabulary,
                'vocabulary_size': self.vocabulary_size,
                'is_trained': self.is_trained
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(vocab_data, f)
            
            self.logger.info(f"Vocabulary saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save vocabulary: {e}")
    
    def load_vocabulary(self, filepath: str) -> bool:
        """Load vocabulary from file"""
        try:
            if not os.path.exists(filepath):
                return False
            
            with open(filepath, 'rb') as f:
                vocab_data = pickle.load(f)
            
            self.vocabulary = vocab_data['vocabulary']
            self.vocabulary_size = vocab_data['vocabulary_size']
            self.is_trained = vocab_data['is_trained']
            
            self.logger.info(f"Vocabulary loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load vocabulary: {e}")
            return False

class LoopDetector:
    """Advanced loop closure detection system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.min_loop_distance = config.get('loop', {}).get('min_loop_distance', 2.0)
        self.similarity_threshold = config.get('loop', {}).get('similarity_threshold', 0.8)
        self.geometric_verification = config.get('loop', {}).get('geometric_verification', True)
        self.max_loop_candidates = config.get('loop', {}).get('max_loop_candidates', 5)
        
        # Visual vocabulary
        self.vocabulary = VisualVocabulary(vocabulary_size=1000)
        self.vocabulary_training_interval = 50  # Train every N keyframes
        
        # Keyframe database
        self.keyframes_db = {}  # keyframe_id -> keyframe_data
        self.bow_histograms = {}  # keyframe_id -> BoW histogram
        self.keyframe_positions = {}  # keyframe_id -> 3D position
        
        # Loop detection state
        self.last_keyframe_id = -1
        self.loop_closures = []
        self.training_descriptors = []
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'loops_detected': 0,
            'false_positives': 0,
            'vocabulary_updates': 0
        }
        
        # Try to load existing vocabulary
        vocab_path = "data/visual_vocabulary.pkl"
        if os.path.exists(vocab_path):
            self.vocabulary.load_vocabulary(vocab_path)
        
        self.logger.info("Loop detector initialized")
    
    def add_keyframe(self, keyframe_id: int, features: FeatureData, 
                    position: np.ndarray) -> bool:
        """Add a new keyframe to the database"""
        try:
            # Store keyframe data
            self.keyframes_db[keyframe_id] = {
                'features': features,
                'descriptors': features.descriptors,
                'position': position.copy()
            }
            
            self.keyframe_positions[keyframe_id] = position.copy()
            
            # Collect descriptors for vocabulary training
            if features.descriptors is not None:
                self.training_descriptors.append(features.descriptors)
            
            # Update vocabulary periodically
            if (len(self.keyframes_db) % self.vocabulary_training_interval == 0 and 
                not self.vocabulary.is_trained):
                self._update_vocabulary()
            
            # Compute BoW histogram if vocabulary is ready
            if self.vocabulary.is_trained and features.descriptors is not None:
                bow_histogram = self.vocabulary.get_bow_histogram(features.descriptors)
                self.bow_histograms[keyframe_id] = bow_histogram
            
            self.last_keyframe_id = keyframe_id
            self.logger.debug(f"Added keyframe {keyframe_id} to loop detector")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add keyframe {keyframe_id}: {e}")
            return False
    
    def check_loop_closure(self, features: FeatureData, 
                          current_position: np.ndarray) -> LoopDetectionResult:
        """Check for loop closure with current frame"""
        try:
            self.stats['total_queries'] += 1
            
            if not self.vocabulary.is_trained or len(self.bow_histograms) < 3:
                return LoopDetectionResult(detected=False)
            
            # Compute BoW histogram for current frame
            if features.descriptors is None:
                return LoopDetectionResult(detected=False)
            
            current_bow = self.vocabulary.get_bow_histogram(features.descriptors)
            
            # Find loop candidates based on visual similarity
            candidates = self._find_visual_candidates(current_bow, current_position)
            
            if not candidates:
                return LoopDetectionResult(detected=False, all_candidates=[])
            
            # Geometric verification
            if self.geometric_verification:
                candidates = self._geometric_verification(features, candidates)
            
            # Select best candidate
            if candidates:
                best_candidate = max(candidates, key=lambda x: x.combined_score)
                
                if best_candidate.combined_score > self.similarity_threshold:
                    self.stats['loops_detected'] += 1
                    self.loop_closures.append({
                        'query_position': current_position.copy(),
                        'matched_keyframe': best_candidate.keyframe_id,
                        'score': best_candidate.combined_score
                    })
                    
                    self.logger.info(f"Loop closure detected with keyframe {best_candidate.keyframe_id}")
                    
                    return LoopDetectionResult(
                        detected=True,
                        candidate=best_candidate,
                        all_candidates=candidates,
                        closure_transformation=best_candidate.transformation
                    )
            
            return LoopDetectionResult(detected=False, all_candidates=candidates)
            
        except Exception as e:
            self.logger.error(f"Loop closure detection failed: {e}")
            return LoopDetectionResult(detected=False)
    
    def _update_vocabulary(self):
        """Update visual vocabulary with collected descriptors"""
        if len(self.training_descriptors) < 10:
            return
        
        try:
            self.logger.info("Updating visual vocabulary...")
            success = self.vocabulary.train_vocabulary(self.training_descriptors)
            
            if success:
                self.stats['vocabulary_updates'] += 1
                
                # Recompute BoW histograms for all keyframes
                self._recompute_bow_histograms()
                
                # Save vocabulary
                os.makedirs("data", exist_ok=True)
                self.vocabulary.save_vocabulary("data/visual_vocabulary.pkl")
                
                self.logger.info("Visual vocabulary updated successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update vocabulary: {e}")
    
    def _recompute_bow_histograms(self):
        """Recompute BoW histograms for all keyframes"""
        try:
            self.bow_histograms.clear()
            
            for keyframe_id, keyframe_data in self.keyframes_db.items():
                descriptors = keyframe_data['descriptors']
                if descriptors is not None:
                    bow_histogram = self.vocabulary.get_bow_histogram(descriptors)
                    self.bow_histograms[keyframe_id] = bow_histogram
            
            self.logger.info(f"Recomputed BoW histograms for {len(self.bow_histograms)} keyframes")
            
        except Exception as e:
            self.logger.error(f"Failed to recompute BoW histograms: {e}")
    
    def _find_visual_candidates(self, query_bow: np.ndarray, 
                              current_position: np.ndarray) -> List[LoopCandidate]:
        """Find loop candidates based on visual similarity"""
        candidates = []
        
        try:
            for keyframe_id, keyframe_bow in self.bow_histograms.items():
                # Skip recent keyframes
                if abs(keyframe_id - self.last_keyframe_id) < 10:
                    continue
                
                # Check spatial distance
                keyframe_position = self.keyframe_positions[keyframe_id]
                spatial_distance = np.linalg.norm(current_position - keyframe_position)
                
                if spatial_distance < self.min_loop_distance:
                    continue
                
                # Calculate visual similarity
                similarity = self._calculate_bow_similarity(query_bow, keyframe_bow)
                
                if similarity > 0.3:  # Minimum similarity threshold
                    candidate = LoopCandidate(
                        keyframe_id=keyframe_id,
                        similarity_score=similarity,
                        geometric_score=0.0,
                        combined_score=similarity
                    )
                    candidates.append(candidate)
            
            # Sort by similarity score
            candidates.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Return top candidates
            return candidates[:self.max_loop_candidates]
            
        except Exception as e:
            self.logger.error(f"Failed to find visual candidates: {e}")
            return []
    
    def _calculate_bow_similarity(self, bow1: np.ndarray, bow2: np.ndarray) -> float:
        """Calculate similarity between two BoW histograms"""
        try:
            # Cosine similarity
            bow1_norm = bow1.reshape(1, -1)
            bow2_norm = bow2.reshape(1, -1)
            
            similarity = cosine_similarity(bow1_norm, bow2_norm)[0, 0]
            
            # Convert to range [0, 1]
            similarity = (similarity + 1.0) / 2.0
            
            return similarity
            
        except Exception as e:
            self.logger.error(f"Failed to calculate BoW similarity: {e}")
            return 0.0
    
    def _geometric_verification(self, query_features: FeatureData, 
                              candidates: List[LoopCandidate]) -> List[LoopCandidate]:
        """Perform geometric verification of loop candidates"""
        verified_candidates = []
        
        try:
            for candidate in candidates:
                keyframe_data = self.keyframes_db[candidate.keyframe_id]
                keyframe_features = keyframe_data['features']
                
                # Match features between query and candidate keyframe
                geometric_score, transformation = self._verify_geometric_consistency(
                    query_features, keyframe_features
                )
                
                if geometric_score > 0.3:  # Minimum geometric score
                    candidate.geometric_score = geometric_score
                    candidate.combined_score = (
                        0.6 * candidate.similarity_score + 
                        0.4 * geometric_score
                    )
                    candidate.transformation = transformation
                    verified_candidates.append(candidate)
            
            return verified_candidates
            
        except Exception as e:
            self.logger.error(f"Geometric verification failed: {e}")
            return candidates  # Return original candidates if verification fails
    
    def _verify_geometric_consistency(self, features1: FeatureData, 
                                    features2: FeatureData) -> Tuple[float, Optional[np.ndarray]]:
        """Verify geometric consistency between two sets of features"""
        try:
            # Match features
            bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf_matcher.match(features1.descriptors, features2.descriptors)
            
            if len(matches) < 20:
                return 0.0, None
            
            # Extract matched points
            pts1 = np.array([features1.keypoints[m.queryIdx].pt for m in matches])
            pts2 = np.array([features2.keypoints[m.trainIdx].pt for m in matches])
            
            # Estimate fundamental matrix with RANSAC
            F, mask = cv2.findFundamentalMat(
                pts1, pts2,
                cv2.FM_RANSAC,
                ransacReprojThreshold=3.0,
                confidence=0.99
            )
            
            if F is None or mask is None:
                return 0.0, None
            
            # Calculate geometric score based on inlier ratio
            num_inliers = np.sum(mask)
            inlier_ratio = num_inliers / len(matches)
            
            # Additional geometric consistency checks
            if inlier_ratio > 0.5:
                # Estimate relative transformation (simplified)
                transformation = np.eye(4)  # Placeholder
                return inlier_ratio, transformation
            else:
                return inlier_ratio, None
            
        except Exception as e:
            self.logger.error(f"Geometric consistency verification failed: {e}")
            return 0.0, None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get loop detection statistics"""
        success_rate = (
            self.stats['loops_detected'] / max(1, self.stats['total_queries'])
        )
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'keyframes_in_db': len(self.keyframes_db),
            'vocabulary_trained': self.vocabulary.is_trained,
            'vocabulary_size': self.vocabulary.vocabulary_size
        }
    
    def reset(self):
        """Reset loop detector"""
        self.keyframes_db.clear()
        self.bow_histograms.clear()
        self.keyframe_positions.clear()
        self.last_keyframe_id = -1
        self.loop_closures.clear()
        self.training_descriptors.clear()
        
        # Reset statistics
        self.stats = {
            'total_queries': 0,
            'loops_detected': 0,
            'false_positives': 0,
            'vocabulary_updates': 0
        }
        
        self.logger.info("Loop detector reset")