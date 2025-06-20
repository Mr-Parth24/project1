"""
Feature Detection and Matching for Visual SLAM
Handles ORB feature detection, description, and matching
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import time

class FeatureDetector:
    """
    ORB-based feature detector for Visual SLAM
    Optimized for real-time performance without IMU
    """
    
    def __init__(self, max_features=1000, quality_level=0.01):
        """
        Initialize feature detector
        
        Args:
            max_features: Maximum number of features to detect
            quality_level: Quality threshold for feature detection
        """
        self.max_features = max_features
        self.quality_level = quality_level
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20
        )
        
        # Initialize matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Previous frame data
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Performance tracking
        self.detection_times = []
        self.matching_times = []
        
        print("Feature Detector initialized with ORB")
    
    def detect_features(self, frame: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect ORB features in frame
        
        Args:
            frame: Input image (BGR or grayscale)
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        start_time = time.time()
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        # Track performance
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        if len(self.detection_times) > 100:  # Keep last 100 measurements
            self.detection_times.pop(0)
        
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, 
                      ratio_threshold: float = 0.75) -> List:
        """
        Match features between two descriptor sets
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            ratio_threshold: Ratio test threshold for good matches
            
        Returns:
            List of good matches
        """
        if desc1 is None or desc2 is None:
            return []
        
        start_time = time.time()
        
        try:
            # Brute force matching
            matches = self.matcher.match(desc1, desc2)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Filter matches using distance threshold
            good_matches = []
            if len(matches) > 0:
                # Use adaptive threshold based on match distribution
                distances = [m.distance for m in matches]
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                threshold = mean_distance + 0.5 * std_distance
                
                good_matches = [m for m in matches if m.distance < threshold]
            
            # Track performance
            matching_time = time.time() - start_time
            self.matching_times.append(matching_time)
            if len(self.matching_times) > 100:
                self.matching_times.pop(0)
            
            return good_matches
            
        except Exception as e:
            print(f"Error in feature matching: {e}")
            return []
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Process a single frame for feature detection and matching
        
        Args:
            frame: Current frame
            
        Returns:
            Dictionary with processing results
        """
        # Detect features in current frame
        keypoints, descriptors = self.detect_features(frame)
        
        results = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'matches': [],
            'num_features': len(keypoints) if keypoints else 0,
            'match_quality': 0.0
        }
        
        # Match with previous frame if available
        if (self.prev_descriptors is not None and 
            descriptors is not None and len(descriptors) > 0):
            
            matches = self.match_features(self.prev_descriptors, descriptors)
            results['matches'] = matches
            results['match_quality'] = len(matches) / max(len(keypoints), 1)
        
        # Update previous frame data
        self.prev_frame = frame.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        return results
    
    def draw_features(self, frame: np.ndarray, keypoints: List, 
                     color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw detected features on frame
        
        Args:
            frame: Image to draw on
            keypoints: Detected keypoints
            color: Color for drawing features (BGR)
            
        Returns:
            Image with features drawn
        """
        if not keypoints:
            return frame
        
        # Draw keypoints
        frame_with_features = cv2.drawKeypoints(
            frame, keypoints, None, 
            color=color, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        return frame_with_features
    
    def draw_matches(self, img1: np.ndarray, kp1: List, 
                    img2: np.ndarray, kp2: List, 
                    matches: List) -> np.ndarray:
        """
        Draw feature matches between two images
        
        Args:
            img1: First image
            kp1: Keypoints from first image
            img2: Second image
            kp2: Keypoints from second image
            matches: List of matches
            
        Returns:
            Image showing matches
        """
        if not matches or not kp1 or not kp2:
            # Return concatenated images if no matches
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
            combined[:h1, :w1] = img1 if len(img1.shape) == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            combined[:h2, w1:w1+w2] = img2 if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
            return combined
        
        # Draw matches
        match_image = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:50],  # Limit to 50 matches for clarity
            None, 
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        
        return match_image
    
    def get_matched_points(self, kp1: List, kp2: List, 
                          matches: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract matched point coordinates
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: List of matches
            
        Returns:
            Tuple of (points1, points2) as numpy arrays
        """
        if not matches or not kp1 or not kp2:
            return np.array([]), np.array([])
        
        # Extract matched points
        points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        return points1, points2
    
    def get_performance_stats(self) -> dict:
        """
        Get performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0.0,
            'avg_matching_time': np.mean(self.matching_times) if self.matching_times else 0.0,
            'total_frames_processed': len(self.detection_times)
        }
        
        return stats
    
    def reset(self):
        """Reset detector state"""
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.detection_times = []
        self.matching_times = []
        print("Feature detector reset")

# Test function
def test_feature_detector():
    """Test the feature detector with sample images"""
    import cv2
    
    detector = FeatureDetector(max_features=500)
    
    # Create test pattern
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add some patterns for feature detection
    cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.circle(test_image, (400, 300), 50, (128, 128, 128), -1)
    cv2.line(test_image, (0, 240), (640, 240), (64, 64, 64), 5)
    
    # Detect features
    keypoints, descriptors = detector.detect_features(test_image)
    
    print(f"Detected {len(keypoints)} features")
    
    # Draw features
    result_image = detector.draw_features(test_image, keypoints)
    
    # Display result
    cv2.imshow('Feature Detection Test', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_feature_detector()