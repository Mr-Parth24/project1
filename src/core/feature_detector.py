"""
Enhanced Feature Detection and Matching for Visual SLAM
Handles ORB feature detection, description, and matching
FIXED: Robust feature detection with fallback mechanisms for real-world use
Date: 2025-06-21 01:26:58 UTC
User: Mr-Parth24
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import time

class FeatureDetector:
    """
    Enhanced ORB-based feature detector for Visual SLAM
    FIXED: Optimized for real-time performance with robust fallback detection
    """
    
    def __init__(self, max_features=1000, quality_level=0.01):
        """
        Initialize enhanced feature detector with robust fallback
        
        Args:
            max_features: Maximum number of features to detect
            quality_level: Quality threshold for feature detection
        """
        self.max_features = max_features
        self.quality_level = quality_level
        
        # ✅ FIXED: Reduced fastThreshold from 20 to 10 for more features
        self.orb = cv2.ORB_create(
            nfeatures=max_features,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=10  # ✅ REDUCED from 20 to 10 for better detection
        )
        
        # ✅ ADDED: Backup FAST detector for fallback when ORB fails
        self.fast_detector = cv2.FastFeatureDetector_create(threshold=10)
        
        # ✅ ADDED: SIFT detector as secondary fallback
        try:
            self.sift_detector = cv2.SIFT_create(nfeatures=max_features//2)
            self.sift_available = True
        except AttributeError:
            self.sift_detector = None
            self.sift_available = False
            print("⚠️  SIFT not available, using ORB+FAST only")
        
        # Initialize matcher with optimized parameters
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Previous frame data
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Performance tracking
        self.detection_times = []
        self.matching_times = []
        self.feature_count_history = []
        
        # ✅ ADDED: Adaptive thresholding
        self.adaptive_threshold = 10
        self.min_threshold = 5
        self.max_threshold = 30
        
        print("✅ Enhanced Feature Detector initialized (FIXED VERSION)")
        print(f"   - ORB threshold: {10} (reduced for better detection)")
        print(f"   - FAST fallback: enabled")
        print(f"   - SIFT fallback: {'enabled' if self.sift_available else 'disabled'}")
    
    def detect_features_robust(self, frame: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        ✅ NEW: Robust feature detection with multiple fallback methods
        
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
        
        # ✅ Method 1: Try ORB first (primary method)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        detection_method = "ORB"
        
        # ✅ Method 2: If insufficient features, try FAST + ORB descriptors
        if len(keypoints) < 30:
            print(f"⚠️  ORB insufficient ({len(keypoints)} features), trying FAST fallback...")
            fast_keypoints = self.fast_detector.detect(gray, None)
            
            if len(fast_keypoints) > 0:
                # Compute ORB descriptors for FAST keypoints
                try:
                    keypoints, descriptors = self.orb.compute(gray, fast_keypoints)
                    detection_method = "FAST+ORB"
                    print(f"✅ FAST fallback successful: {len(keypoints)} features")
                except Exception as e:
                    print(f"❌ FAST+ORB failed: {e}")
        
        # ✅ Method 3: If still insufficient, try SIFT (if available)
        if len(keypoints) < 20 and self.sift_available:
            print(f"⚠️  FAST insufficient ({len(keypoints)} features), trying SIFT fallback...")
            try:
                sift_keypoints, sift_descriptors = self.sift_detector.detectAndCompute(gray, None)
                if len(sift_keypoints) > len(keypoints):
                    # Convert SIFT to ORB-compatible format
                    keypoints = sift_keypoints[:self.max_features]
                    # Recompute ORB descriptors for SIFT keypoints
                    keypoints, descriptors = self.orb.compute(gray, keypoints)
                    detection_method = "SIFT+ORB"
                    print(f"✅ SIFT fallback successful: {len(keypoints)} features")
            except Exception as e:
                print(f"❌ SIFT fallback failed: {e}")
        
        # ✅ Method 4: Last resort - lower ORB threshold dramatically
        if len(keypoints) < 15:
            print(f"⚠️  All methods insufficient ({len(keypoints)} features), lowering ORB threshold...")
            self.orb.setFastThreshold(5)  # Very low threshold
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            detection_method = "ORB_LOWTHRESH"
            print(f"✅ Low threshold ORB: {len(keypoints)} features")
            
            # Reset threshold for next frame
            self.orb.setFastThreshold(self.adaptive_threshold)
        
        # Track performance and adapt
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        self.feature_count_history.append(len(keypoints))
        
        if len(self.detection_times) > 100:
            self.detection_times.pop(0)
            self.feature_count_history.pop(0)
        
        # ✅ Adaptive threshold adjustment
        self._adapt_threshold()
        
        print(f"🎯 Feature Detection: {len(keypoints)} features using {detection_method} ({detection_time:.3f}s)")
        
        return keypoints, descriptors
    
    def detect_features(self, frame: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        ✅ UPDATED: Main detection method now uses robust detection
        
        Args:
            frame: Input image (BGR or grayscale)
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Use the robust detection method
        return self.detect_features_robust(frame)
    
    def _adapt_threshold(self):
        """✅ NEW: Adaptive threshold adjustment based on recent performance"""
        if len(self.feature_count_history) < 10:
            return
        
        recent_counts = self.feature_count_history[-10:]
        avg_features = np.mean(recent_counts)
        
        # Target: 50-200 features per frame
        target_min = 50
        target_max = 200
        
        if avg_features < target_min:
            # Too few features - lower threshold
            self.adaptive_threshold = max(self.min_threshold, self.adaptive_threshold - 2)
            self.orb.setFastThreshold(self.adaptive_threshold)
            print(f"🔽 Lowered ORB threshold to {self.adaptive_threshold} (avg features: {avg_features:.0f})")
            
        elif avg_features > target_max:
            # Too many features - raise threshold
            self.adaptive_threshold = min(self.max_threshold, self.adaptive_threshold + 2)
            self.orb.setFastThreshold(self.adaptive_threshold)
            print(f"🔼 Raised ORB threshold to {self.adaptive_threshold} (avg features: {avg_features:.0f})")
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray, 
                      ratio_threshold: float = 0.8) -> List:
        """
        ✅ ENHANCED: More lenient feature matching for real-world conditions
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            ratio_threshold: Ratio test threshold (RELAXED from 0.75 to 0.8)
            
        Returns:
            List of good matches
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        
        start_time = time.time()
        
        try:
            # ✅ ENHANCED: Multiple matching strategies
            good_matches = []
            
            # Strategy 1: Brute force matching with cross-check
            matches = self.matcher.match(desc1, desc2)
            
            if len(matches) > 0:
                # Sort matches by distance
                matches = sorted(matches, key=lambda x: x.distance)
                
                # ✅ RELAXED: More lenient distance threshold
                distances = [m.distance for m in matches]
                if len(distances) > 0:
                    mean_distance = np.mean(distances)
                    std_distance = np.std(distances)
                    # More permissive threshold
                    threshold = mean_distance + 1.0 * std_distance  # Was 0.5, now 1.0
                    
                    good_matches = [m for m in matches if m.distance < threshold]
            
            # Strategy 2: If insufficient matches, try KNN matching
            if len(good_matches) < 10:
                try:
                    # Create FLANN matcher for KNN
                    FLANN_INDEX_LSH = 6
                    index_params = dict(algorithm=FLANN_INDEX_LSH,
                                       table_number=6,
                                       key_size=12,
                                       multi_probe_level=1)
                    search_params = dict(checks=50)
                    
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    knn_matches = flann.knnMatch(desc1, desc2, k=2)
                    
                    # Lowe's ratio test with relaxed threshold
                    for match_pair in knn_matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < ratio_threshold * n.distance:  # Using relaxed ratio
                                good_matches.append(m)
                    
                    print(f"🔄 KNN matching found {len(good_matches)} matches")
                    
                except Exception as e:
                    print(f"⚠️  KNN matching failed: {e}")
            
            # Track performance
            matching_time = time.time() - start_time
            self.matching_times.append(matching_time)
            if len(self.matching_times) > 100:
                self.matching_times.pop(0)
            
            print(f"🎯 Feature Matching: {len(good_matches)} good matches ({matching_time:.3f}s)")
            
            return good_matches
            
        except Exception as e:
            print(f"❌ Feature matching error: {e}")
            return []
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        ✅ ENHANCED: Process frame with robust detection and matching
        
        Args:
            frame: Current frame
            
        Returns:
            Dictionary with processing results
        """
        # Detect features using robust method
        keypoints, descriptors = self.detect_features_robust(frame)
        
        results = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'matches': [],
            'num_features': len(keypoints) if keypoints else 0,
            'match_quality': 0.0,
            'detection_method': 'robust_multi_fallback'
        }
        
        # Match with previous frame if available
        if (self.prev_descriptors is not None and 
            descriptors is not None and len(descriptors) > 0):
            
            matches = self.match_features(self.prev_descriptors, descriptors)
            results['matches'] = matches
            results['match_quality'] = len(matches) / max(len(keypoints), 1)
            
            print(f"🎯 Frame Processing: {len(keypoints)} features, {len(matches)} matches, quality: {results['match_quality']:.2f}")
        
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
        
        # Draw keypoints with response-based coloring
        frame_with_features = frame.copy()
        
        for kp in keypoints:
            center = (int(kp.pt[0]), int(kp.pt[1]))
            # Color intensity based on response
            response = getattr(kp, 'response', 30)
            intensity = min(255, int(response * 5))
            draw_color = (0, intensity, 255 - intensity)
            
            # Draw circle with size based on response
            radius = max(2, min(6, int(response / 10)))
            cv2.circle(frame_with_features, center, radius, draw_color, 2)
        
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
        
        # Draw matches with quality-based coloring
        match_image = cv2.drawMatches(
            img1, kp1, img2, kp2, matches[:30],  # Show top 30 matches
            None, 
            matchColor=(0, 255, 0),  # Green for good matches
            singlePointColor=(255, 0, 0),  # Red for unmatched
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
        ✅ ENHANCED: Get comprehensive performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        stats = {
            'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0.0,
            'avg_matching_time': np.mean(self.matching_times) if self.matching_times else 0.0,
            'avg_features_detected': np.mean(self.feature_count_history) if self.feature_count_history else 0.0,
            'total_frames_processed': len(self.detection_times),
            'current_orb_threshold': self.adaptive_threshold,
            'feature_consistency': np.std(self.feature_count_history) if self.feature_count_history else 0.0,
            'detection_methods_available': {
                'orb': True,
                'fast': True,
                'sift': self.sift_available
            }
        }
        
        return stats
    
    def reset(self):
        """Reset detector state"""
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.detection_times = []
        self.matching_times = []
        self.feature_count_history = []
        
        # Reset adaptive threshold
        self.adaptive_threshold = 10
        self.orb.setFastThreshold(self.adaptive_threshold)
        
        print("✅ Enhanced Feature detector reset (FIXED VERSION)")

# Test function
def test_enhanced_feature_detector():
    """Test the enhanced feature detector with real scenarios"""
    import cv2
    
    detector = FeatureDetector(max_features=500)
    
    # Test 1: High contrast pattern
    test_image1 = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image1, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.circle(test_image1, (400, 300), 50, (128, 128, 128), -1)
    cv2.line(test_image1, (0, 240), (640, 240), (64, 64, 64), 5)
    
    # Test 2: Low contrast (challenging scenario)
    test_image2 = np.random.randint(120, 140, (480, 640, 3), dtype=np.uint8)
    
    # Test 3: Real-world simulation with noise
    test_image3 = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    test_images = [test_image1, test_image2, test_image3]
    test_names = ["High Contrast", "Low Contrast", "Noisy"]
    
    for i, (image, name) in enumerate(zip(test_images, test_names)):
        print(f"\n🧪 Testing {name} scenario...")
        keypoints, descriptors = detector.detect_features_robust(image)
        print(f"   Result: {len(keypoints)} features detected")
        
        if len(keypoints) > 0:
            result_image = detector.draw_features(image, keypoints)
            print(f"   ✅ {name} test passed")
        else:
            print(f"   ❌ {name} test failed - no features detected")
    
    # Performance stats
    stats = detector.get_performance_stats()
    print(f"\n📊 Performance Summary:")
    print(f"   Average features: {stats['avg_features_detected']:.1f}")
    print(f"   Average detection time: {stats['avg_detection_time']:.3f}s")
    print(f"   Current ORB threshold: {stats['current_orb_threshold']}")

if __name__ == "__main__":
    test_enhanced_feature_detector()