"""
ORB-SLAM3 Agricultural Variant
Specialized implementation for agricultural environments
Based on ORB-SLAM3AB research with agricultural optimizations
"""

import numpy as np
import cv2
import time
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import deque
import threading

@dataclass
class AgriculturalKeyframe:
    """Enhanced keyframe with agricultural context"""
    id: int
    timestamp: float
    pose: np.ndarray  # 4x4 transformation matrix
    features: np.ndarray  # 2D feature points
    descriptors: np.ndarray  # ORB descriptors
    depth_points: np.ndarray  # 3D points
    agricultural_features: Dict  # Crop rows, ground plane, etc.
    scene_complexity: float
    lighting_quality: float

@dataclass
class CropRow:
    """Crop row representation"""
    id: int
    start_point: np.ndarray
    end_point: np.ndarray
    direction_vector: np.ndarray
    confidence: float
    observations: List[int]  # Keyframe IDs that observe this row

@dataclass
class GroundPlane:
    """Ground plane representation"""
    normal: np.ndarray  # Plane normal vector
    distance: float  # Distance from origin
    confidence: float
    last_updated: float

class AgriculturalORBDetector:
    """
    Enhanced ORB detector optimized for agricultural scenes
    Implements adaptive thresholding and agricultural scene understanding
    """
    
    def __init__(self, max_features: int = 2000):
        """Initialize agricultural ORB detector"""
        self.max_features = max_features
        self.base_orb = cv2.ORB_create(
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
        
        # Agricultural scene analysis
        self.crop_row_detector = self._init_crop_row_detector()
        self.ground_plane_estimator = self._init_ground_plane_estimator()
        
        # Adaptive parameters
        self.current_threshold = 20
        self.min_threshold = 5
        self.max_threshold = 50
        self.adaptation_rate = 0.1
        
        # Performance tracking
        self.detection_history = deque(maxlen=50)
        
    def _init_crop_row_detector(self) -> Dict:
        """Initialize crop row detection parameters"""
        return {
            'line_detector': cv2.createLineSegmentDetector(),
            'parallel_threshold': 0.15,  # radians (~8.5 degrees)
            'min_line_length': 100,
            'max_line_gap': 30,
            'clustering_distance': 50  # pixels
        }
    
    def _init_ground_plane_estimator(self) -> Dict:
        """Initialize ground plane estimation parameters"""
        return {
            'ransac_iterations': 1000,
            'distance_threshold': 0.05,  # 5cm
            'min_inliers': 100,
            'plane_update_threshold': 0.1
        }
    
    def detect_agricultural_features(self, frame: np.ndarray, depth_frame: np.ndarray = None) -> Dict:
        """
        Detect features optimized for agricultural scenes
        
        Args:
            frame: Input color/grayscale image
            depth_frame: Optional depth image for 3D analysis
            
        Returns:
            Dictionary with detected features and agricultural context
        """
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Pre-process for agricultural scenes
            enhanced_gray = self._preprocess_agricultural(gray)
            
            # Detect ORB features with adaptive thresholding
            keypoints, descriptors = self._adaptive_orb_detection(enhanced_gray)
            
            # Detect crop rows
            crop_rows = self._detect_crop_rows(enhanced_gray)
            
            # Estimate ground plane if depth available
            ground_plane = None
            if depth_frame is not None:
                ground_plane = self._estimate_ground_plane(gray, depth_frame)
            
            # Analyze scene complexity
            scene_complexity = self._analyze_scene_complexity(gray)
            lighting_quality = self._analyze_lighting_quality(gray)
            
            # Combine results
            results = {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'num_features': len(keypoints) if keypoints else 0,
                'crop_rows': crop_rows,
                'ground_plane': ground_plane,
                'scene_complexity': scene_complexity,
                'lighting_quality': lighting_quality,
                'agricultural_score': self._calculate_agricultural_score(crop_rows, ground_plane),
                'detection_quality': self._assess_detection_quality(keypoints, descriptors)
            }
            
            # Update detection history
            self.detection_history.append({
                'num_features': results['num_features'],
                'agricultural_score': results['agricultural_score'],
                'timestamp': time.time()
            })
            
            # Adapt detection parameters
            self._adapt_detection_parameters(results)
            
            return results
            
        except Exception as e:
            print(f"Agricultural feature detection error: {e}")
            return {
                'keypoints': [],
                'descriptors': np.array([]),
                'num_features': 0,
                'crop_rows': [],
                'ground_plane': None,
                'scene_complexity': 0.5,
                'lighting_quality': 0.5,
                'agricultural_score': 0.0,
                'detection_quality': 0.0
            }
    
    def _preprocess_agricultural(self, gray: np.ndarray) -> np.ndarray:
        """Pre-process image for agricultural feature detection"""
        try:
            # Apply CLAHE for outdoor lighting variations
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Slight Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            return enhanced
            
        except Exception as e:
            print(f"Agricultural preprocessing error: {e}")
            return gray
    
    def _adaptive_orb_detection(self, gray: np.ndarray) -> Tuple[List, np.ndarray]:
        """Adaptive ORB detection with agricultural optimization"""
        try:
            # Update ORB threshold
            self.base_orb.setFastThreshold(int(self.current_threshold))
            
            # Detect keypoints and descriptors
            keypoints, descriptors = self.base_orb.detectAndCompute(gray, None)
            
            # Filter keypoints for agricultural scenes
            if keypoints:
                keypoints = self._filter_agricultural_keypoints(keypoints, gray)
                
                # Recompute descriptors for filtered keypoints
                if len(keypoints) > 0:
                    keypoints, descriptors = self.base_orb.compute(gray, keypoints)
            
            return keypoints, descriptors
            
        except Exception as e:
            print(f"Adaptive ORB detection error: {e}")
            return [], np.array([])
    
    def _filter_agricultural_keypoints(self, keypoints: List, gray: np.ndarray) -> List:
        """Filter keypoints for agricultural relevance"""
        try:
            if not keypoints:
                return keypoints
            
            filtered_keypoints = []
            height, width = gray.shape
            
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                
                # Skip points too close to image borders
                if x < 20 or y < 20 or x > width - 20 or y > height - 20:
                    continue
                
                # Check local contrast (important for agricultural scenes)
                local_patch = gray[y-5:y+6, x-5:x+6]
                if local_patch.size > 0:
                    contrast = np.std(local_patch)
                    if contrast < 10:  # Skip low-contrast areas
                        continue
                
                # Prefer points in lower 2/3 of image (ground level features)
                if y > height * 0.33:
                    kp.response *= 1.2  # Boost response for ground-level features
                
                filtered_keypoints.append(kp)
            
            # Sort by response and keep best features
            filtered_keypoints.sort(key=lambda x: x.response, reverse=True)
            return filtered_keypoints[:self.max_features]
            
        except Exception as e:
            print(f"Keypoint filtering error: {e}")
            return keypoints
    
    def _detect_crop_rows(self, gray: np.ndarray) -> List[CropRow]:
        """Detect crop rows using line detection and clustering"""
        try:
            # Edge detection optimized for crop rows
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect line segments
            detector = self.crop_row_detector['line_detector']
            lines = detector.detect(edges)
            
            if lines is None or len(lines) == 0:
                return []
            
            # Convert to standard format
            line_segments = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if length > self.crop_row_detector['min_line_length']:
                    angle = math.atan2(y2-y1, x2-x1)
                    line_segments.append({
                        'start': np.array([x1, y1]),
                        'end': np.array([x2, y2]),
                        'angle': angle,
                        'length': length
                    })
            
            # Cluster parallel lines to form crop rows
            crop_rows = self._cluster_parallel_lines(line_segments)
            
            return crop_rows
            
        except Exception as e:
            print(f"Crop row detection error: {e}")
            return []
    
    def _cluster_parallel_lines(self, line_segments: List[Dict]) -> List[CropRow]:
        """Cluster parallel lines into crop rows"""
        try:
            if len(line_segments) < 2:
                return []
            
            crop_rows = []
            used_lines = set()
            threshold = self.crop_row_detector['parallel_threshold']
            
            for i, line1 in enumerate(line_segments):
                if i in used_lines:
                    continue
                
                # Find parallel lines
                parallel_group = [line1]
                used_lines.add(i)
                
                for j, line2 in enumerate(line_segments[i+1:], i+1):
                    if j in used_lines:
                        continue
                    
                    angle_diff = abs(line1['angle'] - line2['angle'])
                    angle_diff = min(angle_diff, math.pi - angle_diff)
                    
                    if angle_diff < threshold:
                        parallel_group.append(line2)
                        used_lines.add(j)
                
                # Create crop row if enough parallel lines
                if len(parallel_group) >= 2:
                    crop_row = self._create_crop_row_from_group(parallel_group, len(crop_rows))
                    if crop_row:
                        crop_rows.append(crop_row)
            
            return crop_rows
            
        except Exception as e:
            print(f"Line clustering error: {e}")
            return []
    
    def _create_crop_row_from_group(self, line_group: List[Dict], row_id: int) -> Optional[CropRow]:
        """Create crop row from group of parallel lines"""
        try:
            if not line_group:
                return None
            
            # Calculate average direction
            angles = [line['angle'] for line in line_group]
            avg_angle = np.mean(angles)
            direction_vector = np.array([math.cos(avg_angle), math.sin(avg_angle)])
            
            # Find extent of crop row
            all_points = []
            for line in line_group:
                all_points.extend([line['start'], line['end']])
            
            all_points = np.array(all_points)
            
            # Project points onto direction vector to find extent
            projections = all_points @ direction_vector
            min_proj, max_proj = np.min(projections), np.max(projections)
            
            # Calculate start and end points
            center = np.mean(all_points, axis=0)
            half_length = (max_proj - min_proj) / 2
            
            start_point = center - direction_vector * half_length
            end_point = center + direction_vector * half_length
            
            # Calculate confidence based on line consistency
            confidence = min(1.0, len(line_group) / 5.0)
            
            return CropRow(
                id=row_id,
                start_point=start_point,
                end_point=end_point,
                direction_vector=direction_vector,
                confidence=confidence,
                observations=[]
            )
            
        except Exception as e:
            print(f"Crop row creation error: {e}")
            return None
    
    def _estimate_ground_plane(self, gray: np.ndarray, depth_frame: np.ndarray) -> Optional[GroundPlane]:
        """Estimate ground plane using RANSAC on depth data"""
        try:
            height, width = depth_frame.shape
            
            # Sample points from lower portion of image
            sample_region = depth_frame[height//2:, :]
            valid_mask = (sample_region > 200) & (sample_region < 8000)
            
            if np.sum(valid_mask) < 100:
                return None
            
            # Get 3D points (simplified camera model)
            fx, fy = 615.0, 615.0
            cx, cy = width/2, height/2
            
            y_coords, x_coords = np.where(valid_mask)
            y_coords += height//2  # Adjust for region offset
            depths = sample_region[valid_mask] / 1000.0  # Convert to meters
            
            # Back-project to 3D
            X = (x_coords - cx) * depths / fx
            Y = (y_coords - cy) * depths / fy
            Z = depths
            
            points_3d = np.column_stack([X, Y, Z])
            
            # RANSAC plane fitting
            best_plane = None
            best_inliers = 0
            
            for _ in range(self.ground_plane_estimator['ransac_iterations']):
                # Sample 3 points
                if len(points_3d) < 3:
                    break
                    
                sample_idx = np.random.choice(len(points_3d), 3, replace=False)
                sample_points = points_3d[sample_idx]
                
                # Fit plane
                v1 = sample_points[1] - sample_points[0]
                v2 = sample_points[2] - sample_points[0]
                normal = np.cross(v1, v2)
                
                if np.linalg.norm(normal) < 1e-6:
                    continue
                
                normal = normal / np.linalg.norm(normal)
                distance = -np.dot(normal, sample_points[0])
                
                # Count inliers
                plane_distances = np.abs(np.dot(points_3d, normal) + distance)
                inliers = np.sum(plane_distances < self.ground_plane_estimator['distance_threshold'])
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_plane = (normal, distance)
            
            if best_inliers >= self.ground_plane_estimator['min_inliers']:
                normal, distance = best_plane
                confidence = min(1.0, best_inliers / 500.0)
                
                return GroundPlane(
                    normal=normal,
                    distance=distance,
                    confidence=confidence,
                    last_updated=time.time()
                )
            
            return None
            
        except Exception as e:
            print(f"Ground plane estimation error: {e}")
            return None
    
    def _analyze_scene_complexity(self, gray: np.ndarray) -> float:
        """Analyze agricultural scene complexity"""
        try:
            # Calculate gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Calculate complexity metrics
            mean_gradient = np.mean(gradient_mag)
            std_gradient = np.std(gradient_mag)
            edge_density = np.sum(gradient_mag > 50) / gradient_mag.size
            
            # Combine into complexity score
            complexity = (mean_gradient/100.0 + std_gradient/50.0 + edge_density) / 3.0
            return np.clip(complexity, 0.0, 1.0)
            
        except Exception as e:
            print(f"Scene complexity analysis error: {e}")
            return 0.5
    
    def _analyze_lighting_quality(self, gray: np.ndarray) -> float:
        """Analyze lighting quality for agricultural scenes"""
        try:
            mean_intensity = np.mean(gray)
            std_intensity = np.std(gray)
            
            # Check for over/under exposure
            overexposed = np.sum(gray > 240) / gray.size
            underexposed = np.sum(gray < 15) / gray.size
            
            # Calculate quality metrics
            intensity_score = 1.0 - abs(mean_intensity - 128) / 128.0
            contrast_score = min(std_intensity / 64.0, 1.0)
            exposure_score = 1.0 - (overexposed + underexposed)
            
            quality = (intensity_score + contrast_score + exposure_score) / 3.0
            return np.clip(quality, 0.0, 1.0)
            
        except Exception as e:
            print(f"Lighting quality analysis error: {e}")
            return 0.5
    
    def _calculate_agricultural_score(self, crop_rows: List[CropRow], 
                                   ground_plane: Optional[GroundPlane]) -> float:
        """Calculate overall agricultural scene score"""
        try:
            score = 0.0
            
            # Crop rows contribution
            if crop_rows:
                crop_score = min(len(crop_rows) / 5.0, 1.0)  # Max score for 5+ rows
                avg_confidence = np.mean([row.confidence for row in crop_rows])
                score += 0.6 * crop_score * avg_confidence
            
            # Ground plane contribution
            if ground_plane:
                score += 0.4 * ground_plane.confidence
            
            return min(score, 1.0)
            
        except Exception as e:
            print(f"Agricultural score calculation error: {e}")
            return 0.0
    
    def _assess_detection_quality(self, keypoints: List, descriptors: np.ndarray) -> float:
        """Assess quality of feature detection"""
        try:
            if not keypoints:
                return 0.0
            
            # Response quality
            responses = [kp.response for kp in keypoints]
            avg_response = np.mean(responses)
            response_score = min(avg_response / 50.0, 1.0)
            
            # Spatial distribution
            points = np.array([kp.pt for kp in keypoints])
            spatial_std = np.std(points, axis=0)
            distribution_score = min(np.mean(spatial_std) / 100.0, 1.0)
            
            # Feature count score
            count_score = min(len(keypoints) / 500.0, 1.0)
            
            return (response_score + distribution_score + count_score) / 3.0
            
        except Exception as e:
            print(f"Detection quality assessment error: {e}")
            return 0.5
    
    def _adapt_detection_parameters(self, results: Dict):
        """Adapt detection parameters based on results"""
        try:
            num_features = results['num_features']
            target_features = self.max_features // 2
            
            # Adjust threshold based on feature count
            if num_features < target_features * 0.7:
                adjustment = -5 * (1 - num_features / target_features)
            elif num_features > target_features * 1.3:
                adjustment = 5 * (num_features / target_features - 1)
            else:
                adjustment = 0
            
            # Apply adaptation
            adjustment *= self.adaptation_rate
            new_threshold = self.current_threshold + adjustment
            
            self.current_threshold = np.clip(
                new_threshold, 
                self.min_threshold, 
                self.max_threshold
            )
            
        except Exception as e:
            print(f"Parameter adaptation error: {e}")

class ORBSLAMAgriculturalCore:
    """
    Core ORB-SLAM3 implementation optimized for agricultural environments
    Integrates crop row detection, ground plane estimation, and agricultural scene understanding
    """
    
    def __init__(self, camera_matrix: np.ndarray, max_features: int = 2000):
        """Initialize agricultural ORB-SLAM3 core"""
        self.camera_matrix = camera_matrix
        self.max_features = max_features
        
        # Agricultural ORB detector
        self.orb_detector = AgriculturalORBDetector(max_features)
        
        # SLAM components
        self.keyframes: List[AgriculturalKeyframe] = []
        self.crop_rows: List[CropRow] = []
        self.current_ground_plane: Optional[GroundPlane] = None
        
        # Tracking state
        self.current_pose = np.eye(4)
        self.trajectory = [np.array([0.0, 0.0, 0.0])]
        self.is_initialized = False
        self.tracking_state = "INITIALIZING"  # INITIALIZING, TRACKING, LOST
        
        # Performance monitoring
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)
        self.agricultural_scores = deque(maxlen=100)
        
        # Thread safety
        self.slam_lock = threading.Lock()
        
        print(f"ORB-SLAM3 Agricultural Core initialized:")
        print(f"  - Max features: {max_features}")
        print(f"  - Crop row detection: enabled")
        print(f"  - Ground plane estimation: enabled")
        print(f"  - Agricultural scene analysis: active")
    
    def process_frame_agricultural(self, color_frame: np.ndarray, 
                                 depth_frame: np.ndarray, 
                                 timestamp: float) -> Dict:
        """
        Process frame with agricultural SLAM pipeline
        
        Args:
            color_frame: RGB color image
            depth_frame: Depth image
            timestamp: Frame timestamp
            
        Returns:
            Dictionary with SLAM results and agricultural context
        """
        start_time = time.time()
        
        with self.slam_lock:
            self.frame_count += 1
            
            try:
                # Detect agricultural features
                feature_results = self.orb_detector.detect_agricultural_features(
                    color_frame, depth_frame
                )
                
                # Initialize results
                results = {
                    'timestamp': timestamp,
                    'frame_id': self.frame_count,
                    'tracking_state': self.tracking_state,
                    'pose_estimated': False,
                    'current_pose': self.current_pose.copy(),
                    'position': self.trajectory[-1].copy(),
                    'num_features': feature_results['num_features'],
                    'agricultural_features': {
                        'crop_rows': feature_results['crop_rows'],
                        'ground_plane': feature_results['ground_plane'],
                        'agricultural_score': feature_results['agricultural_score'],
                        'scene_complexity': feature_results['scene_complexity'],
                        'lighting_quality': feature_results['lighting_quality']
                    },
                    'num_keyframes': len(self.keyframes),
                    'num_crop_rows': len(self.crop_rows),
                    'processing_time': 0.0,
                    'debug_info': 'Processing agricultural frame...'
                }
                
                # Process based on current state
                if not self.is_initialized:
                    success = self._initialize_agricultural_slam(
                        color_frame, depth_frame, feature_results, timestamp
                    )
                    if success:
                        self.is_initialized = True
                        self.tracking_state = "TRACKING"
                        results['pose_estimated'] = True
                        results['debug_info'] = "Agricultural SLAM initialized successfully"
                        print("🌾 Agricultural ORB-SLAM3 initialized")
                    else:
                        results['debug_info'] = f"Initialization: {feature_results['num_features']} features detected"
                
                else:
                    # Main tracking
                    tracking_success = self._track_agricultural_frame(
                        color_frame, depth_frame, feature_results, timestamp
                    )
                    
                    if tracking_success:
                        self.tracking_state = "TRACKING"
                        results['pose_estimated'] = True
                        results['current_pose'] = self.current_pose.copy()
                        results['position'] = self.trajectory[-1].copy()
                        results['debug_info'] = f"Tracking: {feature_results['num_features']} features, {len(self.crop_rows)} crop rows"
                    else:
                        self.tracking_state = "LOST"
                        results['debug_info'] = "Tracking lost - attempting recovery"
                
                # Update agricultural map
                self._update_agricultural_map(feature_results)
                
                # Update results
                results.update({
                    'tracking_state': self.tracking_state,
                    'num_crop_rows': len(self.crop_rows),
                    'ground_plane_valid': self.current_ground_plane is not None
                })
                
                # Performance tracking
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self.agricultural_scores.append(feature_results['agricultural_score'])
                
                results['processing_time'] = processing_time
                
                return results
                
            except Exception as e:
                print(f"Agricultural SLAM processing error: {e}")
                return self._get_error_results(timestamp, str(e))
    
    def _initialize_agricultural_slam(self, color_frame: np.ndarray, 
                                   depth_frame: np.ndarray,
                                   feature_results: Dict, 
                                   timestamp: float) -> bool:
        """Initialize agricultural SLAM with first keyframe"""
        try:
            # Require sufficient features
            if feature_results['num_features'] < 100:
                return False
            
            # Require reasonable agricultural scene
            if feature_results['agricultural_score'] < 0.2:
                print("Warning: Low agricultural scene score for initialization")
            
            # Create first keyframe
            keyframe = AgriculturalKeyframe(
                id=0,
                timestamp=timestamp,
                pose=np.eye(4),
                features=np.array([kp.pt for kp in feature_results['keypoints']]),
                descriptors=feature_results['descriptors'],
                depth_points=self._extract_3d_points(feature_results['keypoints'], depth_frame),
                agricultural_features={
                    'crop_rows': feature_results['crop_rows'],
                    'ground_plane': feature_results['ground_plane']
                },
                scene_complexity=feature_results['scene_complexity'],
                lighting_quality=feature_results['lighting_quality']
            )
            
            self.keyframes.append(keyframe)
            
            # Initialize agricultural map
            self.crop_rows = feature_results['crop_rows'].copy()
            self.current_ground_plane = feature_results['ground_plane']
            
            # Initialize trajectory
            self.current_pose = np.eye(4)
            self.trajectory = [np.array([0.0, 0.0, 0.0])]
            
            print(f"🌾 Agricultural keyframe 0 created: {len(keyframe.depth_points)} 3D points")
            if self.crop_rows:
                print(f"🌾 Detected {len(self.crop_rows)} crop rows")
            if self.current_ground_plane:
                print(f"🌾 Ground plane estimated (confidence: {self.current_ground_plane.confidence:.2f})")
            
            return True
            
        except Exception as e:
            print(f"Agricultural SLAM initialization error: {e}")
            return False
    
    def _track_agricultural_frame(self, color_frame: np.ndarray,
                                depth_frame: np.ndarray,
                                feature_results: Dict,
                                timestamp: float) -> bool:
        """Track frame using agricultural features"""
        try:
            # Simplified tracking for now - would implement full ORB-SLAM3 tracking here
            # For this implementation, we'll do basic feature matching and pose estimation
            
            if not self.keyframes:
                return False
            
            last_keyframe = self.keyframes[-1]
            
            # Match features with last keyframe
            matches = self._match_features_agricultural(
                last_keyframe.descriptors,
                feature_results['descriptors']
            )
            
            if len(matches) < 20:  # Insufficient matches
                return False
            
            # Estimate pose (simplified - would use full ORB-SLAM3 pose estimation)
            pose_success = self._estimate_pose_agricultural(
                last_keyframe, feature_results, matches, depth_frame
            )
            
            if pose_success:
                # Update trajectory
                position = self.current_pose[:3, 3]
                self.trajectory.append(position.copy())
                
                # Check if new keyframe needed
                if self._should_create_agricultural_keyframe(feature_results):
                    self._create_agricultural_keyframe(
                        color_frame, depth_frame, feature_results, timestamp
                    )
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Agricultural tracking error: {e}")
            return False
    
    def _extract_3d_points(self, keypoints: List, depth_frame: np.ndarray) -> np.ndarray:
        """Extract 3D points from keypoints using depth"""
        try:
            points_3d = []
            fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
            
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                
                if (0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]):
                    depth = depth_frame[y, x]
                    
                    if 200 <= depth <= 8000:  # Valid depth range
                        z = depth / 1000.0  # Convert to meters
                        x_3d = (x - cx) * z / fx
                        y_3d = (y - cy) * z / fy
                        
                        points_3d.append([x_3d, y_3d, z])
            
            return np.array(points_3d, dtype=np.float32)
            
        except Exception as e:
            print(f"3D point extraction error: {e}")
            return np.array([])
    
    def _match_features_agricultural(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Match features with agricultural optimization"""
        try:
            if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
                return []
            
            # Use brute force matcher with Hamming distance
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(desc1, desc2)
            
            # Sort by distance and take best matches
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Filter matches based on distance threshold
            good_matches = []
            if matches:
                distances = [m.distance for m in matches]
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                threshold = mean_dist + std_dist
                
                good_matches = [m for m in matches if m.distance < threshold]
            
            return good_matches[:100]  # Limit to top 100 matches
            
        except Exception as e:
            print(f"Feature matching error: {e}")
            return []
    
    def _estimate_pose_agricultural(self, last_keyframe: AgriculturalKeyframe,
                                  feature_results: Dict, matches: List,
                                  depth_frame: np.ndarray) -> bool:
        """Estimate pose using agricultural constraints"""
        try:
            if len(matches) < 8:
                return False
            
            # Extract matched points
            points_3d = []
            points_2d = []
            
            for match in matches:
                if match.queryIdx < len(last_keyframe.depth_points):
                    point_3d = last_keyframe.depth_points[match.queryIdx]
                    point_2d = feature_results['keypoints'][match.trainIdx].pt
                    
                    points_3d.append(point_3d)
                    points_2d.append(point_2d)
            
            if len(points_3d) < 8:
                return False
            
            points_3d = np.array(points_3d, dtype=np.float32).reshape(-1, 1, 3)
            points_2d = np.array(points_2d, dtype=np.float32).reshape(-1, 1, 2)
            
            # Solve PnP with RANSAC
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d,
                points_2d,
                self.camera_matrix,
                np.zeros(4),
                iterationsCount=1000,
                reprojectionError=2.0,
                confidence=0.99
            )
            
            if success and inliers is not None and len(inliers) >= 8:
                # Convert to pose matrix
                R, _ = cv2.Rodrigues(rvec)
                
                # Apply agricultural constraints (e.g., ground plane)
                if self.current_ground_plane:
                    tvec = self._apply_ground_plane_constraint(tvec, R)
                
                # Update pose
                self.current_pose[:3, :3] = R
                self.current_pose[:3, 3] = self.trajectory[-1] + tvec.ravel()
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Pose estimation error: {e}")
            return False
    
    def _apply_ground_plane_constraint(self, tvec: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Apply ground plane constraint to pose estimation"""
        try:
            if not self.current_ground_plane:
                return tvec
            
            # Project movement onto ground plane
            movement = tvec.ravel()
            normal = self.current_ground_plane.normal
            
            # Remove component perpendicular to ground plane
            perpendicular_component = np.dot(movement, normal) * normal
            constrained_movement = movement - 0.5 * perpendicular_component
            
            return constrained_movement.reshape(-1, 1)
            
        except Exception as e:
            print(f"Ground plane constraint error: {e}")
            return tvec
    
    def _should_create_agricultural_keyframe(self, feature_results: Dict) -> bool:
        """Determine if new keyframe needed with agricultural criteria"""
        try:
            if not self.keyframes:
                return True
            
            last_keyframe = self.keyframes[-1]
            current_position = self.current_pose[:3, 3]
            last_position = last_keyframe.pose[:3, 3]
            
            # Distance threshold
            distance = np.linalg.norm(current_position - last_position)
            distance_threshold = 0.3  # 30cm
            
            # Feature quality threshold
            feature_ratio = feature_results['num_features'] / max(len(last_keyframe.features), 1)
            
            # Agricultural scene change threshold
            agricultural_change = abs(
                feature_results['agricultural_score'] - 
                last_keyframe.agricultural_features.get('agricultural_score', 0.5)
            )
            
            return (distance > distance_threshold or 
                    feature_ratio < 0.7 or 
                    agricultural_change > 0.3)
            
        except Exception as e:
            print(f"Keyframe decision error: {e}")
            return False
    
    def _create_agricultural_keyframe(self, color_frame: np.ndarray,
                                   depth_frame: np.ndarray,
                                   feature_results: Dict,
                                   timestamp: float):
        """Create new agricultural keyframe"""
        try:
            keyframe_id = len(self.keyframes)
            
            keyframe = AgriculturalKeyframe(
                id=keyframe_id,
                timestamp=timestamp,
                pose=self.current_pose.copy(),
                features=np.array([kp.pt for kp in feature_results['keypoints']]),
                descriptors=feature_results['descriptors'],
                depth_points=self._extract_3d_points(feature_results['keypoints'], depth_frame),
                agricultural_features=feature_results.copy(),
                scene_complexity=feature_results['scene_complexity'],
                lighting_quality=feature_results['lighting_quality']
            )
            
            self.keyframes.append(keyframe)
            print(f"🌾 Agricultural keyframe {keyframe_id} created: {len(keyframe.depth_points)} 3D points")
            
        except Exception as e:
            print(f"Keyframe creation error: {e}")
    
    def _update_agricultural_map(self, feature_results: Dict):
        """Update agricultural map with new observations"""
        try:
            # Update crop rows
            new_crop_rows = feature_results['crop_rows']
            for new_row in new_crop_rows:
                # Find matching existing row or add new one
                matched = False
                for existing_row in self.crop_rows:
                    if self._crop_rows_similar(existing_row, new_row):
                        # Update existing row
                        existing_row.confidence = (existing_row.confidence + new_row.confidence) / 2
                        matched = True
                        break
                
                if not matched:
                    self.crop_rows.append(new_row)
            
            # Update ground plane
            new_ground_plane = feature_results['ground_plane']
            if new_ground_plane:
                if self.current_ground_plane:
                    # Update existing ground plane
                    self.current_ground_plane.confidence = (
                        self.current_ground_plane.confidence + new_ground_plane.confidence
                    ) / 2
                    self.current_ground_plane.last_updated = time.time()
                else:
                    self.current_ground_plane = new_ground_plane
            
        except Exception as e:
            print(f"Agricultural map update error: {e}")
    
    def _crop_rows_similar(self, row1: CropRow, row2: CropRow) -> bool:
        """Check if two crop rows are similar enough to be the same row"""
        try:
            # Check direction similarity
            dot_product = np.dot(row1.direction_vector, row2.direction_vector)
            angle_diff = np.arccos(np.clip(abs(dot_product), 0, 1))
            
            if angle_diff > 0.2:  # ~11 degrees
                return False
            
            # Check spatial proximity
            distance = np.linalg.norm(row1.start_point - row2.start_point)
            if distance > 100:  # 100 pixels
                return False
            
            return True
            
        except Exception as e:
            print(f"Crop row similarity check error: {e}")
            return False
    
    def _get_error_results(self, timestamp: float, error_msg: str) -> Dict:
        """Get error results structure"""
        return {
            'timestamp': timestamp,
            'frame_id': self.frame_count,
            'tracking_state': "ERROR",
            'pose_estimated': False,
            'current_pose': self.current_pose.copy(),
            'position': self.trajectory[-1].copy() if self.trajectory else np.array([0, 0, 0]),
            'num_features': 0,
            'agricultural_features': {
                'crop_rows': [],
                'ground_plane': None,
                'agricultural_score': 0.0,
                'scene_complexity': 0.0,
                'lighting_quality': 0.0
            },
            'num_keyframes': len(self.keyframes),
            'num_crop_rows': len(self.crop_rows),
            'processing_time': 0.0,
            'debug_info': f"Error: {error_msg}"
        }
    
    def get_agricultural_map_data(self) -> Dict:
        """Get comprehensive agricultural map data"""
        with self.slam_lock:
            return {
                'keyframes': self.keyframes.copy(),
                'crop_rows': self.crop_rows.copy(),
                'ground_plane': self.current_ground_plane,
                'trajectory': np.array(self.trajectory),
                'current_pose': self.current_pose.copy(),
                'agricultural_stats': {
                    'avg_agricultural_score': np.mean(self.agricultural_scores) if self.agricultural_scores else 0.0,
                    'num_crop_rows_detected': len(self.crop_rows),
                    'ground_plane_confidence': self.current_ground_plane.confidence if self.current_ground_plane else 0.0,
                    'total_keyframes': len(self.keyframes)
                }
            }
    
    def reset_agricultural(self):
        """Reset agricultural SLAM system"""
        with self.slam_lock:
            self.keyframes.clear()
            self.crop_rows.clear()
            self.current_ground_plane = None
            self.current_pose = np.eye(4)
            self.trajectory = [np.array([0.0, 0.0, 0.0])]
            self.is_initialized = False
            self.tracking_state = "INITIALIZING"
            self.frame_count = 0
            self.processing_times.clear()
            self.agricultural_scores.clear()
            
            print("🌾 Agricultural ORB-SLAM3 reset")

# Test function
def test_orb_slam3_agricultural():
    """Test agricultural ORB-SLAM3 implementation"""
    print("Testing ORB-SLAM3 Agricultural...")
    
    # Create camera matrix
    camera_matrix = np.array([
        [615.0, 0, 320.0],
        [0, 615.0, 240.0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Create SLAM system
    slam = ORBSLAMAgriculturalCore(camera_matrix)
    
    # Process test frames
    for i in range(10):
        color_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth_frame = np.random.randint(500, 3000, (480, 640), dtype=np.uint16)
        timestamp = time.time() + i * 0.033
        
        results = slam.process_frame_agricultural(color_frame, depth_frame, timestamp)
        
        print(f"Frame {i}: {results['tracking_state']}, "
              f"Features: {results['num_features']}, "
              f"Agricultural Score: {results['agricultural_features']['agricultural_score']:.2f}")
    
    # Print final map data
    map_data = slam.get_agricultural_map_data()
    print(f"\nFinal Map:")
    print(f"  Keyframes: {len(map_data['keyframes'])}")
    print(f"  Crop Rows: {len(map_data['crop_rows'])}")
    print(f"  Ground Plane: {'Yes' if map_data['ground_plane'] else 'No'}")

if __name__ == "__main__":
    test_orb_slam3_agricultural()