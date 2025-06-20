"""
Agricultural SLAM Core Engine
Main SLAM pipeline optimized for agricultural environments
Based on ORB-SLAM3 with agricultural scene understanding
FIXED: Data structure compatibility with camera widget
"""

import numpy as np
import cv2
import time
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import math

from src.core.camera_manager import CameraManager
from src.core.feature_detector import FeatureDetector
from src.core.visual_odometry import VisualOdometry
from src.utils.config_manager import get_config_manager
from src.utils.data_logger import get_data_logger

@dataclass
class Keyframe:
    """Enhanced keyframe for agricultural SLAM"""
    id: int
    timestamp: float
    pose: np.ndarray
    features_2d: np.ndarray
    features_3d: np.ndarray
    descriptors: np.ndarray
    frame_quality: float
    scene_type: str  # 'field', 'crop_rows', 'mixed', 'structures'

@dataclass
class MapPoint:
    """3D map point with agricultural context"""
    id: int
    position: np.ndarray
    observations: List[int]  # Keyframe IDs that observe this point
    descriptor: np.ndarray
    reliability: float
    point_type: str  # 'ground', 'crop', 'structure', 'sky'

# ============ CRITICAL FIX: Added structured data classes ============
@dataclass
class CropRowDetection:
    """Structured crop row detection result for camera widget compatibility"""
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    confidence: float
    length: float
    angle: float
    row_id: int

@dataclass
class GroundPlaneDetection:
    """Structured ground plane detection result"""
    plane_equation: np.ndarray
    confidence: float
    inlier_count: int
    distance_to_camera: float
    normal_vector: np.ndarray

class AgriculturalSceneAnalyzer:
    """Analyzes agricultural scenes for SLAM optimization - FIXED DATA STRUCTURES"""
    
    def __init__(self):
        self.crop_row_detector = self._init_crop_row_detector()
        self.ground_plane_estimator = self._init_ground_plane_estimator()
        
    def _init_crop_row_detector(self):
        """Initialize crop row detection using Hough lines"""
        return {
            'hough_threshold': 50,
            'min_line_length': 100,
            'max_line_gap': 20,
            'parallel_threshold': 0.1  # radians
        }
    
    def _init_ground_plane_estimator(self):
        """Initialize ground plane estimation"""
        return {
            'ransac_threshold': 0.05,  # 5cm
            'min_inliers': 100,
            'max_iterations': 1000
        }
    
    def analyze_scene(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> Dict:
        """Analyze agricultural scene characteristics - FIXED OUTPUT FORMAT"""
        try:
            scene_info = {
                'scene_type': 'field',
                'crop_rows_detected': False,
                'crop_rows': [],  # Will contain CropRowDetection objects
                'ground_plane': None,  # Will contain GroundPlaneDetection object
                'structure_density': 0.0,
                'sky_ratio': 0.0,
                'recommended_features': 'orb',
                'scene_complexity': 0.0,
                'lighting_quality': 0.0,
                'agricultural_score': 0.0
            }
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            
            # FIXED: Detect crop rows with structured output
            crop_rows = self._detect_crop_rows(gray)
            if len(crop_rows) >= 2:
                scene_info['crop_rows_detected'] = True
                scene_info['crop_rows'] = crop_rows
                scene_info['scene_type'] = 'crop_rows'
                scene_info['recommended_features'] = 'lines_and_points'
            
            # FIXED: Estimate ground plane with structured output
            ground_plane = self._estimate_ground_plane(depth_frame)
            if ground_plane is not None:
                scene_info['ground_plane'] = ground_plane
            
            # Analyze sky ratio (typically upper portion of image)
            upper_region = gray[:gray.shape[0]//3, :]
            sky_threshold = np.percentile(gray, 85)
            sky_pixels = np.sum(upper_region > sky_threshold)
            scene_info['sky_ratio'] = sky_pixels / upper_region.size
            
            # Structure density (high gradient areas)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            structure_pixels = np.sum(gradient_magnitude > 50)
            scene_info['structure_density'] = structure_pixels / gradient_magnitude.size
            
            # FIXED: Calculate additional metrics for camera widget
            scene_info['scene_complexity'] = min(1.0, scene_info['structure_density'] * 2.0)
            scene_info['lighting_quality'] = self._assess_lighting_quality(gray)
            scene_info['agricultural_score'] = self._calculate_agricultural_score(scene_info)
            
            return scene_info
            
        except Exception as e:
            print(f"Scene analysis error: {e}")
            return {
                'scene_type': 'field', 
                'crop_rows_detected': False,
                'crop_rows': [],
                'ground_plane': None,
                'scene_complexity': 0.0,
                'lighting_quality': 0.5,
                'agricultural_score': 0.0
            }
    
    def _detect_crop_rows(self, gray_frame: np.ndarray) -> List[CropRowDetection]:
        """FIXED: Detect crop rows with structured output for camera widget"""
        try:
            # Edge detection optimized for crop rows
            edges = cv2.Canny(gray_frame, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=self.crop_row_detector['hough_threshold'],
                minLineLength=self.crop_row_detector['min_line_length'],
                maxLineGap=self.crop_row_detector['max_line_gap']
            )
            
            if lines is None:
                return []
            
            # FIXED: Convert raw lines to structured objects
            crop_rows = []
            for row_id, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                
                # Calculate line properties
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.arctan2(y2-y1, x2-x1)
                
                # Calculate confidence based on multiple factors
                length_score = min(1.0, length / 200.0)  # Normalize by expected min length
                position_score = self._calculate_position_score(x1, y1, x2, y2, gray_frame.shape)
                confidence = (length_score + position_score) / 2.0
                
                # Create structured crop row object
                crop_row = CropRowDetection(
                    start_point=(int(x1), int(y1)),
                    end_point=(int(x2), int(y2)),
                    confidence=float(confidence),
                    length=float(length),
                    angle=float(angle),
                    row_id=row_id
                )
                crop_rows.append(crop_row)
            
            # Filter for parallel lines (crop rows are typically parallel)
            parallel_crop_rows = self._filter_parallel_crop_rows(crop_rows)
            
            print(f"🌾 Detected {len(parallel_crop_rows)} crop rows with structured data")
            return parallel_crop_rows
            
        except Exception as e:
            print(f"Crop row detection error: {e}")
            return []
    
    def _calculate_position_score(self, x1: int, y1: int, x2: int, y2: int, frame_shape: Tuple) -> float:
        """Calculate position-based confidence score for crop row"""
        try:
            height, width = frame_shape
            
            # Prefer lines in lower 2/3 of image (where crop rows typically appear)
            avg_y = (y1 + y2) / 2
            vertical_position_score = 1.0 if avg_y > height / 3 else 0.5
            
            # Prefer longer lines that span more of the image width
            line_width = abs(x2 - x1)
            width_coverage_score = min(1.0, line_width / (width * 0.3))  # 30% width coverage is good
            
            return (vertical_position_score + width_coverage_score) / 2.0
            
        except Exception as e:
            print(f"Position score calculation error: {e}")
            return 0.5
    
    def _filter_parallel_crop_rows(self, crop_rows: List[CropRowDetection]) -> List[CropRowDetection]:
        """Filter crop rows to find parallel ones (enhanced from original)"""
        if len(crop_rows) < 2:
            return crop_rows
        
        try:
            parallel_groups = []
            threshold = self.crop_row_detector['parallel_threshold']
            
            for i, crop_row1 in enumerate(crop_rows):
                group = [crop_row1]
                angle1 = crop_row1.angle
                
                for j, crop_row2 in enumerate(crop_rows[i+1:], i+1):
                    angle2 = crop_row2.angle
                    
                    if abs(angle1 - angle2) < threshold:
                        group.append(crop_row2)
                
                if len(group) >= 2:
                    parallel_groups.extend(group)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_parallel_rows = []
            for row in parallel_groups:
                if row.row_id not in seen:
                    seen.add(row.row_id)
                    unique_parallel_rows.append(row)
            
            return unique_parallel_rows
            
        except Exception as e:
            print(f"Parallel filtering error: {e}")
            return crop_rows
    
    def _estimate_ground_plane(self, depth_frame: np.ndarray) -> Optional[GroundPlaneDetection]:
        """FIXED: Estimate ground plane with structured output"""
        try:
            # Sample points from lower portion of depth image
            height, width = depth_frame.shape
            lower_region = depth_frame[height//2:, :]
            
            # Get valid depth points
            valid_mask = (lower_region > 200) & (lower_region < 8000)
            if np.sum(valid_mask) < 100:
                return None
            
            # Convert to 3D points (simplified camera model)
            fx, fy = 615.0, 615.0  # Approximate focal lengths
            cx, cy = width/2, height/2
            
            y_coords, x_coords = np.where(valid_mask)
            y_coords += height//2  # Adjust for cropping
            depths = lower_region[valid_mask] / 1000.0  # Convert to meters
            
            # Back-project to 3D
            X = (x_coords - cx) * depths / fx
            Y = (y_coords - cy) * depths / fy
            Z = depths
            
            points_3d = np.column_stack([X, Y, Z])
            
            # RANSAC plane fitting
            best_plane = None
            best_inliers = 0
            best_normal = None
            
            for _ in range(self.ground_plane_estimator['max_iterations']):
                # Sample 3 random points
                sample_indices = np.random.choice(len(points_3d), 3, replace=False)
                sample_points = points_3d[sample_indices]
                
                # Fit plane
                v1 = sample_points[1] - sample_points[0]
                v2 = sample_points[2] - sample_points[0]
                normal = np.cross(v1, v2)
                
                if np.linalg.norm(normal) < 1e-6:
                    continue
                
                normal = normal / np.linalg.norm(normal)
                d = -np.dot(normal, sample_points[0])
                
                # Count inliers
                distances = np.abs(np.dot(points_3d, normal) + d)
                inliers = np.sum(distances < self.ground_plane_estimator['ransac_threshold'])
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_plane = np.append(normal, d)
                    best_normal = normal.copy()
            
            if best_inliers >= self.ground_plane_estimator['min_inliers']:
                # FIXED: Return structured ground plane object
                confidence = best_inliers / len(points_3d)
                distance_to_camera = abs(best_plane[3])
                
                ground_plane = GroundPlaneDetection(
                    plane_equation=best_plane,
                    confidence=float(confidence),
                    inlier_count=int(best_inliers),
                    distance_to_camera=float(distance_to_camera),
                    normal_vector=best_normal
                )
                
                print(f"🌾 Ground plane detected: confidence={confidence:.2f}, distance={distance_to_camera:.2f}m")
                return ground_plane
            
            return None
            
        except Exception as e:
            print(f"Ground plane estimation error: {e}")
            return None
    
    def _assess_lighting_quality(self, gray_frame: np.ndarray) -> float:
        """Assess lighting quality for agricultural scenes"""
        try:
            # Calculate histogram statistics
            hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
            
            # Good lighting should have well-distributed intensities
            hist_normalized = hist / hist.sum()
            
            # Calculate entropy (higher entropy = better distributed intensities)
            entropy = -np.sum(hist_normalized * np.log(hist_normalized + 1e-10))
            entropy_normalized = entropy / np.log(256)  # Normalize to 0-1
            
            # Check for saturation (very bright or very dark pixels)
            dark_pixels = np.sum(hist_normalized[:50])  # Very dark
            bright_pixels = np.sum(hist_normalized[200:])  # Very bright
            saturation_penalty = dark_pixels + bright_pixels
            
            lighting_quality = entropy_normalized * (1.0 - saturation_penalty)
            return float(np.clip(lighting_quality, 0.0, 1.0))
            
        except Exception as e:
            print(f"Lighting quality assessment error: {e}")
            return 0.5
    
    def _calculate_agricultural_score(self, scene_info: Dict) -> float:
        """Calculate overall agricultural scene score"""
        try:
            score_components = []
            
            # Crop row detection score
            if scene_info['crop_rows_detected']:
                crop_row_score = min(1.0, len(scene_info['crop_rows']) / 5.0)  # Normalize by expected count
                score_components.append(crop_row_score * 0.4)  # 40% weight
            
            # Ground plane score
            if scene_info['ground_plane'] is not None:
                ground_score = scene_info['ground_plane'].confidence
                score_components.append(ground_score * 0.3)  # 30% weight
            
            # Lighting quality score
            lighting_score = scene_info['lighting_quality']
            score_components.append(lighting_score * 0.2)  # 20% weight
            
            # Scene complexity score (moderate complexity is good)
            complexity = scene_info['scene_complexity']
            complexity_score = 1.0 - abs(complexity - 0.5) * 2  # Peak at 0.5 complexity
            score_components.append(complexity_score * 0.1)  # 10% weight
            
            # Calculate weighted average
            if score_components:
                agricultural_score = sum(score_components)
            else:
                agricultural_score = 0.0
            
            return float(np.clip(agricultural_score, 0.0, 1.0))
            
        except Exception as e:
            print(f"Agricultural score calculation error: {e}")
            return 0.0

class AgriSLAMCore:
    """
    Main Agricultural SLAM Core Engine - FIXED FOR CAMERA WIDGET COMPATIBILITY
    Implements real-time SLAM optimized for agricultural environments
    """
    
    def __init__(self, camera_manager: CameraManager):
        """Initialize Agricultural SLAM Core"""
        self.camera_manager = camera_manager
        self.config_manager = get_config_manager()
        self.data_logger = get_data_logger()
        
        # Core SLAM components
        self.visual_odometry = VisualOdometry()
        self.feature_detector = FeatureDetector(max_features=2000)
        self.scene_analyzer = AgriculturalSceneAnalyzer()
        
        # SLAM state
        self.is_initialized = False
        self.is_tracking = False
        self.slam_mode = "INITIALIZING"  # INITIALIZING, TRACKING, LOST
        
        # Map and trajectory data
        self.keyframes: List[Keyframe] = []
        self.map_points: List[MapPoint] = []
        self.trajectory_3d = [np.array([0.0, 0.0, 0.0])]
        self.current_pose = np.eye(4)
        
        # Performance monitoring
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)
        self.tracking_quality_history = deque(maxlen=50)
        
        # Agricultural-specific data
        self.total_distance = 0.0
        self.path_segments = []  # Store path segments with metadata
        self.field_boundaries = []
        self.crop_row_map = []
        
        # Threading for real-time processing
        self.processing_lock = threading.Lock()
        self.latest_results = None
        
        # Distance tracking with validation
        self.distance_validator = self._init_distance_validator()
        
        print("🌾 Agricultural SLAM Core initialized (FIXED VERSION):")
        print(f"  - Visual odometry: Enhanced precision mode")
        print(f"  - Feature detection: Adaptive agricultural mode")
        print(f"  - Scene analysis: Structured crop rows + ground plane")
        print(f"  - Distance tracking: Centimeter precision")
        print(f"  - Camera widget compatibility: ✅ FIXED")
    
    def _init_distance_validator(self) -> Dict:
        """Initialize distance measurement validator"""
        return {
            'movement_window': deque(maxlen=10),
            'scale_estimates': deque(maxlen=20),
            'outlier_threshold': 3.0,  # Standard deviations
            'min_movement': 0.01,  # 1cm minimum
            'max_movement': 2.0    # 2m maximum per frame
        }
    
    def process_frame(self, color_frame: np.ndarray, depth_frame: np.ndarray, 
                     timestamp: float) -> Dict:
        """
        FIXED: Main frame processing pipeline with proper data structures
        
        Args:
            color_frame: RGB color image
            depth_frame: Depth image
            timestamp: Frame timestamp
            
        Returns:
            Dictionary with SLAM results (FIXED FORMAT)
        """
        start_time = time.time()
        
        with self.processing_lock:
            # Initialize results with proper structure
            results = {
                'timestamp': timestamp,
                'slam_mode': self.slam_mode,
                'pose_estimated': False,
                'position': self.trajectory_3d[-1].copy(),
                'rotation': np.eye(3),
                'num_features': 0,
                'num_matches': 0,
                'num_keyframes': len(self.keyframes),
                'num_map_points': len(self.map_points),
                'total_distance': self.total_distance,
                'processing_time': 0.0,
                'tracking_quality': 0.0,
                'agricultural_scene': {},  # FIXED: Proper key name
                'scene_info': {},  # Maintain backward compatibility
                'debug_info': 'Processing...'
            }
            
            try:
                # FIXED: Analyze agricultural scene with structured output
                scene_info = self.scene_analyzer.analyze_scene(color_frame, depth_frame)
                results['agricultural_scene'] = scene_info  # Primary key for camera widget
                results['scene_info'] = scene_info  # Backward compatibility
                
                # Process with visual odometry
                vo_results = self.visual_odometry.process_frame(color_frame, depth_frame)
                
                # Update results with VO data
                results.update({
                    'pose_estimated': vo_results['pose_estimated'],
                    'position': vo_results['position'].copy(),
                    'rotation': vo_results['rotation'].copy(),
                    'num_features': vo_results['num_features'],
                    'num_matches': vo_results['num_matches'],
                    'debug_info': vo_results['debug_info']
                })
                
                # Handle SLAM state machine
                if not self.is_initialized:
                    if vo_results['pose_estimated']:
                        success = self._initialize_slam(color_frame, depth_frame, vo_results, scene_info, timestamp)
                        if success:
                            self.is_initialized = True
                            self.is_tracking = True
                            self.slam_mode = "TRACKING"
                            results['debug_info'] = "Agricultural SLAM initialized successfully"
                            print("🌾 Agricultural SLAM initialized with structured scene understanding")
                        else:
                            self.slam_mode = "INITIALIZING"
                            results['debug_info'] = "Waiting for sufficient features for initialization"
                    else:
                        self.slam_mode = "INITIALIZING"
                        results['debug_info'] = "Initializing - detecting features..."
                
                else:
                    # Main tracking loop
                    if vo_results['pose_estimated']:
                        success = self._process_tracking(color_frame, depth_frame, vo_results, scene_info, timestamp)
                        if success:
                            self.slam_mode = "TRACKING"
                            self.is_tracking = True
                            
                            # Update distance with validation
                            movement = vo_results.get('translation_magnitude', 0.0)
                            if self._validate_movement(movement):
                                self.total_distance += movement
                                results['total_distance'] = self.total_distance
                            
                            # Calculate tracking quality
                            quality = self._calculate_tracking_quality(vo_results)
                            self.tracking_quality_history.append(quality)
                            results['tracking_quality'] = quality
                            
                        else:
                            self.slam_mode = "LOST"
                            self.is_tracking = False
                            results['debug_info'] = "Tracking lost - attempting recovery"
                    else:
                        self.slam_mode = "LOST"
                        self.is_tracking = False
                        results['debug_info'] = f"Tracking lost: {vo_results['debug_info']}"
                
                # Update slam mode in results
                results['slam_mode'] = self.slam_mode
                
                # Performance tracking
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                results['processing_time'] = processing_time
                
                self.frame_count += 1
                self.latest_results = results.copy()
                
                # FIXED: Log structured data occasionally
                if self.frame_count % 100 == 0:
                    crop_count = len(scene_info.get('crop_rows', []))
                    ground_detected = scene_info.get('ground_plane') is not None
                    print(f"🌾 Frame {self.frame_count}: {crop_count} crop rows, ground plane: {ground_detected}")
                
                return results
                
            except Exception as e:
                print(f"SLAM processing error: {e}")
                results['debug_info'] = f"Processing error: {str(e)}"
                results['slam_mode'] = "ERROR"
                return results
    
    # ============ REST OF METHODS UNCHANGED (for brevity) ============
    def _initialize_slam(self, color_frame: np.ndarray, depth_frame: np.ndarray,
                        vo_results: Dict, scene_info: Dict, timestamp: float) -> bool:
        """Initialize SLAM with first keyframe"""
        try:
            # Require sufficient features for initialization
            if vo_results['num_features'] < 100:
                return False
            
            # Create first keyframe
            keyframe = Keyframe(
                id=0,
                timestamp=timestamp,
                pose=np.eye(4),
                features_2d=np.array([]),  # Will be filled by feature detector
                features_3d=np.array([]),  # Will be filled by depth processing
                descriptors=np.array([]),
                frame_quality=0.8,  # Initial quality
                scene_type=scene_info.get('scene_type', 'field')
            )
            
            # Process features
            feature_results = self.feature_detector.process_frame(color_frame)
            if feature_results['keypoints']:
                keyframe.features_2d = np.array([kp.pt for kp in feature_results['keypoints']])
                keyframe.descriptors = feature_results['descriptors']
                
                # Generate 3D points
                points_3d = self.visual_odometry.enhanced_depth_to_3d_points(
                    keyframe.features_2d, depth_frame
                )
                keyframe.features_3d = points_3d
            
            self.keyframes.append(keyframe)
            
            # Initialize map points
            self._create_initial_map_points(keyframe)
            
            print(f"🌾 First keyframe created: {len(keyframe.features_3d)} 3D points, scene: {keyframe.scene_type}")
            return True
            
        except Exception as e:
            print(f"SLAM initialization error: {e}")
            return False
    
    def _process_tracking(self, color_frame: np.ndarray, depth_frame: np.ndarray,
                         vo_results: Dict, scene_info: Dict, timestamp: float) -> bool:
        """Process tracking frame"""
        try:
            # Update current pose
            self.current_pose[:3, :3] = vo_results['rotation']
            self.current_pose[:3, 3] = vo_results['position']
            
            # Update trajectory
            self.trajectory_3d.append(vo_results['position'].copy())
            
            # Check if we need a new keyframe
            if self._should_create_keyframe(vo_results, scene_info):
                self._create_keyframe(color_frame, depth_frame, vo_results, scene_info, timestamp)
            
            return True
            
        except Exception as e:
            print(f"Tracking processing error: {e}")
            return False
    
    def _should_create_keyframe(self, vo_results: Dict, scene_info: Dict) -> bool:
        """Determine if a new keyframe should be created"""
        if len(self.keyframes) == 0:
            return True
        
        last_keyframe = self.keyframes[-1]
        current_pose = np.eye(4)
        current_pose[:3, :3] = vo_results['rotation']
        current_pose[:3, 3] = vo_results['position']
        
        # Calculate distance from last keyframe
        translation_distance = np.linalg.norm(
            current_pose[:3, 3] - last_keyframe.pose[:3, 3]
        )
        
        # Calculate rotation difference
        R_diff = current_pose[:3, :3] @ last_keyframe.pose[:3, :3].T
        rotation_angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        
        # Keyframe creation criteria
        distance_threshold = 0.3  # 30cm
        angle_threshold = 0.2     # ~11 degrees
        
        # Adaptive thresholds based on scene
        if scene_info.get('crop_rows_detected', False):
            distance_threshold = 0.5  # Larger spacing for crop rows
        
        return (translation_distance > distance_threshold or 
                rotation_angle > angle_threshold or
                vo_results['num_matches'] < 50)  # Low matches trigger keyframe
    
    def _create_keyframe(self, color_frame: np.ndarray, depth_frame: np.ndarray,
                        vo_results: Dict, scene_info: Dict, timestamp: float):
        """Create a new keyframe"""
        try:
            keyframe_id = len(self.keyframes)
            
            # Create pose matrix
            pose = np.eye(4)
            pose[:3, :3] = vo_results['rotation']
            pose[:3, 3] = vo_results['position']
            
            # Process features
            feature_results = self.feature_detector.process_frame(color_frame)
            
            keyframe = Keyframe(
                id=keyframe_id,
                timestamp=timestamp,
                pose=pose,
                features_2d=np.array([kp.pt for kp in feature_results['keypoints']]) if feature_results['keypoints'] else np.array([]),
                features_3d=np.array([]),
                descriptors=feature_results['descriptors'],
                frame_quality=0.8,  # Calculate based on feature quality
                scene_type=scene_info.get('scene_type', 'field')
            )
            
            # Generate 3D points
            if len(keyframe.features_2d) > 0:
                points_3d = self.visual_odometry.enhanced_depth_to_3d_points(
                    keyframe.features_2d, depth_frame
                )
                keyframe.features_3d = points_3d
            
            self.keyframes.append(keyframe)
            
            # Create new map points
            self._create_map_points_from_keyframe(keyframe)
            
            print(f"🌾 Keyframe {keyframe_id} created: {len(keyframe.features_3d)} 3D points")
            
        except Exception as e:
            print(f"Keyframe creation error: {e}")
    
    def _create_initial_map_points(self, keyframe: Keyframe):
        """Create initial map points from first keyframe"""
        try:
            for i, point_3d in enumerate(keyframe.features_3d):
                map_point = MapPoint(
                    id=len(self.map_points),
                    position=point_3d.copy(),
                    observations=[keyframe.id],
                    descriptor=keyframe.descriptors[i] if len(keyframe.descriptors) > i else None,
                    reliability=1.0,
                    point_type='unknown'
                )
                self.map_points.append(map_point)
                
        except Exception as e:
            print(f"Initial map points creation error: {e}")
    
    def _create_map_points_from_keyframe(self, keyframe: Keyframe):
        """Create map points from new keyframe"""
        try:
            points_added = 0
            for i, point_3d in enumerate(keyframe.features_3d):
                # Check if point already exists (simple distance check)
                is_new = True
                for existing_point in self.map_points:
                    if np.linalg.norm(existing_point.position - point_3d) < 0.1:  # 10cm threshold
                        existing_point.observations.append(keyframe.id)
                        is_new = False
                        break
                
                if is_new:
                    map_point = MapPoint(
                        id=len(self.map_points),
                        position=point_3d.copy(),
                        observations=[keyframe.id],
                        descriptor=keyframe.descriptors[i] if len(keyframe.descriptors) > i else None,
                        reliability=1.0,
                        point_type='unknown'
                    )
                    self.map_points.append(map_point)
                    points_added += 1
            
            if points_added > 0:
                print(f"🌾 Added {points_added} new map points")
                
        except Exception as e:
            print(f"Map points creation error: {e}")
    
    def _validate_movement(self, movement: float) -> bool:
        """Validate movement measurement for distance accuracy"""
        try:
            # Check basic bounds
            if movement < self.distance_validator['min_movement']:
                return False
            if movement > self.distance_validator['max_movement']:
                return False
            
            # Add to movement window
            self.distance_validator['movement_window'].append(movement)
            
            # Statistical validation if we have enough data
            if len(self.distance_validator['movement_window']) >= 5:
                movements = list(self.distance_validator['movement_window'])
                mean_movement = np.mean(movements)
                std_movement = np.std(movements)
                
                # Outlier detection
                if std_movement > 0:
                    z_score = abs(movement - mean_movement) / std_movement
                    if z_score > self.distance_validator['outlier_threshold']:
                        return False
            
            return True
            
        except Exception as e:
            print(f"Movement validation error: {e}")
            return True  # Default to accepting movement
    
    def _calculate_tracking_quality(self, vo_results: Dict) -> float:
        """Calculate tracking quality score (0-1)"""
        try:
            factors = []
            
            # Feature count factor
            num_features = vo_results.get('num_features', 0)
            feature_factor = min(num_features / 500.0, 1.0)
            factors.append(feature_factor)
            
            # Match count factor
            num_matches = vo_results.get('num_matches', 0)
            match_factor = min(num_matches / 100.0, 1.0)
            factors.append(match_factor)
            
            # Inliers factor
            inliers = vo_results.get('inliers', 0)
            inlier_factor = min(inliers / 50.0, 1.0)
            factors.append(inlier_factor)
            
            # Processing time factor (faster is better)
            processing_time = vo_results.get('processing_time', 0.1)
            time_factor = max(0, 1.0 - processing_time / 0.1)  # 100ms is poor
            factors.append(time_factor)
            
            return np.mean(factors) if factors else 0.0
            
        except Exception as e:
            print(f"Quality calculation error: {e}")
            return 0.5
    
    def get_current_pose(self) -> np.ndarray:
        """Get current camera pose matrix"""
        with self.processing_lock:
            return self.current_pose.copy()
    
    def get_trajectory_3d(self) -> np.ndarray:
        """Get 3D trajectory points"""
        with self.processing_lock:
            return np.array(self.trajectory_3d)
    
    def get_total_distance(self) -> float:
        """Get total distance traveled"""
        with self.processing_lock:
            return self.total_distance
    
    def get_map_data(self) -> Dict:
        """Get complete map data for visualization"""
        with self.processing_lock:
            return {
                'keyframes': self.keyframes.copy(),
                'map_points': self.map_points.copy(),
                'trajectory': self.trajectory_3d.copy(),
                'total_distance': self.total_distance,
                'field_boundaries': self.field_boundaries.copy(),
                'crop_rows': self.crop_row_map.copy()
            }
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        with self.processing_lock:
            vo_stats = self.visual_odometry.get_performance_stats()
            feature_stats = self.feature_detector.get_performance_stats()
            
            return {
                'slam_mode': self.slam_mode,
                'is_initialized': self.is_initialized,
                'is_tracking': self.is_tracking,
                'frame_count': self.frame_count,
                'keyframes_count': len(self.keyframes),
                'map_points_count': len(self.map_points),
                'trajectory_length': len(self.trajectory_3d),
                'total_distance': self.total_distance,
                'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0.0,
                'avg_tracking_quality': np.mean(self.tracking_quality_history) if self.tracking_quality_history else 0.0,
                'vo_stats': vo_stats,
                'feature_stats': feature_stats
            }
    
    def reset(self):
        """Reset SLAM system to initial state"""
        with self.processing_lock:
            # Reset SLAM state
            self.is_initialized = False
            self.is_tracking = False
            self.slam_mode = "INITIALIZING"
            
            # Clear data structures
            self.keyframes.clear()
            self.map_points.clear()
            self.trajectory_3d = [np.array([0.0, 0.0, 0.0])]
            self.current_pose = np.eye(4)
            
            # Reset counters
            self.frame_count = 0
            self.total_distance = 0.0
            self.processing_times.clear()
            self.tracking_quality_history.clear()
            
            # Reset components
            self.visual_odometry.reset()
            self.feature_detector.reset()
            
            # Reset distance validator
            self.distance_validator['movement_window'].clear()
            self.distance_validator['scale_estimates'].clear()
            
            print("🌾 Agricultural SLAM Core reset successfully (FIXED VERSION)")
    
    def save_session(self, filename: str = None) -> str:
        """Save current SLAM session data"""
        try:
            with self.processing_lock:
                session_data = {
                    'trajectory': np.array(self.trajectory_3d),
                    'total_distance': self.total_distance,
                    'keyframes_count': len(self.keyframes),
                    'map_points_count': len(self.map_points),
                    'frame_count': self.frame_count,
                    'session_metadata': {
                        'start_time': time.time(),
                        'slam_version': '1.0_FIXED',
                        'agricultural_optimized': True,
                        'structured_features': True
                    }
                }
                
                filepath = self.data_logger.save_trajectory(
                    session_data['trajectory'],
                    metadata=session_data
                )
                
                print(f"🌾 SLAM session saved (FIXED VERSION): {filepath}")
                return filepath
                
        except Exception as e:
            print(f"Error saving SLAM session: {e}")
            return ""