"""
SLAM (Simultaneous Localization and Mapping) Processor
Advanced loop closure detection and pose graph optimization
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Any, Optional, Tuple
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
import threading
from core.visual_odometry import VisualOdometry

# Alternative implementation without g2o dependency
class SimplePoseGraphOptimizer:
    """Simple pose graph optimization without g2o dependency"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Pose graph data
        self.poses = {}
        self.edges = []
        self.next_vertex_id = 0
        
    def add_pose(self, pose: np.ndarray, vertex_id: Optional[int] = None) -> int:
        """Add a pose vertex to the graph"""
        if vertex_id is None:
            vertex_id = self.next_vertex_id
            self.next_vertex_id += 1
        
        self.poses[vertex_id] = pose.copy()
        return vertex_id
    
    def add_edge(self, vertex_id1: int, vertex_id2: int, measurement: np.ndarray, 
                information: np.ndarray):
        """Add an edge (constraint) between two poses"""
        self.edges.append({
            'vertex1': vertex_id1,
            'vertex2': vertex_id2,
            'measurement': measurement.copy(),
            'information': information.copy()
        })
    
    def optimize(self, iterations: int = 20) -> bool:
        """Optimize the pose graph using simple iterative method"""
        try:
            if len(self.poses) < 2:
                return True
            
            # Simple optimization using least squares
            for iteration in range(iterations):
                total_error = 0
                pose_updates = {}
                
                # Initialize updates
                for vertex_id in self.poses.keys():
                    pose_updates[vertex_id] = np.zeros(6)  # [x, y, z, rx, ry, rz]
                
                # Process each edge
                for edge in self.edges:
                    v1, v2 = edge['vertex1'], edge['vertex2']
                    if v1 not in self.poses or v2 not in self.poses:
                        continue
                    
                    # Current poses
                    pose1 = self.poses[v1]
                    pose2 = self.poses[v2]
                    
                    # Expected relative transformation
                    expected_relative = edge['measurement']
                    
                    # Actual relative transformation
                    actual_relative = np.linalg.inv(pose1) @ pose2
                    
                    # Compute error
                    error_matrix = expected_relative @ np.linalg.inv(actual_relative)
                    
                    # Extract translation and rotation errors
                    translation_error = error_matrix[:3, 3]
                    rotation_error = self._rotation_matrix_to_axis_angle(error_matrix[:3, :3])
                    
                    error_vector = np.hstack([translation_error, rotation_error])
                    total_error += np.linalg.norm(error_vector)
                    
                    # Distribute error correction (simplified)
                    if v1 != 0:  # Don't move the first pose (anchor)
                        pose_updates[v1] -= error_vector * 0.1
                    if v2 != 0:
                        pose_updates[v2] += error_vector * 0.1
                
                # Apply updates
                for vertex_id, update in pose_updates.items():
                    if vertex_id == 0:  # Keep first pose fixed
                        continue
                    
                    # Apply translation update
                    self.poses[vertex_id][:3, 3] += update[:3] * 0.01
                    
                    # Apply rotation update
                    rotation_update = self._axis_angle_to_rotation_matrix(update[3:] * 0.01)
                    self.poses[vertex_id][:3, :3] = self.poses[vertex_id][:3, :3] @ rotation_update
                
                # Check convergence
                if total_error < 1e-6:
                    break
            
            self.logger.info(f"Simple pose graph optimized with {iteration + 1} iterations")
            return True
            
        except Exception as e:
            self.logger.error(f"Pose graph optimization failed: {e}")
            return False
    
    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to axis-angle representation"""
        from scipy.spatial.transform import Rotation as Rot
        r = Rot.from_matrix(R)
        return r.as_rotvec()
    
    def _axis_angle_to_rotation_matrix(self, axis_angle: np.ndarray) -> np.ndarray:
        """Convert axis-angle to rotation matrix"""
        from scipy.spatial.transform import Rotation as Rot
        r = Rot.from_rotvec(axis_angle)
        return r.as_matrix()
    
    def get_optimized_poses(self) -> Dict[int, np.ndarray]:
        """Get optimized poses"""
        return self.poses.copy()

# Try to import g2o, fall back to simple optimizer if not available
try:
    import g2o
    
    class G2OPoseGraphOptimizer:
        """G2O-based pose graph optimization"""
        
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            
            # Create optimizer
            self.optimizer = g2o.SparseOptimizer()
            
            # Try different solver configurations
            try:
                # Try newer g2o API
                solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
            except AttributeError:
                try:
                    # Try alternative solver
                    solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
                except AttributeError:
                    # Fallback to basic solver
                    self.logger.warning("Advanced g2o solvers not available, using basic solver")
                    solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
            
            algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
            self.optimizer.set_algorithm(algorithm)
            
            # Pose graph data
            self.poses = {}
            self.edges = []
            self.next_vertex_id = 0
        
        def add_pose(self, pose: np.ndarray, vertex_id: Optional[int] = None) -> int:
            """Add a pose vertex to the graph"""
            if vertex_id is None:
                vertex_id = self.next_vertex_id
                self.next_vertex_id += 1
            
            # Convert pose to SE3
            se3_pose = g2o.SE3Quat(pose[:3, :3], pose[:3, 3])
            
            # Create vertex
            vertex = g2o.VertexSE3Expmap()
            vertex.set_id(vertex_id)
            vertex.set_estimate(se3_pose)
            
            # Fix the first pose
            if vertex_id == 0:
                vertex.set_fixed(True)
            
            self.optimizer.add_vertex(vertex)
            self.poses[vertex_id] = pose.copy()
            
            return vertex_id
        
        def add_edge(self, vertex_id1: int, vertex_id2: int, measurement: np.ndarray, 
                    information: np.ndarray):
            """Add an edge (constraint) between two poses"""
            
            # Create edge
            edge = g2o.EdgeSE3Expmap()
            edge.set_vertex(0, self.optimizer.vertex(vertex_id1))
            edge.set_vertex(1, self.optimizer.vertex(vertex_id2))
            
            # Set measurement (relative transformation)
            se3_measurement = g2o.SE3Quat(measurement[:3, :3], measurement[:3, 3])
            edge.set_measurement(se3_measurement)
            
            # Set information matrix (inverse of covariance)
            edge.set_information(information)
            
            self.optimizer.add_edge(edge)
            self.edges.append((vertex_id1, vertex_id2, measurement))
        
        def optimize(self, iterations: int = 20) -> bool:
            """Optimize the pose graph"""
            try:
                self.optimizer.initialize_optimization()
                self.optimizer.optimize(iterations)
                
                # Update poses with optimized values
                for vertex_id in self.poses.keys():
                    vertex = self.optimizer.vertex(vertex_id)
                    if vertex:
                        se3_estimate = vertex.estimate()
                        rotation = se3_estimate.rotation().matrix()
                        translation = se3_estimate.translation()
                        
                        optimized_pose = np.eye(4)
                        optimized_pose[:3, :3] = rotation
                        optimized_pose[:3, 3] = translation
                        
                        self.poses[vertex_id] = optimized_pose
                
                self.logger.info(f"G2O pose graph optimized with {iterations} iterations")
                return True
                
            except Exception as e:
                self.logger.error(f"G2O pose graph optimization failed: {e}")
                return False
        
        def get_optimized_poses(self) -> Dict[int, np.ndarray]:
            """Get optimized poses"""
            return self.poses.copy()
    
    # Use G2O optimizer if available
    PoseGraphOptimizer = G2OPoseGraphOptimizer
    
except ImportError:
    # Fall back to simple optimizer
    PoseGraphOptimizer = SimplePoseGraphOptimizer
    logging.getLogger(__name__).warning("g2o not available, using simple pose graph optimizer")

class LoopClosureDetector:
    """Advanced loop closure detection using BoW and geometric verification"""
    
    def __init__(self, vocabulary_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.vocabulary_size = vocabulary_size
        
        # BoW detector and matcher
        self.bow_extractor = None
        self.bow_matcher = None
        self.vocabulary = None
        
        # Keyframe database
        self.keyframes = []
        self.bow_database = []
        
        # Initialize BoW
        self._initialize_bow()
        
    def _initialize_bow(self):
        """Initialize Bag of Words system"""
        try:
            # Create ORB detector for BoW (more reliable than SIFT)
            self.detector = cv2.ORB_create(nfeatures=500)
            
            # Create BoW trainer
            self.bow_trainer = cv2.BOWKMeansTrainer(self.vocabulary_size)
            
            # Create BoW extractor (will be initialized after vocabulary)
            self.bow_extractor = cv2.BOWImgDescriptorExtractor(self.detector, cv2.BFMatcher(cv2.NORM_HAMMING))
            
            self.logger.info("BoW system initialized with ORB detector")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize BoW: {e}")
    
    def add_keyframe(self, image: np.ndarray, pose: np.ndarray, descriptors: np.ndarray):
        """Add a keyframe to the database"""
        
        keyframe_data = {
            'id': len(self.keyframes),
            'image': image,
            'pose': pose.copy(),
            'descriptors': descriptors
        }
        
        self.keyframes.append(keyframe_data)
        
        # Add descriptors to vocabulary trainer
        if descriptors is not None and len(descriptors) > 0:
            self.bow_trainer.add(descriptors.astype(np.float32))
        
        # Update vocabulary if we have enough keyframes
        if len(self.keyframes) % 50 == 0 and len(self.keyframes) > 0:
            self._update_vocabulary()
    
    def _update_vocabulary(self):
        """Update BoW vocabulary"""
        try:
            if self.bow_trainer.descriptorsCount() > self.vocabulary_size:
                self.vocabulary = self.bow_trainer.cluster()
                self.bow_extractor.setVocabulary(self.vocabulary)
                
                # Recompute BoW descriptors for all keyframes
                self._recompute_bow_descriptors()
                
                self.logger.info(f"BoW vocabulary updated with {len(self.vocabulary)} words")
                
        except Exception as e:
            self.logger.error(f"Failed to update vocabulary: {e}")
    
    def _recompute_bow_descriptors(self):
        """Recompute BoW descriptors for all keyframes"""
        self.bow_database = []
        
        for keyframe in self.keyframes:
            try:
                # Convert to grayscale if needed
                image = keyframe['image']
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image
                
                # Compute BoW descriptor
                bow_descriptor = self.bow_extractor.compute(gray, self.detector.detect(gray))
                if bow_descriptor is not None:
                    self.bow_database.append(bow_descriptor.flatten())
                else:
                    self.bow_database.append(np.array([]))
                    
            except Exception as e:
                self.logger.warning(f"Failed to compute BoW for keyframe {keyframe['id']}: {e}")
                self.bow_database.append(np.array([]))
    
    def detect_loop_closure(self, current_image: np.ndarray, current_pose: np.ndarray, 
                          min_keyframe_gap: int = 30) -> List[Tuple[int, float]]:
        """Detect loop closure candidates"""
        
        candidates = []
        
        if self.vocabulary is None or len(self.keyframes) < min_keyframe_gap:
            return candidates
        
        try:
            # Compute BoW descriptor for current image
            if len(current_image.shape) == 3:
                gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = current_image
            
            current_bow = self.bow_extractor.compute(gray, self.detector.detect(gray))
            if current_bow is None:
                return candidates
            
            current_bow = current_bow.flatten()
            
            # Compare with keyframes (excluding recent ones)
            for i in range(len(self.keyframes) - min_keyframe_gap):
                if i >= len(self.bow_database) or len(self.bow_database[i]) == 0:
                    continue
                
                # Compute similarity
                similarity = self._compute_bow_similarity(current_bow, self.bow_database[i])
                
                if similarity > 0.7:  # Similarity threshold
                    # Geometric verification
                    if self._geometric_verification(current_image, self.keyframes[i]['image'], 
                                                  current_pose, self.keyframes[i]['pose']):
                        candidates.append((i, similarity))
            
            # Sort by similarity
            candidates.sort(key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Loop closure detection failed: {e}")
        
        return candidates
    
    def _compute_bow_similarity(self, bow1: np.ndarray, bow2: np.ndarray) -> float:
        """Compute similarity between BoW descriptors"""
        if len(bow1) == 0 or len(bow2) == 0:
            return 0.0
        
        # Normalize descriptors
        bow1_norm = bow1 / (np.linalg.norm(bow1) + 1e-8)
        bow2_norm = bow2 / (np.linalg.norm(bow2) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(bow1_norm, bow2_norm)
        return max(0.0, similarity)
    
    def _geometric_verification(self, img1: np.ndarray, img2: np.ndarray,
                              pose1: np.ndarray, pose2: np.ndarray) -> bool:
        """Geometric verification of loop closure"""
        try:
            # Detect and match features
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
            
            kp1, desc1 = self.detector.detectAndCompute(gray1, None)
            kp2, desc2 = self.detector.detectAndCompute(gray2, None)
            
            if desc1 is None or desc2 is None:
                return False
            
            # Match features
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(desc1, desc2)
            
            if len(matches) < 20:
                return False
            
            # Extract matched points
            pts1 = np.array([kp1[m.queryIdx].pt for m in matches], dtype=np.float32)
            pts2 = np.array([kp2[m.trainIdx].pt for m in matches], dtype=np.float32)
            
            # Compute fundamental matrix
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
            
            if F is None:
                return False
            
            # Count inliers
            inliers = np.sum(mask)
            inlier_ratio = inliers / len(matches)
            
            return inlier_ratio > 0.3
            
        except Exception as e:
            self.logger.error(f"Geometric verification failed: {e}")
            return False

class SLAMProcessor:
    """Main SLAM processor combining visual odometry with loop closure"""
    
    def __init__(self, visual_odometry: VisualOdometry):
        self.logger = logging.getLogger(__name__)
        self.visual_odometry = visual_odometry
        
        # SLAM components
        self.pose_graph = PoseGraphOptimizer()
        self.loop_detector = LoopClosureDetector()
        
        # SLAM state
        self.keyframes = []
        self.current_vertex_id = 0
        self.last_keyframe_pose = None
        
        # Threading
        self.slam_thread = None
        self.slam_running = False
        self.slam_lock = threading.Lock()
        
        # Parameters
        self.keyframe_distance_threshold = 0.2  # 20cm
        self.keyframe_rotation_threshold = 0.2  # ~11 degrees
        
    def process_frame(self, frame_data, vo_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process frame for SLAM"""
        slam_result = vo_result.copy()
        slam_result['loop_closure_detected'] = False
        slam_result['pose_optimized'] = False
        
        # Check if current frame should be a keyframe
        if self._should_add_keyframe(vo_result['pose']):
            keyframe_id = self._add_keyframe(frame_data, vo_result['pose'])
            slam_result['keyframe_added'] = True
            slam_result['keyframe_id'] = keyframe_id
            
            # Detect loop closures
            loop_candidates = self.loop_detector.detect_loop_closure(
                frame_data.color_image, 
                vo_result['pose']
            )
            
            if loop_candidates:
                self.logger.info(f"Loop closure candidates found: {len(loop_candidates)}")
                slam_result['loop_closure_detected'] = True
                slam_result['loop_candidates'] = loop_candidates
                
                # Add loop closure constraints
                for candidate_id, similarity in loop_candidates[:3]:  # Top 3 candidates
                    self._add_loop_closure_constraint(keyframe_id, candidate_id)
                
                # Optimize pose graph
                if self._optimize_pose_graph():
                    slam_result['pose_optimized'] = True
                    
                    # Update visual odometry with optimized poses
                    self._update_visual_odometry_poses()
        
        return slam_result
    
    def _should_add_keyframe(self, current_pose: np.ndarray) -> bool:
        """Determine if current pose should be added as keyframe"""
        if self.last_keyframe_pose is None:
            return True
        
        # Calculate translation distance
        translation_dist = np.linalg.norm(
            current_pose[:3, 3] - self.last_keyframe_pose[:3, 3]
        )
        
        # Calculate rotation angle
        relative_rotation = self.last_keyframe_pose[:3, :3].T @ current_pose[:3, :3]
        rotation_angle = np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))
        
        return (translation_dist > self.keyframe_distance_threshold or 
                rotation_angle > self.keyframe_rotation_threshold)
    
    def _add_keyframe(self, frame_data, pose: np.ndarray) -> int:
        """Add a new keyframe"""
        with self.slam_lock:
            keyframe_data = {
                'id': len(self.keyframes),
                'frame_data': frame_data,
                'pose': pose.copy(),
                'vertex_id': self.current_vertex_id
            }
            
            self.keyframes.append(keyframe_data)
            
            # Add to pose graph
            vertex_id = self.pose_graph.add_pose(pose, self.current_vertex_id)
            
            # Add to loop closure detector
            # Extract features for loop closure
            detector = cv2.ORB_create(nfeatures=500)
            gray = cv2.cvtColor(frame_data.color_image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            
            self.loop_detector.add_keyframe(frame_data.color_image, pose, descriptors)
            
            # Add odometry edge to previous keyframe
            if len(self.keyframes) > 1:
                prev_keyframe = self.keyframes[-2]
                relative_pose = np.linalg.inv(prev_keyframe['pose']) @ pose
                
                # Information matrix (inverse covariance)
                information = np.eye(6) * 100  # High confidence in odometry
                
                self.pose_graph.add_edge(
                    prev_keyframe['vertex_id'],
                    vertex_id,
                    relative_pose,
                    information
                )
            
            self.current_vertex_id += 1
            self.last_keyframe_pose = pose.copy()
            
            return keyframe_data['id']
    
    def _add_loop_closure_constraint(self, keyframe_id1: int, keyframe_id2: int):
        """Add loop closure constraint between keyframes"""
        try:
            kf1 = self.keyframes[keyframe_id1]
            kf2 = self.keyframes[keyframe_id2]
            
            # Compute relative transformation
            relative_pose = np.linalg.inv(kf2['pose']) @ kf1['pose']
            
            # Information matrix (lower confidence than odometry)
            information = np.eye(6) * 50
            
            self.pose_graph.add_edge(
                kf2['vertex_id'],
                kf1['vertex_id'],
                relative_pose,
                information
            )
            
            self.logger.info(f"Added loop closure constraint between keyframes {keyframe_id1} and {keyframe_id2}")
            
        except Exception as e:
            self.logger.error(f"Failed to add loop closure constraint: {e}")
    
    def _optimize_pose_graph(self) -> bool:
        """Optimize the pose graph"""
        return self.pose_graph.optimize(iterations=20)
    
    def _update_visual_odometry_poses(self):
        """Update visual odometry trajectory with optimized poses"""
        try:
            optimized_poses = self.pose_graph.get_optimized_poses()
            
            # Update keyframe poses
            for keyframe in self.keyframes:
                vertex_id = keyframe['vertex_id']
                if vertex_id in optimized_poses:
                    keyframe['pose'] = optimized_poses[vertex_id]
            
            # Update visual odometry trajectory
            # This is a simplified approach - in practice, you'd interpolate between keyframes
            if len(self.keyframes) > 0:
                self.visual_odometry.current_pose = self.keyframes[-1]['pose'].copy()
            
            self.logger.info("Visual odometry poses updated with optimization results")
            
        except Exception as e:
            self.logger.error(f"Failed to update visual odometry poses: {e}")
    
    def get_optimized_trajectory(self) -> List[np.ndarray]:
        """Get optimized trajectory"""
        return [kf['pose'] for kf in self.keyframes]
    
    def reset(self):
        """Reset SLAM system"""
        with self.slam_lock:
            self.keyframes = []
            self.current_vertex_id = 0
            self.last_keyframe_pose = None
            
            # Reset components
            self.pose_graph = PoseGraphOptimizer()
            self.loop_detector = LoopClosureDetector()
            
            self.logger.info("SLAM system reset")