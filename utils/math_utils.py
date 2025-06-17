"""
Mathematical utilities for pose estimation and geometric computations
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import cv2
import logging
from typing import Tuple, List, Optional

class PoseUtils:
    """Utilities for pose and transformation operations"""
    
    @staticmethod
    def matrix_to_pose(transformation_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract position and rotation from transformation matrix"""
        position = transformation_matrix[:3, 3]
        rotation = transformation_matrix[:3, :3]
        return position, rotation
    
    @staticmethod
    def pose_to_matrix(position: np.ndarray, rotation: np.ndarray) -> np.ndarray:
        """Create transformation matrix from position and rotation"""
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = position
        return T
    
    @staticmethod
    def invert_pose(pose: np.ndarray) -> np.ndarray:
        """Invert pose transformation"""
        R_inv = pose[:3, :3].T
        t_inv = -R_inv @ pose[:3, 3]
        
        pose_inv = np.eye(4)
        pose_inv[:3, :3] = R_inv
        pose_inv[:3, 3] = t_inv
        return pose_inv
    
    @staticmethod
    def compose_poses(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
        """Compose two poses"""
        return pose1 @ pose2
    
    @staticmethod
    def relative_pose(pose1: np.ndarray, pose2: np.ndarray) -> np.ndarray:
        """Compute relative pose from pose1 to pose2"""
        return PoseUtils.invert_pose(pose1) @ pose2
    
    @staticmethod
    def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles (roll, pitch, yaw)"""
        r = R.from_matrix(R)
        return r.as_euler('xyz')
    
    @staticmethod
    def euler_to_rotation_matrix(euler: np.ndarray) -> np.ndarray:
        """Convert Euler angles to rotation matrix"""
        r = R.from_euler('xyz', euler)
        return r.as_matrix()
    
    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion [w, x, y, z]"""
        r = R.from_matrix(R)
        quat = r.as_quat()  # Returns [x, y, z, w]
        return np.array([quat[3], quat[0], quat[1], quat[2]])  # Reorder to [w, x, y, z]
    
    @staticmethod
    def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion [w, x, y, z] to rotation matrix"""
        # Reorder from [w, x, y, z] to [x, y, z, w] for scipy
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        r = R.from_quat(quat_scipy)
        return r.as_matrix()
    
    @staticmethod
    def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors"""
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

class GeometryUtils:
    """Geometric computation utilities"""
    
    @staticmethod
    def triangulate_points(points1: np.ndarray, points2: np.ndarray, 
                          P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
        """Triangulate 3D points from two views"""
        points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.T
    
    @staticmethod
    def project_points(points_3d: np.ndarray, camera_matrix: np.ndarray, 
                      rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """Project 3D points to 2D image plane"""
        points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, None)
        return points_2d.reshape(-1, 2)
    
    @staticmethod
    def compute_fundamental_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """Compute fundamental matrix from point correspondences"""
        F, _ = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)
        return F
    
    @staticmethod
    def compute_essential_matrix(points1: np.ndarray, points2: np.ndarray, 
                               camera_matrix: np.ndarray) -> np.ndarray:
        """Compute essential matrix from point correspondences"""
        E, _ = cv2.findEssentialMat(points1, points2, camera_matrix)
        return E
    
    @staticmethod
    def recover_pose_from_essential(E: np.ndarray, points1: np.ndarray, 
                                  points2: np.ndarray, camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Recover pose from essential matrix"""
        _, R, t, _ = cv2.recoverPose(E, points1, points2, camera_matrix)
        return R, t
    
    @staticmethod
    def compute_epipolar_error(points1: np.ndarray, points2: np.ndarray, 
                             F: np.ndarray) -> np.ndarray:
        """Compute epipolar error for point correspondences"""
        # Convert to homogeneous coordinates
        points1_h = np.hstack([points1, np.ones((len(points1), 1))])
        points2_h = np.hstack([points2, np.ones((len(points2), 1))])
        
        # Compute epipolar lines
        lines = (F @ points1_h.T).T
        
        # Compute point-to-line distances
        distances = np.abs(np.sum(lines * points2_h, axis=1)) / np.sqrt(lines[:, 0]**2 + lines[:, 1]**2)
        
        return distances
    
    @staticmethod
    def fit_plane_ransac(points: np.ndarray, threshold: float = 0.01, 
                        max_iterations: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Fit plane to 3D points using RANSAC"""
        best_inliers = []
        best_plane = None
        
        for _ in range(max_iterations):
            # Sample 3 random points
            sample_indices = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_indices]
            
            # Compute plane parameters
            v1 = sample_points[1] - sample_points[0]
            v2 = sample_points[2] - sample_points[0]
            normal = np.cross(v1, v2)
            
            if np.linalg.norm(normal) == 0:
                continue
                
            normal = normal / np.linalg.norm(normal)
            d = -np.dot(normal, sample_points[0])
            
            # Compute distances to plane
            distances = np.abs(np.dot(points, normal) + d)
            inliers = distances < threshold
            
            if np.sum(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = np.append(normal, d)
        
        return best_plane, best_inliers

class FilterUtils:
    """Filtering and smoothing utilities"""
    
    @staticmethod
    def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
        """Apply moving average filter"""
        if len(data) < window_size:
            return data
        
        filtered = np.zeros_like(data)
        for i in range(len(data)):
            start = max(0, i - window_size // 2)
            end = min(len(data), i + window_size // 2 + 1)
            filtered[i] = np.mean(data[start:end])
        
        return filtered
    
    @staticmethod
    def gaussian_filter_1d(data: np.ndarray, sigma: float) -> np.ndarray:
        """Apply 1D Gaussian filter"""
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(data, sigma)
    
    @staticmethod
    def kalman_filter_pose(poses: List[np.ndarray], process_noise: float = 0.01, 
                          measurement_noise: float = 0.1) -> List[np.ndarray]:
        """Apply Kalman filter to pose sequence"""
        if len(poses) < 2:
            return poses
        
        # Simple Kalman filter for position only
        filtered_poses = []
        
        # Initial state
        x = poses[0][:3, 3]  # Position
        v = np.zeros(3)      # Velocity
        P = np.eye(6) * 0.1  # Initial covariance
        
        # Process model (constant velocity)
        dt = 1.0  # Assume unit time step
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        Q = np.eye(6) * process_noise  # Process noise
        R = np.eye(3) * measurement_noise  # Measurement noise
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        state = np.hstack([x, v])
        
        for pose in poses:
            # Predict
            state = F @ state
            P = F @ P @ F.T + Q
            
            # Update
            z = pose[:3, 3]  # Measured position
            y = z - H @ state  # Innovation
            S = H @ P @ H.T + R  # Innovation covariance
            K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
            
            state = state + K @ y
            P = (np.eye(6) - K @ H) @ P
            
            # Create filtered pose
            filtered_pose = pose.copy()
            filtered_pose[:3, 3] = state[:3]
            filtered_poses.append(filtered_pose)
        
        return filtered_poses

class OptimizationUtils:
    """Optimization utilities for pose refinement"""
    
    @staticmethod
    def bundle_adjustment_simple(poses: List[np.ndarray], points_3d: List[np.ndarray], 
                               points_2d: List[List[np.ndarray]], camera_matrix: np.ndarray) -> List[np.ndarray]:
        """Simple bundle adjustment implementation"""
        logger = logging.getLogger(__name__)
        
        try:
            # Convert poses to parameter vector
            def poses_to_params(poses_list):
                params = []
                for pose in poses_list:
                    rvec, _ = cv2.Rodrigues(pose[:3, :3])
                    tvec = pose[:3, 3]
                    params.extend(rvec.flatten())
                    params.extend(tvec.flatten())
                return np.array(params)
            
            def params_to_poses(params):
                poses_list = []
                for i in range(0, len(params), 6):
                    rvec = params[i:i+3]
                    tvec = params[i+3:i+6]
                    R, _ = cv2.Rodrigues(rvec)
                    pose = np.eye(4)
                    pose[:3, :3] = R
                    pose[:3, 3] = tvec
                    poses_list.append(pose)
                return poses_list
            
            # Objective function
            def objective(params):
                poses_opt = params_to_poses(params)
                total_error = 0
                
                for i, (pose, points_3d_frame, points_2d_frame) in enumerate(zip(poses_opt, points_3d, points_2d)):
                    if len(points_3d_frame) == 0:
                        continue
                    
                    rvec, _ = cv2.Rodrigues(pose[:3, :3])
                    tvec = pose[:3, 3]
                    
                    projected_points, _ = cv2.projectPoints(
                        points_3d_frame, rvec, tvec, camera_matrix, None
                    )
                    projected_points = projected_points.reshape(-1, 2)
                    
                    error = np.sum((projected_points - points_2d_frame)**2)
                    total_error += error
                
                return total_error
            
            # Initial parameters
            initial_params = poses_to_params(poses)
            
            # Optimize
            result = minimize(objective, initial_params, method='L-BFGS-B')
            
            if result.success:
                optimized_poses = params_to_poses(result.x)
                logger.info("Bundle adjustment completed successfully")
                return optimized_poses
            else:
                logger.warning("Bundle adjustment failed")
                return poses
                
        except Exception as e:
            logger.error(f"Bundle adjustment error: {e}")
            return poses