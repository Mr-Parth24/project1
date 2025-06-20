"""
Coordinate transformation utilities for SLAM system
Handles transformations between different coordinate frames
"""

import numpy as np
import cv2
from typing import Tuple, List
import math

class CoordinateTransform:
    """
    Coordinate transformation utilities for SLAM
    Handles camera, world, and screen coordinate conversions
    """
    
    def __init__(self):
        """Initialize coordinate transformer"""
        # Transformation matrices  
        self.camera_to_world_matrix = np.eye(4)
        self.world_to_camera_matrix = np.eye(4)
        
        # Camera parameters (defaults for D435i)
        self.camera_matrix = self._get_default_camera_matrix()
        self.dist_coeffs = np.zeros((4, 1))
        
        print("Coordinate Transform initialized")
    
    def _get_default_camera_matrix(self) -> np.ndarray:
        """Get default camera matrix for D435i"""
        fx = 615.0  # Focal length x
        fy = 615.0  # Focal length y
        cx = 320.0  # Principal point x
        cy = 240.0  # Principal point y
        
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def set_camera_matrix(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray = None):
        """
        Set camera intrinsic parameters
        
        Args:
            camera_matrix: 3x3 camera matrix
            dist_coeffs: Distortion coefficients
        """
        self.camera_matrix = camera_matrix.copy()
        if dist_coeffs is not None:
            self.dist_coeffs = dist_coeffs.copy()
    
    def update_camera_pose(self, rotation: np.ndarray, translation: np.ndarray):
        """
        Update camera pose in world coordinates
        
        Args:
            rotation: 3x3 rotation matrix
            translation: 3x1 translation vector
        """
        # Build transformation matrix
        self.camera_to_world_matrix[:3, :3] = rotation
        self.camera_to_world_matrix[:3, 3] = translation.reshape(3)
        
        # Compute inverse
        self.world_to_camera_matrix = np.linalg.inv(self.camera_to_world_matrix)
    
    def pixel_to_camera(self, pixel_coords: np.ndarray, depth: float) -> np.ndarray:
        """
        Convert pixel coordinates to camera coordinates
        
        Args:
            pixel_coords: 2D pixel coordinates [u, v]
            depth: Depth value in meters
            
        Returns:
            3D point in camera coordinates
        """
        if depth <= 0:
            return np.array([0, 0, 0])
        
        u, v = pixel_coords
        
        # Back-project to camera coordinates
        x = (u - self.camera_matrix[0, 2]) * depth / self.camera_matrix[0, 0]
        y = (v - self.camera_matrix[1, 2]) * depth / self.camera_matrix[1, 1]
        z = depth
        
        return np.array([x, y, z])
    
    def camera_to_pixel(self, point_3d: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Project 3D camera coordinates to pixel coordinates
        
        Args:
            point_3d: 3D point in camera coordinates
            
        Returns:
            Tuple of (pixel_coords, depth)
        """
        if point_3d[2] <= 0:
            return np.array([0, 0]), 0.0
        
        # Project to image plane
        x, y, z = point_3d
        u = (x * self.camera_matrix[0, 0] / z) + self.camera_matrix[0, 2]
        v = (y * self.camera_matrix[1, 1] / z) + self.camera_matrix[1, 2]
        
        return np.array([u, v]), z
    
    def camera_to_world(self, point_camera: np.ndarray) -> np.ndarray:
        """
        Transform point from camera to world coordinates
        
        Args:
            point_camera: 3D point in camera coordinates
            
        Returns:
            3D point in world coordinates
        """
        # Convert to homogeneous coordinates
        point_homo = np.append(point_camera, 1.0)
        
        # Transform to world coordinates using the transformation matrix
        world_homo = self.camera_to_world_matrix @ point_homo
        
        return world_homo[:3]
    
    def world_to_camera_coords(self, point_world: np.ndarray) -> np.ndarray:
        """
        Transform point from world to camera coordinates
        
        Args:
            point_world: 3D point in world coordinates
            
        Returns:
            3D point in camera coordinates
        """
        # Convert to homogeneous coordinates
        point_homo = np.append(point_world, 1.0)
        
        # Transform to camera coordinates
        camera_homo = self.world_to_camera_matrix @ point_homo
        
        return camera_homo[:3]
    
    def depth_to_pointcloud(self, depth_image: np.ndarray, color_image: np.ndarray = None) -> np.ndarray:
        """
        Convert depth image to 3D point cloud
        
        Args:
            depth_image: Depth image in millimeters
            color_image: Optional color image for colored point cloud
            
        Returns:
            Point cloud as Nx3 or Nx6 array (XYZ or XYZRGB)
        """
        height, width = depth_image.shape
        
        # Create coordinate grids
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # Flatten coordinates
        u_flat = u_coords.flatten()
        v_flat = v_coords.flatten()
        depth_flat = depth_image.flatten()
        
        # Filter valid depth values
        valid_mask = depth_flat > 0
        u_valid = u_flat[valid_mask]
        v_valid = v_flat[valid_mask]
        depth_valid = depth_flat[valid_mask] / 1000.0  # Convert mm to meters
        
        # Back-project to 3D
        x = (u_valid - self.camera_matrix[0, 2]) * depth_valid / self.camera_matrix[0, 0]
        y = (v_valid - self.camera_matrix[1, 2]) * depth_valid / self.camera_matrix[1, 1]
        z = depth_valid
        
        # Create point cloud
        points_3d = np.stack([x, y, z], axis=1)
        
        # Add color information if available
        if color_image is not None:
            color_flat = color_image.reshape(-1, 3)
            color_valid = color_flat[valid_mask]
            points_3d = np.concatenate([points_3d, color_valid], axis=1)
        
        return points_3d
    
    def transform_pose(self, rotation: np.ndarray, translation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform pose using current camera-to-world transformation
        
        Args:
            rotation: 3x3 rotation matrix
            translation: 3x1 translation vector
            
        Returns:
            Tuple of (transformed_rotation, transformed_translation)
        """
        # Create pose matrix
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation.reshape(3)
        
        # Transform pose
        transformed_pose = self.camera_to_world_matrix @ pose
        
        # Extract rotation and translation
        transformed_rotation = transformed_pose[:3, :3]
        transformed_translation = transformed_pose[:3, 3]
        
        return transformed_rotation, transformed_translation
    
    def euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        Convert Euler angles to rotation matrix
        
        Args:
            roll: Roll angle in radians
            pitch: Pitch angle in radians
            yaw: Yaw angle in radians
            
        Returns:
            3x3 rotation matrix
        """
        # Rotation matrices for each axis
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])
        
        R_y = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        
        R_z = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation
        R = R_z @ R_y @ R_x
        
        return R
    
    def rotation_matrix_to_euler(self, rotation: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles
        
        Args:
            rotation: 3x3 rotation matrix
            
        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        sy = math.sqrt(rotation[0, 0]**2 + rotation[1, 0]**2)
        
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(rotation[2, 1], rotation[2, 2])
            y = math.atan2(-rotation[2, 0], sy)
            z = math.atan2(rotation[1, 0], rotation[0, 0])
        else:
            x = math.atan2(-rotation[1, 2], rotation[1, 1])
            y = math.atan2(-rotation[2, 0], sy)
            z = 0
        
        return x, y, z
    
    def normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to [-pi, pi] range
        
        Args:
            angle: Angle in radians
            
        Returns:
            Normalized angle
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def get_pose_distance(self, pose1: np.ndarray, pose2: np.ndarray) -> Tuple[float, float]:
        """
        Calculate distance between two poses
        
        Args:
            pose1: 4x4 transformation matrix
            pose2: 4x4 transformation matrix
            
        Returns:
            Tuple of (translation_distance, rotation_angle)
        """
        # Translation distance
        trans_diff = pose1[:3, 3] - pose2[:3, 3]
        trans_distance = np.linalg.norm(trans_diff)
        
        # Rotation angle difference
        R_diff = pose1[:3, :3] @ pose2[:3, :3].T
        trace = np.trace(R_diff)
        
        # Avoid numerical issues
        trace = np.clip(trace, -1, 3)
        angle = math.acos((trace - 1) / 2)
        
        return trans_distance, angle
    
    def interpolate_poses(self, pose1: np.ndarray, pose2: np.ndarray, t: float) -> np.ndarray:
        """
        Interpolate between two poses
        
        Args:
            pose1: 4x4 transformation matrix
            pose2: 4x4 transformation matrix
            t: Interpolation parameter [0, 1]
            
        Returns:
            Interpolated 4x4 transformation matrix
        """
        # Linear interpolation for translation
        trans1 = pose1[:3, 3]
        trans2 = pose2[:3, 3]
        interp_trans = (1 - t) * trans1 + t * trans2
        
        # SLERP for rotation (simplified)
        rot1 = pose1[:3, :3]
        rot2 = pose2[:3, :3]
        
        # Convert to axis-angle representation for interpolation
        rot_diff = rot2 @ rot1.T
        rvec, _ = cv2.Rodrigues(rot_diff)
        angle = np.linalg.norm(rvec)
        
        if angle > 1e-6:
            axis = rvec / angle
            interp_angle = t * angle
            interp_rvec = interp_angle * axis
            interp_rot_diff, _ = cv2.Rodrigues(interp_rvec)
            interp_rot = interp_rot_diff @ rot1
        else:
            interp_rot = rot1
        
        # Combine into transformation matrix
        interp_pose = np.eye(4)
        interp_pose[:3, :3] = interp_rot
        interp_pose[:3, 3] = interp_trans
        
        return interp_pose

# Test function
def test_coordinate_transform():
    """Test coordinate transformation utilities"""
    transform = CoordinateTransform()
    
    # Test pixel to camera conversion
    pixel = np.array([320, 240])  # Center pixel
    depth = 1.0  # 1 meter
    camera_point = transform.pixel_to_camera(pixel, depth)
    print(f"Pixel {pixel} at depth {depth}m -> Camera: {camera_point}")
    
    # Test back projection
    back_pixel, back_depth = transform.camera_to_pixel(camera_point)
    print(f"Camera {camera_point} -> Pixel: {back_pixel}, Depth: {back_depth}")
    
    # Test coordinate transformation
    world_point = transform.camera_to_world(camera_point)
    print(f"Camera {camera_point} -> World: {world_point}")
    
    # Test Euler angles
    roll, pitch, yaw = 0.1, 0.2, 0.3
    R = transform.euler_to_rotation_matrix(roll, pitch, yaw)
    back_roll, back_pitch, back_yaw = transform.rotation_matrix_to_euler(R)
    print(f"Euler: ({roll:.3f}, {pitch:.3f}, {yaw:.3f}) -> ({back_roll:.3f}, {back_pitch:.3f}, {back_yaw:.3f})")

if __name__ == "__main__":
    test_coordinate_transform()