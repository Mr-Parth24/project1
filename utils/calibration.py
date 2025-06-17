"""
Camera calibration utilities for RealSense D435i
"""

import cv2
import numpy as np
import os
import glob
import logging
from typing import List, Tuple, Optional
import pyrealsense2 as rs

class CameraCalibrator:
    """Camera calibration utility for RealSense D435i"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Calibration parameters
        self.chessboard_size = (9, 6)  # Internal corners
        self.square_size = 0.025  # 25mm squares
        
        # Calibration data
        self.object_points = []  # 3D points
        self.image_points = []   # 2D points
        self.gray_images = []
        
        # Results
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.rotation_vectors = None
        self.translation_vectors = None
        
    def prepare_object_points(self) -> np.ndarray:
        """Prepare 3D object points for chessboard"""
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def detect_chessboard_corners(self, image: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Detect chessboard corners in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
        return ret, corners
    
    def add_calibration_image(self, image: np.ndarray) -> bool:
        """Add image for calibration"""
        ret, corners = self.detect_chessboard_corners(image)
        
        if ret:
            self.object_points.append(self.prepare_object_points())
            self.image_points.append(corners)
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            self.gray_images.append(gray)
            
            self.logger.info(f"Added calibration image {len(self.image_points)}")
            return True
        else:
            self.logger.warning("Chessboard not found in image")
            return False
    
    def calibrate_camera(self) -> bool:
        """Perform camera calibration"""
        if len(self.object_points) < 10:
            self.logger.error("Need at least 10 calibration images")
            return False
        
        try:
            # Get image size
            h, w = self.gray_images[0].shape[:2]
            
            # Perform calibration
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points, 
                self.image_points, 
                (w, h), 
                None, 
                None
            )
            
            if ret:
                self.camera_matrix = camera_matrix
                self.distortion_coeffs = dist_coeffs
                self.rotation_vectors = rvecs
                self.translation_vectors = tvecs
                
                # Calculate reprojection error
                total_error = 0
                for i in range(len(self.object_points)):
                    proj_points, _ = cv2.projectPoints(
                        self.object_points[i], 
                        rvecs[i], 
                        tvecs[i], 
                        camera_matrix, 
                        dist_coeffs
                    )
                    error = cv2.norm(self.image_points[i], proj_points, cv2.NORM_L2) / len(proj_points)
                    total_error += error
                
                mean_error = total_error / len(self.object_points)
                self.logger.info(f"Calibration successful! Mean reprojection error: {mean_error:.4f}")
                
                return True
            else:
                self.logger.error("Calibration failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Calibration error: {e}")
            return False
    
    def save_calibration(self, filename: str):
        """Save calibration results"""
        if self.camera_matrix is None:
            self.logger.error("No calibration data to save")
            return False
        
        try:
            calibration_data = {
                'camera_matrix': self.camera_matrix.tolist(),
                'distortion_coefficients': self.distortion_coeffs.tolist(),
                'reprojection_error': self.calculate_reprojection_error(),
                'calibration_images': len(self.image_points),
                'image_size': [self.gray_images[0].shape[1], self.gray_images[0].shape[0]]
            }
            
            import json
            with open(filename, 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            self.logger.info(f"Calibration saved to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save calibration: {e}")
            return False
    
    def load_calibration(self, filename: str) -> bool:
        """Load calibration from file"""
        try:
            import json
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.camera_matrix = np.array(data['camera_matrix'])
            self.distortion_coeffs = np.array(data['distortion_coefficients'])
            
            self.logger.info(f"Calibration loaded from {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load calibration: {e}")
            return False
    
    def calculate_reprojection_error(self) -> float:
        """Calculate mean reprojection error"""
        if not self.object_points or self.camera_matrix is None:
            return 0.0
        
        total_error = 0
        for i in range(len(self.object_points)):
            proj_points, _ = cv2.projectPoints(
                self.object_points[i],
                self.rotation_vectors[i],
                self.translation_vectors[i],
                self.camera_matrix,
                self.distortion_coeffs
            )
            error = cv2.norm(self.image_points[i], proj_points, cv2.NORM_L2) / len(proj_points)
            total_error += error
        
        return total_error / len(self.object_points)
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Undistort image using calibration"""
        if self.camera_matrix is None:
            return image
        
        return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
    
    def get_calibration_quality(self) -> str:
        """Get calibration quality assessment"""
        if self.camera_matrix is None:
            return "No calibration data"
        
        error = self.calculate_reprojection_error()
        
        if error < 0.5:
            return "Excellent"
        elif error < 1.0:
            return "Good"
        elif error < 2.0:
            return "Fair"
        else:
            return "Poor"

class RealSenseIntrinsicsExtractor:
    """Extract intrinsic parameters from RealSense camera"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_camera_intrinsics(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get camera intrinsics from RealSense device"""
        try:
            # Create pipeline
            pipeline = rs.pipeline()
            config = rs.config()
            
            # Configure streams
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start pipeline
            profile = pipeline.start(config)
            
            try:
                # Get color stream intrinsics
                color_stream = profile.get_stream(rs.stream.color)
                color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
                
                # Get depth stream intrinsics  
                depth_stream = profile.get_stream(rs.stream.depth)
                depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
                
                # Convert to OpenCV format
                color_matrix = np.array([
                    [color_intrinsics.fx, 0, color_intrinsics.ppx],
                    [0, color_intrinsics.fy, color_intrinsics.ppy],
                    [0, 0, 1]
                ])
                
                depth_matrix = np.array([
                    [depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                    [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                    [0, 0, 1]
                ])
                
                # Get distortion coefficients
                color_dist = np.array(color_intrinsics.coeffs)
                depth_dist = np.array(depth_intrinsics.coeffs)
                
                self.logger.info("Successfully extracted RealSense intrinsics")
                
                return (color_matrix, color_dist), (depth_matrix, depth_dist)
                
            finally:
                pipeline.stop()
                
        except Exception as e:
            self.logger.error(f"Failed to get RealSense intrinsics: {e}")
            return None, None
    
    def save_intrinsics(self, color_data: Tuple[np.ndarray, np.ndarray], 
                       depth_data: Tuple[np.ndarray, np.ndarray], filename: str):
        """Save intrinsics to file"""
        try:
            intrinsics_data = {
                'color': {
                    'camera_matrix': color_data[0].tolist(),
                    'distortion_coefficients': color_data[1].tolist()
                },
                'depth': {
                    'camera_matrix': depth_data[0].tolist(),
                    'distortion_coefficients': depth_data[1].tolist()
                },
                'source': 'RealSense_SDK',
                'extraction_time': str(np.datetime64('now'))
            }
            
            import json
            with open(filename, 'w') as f:
                json.dump(intrinsics_data, f, indent=2)
            
            self.logger.info(f"Intrinsics saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save intrinsics: {e}")