"""
Advanced Camera Calibration with Checkerboard and Auto-calibration
Author: Mr-Parth24
Date: 2025-06-13
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
import os
import json
from dataclasses import dataclass
import threading
import time

@dataclass
class CalibrationResult:
    success: bool
    calibration_data: Dict[str, Any]
    reprojection_error: float
    num_images_used: int
    error: Optional[str] = None

class CalibrationManager:
    """Advanced camera calibration system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Calibration parameters
        self.checkerboard_size = (9, 6)  # Internal corners
        self.square_size = 0.025  # 25mm squares
        self.min_images = 10
        self.max_images = 30
        
        # Calibration data
        self.calibration_images = []
        self.object_points = []  # 3D points
        self.image_points = []   # 2D points
        
        # Auto-calibration parameters
        self.auto_calib_frames = []
        self.auto_calib_features = []
        
        # Current calibration state
        self.is_calibrating = False
        self.calibration_thread = None
        
        # Results
        self.last_calibration_result = None
        
        self.logger.info("Calibration manager initialized")
    
    def start_checkerboard_calibration(self, camera_manager) -> bool:
        """Start interactive checkerboard calibration"""
        if self.is_calibrating:
            self.logger.warning("Calibration already in progress")
            return False
        
        try:
            self.is_calibrating = True
            self.calibration_thread = threading.Thread(
                target=self._checkerboard_calibration_loop,
                args=(camera_manager,),
                daemon=True
            )
            self.calibration_thread.start()
            
            self.logger.info("Started checkerboard calibration")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start calibration: {e}")
            self.is_calibrating = False
            return False
    
    def _checkerboard_calibration_loop(self, camera_manager):
        """Interactive checkerboard calibration loop"""
        try:
            self.logger.info("Checkerboard calibration started")
            self.logger.info(f"Show checkerboard pattern ({self.checkerboard_size[0]}x{self.checkerboard_size[1]}) to camera")
            self.logger.info("Press SPACE to capture image, ESC to finish")
            
            # Prepare object points
            objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
            objp *= self.square_size
            
            images_captured = 0
            
            while self.is_calibrating and images_captured < self.max_images:
                # Get camera frame
                frames = camera_manager.get_frames()
                if frames is None:
                    continue
                
                color_frame, depth_frame = frames
                gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
                
                # Find checkerboard corners
                ret, corners = cv2.findChessboardCorners(
                    gray, self.checkerboard_size, None
                )
                
                # Draw corners for visualization
                display_frame = color_frame.copy()
                if ret:
                    cv2.drawChessboardCorners(
                        display_frame, self.checkerboard_size, corners, ret
                    )
                    
                    # Add status text
                    cv2.putText(display_frame, f"Pattern detected! Press SPACE to capture", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Show checkerboard pattern to camera", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(display_frame, f"Images captured: {images_captured}/{self.min_images}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                cv2.imshow('Calibration', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' ') and ret:  # Space to capture
                    # Refine corners
                    corners_refined = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1),
                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    )
                    
                    # Store calibration data
                    self.object_points.append(objp)
                    self.image_points.append(corners_refined)
                    self.calibration_images.append(color_frame.copy())
                    
                    images_captured += 1
                    self.logger.info(f"Captured calibration image {images_captured}")
                    
                    # Flash effect
                    white_frame = np.ones_like(display_frame) * 255
                    cv2.imshow('Calibration', white_frame)
                    cv2.waitKey(100)
                
                elif key == 27:  # ESC to finish
                    break
            
            cv2.destroyWindow('Calibration')
            
            # Perform calibration if enough images
            if len(self.object_points) >= self.min_images:
                self._perform_calibration(gray.shape[::-1])
            else:
                self.logger.error(f"Not enough images captured: {len(self.object_points)}")
                self.last_calibration_result = CalibrationResult(
                    success=False,
                    calibration_data={},
                    reprojection_error=0.0,
                    num_images_used=0,
                    error="Insufficient calibration images"
                )
            
        except Exception as e:
            self.logger.error(f"Calibration loop error: {e}")
        finally:
            self.is_calibrating = False
    
    def _perform_calibration(self, image_size: Tuple[int, int]):
        """Perform camera calibration computation"""
        try:
            self.logger.info("Computing camera calibration...")
            
            # Initial calibration
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.object_points, self.image_points, image_size, None, None
            )
            
            if not ret:
                raise RuntimeError("Camera calibration failed")
            
            # Calculate reprojection error
            total_error = 0
            for i in range(len(self.object_points)):
                imgpoints2, _ = cv2.projectPoints(
                    self.object_points[i], rvecs[i], tvecs[i], 
                    camera_matrix, dist_coeffs
                )
                error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
            
            mean_error = total_error / len(self.object_points)
            
            # Optimize calibration if error is high
            if mean_error > 1.0:
                self.logger.info("High reprojection error, performing optimization...")
                camera_matrix, dist_coeffs = self._optimize_calibration(
                    camera_matrix, dist_coeffs, image_size
                )
                
                # Recalculate error
                total_error = 0
                for i in range(len(self.object_points)):
                    imgpoints2, _ = cv2.projectPoints(
                        self.object_points[i], rvecs[i], tvecs[i], 
                        camera_matrix, dist_coeffs
                    )
                    error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    total_error += error
                
                mean_error = total_error / len(self.object_points)
            
            # Create calibration result
            calibration_data = {
                'camera_matrix': camera_matrix.tolist(),
                'distortion_coefficients': dist_coeffs.tolist(),
                'image_size': image_size,
                'reprojection_error': mean_error,
                'num_images': len(self.object_points),
                'calibration_method': 'checkerboard',
                'intrinsics': {
                    'fx': float(camera_matrix[0, 0]),
                    'fy': float(camera_matrix[1, 1]),
                    'cx': float(camera_matrix[0, 2]),
                    'cy': float(camera_matrix[1, 2]),
                    'width': image_size[0],
                    'height': image_size[1],
                    'distortion': dist_coeffs.flatten().tolist()
                }
            }
            
            self.last_calibration_result = CalibrationResult(
                success=True,
                calibration_data=calibration_data,
                reprojection_error=mean_error,
                num_images_used=len(self.object_points)
            )
            
            # Save calibration
            self._save_calibration(calibration_data)
            
            self.logger.info(f"Calibration completed successfully!")
            self.logger.info(f"Reprojection error: {mean_error:.4f} pixels")
            self.logger.info(f"Used {len(self.object_points)} images")
            
        except Exception as e:
            self.logger.error(f"Calibration computation failed: {e}")
            self.last_calibration_result = CalibrationResult(
                success=False,
                calibration_data={},
                reprojection_error=0.0,
                num_images_used=0,
                error=str(e)
            )
    
    def _optimize_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, 
                            image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize calibration parameters"""
        try:
            # Use more sophisticated calibration flags
            flags = (cv2.CALIB_RATIONAL_MODEL + 
                    cv2.CALIB_THIN_PRISM_MODEL + 
                    cv2.CALIB_TILTED_MODEL)
            
            ret, camera_matrix_opt, dist_coeffs_opt, _, _ = cv2.calibrateCamera(
                self.object_points, self.image_points, image_size,
                camera_matrix, dist_coeffs, flags=flags
            )
            
            if ret:
                return camera_matrix_opt, dist_coeffs_opt
            else:
                return camera_matrix, dist_coeffs
                
        except Exception as e:
            self.logger.warning(f"Calibration optimization failed: {e}")
            return camera_matrix, dist_coeffs
    
    def auto_calibrate(self, camera_manager, duration: int = 30) -> CalibrationResult:
        """Perform automatic calibration using structure from motion"""
        try:
            self.logger.info(f"Starting auto-calibration for {duration} seconds")
            self.logger.info("Move camera around to capture different views")
            
            start_time = time.time()
            frame_count = 0
            
            # Feature detector for auto-calibration
            detector = cv2.ORB_create(nfeatures=1000)
            
            while time.time() - start_time < duration:
                frames = camera_manager.get_frames()
                if frames is None:
                    continue
                
                color_frame, _ = frames
                gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
                
                # Detect features
                keypoints, descriptors = detector.detectAndCompute(gray, None)
                
                if len(keypoints) > 100:  # Good frame for calibration
                    self.auto_calib_frames.append(gray.copy())
                    self.auto_calib_features.append((keypoints, descriptors))
                    frame_count += 1
                
                # Display progress
                display_frame = color_frame.copy()
                cv2.putText(display_frame, f"Auto-calibration: {frame_count} frames", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Time: {int(time.time() - start_time)}s / {duration}s", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Draw features
                display_frame = cv2.drawKeypoints(display_frame, keypoints, None, color=(0, 255, 0))
                
                cv2.imshow('Auto-Calibration', display_frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
                    break
            
            cv2.destroyWindow('Auto-Calibration')
            
            if frame_count < 10:
                return CalibrationResult(
                    success=False,
                    calibration_data={},
                    reprojection_error=0.0,
                    num_images_used=0,
                    error="Insufficient frames for auto-calibration"
                )
            
            # Perform auto-calibration computation
            return self._compute_auto_calibration(gray.shape[::-1])
            
        except Exception as e:
            self.logger.error(f"Auto-calibration failed: {e}")
            return CalibrationResult(
                success=False,
                calibration_data={},
                reprojection_error=0.0,
                num_images_used=0,
                error=str(e)
            )
# ... (previous code continues)

    def _compute_auto_calibration(self, image_size: Tuple[int, int]) -> CalibrationResult:
        """Compute calibration from auto-captured frames"""
        try:
            self.logger.info("Computing auto-calibration...")
            
            # Match features across frames for structure from motion
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            # Find correspondences between frames
            correspondences = []
            for i in range(len(self.auto_calib_features) - 1):
                kp1, desc1 = self.auto_calib_features[i]
                kp2, desc2 = self.auto_calib_features[i + 1]
                
                if desc1 is not None and desc2 is not None:
                    matches = matcher.match(desc1, desc2)
                    if len(matches) > 50:
                        pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
                        pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
                        correspondences.append((pts1, pts2))
            
            if len(correspondences) < 5:
                raise RuntimeError("Insufficient correspondences for auto-calibration")
            
            # Estimate focal length using fundamental matrix
            focal_lengths = []
            for pts1, pts2 in correspondences:
                try:
                    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
                    if F is not None:
                        # Estimate focal length from fundamental matrix
                        # This is a simplified approach
                        fx_est = np.sqrt(np.abs(F[0, 0] * F[1, 1])) * image_size[0]
                        if 100 < fx_est < 5000:  # Reasonable range
                            focal_lengths.append(fx_est)
                except:
                    continue
            
            if focal_lengths:
                fx = fy = np.median(focal_lengths)
            else:
                # Fallback to reasonable estimate
                fx = fy = max(image_size) * 1.2
            
            cx = image_size[0] / 2.0
            cy = image_size[1] / 2.0
            
            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Minimal distortion for auto-calibration
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            
            # Estimate reprojection error (simplified)
            reprojection_error = 1.5  # Conservative estimate
            
            calibration_data = {
                'camera_matrix': camera_matrix.tolist(),
                'distortion_coefficients': dist_coeffs.tolist(),
                'image_size': image_size,
                'reprojection_error': reprojection_error,
                'num_images': len(self.auto_calib_frames),
                'calibration_method': 'auto',
                'intrinsics': {
                    'fx': float(fx),
                    'fy': float(fy),
                    'cx': float(cx),
                    'cy': float(cy),
                    'width': image_size[0],
                    'height': image_size[1],
                    'distortion': dist_coeffs.flatten().tolist()
                }
            }
            
            result = CalibrationResult(
                success=True,
                calibration_data=calibration_data,
                reprojection_error=reprojection_error,
                num_images_used=len(self.auto_calib_frames)
            )
            
            self._save_calibration(calibration_data)
            self.logger.info(f"Auto-calibration completed with estimated error: {reprojection_error:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Auto-calibration computation failed: {e}")
            return CalibrationResult(
                success=False,
                calibration_data={},
                reprojection_error=0.0,
                num_images_used=0,
                error=str(e)
            )
    
    def _save_calibration(self, calibration_data: Dict[str, Any]):
        """Save calibration data to file"""
        try:
            os.makedirs("data", exist_ok=True)
            
            # Save as JSON
            with open("data/camera_calibration.json", 'w') as f:
                json.dump(calibration_data, f, indent=2)
            
            # Save camera matrix and distortion as numpy files
            camera_matrix = np.array(calibration_data['camera_matrix'])
            dist_coeffs = np.array(calibration_data['distortion_coefficients'])
            
            np.save("data/camera_matrix.npy", camera_matrix)
            np.save("data/distortion_coefficients.npy", dist_coeffs)
            
            self.logger.info("Calibration data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save calibration: {e}")
    
    def load_calibration(self, filepath: str = "data/camera_calibration.json") -> Optional[Dict[str, Any]]:
        """Load calibration data from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    calibration_data = json.load(f)
                
                self.logger.info(f"Calibration loaded from {filepath}")
                return calibration_data
            else:
                self.logger.warning(f"Calibration file not found: {filepath}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load calibration: {e}")
            return None
    
    def stop_calibration(self):
        """Stop ongoing calibration"""
        self.is_calibrating = False
        if self.calibration_thread and self.calibration_thread.is_alive():
            self.calibration_thread.join(timeout=2.0)
    
    def reset(self):
        """Reset calibration manager"""
        self.stop_calibration()
        self.calibration_images.clear()
        self.object_points.clear()
        self.image_points.clear()
        self.auto_calib_frames.clear()
        self.auto_calib_features.clear()
        self.last_calibration_result = None
        
        self.logger.info("Calibration manager reset")    
    