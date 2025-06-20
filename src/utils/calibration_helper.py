"""
Camera Calibration Helper for Intel RealSense D435i
Specialized calibration utilities for agricultural SLAM applications
Optimized for 10x10 checkerboard with 40mm squares
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import time
import json
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

class D435iCalibrationHelper:
    """
    Calibration helper for Intel RealSense D435i in agricultural environments
    Optimized for 10x10 checkerboard (18" x 18") with 40mm squares
    """
    
    def __init__(self):
        """Initialize calibration helper for 10x10 checkerboard"""
        self.pipeline = None
        self.config = None
        
        # Calibration parameters for your specific checkerboard
        self.checkerboard_size = (9, 9)  # 10x10 board = 9x9 internal corners
        self.square_size = 0.040  # 40mm squares
        self.board_physical_size = (0.36, 0.36)  # 18" x 18" in meters (457mm actual)
        
        self.calibration_frames = []
        self.calibration_points_2d = []
        self.calibration_points_3d = []
        
        # Agricultural-specific parameters
        self.outdoor_exposure_compensation = True
        self.vibration_filtering = True
        self.dust_compensation = True
        
        # Enhanced parameters for larger checkerboard
        self.min_board_distance = 0.8  # Minimum 80cm for large board
        self.max_board_distance = 3.0  # Maximum 3m distance
        self.optimal_distance_range = (1.2, 2.0)  # Optimal range 1.2-2.0m
        
        # Calibration quality metrics
        self.calibration_quality = {
            'reprojection_error': 0.0,
            'coverage_score': 0.0,
            'stability_score': 0.0,
            'outdoor_suitability': 0.0,
            'distance_validation': 0.0
        }
        
        print("D435i Calibration Helper initialized for 10x10 checkerboard (40mm squares)")
        print(f"Board size: {self.checkerboard_size} internal corners")
        print(f"Square size: {self.square_size*1000:.0f}mm")
        print(f"Physical board size: {self.board_physical_size[0]*1000:.0f}mm x {self.board_physical_size[1]*1000:.0f}mm")
    
    def setup_agricultural_calibration(self) -> bool:
        """Setup camera for agricultural calibration with 10x10 board"""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure for calibration (high quality settings for large board)
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            
            # Start pipeline
            profile = self.pipeline.start(self.config)
            
            # Configure camera for outdoor conditions with large board
            device = profile.get_device()
            color_sensor = device.first_color_sensor()
            depth_sensor = device.first_depth_sensor()
            
            # Agricultural environment settings optimized for large checkerboard
            if self.outdoor_exposure_compensation:
                # Enable auto exposure for varying outdoor light
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                color_sensor.set_option(rs.option.auto_exposure_priority, 0)  # Prioritize frame rate
                
                # Set exposure limits for outdoor use with large board
                if color_sensor.supports(rs.option.exposure):
                    color_sensor.set_option(rs.option.exposure, 6000)  # Lower exposure for large board
                
                if color_sensor.supports(rs.option.gain):
                    color_sensor.set_option(rs.option.gain, 48)  # Lower gain for better quality
            
            # Depth sensor optimization for large board detection
            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, 360)  # Max laser power
            
            if depth_sensor.supports(rs.option.accuracy):
                depth_sensor.set_option(rs.option.accuracy, 1)  # Highest accuracy
            
            print("✅ Camera configured for 10x10 checkerboard calibration")
            print("📋 Enhanced Calibration Requirements:")
            print("   - Use 10x10 checkerboard (9x9 internal corners)")
            print("   - Square size: 40mm x 40mm") 
            print("   - Board size: 18\" x 18\" (457mm x 457mm)")
            print("   - Optimal distance: 1.2-2.0 meters from camera")
            print("   - Ensure good outdoor lighting")
            print("   - Hold board steady during capture")
            print("   - Cover different areas and angles")
            
            return True
            
        except Exception as e:
            print(f"❌ Calibration setup failed: {e}")
            return False
    
    def capture_calibration_frame(self, display_feedback: bool = True) -> Dict:
        """Capture and analyze calibration frame for 10x10 checkerboard"""
        try:
            if not self.pipeline:
                return {'success': False, 'error': 'Camera not initialized'}
            
            # Get frames
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame:
                return {'success': False, 'error': 'No color frame'}
            
            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
            
            # Apply agricultural image processing
            processed_image = self._preprocess_for_calibration(color_image)
            
            # Detect 10x10 checkerboard with enhanced parameters
            ret, corners = cv2.findChessboardCorners(
                processed_image, 
                self.checkerboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                cv2.CALIB_CB_NORMALIZE_IMAGE +
                cv2.CALIB_CB_FILTER_QUADS
            )
            
            result = {
                'success': ret,
                'frame': color_image,
                'processed_frame': processed_image,
                'corners': corners if ret else None,
                'frame_quality': self._assess_frame_quality(color_image, depth_image),
                'timestamp': time.time()
            }
            
            if ret:
                # Refine corner positions with enhanced criteria for large board
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (15, 15), (-1, -1),  # Larger window for 40mm squares
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
                )
                
                result['corners_refined'] = corners_refined
                
                # Enhanced quality assessment for large checkerboard
                coverage_score = self._calculate_coverage_score(corners_refined, color_image.shape)
                distance_score = self._validate_board_distance(corners_refined, depth_image)
                angle_score = self._assess_board_angle(corners_refined)
                
                result['coverage_score'] = coverage_score
                result['distance_score'] = distance_score
                result['angle_score'] = angle_score
                result['board_distance'] = self._estimate_board_distance(corners_refined)
                
                if display_feedback:
                    # Enhanced feedback display for large board
                    display_image = color_image.copy()
                    cv2.drawChessboardCorners(display_image, self.checkerboard_size, corners_refined, ret)
                    
                    # Add distance and quality info
                    distance = result['board_distance']
                    quality_text = f"Distance: {distance:.2f}m | Quality: {result['frame_quality']:.2f}"
                    cv2.putText(display_image, quality_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Distance guidance
                    if distance < self.optimal_distance_range[0]:
                        guidance = "Move camera BACK"
                        color = (0, 0, 255)  # Red
                    elif distance > self.optimal_distance_range[1]:
                        guidance = "Move camera CLOSER"
                        color = (0, 165, 255)  # Orange
                    else:
                        guidance = "Distance OPTIMAL"
                        color = (0, 255, 0)  # Green
                    
                    cv2.putText(display_image, guidance, (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    result['display_image'] = display_image
                
                print(f"✅ 10x10 Checkerboard detected - Quality: {result['frame_quality']:.2f}, "
                      f"Coverage: {coverage_score:.2f}, Distance: {result['board_distance']:.2f}m")
            else:
                if display_feedback:
                    display_image = color_image.copy()
                    cv2.putText(display_image, "10x10 Checkerboard NOT detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(display_image, "Ensure good lighting and board visibility", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    result['display_image'] = display_image
                print("❌ 10x10 Checkerboard not detected")
            
            return result
            
        except Exception as e:
            print(f"Calibration frame capture error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _estimate_board_distance(self, corners: np.ndarray) -> float:
        """Estimate distance to checkerboard based on corner positions"""
        try:
            if corners is None or len(corners) < 4:
                return 0.0
            
            # Calculate board size in pixels
            corner_points = corners.reshape(-1, 2)
            
            # Get corner positions (approximate)
            top_left = corner_points[0]
            top_right = corner_points[8]  # 9 corners per row, index 8 is top-right
            bottom_left = corner_points[-9]  # Bottom-left corner
            
            # Calculate pixel dimensions
            width_pixels = np.linalg.norm(top_right - top_left)
            height_pixels = np.linalg.norm(bottom_left - top_left)
            avg_pixels = (width_pixels + height_pixels) / 2
            
            # Known physical size: 9 squares * 40mm = 360mm = 0.36m
            physical_size = 9 * self.square_size  # 0.36m
            
            # Estimate distance using pinhole camera model
            # distance = (physical_size * focal_length) / pixel_size
            focal_length = 615.0  # Approximate D435i focal length at 1280x720
            estimated_distance = (physical_size * focal_length) / avg_pixels
            
            return estimated_distance
            
        except Exception as e:
            print(f"Distance estimation error: {e}")
            return 1.5  # Default distance
    
    def _validate_board_distance(self, corners: np.ndarray, depth_image: np.ndarray) -> float:
        """Validate board distance using depth information"""
        try:
            if depth_image is None or corners is None:
                return 0.5
            
            # Sample depth at corner positions
            corner_points = corners.reshape(-1, 2).astype(int)
            valid_depths = []
            
            for point in corner_points:
                x, y = point
                if (0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]):
                    depth = depth_image[y, x]
                    if depth > 0:
                        valid_depths.append(depth / 1000.0)  # Convert to meters
            
            if len(valid_depths) >= 4:
                avg_depth = np.median(valid_depths)
                
                # Score based on optimal distance range
                if self.optimal_distance_range[0] <= avg_depth <= self.optimal_distance_range[1]:
                    return 1.0  # Perfect
                elif self.min_board_distance <= avg_depth <= self.max_board_distance:
                    return 0.7  # Good
                else:
                    return 0.3  # Poor
            
            return 0.5
            
        except Exception as e:
            print(f"Distance validation error: {e}")
            return 0.5
    
    def _assess_board_angle(self, corners: np.ndarray) -> float:
        """Assess if checkerboard is at a good angle"""
        try:
            if corners is None or len(corners) < 4:
                return 0.5
            
            corner_points = corners.reshape(-1, 2)
            
            # Get the four outer corners (approximate)
            top_left = corner_points[0]
            top_right = corner_points[8]
            bottom_left = corner_points[-9]
            bottom_right = corner_points[-1]
            
            # Calculate if board appears rectangular (not too skewed)
            # Top and bottom edges should be roughly parallel
            top_edge = top_right - top_left
            bottom_edge = bottom_right - bottom_left
            
            # Calculate angle between edges
            top_angle = np.arctan2(top_edge[1], top_edge[0])
            bottom_angle = np.arctan2(bottom_edge[1], bottom_edge[0])
            angle_diff = abs(top_angle - bottom_angle)
            
            # Score based on parallelism
            if angle_diff < 0.1:  # ~6 degrees
                return 1.0
            elif angle_diff < 0.2:  # ~11 degrees
                return 0.8
            elif angle_diff < 0.3:  # ~17 degrees
                return 0.6
            else:
                return 0.3
            
        except Exception as e:
            print(f"Angle assessment error: {e}")
            return 0.5
    
    def add_calibration_frame(self, frame_data: Dict) -> bool:
        """Add frame to calibration dataset with enhanced validation"""
        try:
            if not frame_data['success'] or frame_data['corners'] is None:
                return False
            
            # Enhanced quality checks for large checkerboard
            if frame_data['frame_quality'] < 0.5:
                print("⚠️  Frame quality too low for large checkerboard")
                return False
            
            if frame_data.get('coverage_score', 0) < 0.4:
                print("⚠️  Coverage score too low - board too small in frame")
                return False
            
            if frame_data.get('distance_score', 0) < 0.5:
                distance = frame_data.get('board_distance', 0)
                print(f"⚠️  Distance not optimal: {distance:.2f}m (optimal: {self.optimal_distance_range[0]:.1f}-{self.optimal_distance_range[1]:.1f}m)")
                return False
            
            # Check for similar frames (avoid redundancy)
            if self._is_similar_to_existing(frame_data):
                print("⚠️  Similar frame already exists")
                return False
            
            # Generate 3D calibration points for 10x10 board
            object_points = self._generate_object_points()
            
            # Store calibration data
            self.calibration_frames.append(frame_data)
            self.calibration_points_2d.append(frame_data['corners_refined'])
            self.calibration_points_3d.append(object_points)
            
            print(f"✅ Calibration frame added ({len(self.calibration_frames)}/12 recommended for large board)")
            
            return True
            
        except Exception as e:
            print(f"Add calibration frame error: {e}")
            return False
    
    def _generate_object_points(self) -> np.ndarray:
        """Generate 3D object points for 10x10 checkerboard"""
        try:
            # 9x9 internal corners for 10x10 checkerboard
            object_points = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
            object_points[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
            object_points *= self.square_size  # 40mm squares
            
            return object_points
            
        except Exception as e:
            print(f"Object points generation error: {e}")
            return np.array([])
    
    def perform_agricultural_calibration(self) -> Dict:
        """Perform camera calibration optimized for 10x10 checkerboard"""
        try:
            if len(self.calibration_frames) < 8:  # Reduced minimum for large board
                return {
                    'success': False,
                    'error': f'Insufficient frames: {len(self.calibration_frames)} (minimum 8 for large board)'
                }
            
            print(f"🔍 Performing calibration with {len(self.calibration_frames)} frames...")
            print("   Using 10x10 checkerboard (40mm squares)")
            
            # Get image size
            image_size = self.calibration_frames[0]['frame'].shape[:2][::-1]
            
            # Perform calibration with enhanced flags for large checkerboard
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.calibration_points_3d,
                self.calibration_points_2d,
                image_size,
                None,
                None,
                flags=cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL
            )
            
            if not ret:
                return {'success': False, 'error': 'Calibration failed'}
            
            # Calculate reprojection error
            total_error = 0
            for i in range(len(self.calibration_points_3d)):
                projected_points, _ = cv2.projectPoints(
                    self.calibration_points_3d[i], rvecs[i], tvecs[i], 
                    camera_matrix, dist_coeffs
                )
                error = cv2.norm(self.calibration_points_2d[i], projected_points, cv2.NORM_L2) / len(projected_points)
                total_error += error
            
            mean_error = total_error / len(self.calibration_points_3d)
            
            # Enhanced validation for agricultural use with large board
            agricultural_score = self._validate_agricultural_suitability(
                camera_matrix, dist_coeffs, mean_error
            )
            
            # Store enhanced calibration quality
            self.calibration_quality = {
                'reprojection_error': mean_error,
                'coverage_score': self._calculate_overall_coverage(),
                'stability_score': self._calculate_stability_score(rvecs, tvecs),
                'outdoor_suitability': agricultural_score,
                'distance_validation': self._calculate_distance_validation(),
                'large_board_optimized': True
            }
            
            calibration_result = {
                'success': True,
                'camera_matrix': camera_matrix,
                'distortion_coefficients': dist_coeffs,
                'reprojection_error': mean_error,
                'calibration_quality': self.calibration_quality,
                'frames_used': len(self.calibration_frames),
                'image_size': image_size,
                'checkerboard_info': {
                    'size': self.checkerboard_size,
                    'square_size_mm': self.square_size * 1000,
                    'board_type': '10x10_agricultural'
                },
                'calibration_timestamp': time.time()
            }
            
            print(f"✅ 10x10 Checkerboard calibration completed!")
            print(f"   Reprojection error: {mean_error:.3f} pixels")
            print(f"   Agricultural suitability: {agricultural_score:.2f}/1.0")
            print(f"   Coverage score: {self.calibration_quality['coverage_score']:.2f}")
            print(f"   Large board optimization: ✅")
            
            return calibration_result
            
        except Exception as e:
            print(f"❌ Calibration error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_distance_validation(self) -> float:
        """Calculate distance validation score"""
        try:
            if not self.calibration_frames:
                return 0.0
            
            distance_scores = [frame.get('distance_score', 0.5) for frame in self.calibration_frames]
            return np.mean(distance_scores)
            
        except Exception as e:
            print(f"Distance validation calculation error: {e}")
            return 0.5
    
    # ... (rest of the methods remain the same as the original file)
    # I'll include the essential ones for space:
    
    def save_calibration(self, calibration_data: Dict, filepath: str = None) -> str:
        """Save calibration data with 10x10 board metadata"""
        try:
            if filepath is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filepath = f"d435i_10x10_agricultural_calibration_{timestamp}.json"
            
            # Prepare data for JSON serialization
            save_data = calibration_data.copy()
            save_data['camera_matrix'] = calibration_data['camera_matrix'].tolist()
            save_data['distortion_coefficients'] = calibration_data['distortion_coefficients'].tolist()
            
            # Add enhanced metadata for 10x10 board
            save_data['metadata'] = {
                'calibration_type': 'agricultural_d435i_10x10',
                'checkerboard_size': self.checkerboard_size,
                'square_size_mm': self.square_size * 1000,
                'board_physical_size_mm': [int(s * 1000) for s in self.board_physical_size],
                'calibration_environment': 'outdoor_agricultural',
                'software_version': 'agricultural_slam_v2.0',
                'large_board_optimized': True,
                'optimal_distance_range_m': self.optimal_distance_range
            }
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"📁 10x10 Board calibration saved: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Calibration save error: {e}")
            return ""
    
    def cleanup(self):
        """Cleanup calibration resources"""
        try:
            if self.pipeline:
                self.pipeline.stop()
                self.pipeline = None
            
            self.calibration_frames.clear()
            self.calibration_points_2d.clear()
            self.calibration_points_3d.clear()
            
            print("10x10 Checkerboard calibration helper cleaned up")
            
        except Exception as e:
            print(f"Cleanup error: {e}")

# Global calibration helper
calibration_helper = D435iCalibrationHelper()

def get_calibration_helper() -> D435iCalibrationHelper:
    """Get global calibration helper instance"""
    return calibration_helper

def quick_agricultural_calibration() -> Optional[Dict]:
    """
    Quick calibration workflow - now redirects to automatic system
    DEPRECATED: Use automatic_calibration.py for 10x10 board
    """
    print("🔄 Redirecting to Automatic Calibration System...")
    print("   The manual calibration has been replaced with intelligent auto-capture")
    print("   for your 10x10 checkerboard (40mm squares)")
    print()
    
    try:
        from .automatic_calibration import start_automatic_calibration
        return start_automatic_calibration()
    except ImportError:
        print("❌ Automatic calibration not available, using manual fallback")
        
        helper = get_calibration_helper()
        
        if not helper.setup_agricultural_calibration():
            return None
        
        print("📋 Manual calibration mode for 10x10 checkerboard:")
        print("1. Hold 10x10 checkerboard (40mm squares) steady")
        print("2. Maintain 1.2-2.0m distance from camera")
        print("3. Press 'c' to capture, 'q' to quit, 'd' when finished")
        
        try:
            while len(helper.calibration_frames) < 12:
                frame_data = helper.capture_calibration_frame()
                
                if 'display_image' in frame_data:
                    cv2.imshow('Manual 10x10 Calibration (Fallback)', frame_data['display_image'])
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c') and frame_data['success']:
                    helper.add_calibration_frame(frame_data)
                elif key == ord('q'):
                    break
                elif key == ord('d'):
                    break
            
            cv2.destroyAllWindows()
            
            if len(helper.calibration_frames) >= 8:
                return helper.perform_agricultural_calibration()
            else:
                print("❌ Insufficient calibration frames")
                return None
                
        except KeyboardInterrupt:
            print("\n🛑 Calibration interrupted")
            return None
        finally:
            helper.cleanup()
    """Quick calibration workflow for 10x10 agricultural checkerboard"""
    helper = get_calibration_helper()
    
    print("🌾 Agricultural Camera Calibration Wizard - 10x10 Checkerboard")
    print("=" * 70)
    
    if not helper.setup_agricultural_calibration():
        return None
    
    print("📋 Instructions for 10x10 Checkerboard:")
    print("1. Hold 10x10 checkerboard (40mm squares) steady")
    print("2. Maintain 1.2-2.0m distance from camera (optimal)")
    print("3. Cover different areas and angles")
    print("4. Ensure good lighting and board visibility")
    print("5. Press 'c' to capture, 'q' to quit, 'd' when finished")
    
    try:
        while len(helper.calibration_frames) < 12:  # 12 frames for large board
            frame_data = helper.capture_calibration_frame()
            
            if 'display_image' in frame_data:
                cv2.imshow('10x10 Agricultural Calibration', frame_data['display_image'])
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and frame_data['success']:
                helper.add_calibration_frame(frame_data)
            elif key == ord('q'):
                break
            elif key == ord('d'):  # 'd' for done
                break
        
        cv2.destroyAllWindows()
        
        if len(helper.calibration_frames) >= 8:  # Minimum for large board
            return helper.perform_agricultural_calibration()
        else:
            print("❌ Insufficient calibration frames for 10x10 board")
            return None
            
    except KeyboardInterrupt:
        print("\n🛑 Calibration interrupted")
        return None
    finally:
        helper.cleanup()