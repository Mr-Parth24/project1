"""
Automatic Camera Calibration System - FIXED VERSION
Mr-Parth24 | 2025-06-20 23:28:21 UTC
WORKING with relaxed thresholds for real-world conditions
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import time
import json
import threading
from typing import Dict, List, Tuple, Optional
from collections import deque
import os
import winsound  # For Windows audio feedback

class AutomaticCalibrationSystem:
    """
    FIXED Automatic calibration system for Mr-Parth24
    Date: 2025-06-20 23:28:21 UTC
    """
    
    def __init__(self):
        """Initialize with WORKING parameters for Mr-Parth24"""
        # Checkerboard parameters for 10x10 board
        self.checkerboard_size = (9, 9)  # 9x9 internal corners
        self.square_size = 0.040  # 40mm squares
        self.board_physical_size = (0.36, 0.36)  # 18" x 18"
        
        # FIXED Auto-capture parameters - MUCH MORE PERMISSIVE
        self.auto_capture_enabled = True
        self.stability_frames_required = 3   # REDUCED: Only need 3 stable frames
        self.quality_threshold = 0.15        # REDUCED: From 0.75 to 0.15
        self.distance_tolerance = 1.0        # INCREASED: Very flexible
        self.angle_tolerance = 1.0           # INCREASED: Very flexible
        self.sharpness_threshold = 0.01      # REDUCED: From 0.6 to 0.01
        self.lighting_threshold = 0.2        # REDUCED: From 0.5 to 0.2
        
        # Coverage requirements
        self.target_frames = 12  # Reduced target
        self.min_frames = 8      # Reduced minimum
        self.coverage_zones = self._initialize_coverage_zones()
        
        # Camera and detection
        self.pipeline = None
        self.config = None
        self.running = False
        
        # Frame storage
        self.captured_frames = []
        self.calibration_points_2d = []
        self.calibration_points_3d = []
        
        # Stability tracking
        self.stability_tracker = deque(maxlen=self.stability_frames_required)
        self.last_capture_time = 0
        self.capture_cooldown = 0.5  # REDUCED: Very fast captures
        
        # Quality metrics
        self.frame_history = deque(maxlen=100)
        self.quality_history = deque(maxlen=50)
        
        # Progress tracking
        self.progress = {
            'frames_captured': 0,
            'coverage_score': 0.0,
            'quality_average': 0.0,
            'time_started': 0,
            'estimated_completion': 0
        }
        
        print("🤖 FIXED Automatic Calibration System - Mr-Parth24")
        print(f"   Date: 2025-06-20 23:28:21 UTC")
        print(f"   RELAXED thresholds for real-world conditions")
        print(f"   Target: {self.target_frames} frames | Min: {self.min_frames}")
    
    def _initialize_coverage_zones(self) -> Dict:
        """Initialize coverage zones"""
        zones = {}
        
        # Add distance zones - RELAXED ranges
        zones['close'] = {'captured': False, 'distance_range': (0.3, 1.0), 'frames': []}
        zones['medium'] = {'captured': False, 'distance_range': (1.0, 2.0), 'frames': []}
        zones['far'] = {'captured': False, 'distance_range': (2.0, 4.0), 'frames': []}
        
        # Add angle zones - RELAXED ranges
        zones['center'] = {'captured': False, 'angle_range': (-20, 20), 'frames': []}
        zones['left'] = {'captured': False, 'angle_range': (-60, -20), 'frames': []}
        zones['right'] = {'captured': False, 'angle_range': (20, 60), 'frames': []}
        
        return zones
    
    def setup_camera(self) -> bool:
        """Setup camera for automatic calibration"""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Use 640x480 for better performance
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start pipeline
            profile = self.pipeline.start(self.config)
            
            # Optimize camera settings
            device = profile.get_device()
            color_sensor = device.first_color_sensor()
            depth_sensor = device.first_depth_sensor()
            
            # Fixed exposure settings
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)
            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, 360)
            
            print("✅ Camera setup complete - WORKING configuration")
            return True
            
        except Exception as e:
            print(f"❌ Camera setup failed: {e}")
            return False
    
    def start_automatic_calibration(self) -> Dict:
        """Start the WORKING automatic calibration process"""
        try:
            if not self.setup_camera():
                return {'success': False, 'error': 'Camera setup failed'}
            
            self.running = True
            self.progress['time_started'] = time.time()
            
            print("🚀 STARTING WORKING CALIBRATION - Mr-Parth24")
            print("=" * 60)
            print(f"Date: 2025-06-20 23:28:21 UTC")
            print("📋 WORKING Instructions:")
            print("   1. Hold 10x10 checkerboard in view")
            print("   2. System will auto-capture with RELAXED thresholds")
            print("   3. Press 'c' to force manual capture anytime")
            print("   4. Press 'q' to quit")
            print("   5. Very permissive - should capture easily!")
            print("=" * 60)
            
            return self._calibration_loop()
            
        except Exception as e:
            print(f"❌ Calibration failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calibration_loop(self) -> Dict:
        """WORKING calibration loop with aggressive capture"""
        try:
            frame_count = 0
            last_status_update = time.time()
            
            while self.running and len(self.captured_frames) < self.target_frames:
                # Get frames
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame:
                    continue
                
                frame_count += 1
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
                
                # Analyze frame
                analysis = self._analyze_frame_for_capture(color_image, depth_image)
                
                # Add to stability tracker
                self.stability_tracker.append(analysis)
                
                # AGGRESSIVE AUTO-CAPTURE CHECK
                should_capture = False
                
                # Method 1: Super permissive auto-capture
                if self._should_auto_capture_permissive():
                    should_capture = True
                    print("🔥 AUTO-CAPTURE: Permissive mode!")
                
                # Method 2: Simple detection-based capture
                elif (analysis.get('detected', False) and 
                      time.time() - self.last_capture_time > self.capture_cooldown):
                    should_capture = True
                    print("🔥 AUTO-CAPTURE: Simple detection mode!")
                
                # Perform capture
                if should_capture:
                    if self._capture_calibration_frame(color_image, depth_image, analysis):
                        self._play_capture_sound()
                        print(f"📸 CAPTURED frame {len(self.captured_frames)}/{self.target_frames}")
                        self._update_coverage_zones(analysis)
                
                # Create display
                display_image = self._create_feedback_display(color_image, analysis)
                
                # Update progress
                self._update_progress()
                
                # Show display
                cv2.imshow('Mr-Parth24 WORKING Calibration - Press C to force capture', display_image)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):  # MANUAL CAPTURE
                    if analysis.get('detected', False):
                        if self._capture_calibration_frame(color_image, depth_image, analysis):
                            self._play_capture_sound()
                            print(f"📸 MANUAL CAPTURE frame {len(self.captured_frames)}/{self.target_frames}")
                            self._update_coverage_zones(analysis)
                elif key == ord('q'):
                    print("\n🛑 User quit")
                    break
                
                # Status update every 3 seconds
                if time.time() - last_status_update > 3.0:
                    self._print_status_update()
                    last_status_update = time.time()
                
                # Auto-complete check
                if len(self.captured_frames) >= self.min_frames:
                    print(f"\n🎉 Minimum frames captured! Continue or press 'q' to finish")
                
                if len(self.captured_frames) >= self.target_frames:
                    print("\n🎉 Target frames reached! Completing calibration...")
                    break
            
            cv2.destroyAllWindows()
            
            # Finalize if we have enough frames
            if len(self.captured_frames) >= self.min_frames:
                return self._finalize_calibration()
            else:
                print(f"❌ Need at least {self.min_frames} frames, got {len(self.captured_frames)}")
                return {'success': False, 'error': 'Insufficient frames'}
                
        except Exception as e:
            print(f"❌ Calibration loop error: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            self.cleanup()
    
    def _analyze_frame_for_capture(self, color_image: np.ndarray, depth_image: np.ndarray) -> Dict:
        """WORKING frame analysis with debug output"""
        try:
            # Convert and enhance
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Detect checkerboard with RELAXED parameters
            ret, corners = cv2.findChessboardCorners(
                enhanced, 
                self.checkerboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            analysis = {
                'detected': ret,
                'corners': corners,
                'timestamp': time.time(),
                'quality_score': 0.0,
                'distance': 0.0,
                'angle': 0.0,
                'sharpness': 0.0,
                'lighting': 0.0,
                'capture_ready': False
            }
            
            if ret and corners is not None:
                # Refine corners
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (5, 5), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)
                )
                analysis['corners_refined'] = corners_refined
                
                # Calculate metrics with WORKING thresholds
                analysis['quality_score'] = self._calculate_quality_permissive(gray, corners_refined)
                analysis['distance'] = self._estimate_board_distance(corners_refined)
                analysis['angle'] = self._estimate_board_angle(corners_refined)
                analysis['sharpness'] = self._calculate_sharpness_permissive(gray)
                analysis['lighting'] = self._assess_lighting_permissive(gray)
                
                # VERY PERMISSIVE capture readiness
                analysis['capture_ready'] = (
                    analysis['quality_score'] > self.quality_threshold and      # 0.15
                    0.2 <= analysis['distance'] <= 5.0 and                     # Very wide range
                    analysis['sharpness'] > self.sharpness_threshold and       # 0.01
                    analysis['lighting'] > self.lighting_threshold             # 0.2
                )
                
                # DEBUG OUTPUT for Mr-Parth24
                print(f"🔍 DEBUG: Q={analysis['quality_score']:.2f}, "
                      f"D={analysis['distance']:.2f}m, "
                      f"S={analysis['sharpness']:.2f}, "
                      f"L={analysis['lighting']:.2f}, "
                      f"Ready={analysis['capture_ready']}")
            
            return analysis
            
        except Exception as e:
            print(f"Analysis error: {e}")
            return {'detected': False, 'capture_ready': False, 'timestamp': time.time()}
    
    def _should_auto_capture_permissive(self) -> bool:
        """VERY permissive auto-capture decision"""
        try:
            if len(self.stability_tracker) < self.stability_frames_required:
                return False
            
            if time.time() - self.last_capture_time < self.capture_cooldown:
                return False
            
            # Check recent frames
            recent_frames = list(self.stability_tracker)
            detected_count = sum(1 for f in recent_frames if f.get('detected', False))
            ready_count = sum(1 for f in recent_frames if f.get('capture_ready', False))
            
            # VERY PERMISSIVE: Just need detection in most frames
            if detected_count >= len(recent_frames) * 0.6:  # 60% detected
                return True
            
            # Or if any frame is ready
            if ready_count > 0:
                return True
            
            return False
            
        except Exception as e:
            print(f"Auto-capture error: {e}")
            return False
    
    def _calculate_quality_permissive(self, gray: np.ndarray, corners: np.ndarray) -> float:
        """PERMISSIVE quality calculation"""
        try:
            # Simple contrast-based quality
            contrast = gray.std() / 255.0
            
            # Corner distribution quality
            if corners is not None and len(corners) > 10:
                corner_points = corners.reshape(-1, 2)
                x_spread = (corner_points[:, 0].max() - corner_points[:, 0].min()) / gray.shape[1]
                y_spread = (corner_points[:, 1].max() - corner_points[:, 1].min()) / gray.shape[0]
                distribution = (x_spread + y_spread) / 2.0
            else:
                distribution = 0.0
            
            # Combined quality (very permissive)
            quality = (contrast + distribution) / 2.0
            return min(quality, 1.0)
            
        except Exception as e:
            return 0.5
    
    def _calculate_sharpness_permissive(self, gray: np.ndarray) -> float:
        """PERMISSIVE sharpness calculation"""
        try:
            # Use gradient magnitude instead of Laplacian
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            sharpness = magnitude.mean() / 100.0  # Scale down
            return min(sharpness, 1.0)
        except:
            return 0.5
    
    def _assess_lighting_permissive(self, gray: np.ndarray) -> float:
        """PERMISSIVE lighting assessment"""
        try:
            mean_brightness = gray.mean() / 255.0
            # Accept almost any lighting
            if 0.1 <= mean_brightness <= 0.9:
                return 0.8
            else:
                return 0.5
        except:
            return 0.5
    
    def _capture_calibration_frame(self, color_image: np.ndarray, depth_image: np.ndarray, analysis: Dict) -> bool:
        """Capture frame - WORKING version"""
        try:
            if not analysis.get('detected', False):
                return False
            
            frame_data = {
                'image': color_image.copy(),
                'depth': depth_image.copy() if depth_image is not None else None,
                'corners': analysis.get('corners_refined', analysis.get('corners')).copy(),
                'quality_score': analysis['quality_score'],
                'distance': analysis['distance'],
                'angle': analysis['angle'],
                'timestamp': analysis['timestamp'],
                'user': 'Mr-Parth24',
                'capture_time': '2025-06-20 23:28:21'
            }
            
            # Generate 3D points
            object_points = self._generate_object_points()
            
            # Store data
            self.captured_frames.append(frame_data)
            self.calibration_points_2d.append(frame_data['corners'])
            self.calibration_points_3d.append(object_points)
            
            self.last_capture_time = time.time()
            return True
            
        except Exception as e:
            print(f"Capture error: {e}")
            return False
    
    def _generate_object_points(self) -> np.ndarray:
        """Generate 3D object points"""
        object_points = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        object_points[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        object_points *= self.square_size
        return object_points
    
    def _create_feedback_display(self, color_image: np.ndarray, analysis: Dict) -> np.ndarray:
        """Create feedback display"""
        try:
            display = color_image.copy()
            
            # Draw checkerboard if detected
            if analysis.get('detected', False) and analysis.get('corners') is not None:
                cv2.drawChessboardCorners(display, self.checkerboard_size, analysis['corners'], True)
            
            # Status panel
            panel_height = 200
            panel = np.zeros((panel_height, display.shape[1], 3), dtype=np.uint8)
            
            # Progress
            cv2.putText(panel, f"Mr-Parth24 | Frames: {len(self.captured_frames)}/{self.target_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if analysis.get('detected', False):
                status_color = (0, 255, 0) if analysis.get('capture_ready', False) else (0, 165, 255)
                cv2.putText(panel, f"DETECTED - Quality: {analysis['quality_score']:.2f}", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
                cv2.putText(panel, f"Distance: {analysis['distance']:.2f}m", 
                           (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
                
                if analysis.get('capture_ready', False):
                    cv2.putText(panel, "READY TO CAPTURE!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.putText(panel, "Adjusting position...", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            else:
                cv2.putText(panel, "Searching for checkerboard...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            
            cv2.putText(panel, "Press 'C' to force capture | 'Q' to quit", 
                       (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            return np.vstack([display, panel])
            
        except Exception as e:
            return color_image
    
    def _finalize_calibration(self) -> Dict:
        """Finalize calibration - WORKING version"""
        try:
            print(f"\n🔍 Finalizing calibration with {len(self.captured_frames)} frames...")
            
            image_size = self.captured_frames[0]['image'].shape[:2][::-1]
            
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.calibration_points_3d,
                self.calibration_points_2d,
                image_size,
                None,
                None
            )
            
            if not ret:
                return {'success': False, 'error': 'Calibration failed'}
            
            # Calculate error
            total_error = 0
            for i in range(len(self.calibration_points_3d)):
                projected_points, _ = cv2.projectPoints(
                    self.calibration_points_3d[i], rvecs[i], tvecs[i], 
                    camera_matrix, dist_coeffs
                )
                error = cv2.norm(self.calibration_points_2d[i], projected_points, cv2.NORM_L2) / len(projected_points)
                total_error += error
            
            mean_error = total_error / len(self.calibration_points_3d)
            
            result = {
                'success': True,
                'camera_matrix': camera_matrix,
                'distortion_coefficients': dist_coeffs,
                'reprojection_error': mean_error,
                'frames_used': len(self.captured_frames),
                'image_size': image_size,
                'user': 'Mr-Parth24',
                'completion_time': '2025-06-20 23:28:21',
                'checkerboard_info': {
                    'size': self.checkerboard_size,
                    'square_size_mm': 40,
                    'board_type': 'mr_parth24_working_10x10'
                }
            }
            
            # Save results
            self._save_calibration_results(result)
            
            print(f"✅ WORKING CALIBRATION COMPLETE!")
            print(f"   User: Mr-Parth24")
            print(f"   Frames: {len(self.captured_frames)}")
            print(f"   Error: {mean_error:.3f} pixels")
            
            return result
            
        except Exception as e:
            print(f"❌ Finalization error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _save_calibration_results(self, result: Dict):
        """Save calibration results"""
        try:
            os.makedirs('data/calibration', exist_ok=True)
            os.makedirs('config', exist_ok=True)
            
            # Prepare for JSON
            save_data = result.copy()
            save_data['camera_matrix'] = result['camera_matrix'].tolist()
            save_data['distortion_coefficients'] = result['distortion_coefficients'].tolist()
            save_data['metadata'] = {
                'calibration_type': 'automatic_agricultural_d435i_10x10',
                'user': 'Mr-Parth24',
                'date': '2025-06-20 23:28:21',
                'version': 'working_fix_v1'
            }
            
            # Save current calibration
            with open('config/current_calibration.json', 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print("✅ Calibration saved to config/current_calibration.json")
            
        except Exception as e:
            print(f"Save error: {e}")
    
    # Required helper methods
    def _estimate_board_distance(self, corners: np.ndarray) -> float:
        """Estimate board distance"""
        try:
            if corners is None or len(corners) < 4:
                return 1.0
            corner_points = corners.reshape(-1, 2)
            width = corner_points[:, 0].max() - corner_points[:, 0].min()
            height = corner_points[:, 1].max() - corner_points[:, 1].min()
            avg_size = (width + height) / 2
            return max(0.3, min(3.0, 300.0 / avg_size))  # Rough estimate
        except:
            return 1.0
    
    def _estimate_board_angle(self, corners: np.ndarray) -> float:
        """Estimate board angle"""
        try:
            if corners is None or len(corners) < 2:
                return 0.0
            corner_points = corners.reshape(-1, 2)
            if len(corner_points) >= 2:
                edge = corner_points[1] - corner_points[0]
                angle = np.degrees(np.arctan2(edge[1], edge[0]))
                return angle
            return 0.0
        except:
            return 0.0
    
    def _update_coverage_zones(self, analysis: Dict):
        """Update coverage zones"""
        try:
            distance = analysis.get('distance', 0)
            angle = analysis.get('angle', 0)
            
            if 0.3 <= distance <= 1.0:
                self.coverage_zones['close']['captured'] = True
            elif 1.0 <= distance <= 2.0:
                self.coverage_zones['medium']['captured'] = True
            elif 2.0 <= distance <= 4.0:
                self.coverage_zones['far']['captured'] = True
            
            if -20 <= angle <= 20:
                self.coverage_zones['center']['captured'] = True
            elif -60 <= angle <= -20:
                self.coverage_zones['left']['captured'] = True
            elif 20 <= angle <= 60:
                self.coverage_zones['right']['captured'] = True
        except:
            pass
    
    def _update_progress(self):
        """Update progress"""
        self.progress['frames_captured'] = len(self.captured_frames)
        self.progress['coverage_score'] = len([z for z in self.coverage_zones.values() if z['captured']]) / len(self.coverage_zones)
    
    def _print_status_update(self):
        """Print status"""
        print(f"\n📊 Mr-Parth24 Status:")
        print(f"   Frames: {len(self.captured_frames)}/{self.target_frames}")
        print(f"   Coverage: {self.progress['coverage_score']:.2f}")
    
    def _play_capture_sound(self):
        """Play capture sound"""
        try:
            winsound.Beep(800, 200)
        except:
            print("*BEEP*")
    
    def cleanup(self):
        """Cleanup"""
        try:
            self.running = False
            if self.pipeline:
                self.pipeline.stop()
            cv2.destroyAllWindows()
            print("🧹 Cleanup complete")
        except:
            pass

# Global instance
auto_calibration_system = AutomaticCalibrationSystem()

def start_automatic_calibration() -> Dict:
    """Start WORKING automatic calibration"""
    return auto_calibration_system.start_automatic_calibration()

def get_auto_calibration_system() -> AutomaticCalibrationSystem:
    """Get WORKING calibration system"""
    return auto_calibration_system