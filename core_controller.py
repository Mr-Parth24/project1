"""
Core System Controller - Main orchestrator for the Visual Odometry System
Author: Enhanced Visual SLAM System
Date: 2025-06-13
"""

import threading
import time
import queue
import json
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
import logging

from camera_manager import CameraManager
from feature_processor import FeatureProcessor
from enhanced_pose_estimator import PoseEstimator
from loop_detector import LoopDetector
from visualizer import Visualizer
from data_manager import DataManager
from calibration_manager import CalibrationManager
from performance_monitor import PerformanceMonitor
from error_handler import ErrorHandler
import config

class SystemState(Enum):
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    CALIBRATING = "calibrating"

@dataclass
class SystemStatus:
    state: SystemState
    fps: float
    features_detected: int
    features_matched: int
    total_distance: float
    current_position: np.ndarray
    loop_detected: bool
    error_message: Optional[str] = None

class CoreController:
    """Main system controller orchestrating all components"""
    
    def __init__(self, config_file: str = "config.json"):
        # Initialize logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = config.load_config(config_file)
        
        # Initialize components
        self.camera_manager = None
        self.feature_processor = None
        self.pose_estimator = None
        self.loop_detector = None
        self.visualizer = None
        self.data_manager = None
        self.calibration_manager = None
        self.performance_monitor = None
        self.error_handler = None
        
        # System state
        self.state = SystemState.STOPPED
        self.status = SystemStatus(
            state=SystemState.STOPPED,
            fps=0.0,
            features_detected=0,
            features_matched=0,
            total_distance=0.0,
            current_position=np.zeros(3),
            loop_detected=False
        )
        
        # Threading
        self.main_thread = None
        self.running = False
        self.paused = False
        
        # Data queues
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=100)
        
        # Callbacks
        self.callbacks = {}
        
        self.logger.info("Core controller initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/system_{time.strftime("%Y%m%d_%H%M%S")}.log'),
                logging.StreamHandler()
            ]
        )
    
    def initialize_components(self):
        """Initialize all system components"""
        try:
            self.state = SystemState.INITIALIZING
            self.logger.info("Initializing system components...")
            
            # Initialize error handler first
            self.error_handler = ErrorHandler()
            
            # Initialize camera
            self.camera_manager = CameraManager(self.config)
            if not self.camera_manager.initialize():
                raise RuntimeError("Failed to initialize camera")
            
            # Initialize other components
            self.feature_processor = FeatureProcessor(self.config)
            self.pose_estimator = PoseEstimator(
                self.camera_manager.get_intrinsics(), 
                self.config
            )
            self.loop_detector = LoopDetector(self.config)
            self.visualizer = Visualizer(self.config)
            self.data_manager = DataManager(self.config)
            self.calibration_manager = CalibrationManager(self.config)
            self.performance_monitor = PerformanceMonitor()
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            self.state = SystemState.ERROR
            self.status.error_message = str(e)
            return False
    
    def start(self):
        """Start the visual odometry system"""
        if self.state == SystemState.RUNNING:
            self.logger.warning("System is already running")
            return False
        
        if not self.initialize_components():
            return False
        
        self.running = True
        self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_thread.start()
        
        self.logger.info("System started")
        return True
    
    def stop(self):
        """Stop the visual odometry system"""
        self.running = False
        self.paused = False
        
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5.0)
        
        # Cleanup components
        if self.camera_manager:
            self.camera_manager.cleanup()
        if self.visualizer:
            self.visualizer.cleanup()
        if self.data_manager:
            self.data_manager.save_session()
        
        self.state = SystemState.STOPPED
        self.logger.info("System stopped")
    
    def pause(self):
        """Pause the system"""
        if self.state == SystemState.RUNNING:
            self.paused = True
            self.state = SystemState.PAUSED
            self.logger.info("System paused")
    
    def resume(self):
        """Resume the system"""
        if self.state == SystemState.PAUSED:
            self.paused = False
            self.state = SystemState.RUNNING
            self.logger.info("System resumed")
    
    def reset(self):
        """Reset the system state"""
        self.pause()
        
        # Reset components
        if self.pose_estimator:
            self.pose_estimator.reset()
        if self.loop_detector:
            self.loop_detector.reset()
        if self.data_manager:
            self.data_manager.reset()
        
        self.status.total_distance = 0.0
        self.status.current_position = np.zeros(3)
        self.status.loop_detected = False
        
        self.logger.info("System reset")
    
    def _main_loop(self):
        """Main processing loop"""
        self.state = SystemState.RUNNING
        frame_count = 0
        
        try:
            while self.running:
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Start performance monitoring
                self.performance_monitor.start_frame()
                
                # Get camera frames
                frames = self.camera_manager.get_frames()
                if frames is None:
                    continue
                
                color_frame, depth_frame = frames
                
                # Process features
                features = self.feature_processor.process_frame(color_frame)
                if features is None:
                    continue
                
                # Estimate pose
                pose_result = self.pose_estimator.estimate_pose(
                    features, depth_frame, frame_count
                )
                
                if pose_result.success:
                    # Update status
                    self.status.features_detected = len(features.keypoints)
                    self.status.features_matched = pose_result.num_matches
                    self.status.total_distance = pose_result.total_distance
                    self.status.current_position = pose_result.position
                    
                    # Check for loop closure
                    loop_result = self.loop_detector.check_loop_closure(
                        features, pose_result.position
                    )
                    self.status.loop_detected = loop_result.detected
                    
                    # Update visualization
                    if self.visualizer:
                        self.visualizer.update(
                            color_frame, 
                            features, 
                            pose_result, 
                            loop_result
                        )
                    
                    # Log data
                    if self.data_manager:
                        self.data_manager.log_frame(
                            frame_count, 
                            color_frame, 
                            features, 
                            pose_result, 
                            loop_result
                        )
                
                # End performance monitoring
                self.performance_monitor.end_frame(
                    self.status.features_detected,
                    self.status.features_matched
                )
                self.status.fps = self.performance_monitor.get_fps()
                
                # Execute callbacks
                self._execute_callbacks()
                
                frame_count += 1
                
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.error_handler.handle_error(e)
            self.state = SystemState.ERROR
            self.status.error_message = str(e)
    
    def _execute_callbacks(self):
        """Execute registered callbacks"""
        for callback_name, callback_func in self.callbacks.items():
            try:
                callback_func(self.status)
            except Exception as e:
                self.logger.error(f"Error in callback {callback_name}: {e}")
    
    def register_callback(self, name: str, callback: Callable):
        """Register a status callback"""
        self.callbacks[name] = callback
    
    def unregister_callback(self, name: str):
        """Unregister a callback"""
        if name in self.callbacks:
            del self.callbacks[name]
    
    def get_status(self) -> SystemStatus:
        """Get current system status"""
        return self.status
    
    def calibrate_camera(self, use_checkerboard: bool = True):
        """Start camera calibration"""
        if self.state == SystemState.RUNNING:
            self.pause()
        
        self.state = SystemState.CALIBRATING
        
        try:
            if use_checkerboard:
                result = self.calibration_manager.calibrate_with_checkerboard()
            else:
                result = self.calibration_manager.auto_calibrate()
            
            if result.success:
                self.logger.info("Camera calibration successful")
                # Update camera parameters
                self.camera_manager.update_calibration(result.calibration_data)
            else:
                self.logger.error(f"Camera calibration failed: {result.error}")
                
        except Exception as e:
            self.logger.error(f"Calibration error: {e}")
        finally:
            self.state = SystemState.STOPPED