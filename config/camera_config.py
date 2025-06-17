"""
Camera configuration and calibration parameters
"""

import numpy as np
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: np.ndarray
    
    def to_matrix(self) -> np.ndarray:
        """Convert to camera matrix format"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

@dataclass
class IMUParameters:
    """IMU calibration parameters"""
    accel_noise: float = 0.01
    gyro_noise: float = 0.001
    accel_bias: np.ndarray = None
    gyro_bias: np.ndarray = None
    
    def __post_init__(self):
        if self.accel_bias is None:
            self.accel_bias = np.zeros(3)
        if self.gyro_bias is None:
            self.gyro_bias = np.zeros(3)

class CameraConfig:
    """Main camera configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/camera_params.json"
        
        # Default RealSense D435i parameters
        self.rgb_intrinsics = CameraIntrinsics(
            fx=615.0, fy=615.0, cx=320.0, cy=240.0,
            width=640, height=480,
            distortion=np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        )
        
        self.depth_intrinsics = CameraIntrinsics(
            fx=615.0, fy=615.0, cx=320.0, cy=240.0,
            width=640, height=480,
            distortion=np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        )
        
        self.imu_params = IMUParameters()
        
        # Depth to RGB extrinsics (baseline for D435i is ~50mm)
        self.depth_to_rgb_extrinsics = np.array([
            [1.0, 0.0, 0.0, -0.05],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Visual odometry parameters
        self.vo_params = {
            'feature_detector': 'ORB',  # ORB, SIFT, SURF, SUPERPOINT
            'max_features': 1000,
            'ransac_threshold': 1.0,
            'ransac_confidence': 0.99,
            'min_inliers': 50,
            'keyframe_threshold': 0.1,
            'use_depth': True,
            'use_imu': True,
            'depth_scale': 0.001,  # Convert mm to meters
            'max_depth': 10.0,  # Maximum depth in meters
            'min_depth': 0.1,   # Minimum depth in meters
        }
        
        # SLAM parameters
        self.slam_params = {
            'loop_closure_threshold': 0.15,
            'relocalization_threshold': 0.2,
            'bundle_adjustment_interval': 10,
            'keyframe_culling_threshold': 0.9,
            'map_point_culling_threshold': 2,
        }
        
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Update parameters from config file
                if 'rgb_intrinsics' in config:
                    rgb_params = config['rgb_intrinsics']
                    self.rgb_intrinsics = CameraIntrinsics(**rgb_params)
                
                if 'depth_intrinsics' in config:
                    depth_params = config['depth_intrinsics']
                    self.depth_intrinsics = CameraIntrinsics(**depth_params)
                
                if 'vo_params' in config:
                    self.vo_params.update(config['vo_params'])
                
                if 'slam_params' in config:
                    self.slam_params.update(config['slam_params'])
                    
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        config = {
            'rgb_intrinsics': {
                'fx': self.rgb_intrinsics.fx,
                'fy': self.rgb_intrinsics.fy,  
                'cx': self.rgb_intrinsics.cx,
                'cy': self.rgb_intrinsics.cy,
                'width': self.rgb_intrinsics.width,
                'height': self.rgb_intrinsics.height,
                'distortion': self.rgb_intrinsics.distortion.tolist()
            },
            'depth_intrinsics': {
                'fx': self.depth_intrinsics.fx,
                'fy': self.depth_intrinsics.fy,
                'cx': self.depth_intrinsics.cx,
                'cy': self.depth_intrinsics.cy,
                'width': self.depth_intrinsics.width,
                'height': self.depth_intrinsics.height,
                'distortion': self.depth_intrinsics.distortion.tolist()
            },
            'vo_params': self.vo_params,
            'slam_params': self.slam_params
        }
        
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)