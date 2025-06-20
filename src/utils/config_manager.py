"""
Configuration Manager for SLAM System
Handles loading and managing configuration files
"""

import yaml
import os
import numpy as np
from typing import Dict, Any, Optional

class ConfigManager:
    """
    Manages configuration files for the SLAM system
    Loads camera, SLAM, and GUI configurations
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir
        self.camera_config = None
        self.slam_config = None
        self.gui_config = None
        
        # Load all configurations
        self.load_all_configs()
    
    def load_all_configs(self):
        """Load all configuration files"""
        try:
            # Load camera configuration
            camera_path = os.path.join(self.config_dir, "camera_config.yaml")
            self.camera_config = self.load_config(camera_path)
            
            # Load SLAM configuration
            slam_path = os.path.join(self.config_dir, "slam_config.yaml")
            self.slam_config = self.load_config(slam_path)
            
            # Merge GUI config from camera config for backward compatibility
            self.gui_config = self.camera_config.get('gui', {})
            
            print("All configurations loaded successfully")
            
        except Exception as e:
            print(f"Error loading configurations: {e}")
            self._load_default_configs()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load a single configuration file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                print(f"Loaded config: {config_path}")
                return config
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            return {}
        except Exception as e:
            print(f"Error loading config {config_path}: {e}")
            return {}
    
    def _load_default_configs(self):
        """Load default configurations if files are missing"""
        print("Loading default configurations")
        
        # Default camera config
        self.camera_config = {
            'camera': {
                'color_width': 640,
                'color_height': 480,
                'depth_width': 640,
                'depth_height': 480,
                'fps': 30,
                'enable_depth': True,
                'enable_color': True,
                'enable_imu': False,
                'auto_exposure': True
            },
            'gui': {
                'window_title': 'Agricultural SLAM System',
                'window_width': 1400,
                'window_height': 900,
                'camera_display_width': 640,
                'camera_display_height': 480
            }
        }
        
        # Default SLAM config
        self.slam_config = {
            'slam': {
                'max_features': 1000,
                'min_features_for_tracking': 30,
                'min_matches': 6,
                'ransac_threshold': 5.0,
                'max_translation_per_frame': 10.0,
                'keyframe_distance_threshold': 0.3,
                'keyframe_angle_threshold': 0.15
            },
            'camera_calibration': {
                'fx': 607.4,
                'fy': 607.4,
                'cx': 319.5,
                'cy': 239.5
            }
        }
        
        # GUI config from camera config
        self.gui_config = self.camera_config['gui']
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera configuration"""
        return self.camera_config or {}
    
    def get_slam_config(self) -> Dict[str, Any]:
        """Get SLAM configuration"""
        return self.slam_config or {}
    
    def get_gui_config(self) -> Dict[str, Any]:
        """Get GUI configuration"""
        return self.gui_config or {}
    
    def get_camera_matrix(self) -> np.ndarray:
        """Get camera intrinsic matrix"""
        try:
            calib = self.slam_config.get('camera_calibration', {})
            fx = calib.get('fx', 607.4)
            fy = calib.get('fy', 607.4)
            cx = calib.get('cx', 319.5)
            cy = calib.get('cy', 239.5)
            
            return np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
        except Exception as e:
            print(f"Error creating camera matrix: {e}")
            # Return default matrix
            return np.array([
                [607.4, 0, 319.5],
                [0, 607.4, 239.5],
                [0, 0, 1]
            ], dtype=np.float32)
    
    def get_distortion_coefficients(self) -> np.ndarray:
        """Get camera distortion coefficients"""
        try:
            calib = self.slam_config.get('camera_calibration', {})
            k1 = calib.get('k1', 0.0)
            k2 = calib.get('k2', 0.0)
            p1 = calib.get('p1', 0.0)
            p2 = calib.get('p2', 0.0)
            k3 = calib.get('k3', 0.0)
            
            return np.array([k1, k2, p1, p2, k3], dtype=np.float32)
            
        except Exception as e:
            print(f"Error creating distortion coefficients: {e}")
            return np.zeros(5, dtype=np.float32)
    
    def get_slam_parameter(self, parameter_name: str, default_value: Any = None) -> Any:
        """
        Get a specific SLAM parameter
        
        Args:
            parameter_name: Name of the parameter
            default_value: Default value if parameter not found
            
        Returns:
            Parameter value
        """
        try:
            return self.slam_config.get('slam', {}).get(parameter_name, default_value)
        except:
            return default_value
    
    def save_config(self, config_dict: Dict[str, Any], config_name: str):
        """
        Save configuration to file
        
        Args:
            config_dict: Configuration dictionary to save
            config_name: Name of configuration file (without .yaml)
        """
        try:
            os.makedirs(self.config_dir, exist_ok=True)
            config_path = os.path.join(self.config_dir, f"{config_name}.yaml")
            
            with open(config_path, 'w') as file:
                yaml.safe_dump(config_dict, file, default_flow_style=False, indent=2)
            
            print(f"Configuration saved: {config_path}")
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def update_slam_parameter(self, parameter_name: str, value: Any):
        """
        Update a SLAM parameter and save configuration
        
        Args:
            parameter_name: Name of parameter to update
            value: New value
        """
        try:
            if 'slam' not in self.slam_config:
                self.slam_config['slam'] = {}
            
            self.slam_config['slam'][parameter_name] = value
            self.save_config(self.slam_config, 'slam_config')
            
            print(f"Updated SLAM parameter: {parameter_name} = {value}")
            
        except Exception as e:
            print(f"Error updating SLAM parameter: {e}")

# Global configuration manager instance
config_manager = ConfigManager()

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance"""
    return config_manager

# Test function
def test_config_manager():
    """Test configuration manager"""
    cm = ConfigManager()
    
    print("Camera Config:")
    print(cm.get_camera_config())
    
    print("\nSLAM Config:")
    print(cm.get_slam_config())
    
    print("\nCamera Matrix:")
    print(cm.get_camera_matrix())
    
    print("\nSLAM Parameters:")
    print(f"Max features: {cm.get_slam_parameter('max_features')}")
    print(f"Min matches: {cm.get_slam_parameter('min_matches')}")

if __name__ == "__main__":
    test_config_manager()