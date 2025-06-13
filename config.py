"""
Configuration Management System
Handles loading, validation, and updating of system configuration
"""

import json
import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

@dataclass
class CameraConfig:
    width: int = 640
    height: int = 480
    fps: int = 30
    auto_exposure: bool = True
    exposure_value: int = 100
    gain: int = 50

@dataclass
class FeatureConfig:
    type: str = "orb"  # orb, sift, surf, fast
    max_features: int = 1000
    quality_threshold: float = 0.01
    match_ratio: float = 0.75
    min_matches: int = 20

@dataclass
class PoseConfig:
    min_triangulation_points: int = 8
    ransac_threshold: float = 1.0
    ransac_confidence: float = 0.99
    max_reprojection_error: float = 2.0
    keyframe_distance_threshold: float = 0.1
    keyframe_angle_threshold: float = 0.1

@dataclass
class LoopConfig:
    min_loop_distance: float = 2.0
    similarity_threshold: float = 0.8
    geometric_verification: bool = True
    max_loop_candidates: int = 5

@dataclass
class VisualizationConfig:
    enable_3d_plot: bool = True
    enable_feature_display: bool = True
    enable_match_display: bool = True
    update_frequency: int = 5
    plot_window_size: int = 500

@dataclass
class SystemConfig:
    camera: CameraConfig
    features: FeatureConfig
    pose: PoseConfig
    loop: LoopConfig
    visualization: VisualizationConfig
    log_level: str = "INFO"
    data_dir: str = "data"
    log_dir: str = "logs"

class ConfigManager:
    """Configuration management with validation and hot-reloading"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        self.config = self._load_default_config()
        
        # Create directories
        os.makedirs(self.config.data_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
    
    def _load_default_config(self) -> SystemConfig:
        """Load default configuration"""
        return SystemConfig(
            camera=CameraConfig(),
            features=FeatureConfig(),
            pose=PoseConfig(),
            loop=LoopConfig(),
            visualization=VisualizationConfig()
        )
    
    def load_config(self, config_file: Optional[str] = None) -> SystemConfig:
        """Load configuration from file"""
        file_path = config_file or self.config_file
        
        if not os.path.exists(file_path):
            self.logger.warning(f"Config file {file_path} not found, using defaults")
            self.save_config(file_path)
            return self.config
        
        try:
            with open(file_path, 'r') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Merge with defaults
            merged_config = self._merge_config(data)
            self.config = self._dict_to_config(merged_config)
            
            self.logger.info(f"Configuration loaded from {file_path}")
            return self.config
            
        except Exception as e:
            self.logger.error(f"Failed to load config from {file_path}: {e}")
            return self.config
    
    def save_config(self, config_file: Optional[str] = None):
        """Save current configuration to file"""
        file_path = config_file or self.config_file
        
        try:
            config_dict = asdict(self.config)
            
            with open(file_path, 'w') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save config to {file_path}: {e}")
    
    def _merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user config with defaults"""
        default_dict = asdict(self.config)
        
        def deep_merge(default: Dict, user: Dict) -> Dict:
            result = default.copy()
            for key, value in user.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(default_dict, user_config)
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """Convert dictionary to configuration objects"""
        return SystemConfig(
            camera=CameraConfig(**config_dict.get('camera', {})),
            features=FeatureConfig(**config_dict.get('features', {})),
            pose=PoseConfig(**config_dict.get('pose', {})),
            loop=LoopConfig(**config_dict.get('loop', {})),
            visualization=VisualizationConfig(**config_dict.get('visualization', {})),
            log_level=config_dict.get('log_level', 'INFO'),
            data_dir=config_dict.get('data_dir', 'data'),
            log_dir=config_dict.get('log_dir', 'logs')
        )
    
    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        try:
            # Validate camera config
            if self.config.camera.width <= 0 or self.config.camera.height <= 0:
                raise ValueError("Invalid camera resolution")
            
            if self.config.camera.fps <= 0:
                raise ValueError("Invalid camera FPS")
            
            # Validate feature config
            if self.config.features.max_features <= 0:
                raise ValueError("Invalid max features")
            
            if not 0 < self.config.features.match_ratio < 1:
                raise ValueError("Invalid match ratio")
            
            # Validate pose config
            if self.config.pose.ransac_confidence <= 0 or self.config.pose.ransac_confidence >= 1:
                raise ValueError("Invalid RANSAC confidence")
            
            self.logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def update_parameter(self, section: str, parameter: str, value: Any):
        """Update a specific configuration parameter"""
        try:
            section_obj = getattr(self.config, section)
            if hasattr(section_obj, parameter):
                setattr(section_obj, parameter, value)
                self.logger.info(f"Updated {section}.{parameter} = {value}")
            else:
                raise ValueError(f"Parameter {parameter} not found in section {section}")
        except Exception as e:
            self.logger.error(f"Failed to update parameter: {e}")
    
    def get_parameter(self, section: str, parameter: str) -> Any:
        """Get a specific configuration parameter"""
        try:
            section_obj = getattr(self.config, section)
            return getattr(section_obj, parameter)
        except Exception as e:
            self.logger.error(f"Failed to get parameter {section}.{parameter}: {e}")
            return None

# Global configuration instance
_config_manager = None

def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """Load configuration and return as dictionary"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    
    config_obj = _config_manager.load_config(config_file)
    return asdict(config_obj)

def get_config() -> Dict[str, Any]:
    """Get current configuration"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return asdict(_config_manager.config)

def save_config(config_file: str = "config.json"):
    """Save current configuration"""
    global _config_manager
    if _config_manager is not None:
        _config_manager.save_config(config_file)