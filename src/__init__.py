"""
Agricultural SLAM System - Enhanced Visual SLAM for Agricultural Applications
Version 2.0 - Complete agricultural optimization with precision tracking
"""

__version__ = "2.0.0"
__author__ = "Agricultural Technology Solutions"
__description__ = "Enhanced Visual SLAM System for Agricultural Equipment Tracking"

# Make key components easily importable
from .core import CameraManager, AgriSLAMCore, PrecisionDistanceTracker
from .gui import EnhancedMainWindow as MainWindow
from .utils import get_config_manager, get_data_logger, get_performance_monitor
from .algorithms import EnhancedCustomVisualSLAM

__all__ = [
    'CameraManager',
    'AgriSLAMCore', 
    'PrecisionDistanceTracker',
    'MainWindow',
    'EnhancedCustomVisualSLAM',
    'get_config_manager',
    'get_data_logger',
    'get_performance_monitor'
]