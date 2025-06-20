"""
GUI components for the Agricultural SLAM System
Enhanced interface with 3D visualization and real-time monitoring
"""

from .main_window import EnhancedMainWindow
from .camera_widget import EnhancedCameraWidget, CameraWidget
from .trajectory_widget import EnhancedTrajectoryWidget, TrajectoryWidget

# Backward compatibility
MainWindow = EnhancedMainWindow

__all__ = [
    'EnhancedMainWindow',
    'MainWindow',  # Backward compatibility
    'EnhancedCameraWidget', 
    'CameraWidget',
    'EnhancedTrajectoryWidget',
    'TrajectoryWidget'
]