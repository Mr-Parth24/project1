"""
Configuration file for SLAM Tracker
Adjust these settings based on your setup
"""

# Camera Settings
CAMERA_CONFIG = {
    "width": 640,
    "height": 480,
    "fps": 30
}

# Visual Odometry Settings
VO_CONFIG = {
    "max_features": 1000,
    "min_features": 50,
    "match_threshold": 0.7,
    "ransac_threshold": 0.05,  # 5cm
    "max_movement_per_frame": 1.0,  # 1 meter
    "movement_history_size": 5
}

# Plotting Settings
PLOT_CONFIG = {
    "max_trajectory_points": 1000,
    "update_interval": 0.1,
    "plot_size": (10, 8)
}

# Data Export Settings
EXPORT_CONFIG = {
    "output_directory": "data/output_paths",
    "auto_save_interval": 300  # Auto-save every 5 minutes
}