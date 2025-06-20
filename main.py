"""
Agricultural SLAM System - Main Entry Point
Intel RealSense D435i Visual SLAM for Equipment Tracking

Phase 2: Visual SLAM with Feature Detection and Trajectory Tracking
"""

import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import our main window
from src.gui.main_window import MainWindow

def main():
    """Main application entry point"""
    print("=" * 60)
    print("Agricultural SLAM System - Phase 2")
    print("Intel RealSense D435i Visual SLAM")
    print("Features: Visual Odometry + Trajectory Tracking")
    print("=" * 60)
    
    # Create Qt application
    app = QApplication(sys.argv)
      # Set application properties
    app.setApplicationName("Agricultural SLAM System")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Agricultural Technology Solutions")
    
    # Enable high DPI support (PyQt6 compatible)
    app.setAttribute(Qt.ApplicationAttribute.AA_Use96Dpi, False)
    
    try:
        # Create and show main window
        window = MainWindow()
        window.show()
        
        print("Phase 2 application started successfully")
        print("Features available:")
        print("- Live camera feed with feature overlay")
        print("- Real-time visual odometry")
        print("- 2D trajectory visualization")
        print("- Distance measurement")
        print("- SLAM performance monitoring")
        print("")
        print("Instructions:")
        print("1. Click 'Start Camera' to begin")
        print("2. Check 'Enable SLAM' to start tracking")
        print("3. Check 'Show Features' to see detected features")
        print("4. Move camera to see trajectory building")
        print("5. Press Ctrl+Q to quit")
        
        # Run application
        return app.exec()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())