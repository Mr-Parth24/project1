#!/usr/bin/env python3
"""
RealSense D435i Visual Odometry GUI - Enhanced Main Application
Fixed tracking issues with comprehensive logging and debugging

Author: Enhanced for Mr-Parth24
Date: 2025-06-16
"""

import sys
import os
import logging
import traceback
from PyQt5.QtWidgets import QApplication, QStyleFactory, QMessageBox
from PyQt5.QtCore import Qt
from gui.main_window import MainWindow
from config.camera_config import CameraConfig
from utils.enhanced_logging import setup_enhanced_logging, PerformanceLogger

def setup_application_environment():
    """Setup application environment and directories"""
    # Create necessary directories
    directories = [
        'logs',
        'data/exports', 
        'data/calibration',
        'data/sessions',
        'config'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_dependencies():
    """Check if all required dependencies are available"""
    required_modules = [
        ('pyrealsense2', 'Intel RealSense SDK'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('PyQt5', 'PyQt5'),
        ('pyqtgraph', 'PyQtGraph')
    ]
    
    missing_modules = []
    
    for module, name in required_modules:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            missing_modules.append(name)
    
    if missing_modules:
        print(f"\n❌ Missing dependencies: {', '.join(missing_modules)}")
        print("Please install missing dependencies before running the application.")
        return False
    
    return True

def check_camera_connection():
    """Check if RealSense camera is connected"""
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("⚠️  No RealSense devices detected")
            print("   Connect your D435i camera to use live tracking")
            return False
        else:
            for dev in devices:
                name = dev.get_info(rs.camera_info.name)
                serial = dev.get_info(rs.camera_info.serial_number)
                print(f"✅ Found camera: {name} (Serial: {serial})")
            return True
            
    except Exception as e:
        print(f"❌ Camera check failed: {e}")
        return False

def main():
    """Enhanced main application entry point"""
    print("🚀 RealSense D435i Visual Odometry - Enhanced Version")
    print("=" * 60)
    print(f"User: Mr-Parth24")
    print(f"Date: 2025-06-16 22:26:01 UTC")
    print("=" * 60)
    
    # Setup environment
    print("\n📁 Setting up environment...")
    setup_application_environment()
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        input("Press Enter to exit...")
        return 1
    
    # Setup enhanced logging
    print("\n📝 Setting up logging...")
    try:
        performance_logger = setup_enhanced_logging(
            log_level="DEBUG",  # More verbose logging for debugging
            console_output=True,
            file_output=True,
            performance_logging=True
        )
        print("✅ Enhanced logging system initialized")
    except Exception as e:
        print(f"❌ Failed to setup logging: {e}")
        performance_logger = None
    
    # Get main logger
    logger = logging.getLogger(__name__)
    logger.info("=== Application Starting ===")
    logger.info(f"User: Mr-Parth24")
    logger.info(f"Session started at: 2025-06-16 22:26:01 UTC")
    
    # Check camera
    print("\n📷 Checking camera connection...")
    camera_available = check_camera_connection()
    
    try:
        # Create QApplication
        print("\n🎨 Creating application...")
        app = QApplication(sys.argv)
        app.setApplicationName("RealSense D435i Visual Odometry - Enhanced")
        app.setApplicationVersion("1.1.0")
        app.setOrganizationName("Mr-Parth24")
        app.setOrganizationDomain("github.com/Mr-Parth24")
        
        # Set application style
        app.setStyle(QStyleFactory.create('Fusion'))
        
        # Apply enhanced dark theme
        apply_enhanced_theme(app)
        
        # Create and show main window
        print("🏠 Creating main window...")
        window = MainWindow()
        
        # Pass performance logger to window if available
        if performance_logger:
            window.performance_logger = performance_logger
        
        # Show camera warning if not available
        if not camera_available:
            QMessageBox.warning(
                window,
                "Camera Warning",
                "No RealSense D435i camera detected.\n\n"
                "You can still explore the interface, but live tracking will not be available.\n"
                "Connect your D435i camera and restart the application for full functionality."
            )
        
        window.show()
        
        logger.info("Application window created and displayed")
        print("✅ Application started successfully!")
        print("\nInstructions:")
        print("1. Connect your RealSense D435i camera")
        print("2. Click 'Start Tracking' to begin")
        print("3. Move the camera to see trajectory visualization")
        print("4. Check the Debug Information panel for detailed logs")
        print("5. Use 'Export Data' to save your trajectory")
        print("\nTroubleshooting:")
        print("- If tracking doesn't update, check Debug panel for errors")
        print("- Ensure good lighting and textured environment")
        print("- Try different feature detectors (ORB, SIFT) in settings")
        
        # Run application
        exit_code = app.exec_()
        
        # Save performance metrics before exit
        if performance_logger:
            performance_logger.save_metrics()
            summary = performance_logger.get_summary()
            logger.info(f"Session summary: {summary}")
            print(f"\n📊 Session Summary: {summary}")
        
        logger.info("=== Application Ending ===")
        return exit_code
        
    except Exception as e:
        error_msg = f"Failed to start application: {e}"
        print(f"❌ {error_msg}")
        
        if 'logger' in locals():
            logger.error(error_msg, exc_info=True)
        else:
            traceback.print_exc()
        
        # Try to show error dialog
        try:
            if 'app' in locals():
                QMessageBox.critical(
                    None,
                    "Application Error",
                    f"Failed to start application:\n\n{e}\n\n"
                    "Check the log files in the 'logs' directory for more details."
                )
        except:
            pass
        
        return 1

def apply_enhanced_theme(app):
    """Apply enhanced dark theme with better colors"""
    
    # Enhanced dark palette
    dark_palette = app.palette()
    
    # Main colors
    dark_palette.setColor(dark_palette.Window, Qt.black)
    dark_palette.setColor(dark_palette.WindowText, Qt.white)
    dark_palette.setColor(dark_palette.Base, Qt.darkGray)
    dark_palette.setColor(dark_palette.AlternateBase, Qt.gray)
    dark_palette.setColor(dark_palette.ToolTipBase, Qt.white)
    dark_palette.setColor(dark_palette.ToolTipText, Qt.black)
    dark_palette.setColor(dark_palette.Text, Qt.white)
    dark_palette.setColor(dark_palette.Button, Qt.darkGray)
    dark_palette.setColor(dark_palette.ButtonText, Qt.white)
    dark_palette.setColor(dark_palette.BrightText, Qt.red)
    dark_palette.setColor(dark_palette.Link, Qt.blue)
    dark_palette.setColor(dark_palette.Highlight, Qt.darkBlue)
    dark_palette.setColor(dark_palette.HighlightedText, Qt.white)
    
    app.setPalette(dark_palette)
    
    # Enhanced stylesheet
    enhanced_style = """
    QMainWindow {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    QWidget {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    QLabel {
        color: #c9d1d9;
    }
    
    QPushButton {
        background-color: #21262d;
        border: 2px solid #30363d;
        border-radius: 6px;
        padding: 8px 16px;
        color: #c9d1d9;
        font-weight: bold;
        min-height: 20px;
    }
    
    QPushButton:hover {
        background-color: #30363d;
        border-color: #58a6ff;
    }
    
    QPushButton:pressed {
        background-color: #161b22;
    }
    
    QPushButton:disabled {
        background-color: #161b22;
        color: #484f58;
        border-color: #21262d;
    }
    
    QGroupBox {
        font-weight: bold;
        border: 2px solid #30363d;
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 10px;
        color: #f0f6fc;
    }
    
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 8px 0 8px;
        color: #58a6ff;
    }
    
    QProgressBar {
        border: 2px solid #30363d;
        border-radius: 4px;
        text-align: center;
        background-color: #161b22;
        color: #c9d1d9;
    }
    
    QProgressBar::chunk {
        background-color: #238636;
        border-radius: 2px;
    }
    
    QTextEdit {
        background-color: #0d1117;
        border: 2px solid #30363d;
        border-radius: 6px;
        color: #c9d1d9;
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    }
    
    QComboBox {
        background-color: #21262d;
        border: 2px solid #30363d;
        border-radius: 4px;
        padding: 4px 8px;
        color: #c9d1d9;
    }
    
    QComboBox:hover {
        border-color: #58a6ff;
    }
    
    QComboBox::drop-down {
        border: none;
    }
    
    QComboBox::down-arrow {
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 6px solid #c9d1d9;
        margin-right: 8px;
    }
    
    QSpinBox, QDoubleSpinBox {
        background-color: #21262d;
        border: 2px solid #30363d;
        border-radius: 4px;
        padding: 4px;
        color: #c9d1d9;
    }
    
    QSpinBox:hover, QDoubleSpinBox:hover {
        border-color: #58a6ff;
    }
    
    QCheckBox {
        color: #c9d1d9;
        spacing: 8px;
    }
    
    QCheckBox::indicator {
        width: 16px;
        height: 16px;
        border: 2px solid #30363d;
        border-radius: 3px;
        background-color: #21262d;
    }
    
    QCheckBox::indicator:checked {
        background-color: #238636;
        border-color: #238636;
    }
    
    QCheckBox::indicator:checked:pressed {
        background-color: #1f6feb;
    }
    
    QStatusBar {
        background-color: #161b22;
        border-top: 1px solid #30363d;
        color: #8b949e;
    }
    
    QMenuBar {
        background-color: #161b22;
        color: #c9d1d9;
        border-bottom: 1px solid #30363d;
    }
    
    QMenuBar::item {
        background-color: transparent;
        padding: 4px 8px;
    }
    
    QMenuBar::item:selected {
        background-color: #30363d;
        border-radius: 4px;
    }
    
    QMenu {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        color: #c9d1d9;
    }
    
    QMenu::item {
        padding: 6px 12px;
    }
    
    QMenu::item:selected {
        background-color: #30363d;
        border-radius: 4px;
    }
    """
    
    app.setStyleSheet(enhanced_style)

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)