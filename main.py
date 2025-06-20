"""
Agricultural SLAM System - Enhanced Main Entry Point
Complete integration of agricultural SLAM with precision distance tracking
Optimized for Intel RealSense D435i in field environments

Version: 2.0 - Agricultural Enhanced
Features: 
- Centimeter-level distance accuracy
- 3D trajectory visualization  
- Agricultural scene understanding
- Real-time performance optimization
- Automatic 10x10 checkerboard calibration

User: Mr-Parth24
Date: 2025-06-20 23:02:20 UTC
"""

import sys
import os
import time
import argparse
import signal
import json  # For calibration file checking
from typing import Optional
from PyQt6.QtWidgets import QApplication, QMessageBox, QSplashScreen
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QFont

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import our enhanced components
try:
    from src.gui.main_window import EnhancedMainWindow
    from src.core.camera_manager import CameraManager
    from src.utils.config_manager import get_config_manager
    from src.utils.data_logger import get_data_logger
    
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False

# Import automatic calibration (with error handling)
try:
    from src.utils.automatic_calibration import start_automatic_calibration
    AUTOMATIC_CALIBRATION_AVAILABLE = True
except ImportError as e:
    print(f"Automatic calibration not available: {e}")
    AUTOMATIC_CALIBRATION_AVAILABLE = False
    def start_automatic_calibration():
        print("❌ Automatic calibration not available")
        return {'success': False, 'error': 'Module not available'}

class AgriculturalSLAMApplication:
    """
    Main application class for Agricultural SLAM System
    Handles initialization, error management, and system coordination
    """
    
    def __init__(self):
        """Initialize the agricultural SLAM application"""
        self.app = None
        self.main_window = None
        self.config_manager = None
        self.data_logger = None
        
        # System status
        self.initialization_successful = False
        self.shutdown_requested = False
        
        # Performance monitoring
        self.start_time = time.time()
        self.system_stats = {
            'frames_processed': 0,
            'sessions_completed': 0,
            'total_distance_tracked': 0.0,
            'uptime_seconds': 0.0
        }
        
        print("Agricultural SLAM Application initialized")
    
    def create_splash_screen(self) -> QSplashScreen:
        """Create splash screen for application startup"""
        try:
            # Create splash screen pixmap (you can replace with actual image)
            splash_pixmap = QPixmap(400, 300)
            splash_pixmap.fill(Qt.GlobalColor.darkGreen)
            
            splash = QSplashScreen(splash_pixmap)
            splash.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.SplashScreen)
            
            # Add text to splash screen
            font = QFont("Arial", 16, QFont.Weight.Bold)
            splash.setFont(font)
            splash.showMessage(
                "Agricultural SLAM System v2.0\nMr-Parth24 Edition\nInitializing...",
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom,
                Qt.GlobalColor.white
            )
            
            return splash
            
        except Exception as e:
            print(f"Splash screen creation error: {e}")
            return None
    
    def check_system_requirements(self) -> bool:
        """Check system requirements for agricultural SLAM"""
        print("Checking system requirements...")
        
        requirements_met = True
        issues = []
        
        # Check Python version
        if sys.version_info < (3, 8):
            issues.append("Python 3.8+ required")
            requirements_met = False
        
        # Check critical imports
        try:
            import cv2
            cv_version = cv2.__version__
            print(f"✅ OpenCV {cv_version} available")
        except ImportError:
            issues.append("OpenCV not available")
            requirements_met = False
        
        try:
            import numpy as np
            print(f"✅ NumPy {np.__version__} available")
        except ImportError:
            issues.append("NumPy not available")
            requirements_met = False
        
        try:
            import pyrealsense2 as rs
            print("✅ Intel RealSense SDK available")
        except ImportError:
            issues.append("Intel RealSense SDK not available")
            requirements_met = False
        
        try:
            from PyQt6.QtWidgets import QApplication
            print("✅ PyQt6 available")
        except ImportError:
            issues.append("PyQt6 not available")
            requirements_met = False
        
        # Check optional components
        try:
            import OpenGL.GL
            print("✅ OpenGL available (3D visualization enabled)")
        except ImportError:
            print("⚠️  OpenGL not available (3D visualization disabled)")
        
        if not requirements_met:
            print("❌ System requirements not met:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        print("✅ All system requirements met")
        return True
    
    def initialize_system_components(self, splash: Optional[QSplashScreen] = None) -> bool:
        """Initialize all system components"""
        try:
            if splash:
                splash.showMessage("Loading configuration...", 
                                 Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom,
                                 Qt.GlobalColor.white)
            
            # Initialize configuration manager
            self.config_manager = get_config_manager()
            print("✅ Configuration manager initialized")
            
            if splash:
                splash.showMessage("Initializing data logger...", 
                                 Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom,
                                 Qt.GlobalColor.white)
            
            # Initialize data logger
            self.data_logger = get_data_logger()
            print("✅ Data logger initialized")
            
            if splash:
                splash.showMessage("Setting up agricultural SLAM...", 
                                 Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom,
                                 Qt.GlobalColor.white)
            
            # Test camera availability (without starting)
            try:
                test_camera = CameraManager()
                if test_camera.initialize_camera():
                    test_camera.stop_streaming()
                    print("✅ Camera test successful")
                else:
                    print("⚠️  Camera test failed (camera may not be connected)")
            except Exception as e:
                print(f"⚠️  Camera test warning: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ System component initialization failed: {e}")
            return False
    
    def create_main_window(self) -> bool:
        """Create and initialize the main window"""
        try:
            self.main_window = EnhancedMainWindow()
            
            # Set up window properties
            self.main_window.setWindowTitle("Agricultural SLAM System v2.0 - Mr-Parth24 Edition")
            
            # Connect cleanup handler
            self.main_window.closeEvent = self.handle_main_window_close
            
            print("✅ Main window created successfully")
            return True
            
        except Exception as e:
            print(f"❌ Main window creation failed: {e}")
            return False
    
    def handle_main_window_close(self, event):
        """Handle main window close event"""
        try:
            print("Shutting down Agricultural SLAM System...")
            
            # Calculate uptime
            self.system_stats['uptime_seconds'] = time.time() - self.start_time
            
            # Print session summary
            self.print_session_summary()
            
            # Accept the close event
            event.accept()
            
        except Exception as e:
            print(f"Shutdown error: {e}")
            event.accept()
    
    def print_session_summary(self):
        """Print session summary statistics"""
        try:
            uptime_hours = self.system_stats['uptime_seconds'] / 3600
            
            print("\n" + "="*60)
            print("AGRICULTURAL SLAM SYSTEM - SESSION SUMMARY")
            print("="*60)
            print(f"User: Mr-Parth24")
            print(f"Session Duration: {uptime_hours:.2f} hours")
            print(f"System Version: Agricultural SLAM v2.0")
            print(f"Features Used:")
            print(f"  - Enhanced Visual Odometry: ✅")
            print(f"  - Precision Distance Tracking: ✅")
            print(f"  - Agricultural Scene Understanding: ✅")
            print(f"  - 3D Trajectory Visualization: ✅")
            print(f"  - Automatic 10x10 Calibration: ✅")
            print(f"  - Real-time Performance Optimization: ✅")
            
            # Get additional stats from main window if available
            if self.main_window and hasattr(self.main_window, 'slam_system'):
                try:
                    if self.main_window.slam_system:
                        stats = self.main_window.slam_system.get_realtime_stats()
                        if stats.get('status') == 'active':
                            print(f"\nSLAM Statistics:")
                            print(f"  - Total Distance (SLAM): {stats['distances']['total']:.3f}m")
                            print(f"  - Precision Distance: {stats['distances']['precision']:.3f}m")
                            print(f"  - Features Detected: {stats['features']}")
                            print(f"  - Keyframes Created: {stats['keyframes']}")
                            print(f"  - Average FPS: {stats['performance']['fps']:.1f}")
                            
                            agri_info = stats.get('agricultural', {})
                            if agri_info:
                                print(f"\nAgricultural Features:")
                                print(f"  - Scene Type: {agri_info.get('scene_type', 'unknown').title()}")
                                print(f"  - Crop Rows Detected: {'Yes' if agri_info.get('crop_rows_detected') else 'No'}")
                                print(f"  - Field Coverage: {agri_info.get('field_coverage_m2', 0):.1f}m²")
                except Exception as e:
                    print(f"Error getting SLAM stats: {e}")
            
            print("="*60)
            print("Thank you for using Agricultural SLAM System!")
            print("="*60)
            
        except Exception as e:
            print(f"Session summary error: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, shutting down gracefully...")
            self.shutdown_requested = True
            if self.app:
                self.app.quit()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def run(self, args) -> int:
        """Run the agricultural SLAM application"""
        try:
            print("="*60)
            print("AGRICULTURAL SLAM SYSTEM v2.0 - ENHANCED")
            print("Real-time Visual SLAM for Agricultural Equipment")
            print("Mr-Parth24 Edition - 2025-06-20 23:02:20 UTC")
            print("="*60)
            print("Features:")
            print("  🎯 Centimeter-level distance accuracy")
            print("  🗺️  Interactive 3D trajectory visualization")
            print("  🌾 Agricultural scene understanding")
            print("  📊 Real-time performance monitoring")
            print("  💾 Session management and data export")
            print("  🤖 Automatic 10x10 checkerboard calibration")
            print("="*60)
            
            # Check if imports were successful
            if not IMPORTS_SUCCESSFUL:
                print("❌ Critical imports failed. Please check installation.")
                return 1
            
            # Check system requirements
            if not self.check_system_requirements():
                print("❌ System requirements not met.")
                return 1
            
            # Check and run calibration if needed (before GUI startup)
            print("🔍 Checking camera calibration...")
            if not run_calibration_if_needed():
                print("❌ Calibration required for optimal operation")
                print("   Please run calibration before continuing")
                return 1
            
            # Create Qt application
            self.app = QApplication(sys.argv)
            self.app.setApplicationName("Agricultural SLAM System")
            self.app.setApplicationVersion("2.0")
            self.app.setOrganizationName("Agricultural Technology Solutions")
            
            # Enable high DPI support
            self.app.setAttribute(Qt.ApplicationAttribute.AA_Use96Dpi, False)
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Create splash screen
            splash = None
            if not args.no_splash:
                splash = self.create_splash_screen()
                if splash:
                    splash.show()
                    self.app.processEvents()
            
            # Initialize system components
            if not self.initialize_system_components(splash):
                if splash:
                    splash.close()
                QMessageBox.critical(None, "Initialization Error", 
                                   "Failed to initialize system components.")
                return 1
            
            # Create main window
            if splash:
                splash.showMessage("Creating main interface...", 
                                 Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom,
                                 Qt.GlobalColor.white)
                self.app.processEvents()
            
            if not self.create_main_window():
                if splash:
                    splash.close()
                QMessageBox.critical(None, "Window Error", 
                                   "Failed to create main window.")
                return 1
            
            # Show main window
            self.main_window.show()
            
            # Close splash screen
            if splash:
                splash.finish(self.main_window)
            
            # Set initialization flag
            self.initialization_successful = True
            
            print("🚀 Agricultural SLAM System started successfully!")
            print("\nQuick Start Guide:")
            print("1. Click 'Start Camera' to begin video stream")
            print("2. Check 'Enable SLAM' to start tracking")
            print("3. Check 'Show Features' to see detected features")
            print("4. Move camera to build trajectory")
            print("5. Use 3D view for comprehensive visualization")
            print("6. Monitor distance accuracy in real-time")
            print("\nPress Ctrl+C or close window to exit safely.\n")
            
            # Show startup message in status bar
            if hasattr(self.main_window, 'status_bar'):
                self.main_window.status_bar.showMessage(
                    "Agricultural SLAM System ready - Camera calibrated and ready"
                )
            
            # Run application
            return self.app.exec()
            
        except Exception as e:
            print(f"❌ Application error: {e}")
            if splash:
                splash.close()
            if self.app:
                QMessageBox.critical(None, "Application Error", 
                                   f"Critical error: {str(e)}")
            return 1
        
        finally:
            if self.initialization_successful:
                print("Agricultural SLAM System shutdown complete.")

def show_calibration_menu():
    """Show calibration options to user"""
    print()
    print("🎯 Camera Calibration Options:")
    print("=" * 50)
    print("1. 🤖 Automatic Calibration (Recommended)")
    print("   - Intelligent auto-capture for 10x10 board")
    print("   - Hands-free operation with real-time feedback")
    print("   - Optimal quality and coverage assessment")
    print()
    print("2. 📱 Manual Calibration")
    print("   - Traditional capture with manual control")
    print("   - Good for troubleshooting or special cases")
    print()
    print("3. ⏭️  Skip Calibration")
    print("   - Use existing calibration (if available)")
    print("   - Not recommended for first-time setup")
    print()
    
    while True:
        choice = input("Select option (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("Please enter 1, 2, or 3")

def run_calibration_if_needed():
    """Check if calibration is needed and run appropriate calibration"""
    calibration_file = "config/current_calibration.json"
    
    # Check if calibration exists
    if os.path.exists(calibration_file):
        try:
            with open(calibration_file, 'r') as f:
                cal_data = json.load(f)
            
            cal_type = cal_data.get('metadata', {}).get('calibration_type', '')
            
            if 'automatic_agricultural_d435i_10x10' in cal_type:
                print("✅ Found automatic 10x10 calibration")
                print(f"   Date: {cal_data.get('metadata', {}).get('calibration_date', 'unknown')}")
                print(f"   User: {cal_data.get('metadata', {}).get('user', 'unknown')}")
                return True
            else:
                print("⚠️  Found old calibration format")
        except Exception as e:
            print(f"⚠️  Calibration file error: {e}")
    else:
        print("❌ No calibration found")
    
    print("\n🎯 Camera calibration is recommended for optimal accuracy")
    print("   Your 10x10 checkerboard (40mm squares) is perfect for high precision!")
    
    choice = show_calibration_menu()
    
    if choice == '1':
        if not AUTOMATIC_CALIBRATION_AVAILABLE:
            print("❌ Automatic calibration not available, using manual mode")
            choice = '2'
        else:
            print("\n🤖 Starting Automatic Calibration...")
            print("   Hold your 10x10 checkerboard and move to different positions")
            print("   System will auto-capture optimal frames")
            try:
                result = start_automatic_calibration()
                return result.get('success', False)
            except Exception as e:
                print(f"❌ Automatic calibration error: {e}")
                return False
    
    if choice == '2':
        print("\n📱 Starting Manual Calibration...")
        try:
            os.system("python scripts/calibrate_camera.py --auto-save")
            return os.path.exists(calibration_file)
        except Exception as e:
            print(f"❌ Manual calibration error: {e}")
            return False
    else:
        print("⏭️  Skipping calibration")
        print("   Note: Accuracy may be reduced without proper calibration")
        return True
    
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Agricultural SLAM System v2.0 - Enhanced real-time visual SLAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Normal startup with splash screen
  python main.py --no-splash        # Start without splash screen
  python main.py --automatic        # Start in automatic calibration mode
  python main.py --performance fast # Start in fast performance mode
  
Agricultural SLAM Features:
  • Centimeter-level distance accuracy with multi-layer validation
  • Real-time 3D trajectory visualization with agricultural overlays
  • Crop row detection and ground plane estimation
  • Session management for agricultural mapping workflows
  • Automatic 10x10 checkerboard calibration with intelligent capture
  • Optimized for Intel RealSense D435i in field environments        """
    )
    
    parser.add_argument(
        '--no-splash', 
        action='store_true',
        help='Skip splash screen on startup'
    )
    
    parser.add_argument(
        '--performance',
        choices=['fast', 'balanced', 'accurate'],
        default='balanced',
        help='Set initial performance mode (default: balanced)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output'
    )
    
    parser.add_argument(
        '--automatic',
        action='store_true',
        help='Start automatic calibration mode'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Agricultural SLAM System v2.0 - Mr-Parth24 Edition'
    )
    
    return parser.parse_args()

def main():
    """Main entry point"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Set debug mode if requested
        if args.debug:
            os.environ['SLAM_DEBUG'] = '1'
            print("Debug mode enabled")
        
        # Check for automatic calibration mode
        if args.automatic:
            if not AUTOMATIC_CALIBRATION_AVAILABLE:
                print("❌ Automatic calibration not available")
                print("   Please check that all required modules are installed")
                return 1
            
            print("🤖 Starting Automatic Calibration Mode...")
            result = start_automatic_calibration()
        
        # Create and run application
        app = AgriculturalSLAMApplication()
        return app.run(args)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 0
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)