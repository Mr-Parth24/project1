"""
Simple launcher script for the GUI application
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main_gui import main
    print("🚀 Starting SLAM Tracker GUI...")
    main()
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\n📋 Please install required packages:")
    print("pip install PyQt5 pyqtgraph opencv-python pyrealsense2 numpy matplotlib scipy pillow")
except Exception as e:
    print(f"❌ Error starting application: {e}")
    input("Press Enter to exit...")