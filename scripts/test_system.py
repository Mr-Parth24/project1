#!/usr/bin/env python3
"""
System test script for Agricultural SLAM System
Validates all components and dependencies
"""

import sys
import os
import time
import traceback

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing imports...")
    
    tests = [
        ("numpy", "import numpy as np"),
        ("OpenCV", "import cv2"),
        ("PyQt6", "from PyQt6.QtWidgets import QApplication"),
        ("Intel RealSense", "import pyrealsense2 as rs"),
        ("YAML", "import yaml"),
        ("psutil", "import psutil"),
    ]
    
    optional_tests = [
        ("OpenGL", "from OpenGL.GL import *"),
        ("GPUtil", "import GPUtil"),
        ("scipy", "import scipy"),
        ("scikit-learn", "from sklearn import cluster"),
        ("matplotlib", "import matplotlib.pyplot as plt"),
    ]
    
    failed_core = []
    failed_optional = []
    
    # Test core dependencies
    for name, import_str in tests:
        try:
            exec(import_str)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_core.append(name)
    
    # Test optional dependencies
    print("\n🔧 Testing optional dependencies...")
    for name, import_str in optional_tests:
        try:
            exec(import_str)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name}: Not installed (optional)")
            failed_optional.append(name)
    
    return failed_core, failed_optional

def test_camera_connection():
    """Test Intel RealSense camera connection"""
    print("\n📷 Testing camera connection...")
    
    try:
        import pyrealsense2 as rs
        
        # Create context
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("❌ No RealSense cameras detected")
            return False
        
        print(f"✅ Found {len(devices)} RealSense device(s)")
        
        for i, device in enumerate(devices):
            print(f"   Device {i}: {device.get_info(rs.camera_info.name)}")
            print(f"   Serial: {device.get_info(rs.camera_info.serial_number)}")
            print(f"   Firmware: {device.get_info(rs.camera_info.firmware_version)}")
        
        # Test basic pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        profile = pipeline.start(config)
        
        # Try to get a few frames
        for i in range(5):
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                print(f"❌ Failed to get frames on attempt {i+1}")
                pipeline.stop()
                return False
        
        pipeline.stop()
        print("✅ Camera connection and basic streaming test passed")
        return True
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_system_components():
    """Test Agricultural SLAM system components"""
    print("\n🌾 Testing Agricultural SLAM components...")
    
    try:
        # Test core components
        print("Testing core components...")
        from src.core import CameraManager, AgriSLAMCore
        from src.utils import get_config_manager, get_data_logger
        print("✅ Core components import successful")
        
        # Test GUI components
        print("Testing GUI components...")
        from src.gui import EnhancedMainWindow
        print("✅ GUI components import successful")
        
        # Test algorithms
        print("Testing algorithm components...")
        from src.algorithms import EnhancedCustomVisualSLAM
        print("✅ Algorithm components import successful")
        
        # Test utilities
        print("Testing utility components...")
        from src.utils import get_performance_monitor, get_calibration_helper
        from src.filters import filter_slam_data, get_adaptive_threshold_manager
        print("✅ Utility components import successful")
        
        # Test configuration loading
        print("Testing configuration...")
        config_manager = get_config_manager()
        camera_config = config_manager.get_camera_config()
        slam_config = config_manager.get_slam_config()
        
        if camera_config and slam_config:
            print("✅ Configuration loading successful")
        else:
            print("⚠️  Configuration loading incomplete")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        traceback.print_exc()
        return False

def test_performance():
    """Test system performance capabilities"""
    print("\n⚡ Testing performance capabilities...")
    
    try:
        import psutil
        
        # CPU info
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        print(f"✅ CPU: {cpu_count} cores @ {cpu_freq.current:.0f}MHz")
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"✅ Memory: {memory_gb:.1f}GB total, {memory.percent}% used")
        
        # Test OpenGL availability
        try:
            from OpenGL.GL import glGetString, GL_VERSION
            print("✅ OpenGL available for 3D visualization")
        except ImportError:
            print("⚠️  OpenGL not available (3D visualization disabled)")
        
        # Performance recommendations
        if cpu_count >= 4 and memory_gb >= 8:
            print("✅ System meets recommended specifications")
        elif cpu_count >= 2 and memory_gb >= 4:
            print("⚠️  System meets minimum specifications")
        else:
            print("❌ System below minimum specifications")
            print("   Recommended: 4+ cores, 8GB+ RAM")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_data_directories():
    """Test data directory structure"""
    print("\n📁 Testing data directories...")
    
    directories = [
        "data",
        "data/trajectories", 
        "data/maps",
        "data/logs",
        "data/calibration",
        "config"
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory}")
        else:
            print(f"⚠️  {directory} (will be created)")
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"   Created {directory}")
            except Exception as e:
                print(f"❌ Failed to create {directory}: {e}")
                return False
    
    return True

def main():
    """Main test routine"""
    print("🌾 Agricultural SLAM System - System Test")
    print("=" * 60)
    print(f"Test started: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    # Run all tests
    tests = [
        ("Import Test", test_imports),
        ("Camera Test", test_camera_connection),
        ("Component Test", test_system_components),
        ("Performance Test", test_performance),
        ("Directory Test", test_data_directories),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_name == "Import Test":
                failed_core, failed_optional = test_func()
                results[test_name] = len(failed_core) == 0
                if failed_core:
                    print(f"\n❌ Critical imports failed: {', '.join(failed_core)}")
            else:
                results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Test summary
    print("\n" + "="*60)
    print("🎯 TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your Agricultural SLAM System is ready for use!")
        print()
        print("🚀 Quick start:")
        print("   python main.py")
        print()
        print("📚 Next steps:")
        print("   1. Run camera calibration (recommended):")
        print("      python scripts/calibrate_camera.py")
        print("   2. Start the main application")
        print("   3. Connect Intel RealSense D435i camera")
        print("   4. Begin agricultural SLAM mapping")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please address the failed tests before using the system.")
        print()
        print("💡 Common solutions:")
        print("   - Install missing dependencies: pip install -r requirements.txt")
        print("   - Install Intel RealSense SDK")
        print("   - Check camera connection")
        print("   - Verify system meets minimum requirements")
        
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())