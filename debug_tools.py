#!/usr/bin/env python3
"""
Debug and diagnostics tools for RealSense Visual Odometry system
Comprehensive testing and troubleshooting utilities
"""

import sys
import time
import cv2
import numpy as np
import logging
from datetime import datetime
import json
import traceback

def test_camera_basic():
    """Test basic camera functionality"""
    print("🔍 Testing RealSense D435i Camera...")
    
    try:
        import pyrealsense2 as rs
        
        # Create pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure streams
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        print("  ✅ RealSense SDK imported successfully")
        
        # Start pipeline
        profile = pipeline.start(config)
        print("  ✅ Camera pipeline started")
        
        # Get device info
        device = profile.get_device()
        print(f"  📷 Device: {device.get_info(rs.camera_info.name)}")
        print(f"  🔢 Serial: {device.get_info(rs.camera_info.serial_number)}")
        
        # Test frame capture
        frames_captured = 0
        start_time = time.time()
        
        for i in range(30):  # Capture 30 frames
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if color_frame and depth_frame:
                frames_captured += 1
        
        end_time = time.time()
        fps = frames_captured / (end_time - start_time)
        
        print(f"  ✅ Captured {frames_captured} frames")
        print(f"  📊 Actual FPS: {fps:.1f}")
        
        # Test intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        print(f"  🔧 Color intrinsics:")
        print(f"     fx: {color_intrinsics.fx:.1f}")
        print(f"     fy: {color_intrinsics.fy:.1f}")
        print(f"     cx: {color_intrinsics.ppx:.1f}")
        print(f"     cy: {color_intrinsics.ppy:.1f}")
        
        pipeline.stop()
        print("  ✅ Camera test completed successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Camera test failed: {e}")
        return False

def test_feature_detection():
    """Test feature detection algorithms"""
    print("\n🔍 Testing Feature Detection...")
    
    try:
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add some features
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)
        cv2.circle(test_image, (400, 300), 50, (128, 128, 128), -1)
        cv2.line(test_image, (0, 240), (640, 240), (64, 64, 64), 5)
        
        # Test different detectors
        detectors = {
            'ORB': cv2.ORB_create(nfeatures=1000),
            'AKAZE': cv2.AKAZE_create()
        }
        
        # Try SIFT if available
        try:
            detectors['SIFT'] = cv2.SIFT_create(nfeatures=1000)
        except AttributeError:
            print("  ⚠️  SIFT not available in this OpenCV build")
        
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        
        for name, detector in detectors.items():
            try:
                keypoints, descriptors = detector.detectAndCompute(gray, None)
                print(f"  ✅ {name}: {len(keypoints)} features detected")
                
                if descriptors is not None:
                    print(f"     Descriptor shape: {descriptors.shape}")
                else:
                    print(f"     ❌ No descriptors computed")
                    
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feature detection test failed: {e}")
        return False

def test_pose_estimation():
    """Test pose estimation with synthetic data"""
    print("\n🔍 Testing Pose Estimation...")
    
    try:
        # Create synthetic camera matrix
        camera_matrix = np.array([
            [615.0, 0, 320.0],
            [0, 615.0, 240.0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Create synthetic 3D points
        points_3d = np.array([
            [0, 0, 2],
            [1, 0, 2],
            [0, 1, 2],
            [1, 1, 2],
            [0.5, 0.5, 2]
        ], dtype=np.float32)
        
        # Create synthetic transformation
        rvec_true = np.array([0.1, 0.05, 0.02], dtype=np.float32)
        tvec_true = np.array([0.1, 0.05, 0.1], dtype=np.float32)
        
        # Project points
        points_2d, _ = cv2.projectPoints(points_3d, rvec_true, tvec_true, camera_matrix, None)
        points_2d = points_2d.reshape(-1, 2)
        
        # Add some noise
        noise = np.random.normal(0, 0.5, points_2d.shape)
        points_2d_noisy = points_2d + noise
        
        # Estimate pose
        success, rvec_est, tvec_est = cv2.solvePnP(
            points_3d, points_2d_noisy, camera_matrix, None
        )
        
        if success:
            print("  ✅ PnP estimation successful")
            print(f"     True rotation: {rvec_true.flatten()}")
            print(f"     Estimated rotation: {rvec_est.flatten()}")
            print(f"     True translation: {tvec_true.flatten()}")
            print(f"     Estimated translation: {tvec_est.flatten()}")
            
            # Calculate errors
            rot_error = np.linalg.norm(rvec_true - rvec_est.flatten())
            trans_error = np.linalg.norm(tvec_true - tvec_est.flatten())
            
            print(f"     Rotation error: {rot_error:.4f} rad")
            print(f"     Translation error: {trans_error:.4f} m")
            
            return True
        else:
            print("  ❌ PnP estimation failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Pose estimation test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage patterns"""
    print("\n🔍 Testing Memory Usage...")
    
    try:
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f"  📊 Initial memory: {initial_memory:.1f} MB")
        
        # Create large arrays to simulate processing
        arrays = []
        for i in range(10):
            # Simulate camera frames
            color_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            depth_frame = np.random.randint(0, 65535, (480, 640), dtype=np.uint16)
            arrays.append((color_frame, depth_frame))
            
            current_memory = process.memory_info().rss / 1024 / 1024
            print(f"  📊 After {i+1} frames: {current_memory:.1f} MB")
        
        # Cleanup
        del arrays
        
        import gc
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        print(f"  📊 After cleanup: {final_memory:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Memory test failed: {e}")
        return False

def run_comprehensive_diagnostics():
    """Run comprehensive system diagnostics"""
    print("🧪 RealSense Visual Odometry - Comprehensive Diagnostics")
    print("=" * 60)
    print(f"User: Mr-Parth24")
    print(f"Date: 2025-06-16 22:26:01 UTC")
    print("=" * 60)
    
    # Create diagnostics report
    report = {
        'timestamp': datetime.now().isoformat(),
        'user': 'Mr-Parth24',
        'tests': {}
    }
    
    # Run tests
    tests = [
        ('Camera Basic Test', test_camera_basic),
        ('Feature Detection Test', test_feature_detection), 
        ('Pose Estimation Test', test_pose_estimation),
        ('Memory Usage Test', test_memory_usage)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            result = test_func()
            report['tests'][test_name] = {
                'passed': result,
                'error': None
            }
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            report['tests'][test_name] = {
                'passed': False,
                'error': str(e)
            }
            all_passed = False
    
    # Save report
    try:
        import os
        os.makedirs('logs', exist_ok=True)
        
        report_file = f"logs/diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Diagnostics report saved to: {report_file}")
    except Exception as e:
        print(f"⚠️  Could not save diagnostics report: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTICS SUMMARY")
    print("=" * 60)
    
    for test_name, result in report['tests'].items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{status} {test_name}")
        if result['error']:
            print(f"      Error: {result['error']}")
    
    if all_passed:
        print("\n🎉 All tests passed! System appears to be working correctly.")
        print("\nNext steps:")
        print("1. Run the main application: python main.py")
        print("2. Connect your RealSense D435i camera") 
        print("3. Start tracking and monitor the debug panel")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")
        print("\nTroubleshooting steps:")
        print("1. Ensure RealSense SDK is properly installed")
        print("2. Check camera connection (USB 3.0+ required)")
        print("3. Install missing Python packages")
        print("4. Check logs/errors.log for detailed error information")
    
    return all_passed

def create_test_data():
    """Create test data for development and debugging"""
    print("\n🔧 Creating Test Data...")
    
    try:
        import os
        os.makedirs('data/test', exist_ok=True)
        
        # Create synthetic trajectory
        trajectory = []
        for i in range(100):
            t = i * 0.1
            x = np.sin(t * 0.5) * 2
            y = t * 0.1
            z = np.cos(t * 0.5) * 0.5
            
            pose = np.eye(4)
            pose[:3, 3] = [x, y, z]
            
            trajectory.append({
                'timestamp': time.time() + i * 0.1,
                'position': [x, y, z],
                'pose': pose.tolist()
            })
        
        # Save test trajectory
        with open('data/test/synthetic_trajectory.json', 'w') as f:
            json.dump({
                'metadata': {
                    'description': 'Synthetic trajectory for testing',
                    'points': len(trajectory),
                    'pattern': 'sine wave with forward motion'
                },
                'trajectory': trajectory
            }, f, indent=2)
        
        print("  ✅ Synthetic trajectory created")
        
        # Create test images
        for i in range(5):
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(f'data/test/test_image_{i:03d}.png', test_image)
        
        print("  ✅ Test images created")
        print(f"  📁 Test data saved to: data/test/")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test data creation failed: {e}")
        return False

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'camera':
            test_camera_basic()
        elif sys.argv[1] == 'features':
            test_feature_detection()
        elif sys.argv[1] == 'pose':
            test_pose_estimation()
        elif sys.argv[1] == 'memory':
            test_memory_usage()
        elif sys.argv[1] == 'testdata':
            create_test_data()
        else:
            print("Usage: python debug_tools.py [camera|features|pose|memory|testdata]")
    else:
        run_comprehensive_diagnostics()