#!/usr/bin/env python3
"""
Debug Camera Test - Isolate camera initialization issues
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import time

def test_basic_camera():
    """Test basic camera functionality"""
    print("🔍 Testing basic RealSense camera functionality...")
    
    try:
        # Test 1: Context and devices
        print("Step 1: Testing context and device enumeration...")
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("❌ No RealSense devices found")
            return False
        
        print(f"✅ Found {len(devices)} device(s)")
        
        for i, device in enumerate(devices):
            print(f"   Device {i}: {device.get_info(rs.camera_info.name)}")
            print(f"   Serial: {device.get_info(rs.camera_info.serial_number)}")
            print(f"   Firmware: {device.get_info(rs.camera_info.firmware_version)}")
        
        # Test 2: Basic pipeline configuration
        print("Step 2: Testing pipeline configuration...")
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Use simple, guaranteed compatible settings
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        print("✅ Pipeline configured")
        
        # Test 3: Start pipeline
        print("Step 3: Testing pipeline start...")
        profile = pipeline.start(config)
        print("✅ Pipeline started successfully")
        
        # Test 4: Try to get a few frames
        print("Step 4: Testing frame acquisition...")
        frames_received = 0
        max_attempts = 10
        
        for attempt in range(max_attempts):
            try:
                print(f"   Attempt {attempt + 1}/{max_attempts}...")
                frames = pipeline.wait_for_frames(timeout_ms=2000)  # 2 second timeout
                
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    frames_received += 1
                    print(f"   ✅ Frame {frames_received}: Color {color_frame.get_width()}x{color_frame.get_height()}, Depth {depth_frame.get_width()}x{depth_frame.get_height()}")
                    
                    if frames_received >= 3:
                        break
                else:
                    print(f"   ⚠️  Incomplete frame set")
                    
            except Exception as e:
                print(f"   ❌ Frame acquisition failed: {e}")
                break
        
        # Test 5: Stop pipeline
        print("Step 5: Stopping pipeline...")
        pipeline.stop()
        print("✅ Pipeline stopped successfully")
        
        if frames_received >= 3:
            print(f"🎉 Camera test SUCCESSFUL: {frames_received} frames received")
            return True
        else:
            print(f"❌ Camera test FAILED: Only {frames_received} frames received")
            return False
            
    except Exception as e:
        print(f"❌ Camera test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sensor_configuration():
    """Test sensor configuration options"""
    print("\n🔧 Testing sensor configuration...")
    
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        profile = pipeline.start(config)
        device = profile.get_device()
        
        # Test color sensor options
        print("Testing color sensor options...")
        if device.first_color_sensor():
            color_sensor = device.first_color_sensor()
            
            # Test exposure settings
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 1)
                print("✅ Auto exposure enabled")
            
            if color_sensor.supports(rs.option.exposure):
                current_exposure = color_sensor.get_option(rs.option.exposure)
                print(f"✅ Current exposure: {current_exposure}")
            
            if color_sensor.supports(rs.option.gain):
                current_gain = color_sensor.get_option(rs.option.gain)
                print(f"✅ Current gain: {current_gain}")
        
        # Test depth sensor options
        print("Testing depth sensor options...")
        if device.first_depth_sensor():
            depth_sensor = device.first_depth_sensor()
            
            if depth_sensor.supports(rs.option.laser_power):
                current_laser = depth_sensor.get_option(rs.option.laser_power)
                print(f"✅ Current laser power: {current_laser}")
                
                # Try setting laser power
                depth_sensor.set_option(rs.option.laser_power, 240)
                new_laser = depth_sensor.get_option(rs.option.laser_power)
                print(f"✅ Laser power set to: {new_laser}")
            
            # Check for problematic preset options
            print("Checking for visual preset support...")
            try:
                if hasattr(rs.option, 'visual_preset'):
                    if depth_sensor.supports(rs.option.visual_preset):
                        current_preset = depth_sensor.get_option(rs.option.visual_preset)
                        print(f"✅ Current visual preset: {current_preset}")
                    else:
                        print("ℹ️  Visual preset not supported")
                else:
                    print("ℹ️  Visual preset option not available in this SDK version")
            except Exception as preset_error:
                print(f"⚠️  Visual preset test failed: {preset_error}")
        
        pipeline.stop()
        print("✅ Sensor configuration test completed")
        
    except Exception as e:
        print(f"❌ Sensor configuration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🌾 Agricultural SLAM - Camera Debug Test")
    print("=" * 50)
    
    # Run basic camera test
    camera_works = test_basic_camera()
    
    if camera_works:
        test_sensor_configuration()
    else:
        print("❌ Basic camera test failed, skipping advanced tests")
    
    print("\n" + "=" * 50)
    if camera_works:
        print("🎉 Camera appears to be working - the issue might be in the application logic")
    else:
        print("❌ Camera has fundamental issues that need to be resolved")
