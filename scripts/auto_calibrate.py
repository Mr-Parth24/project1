#!/usr/bin/env python3
"""
Automatic Calibration Script for 10x10 Checkerboard
Intelligent auto-capture with hands-free operation

User: Mr-Parth24
Date: 2025-06-20 23:15:48 UTC
Version: 2.1 - Complete Auto-Calibration
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.automatic_calibration import start_automatic_calibration
import argparse

def main():
    """Main automatic calibration process"""
    parser = argparse.ArgumentParser(
        description="Mr-Parth24 Automatic Agricultural Camera Calibration for 10x10 Checkerboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🤖 AUTOMATIC CALIBRATION FEATURES:
  - Intelligent auto-capture based on quality
  - Real-time feedback and guidance
  - Coverage zone optimization
  - Audio feedback for captures
  - Automatic saving and configuration update

📋 REQUIREMENTS:
  - Intel RealSense D435i camera connected
  - 10x10 checkerboard (40mm squares, 18"x18")
  - Good lighting conditions
  - Ability to move camera to different positions

🎯 PROCESS:
  1. System detects checkerboard automatically
  2. Assesses quality, distance, and angle
  3. Auto-captures when conditions are optimal
  4. Provides real-time feedback and guidance
  5. Automatically saves and updates configuration

Mr-Parth24 © 2025-06-20 23:15:48 UTC - Agricultural SLAM v2.1
        """
    )
    
    parser.add_argument(
        '--target-frames',
        type=int,
        default=15,
        help='Target number of frames to capture (default: 15)'
    )
    
    args = parser.parse_args()
    
    print("🤖 Mr-Parth24 Automatic Agricultural Camera Calibration")
    print("=" * 70)
    print(f"User: Mr-Parth24")
    print(f"Date: 2025-06-20 23:15:48 UTC")
    print(f"Board: 10x10 Checkerboard (40mm squares)")
    print(f"Target frames: {args.target_frames}")
    print()
    
    try:
        print("🚀 Starting automatic calibration...")
        print("   Hold camera steady and move to different positions")
        print("   System will auto-capture optimal frames")
        print("   Listen for audio feedback on captures")
        print()
        
        # Run automatic calibration
        result = start_automatic_calibration()
        
        if result.get('success', False):
            print("\n🎉 MR-PARTH24 AUTOMATIC CALIBRATION COMPLETED SUCCESSFULLY!")
            print(f"   User: Mr-Parth24")
            print(f"   Completion time: 2025-06-20 23:15:48 UTC")
            print(f"   Frames captured: {result.get('frames_used', 0)}")
            print(f"   Reprojection error: {result.get('reprojection_error', 0):.3f} pixels")
            print(f"   Coverage achieved: {result.get('coverage_achieved', 0):.2f}")
            print()
            print("✅ System configuration updated automatically")
            print("✅ Calibration data saved")
            print("✅ Ready to use with Agricultural SLAM System")
            print()
            print("🚀 Next step: Run 'python main.py' to start SLAM system")
            
        else:
            print("\n❌ AUTOMATIC CALIBRATION FAILED")
            error = result.get('error', 'Unknown error')
            print(f"   Error: {error}")
            print()
            print("💡 Troubleshooting:")
            print("   - Ensure camera is connected")
            print("   - Check checkerboard visibility")
            print("   - Improve lighting conditions")
            print("   - Try manual calibration: python scripts/calibrate_camera.py")
            
            return 1
        
    except KeyboardInterrupt:
        print("\n🛑 Automatic calibration interrupted")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())