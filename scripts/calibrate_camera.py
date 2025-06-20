#!/usr/bin/env python3
"""
Standalone camera calibration script for Agricultural SLAM System
Can be run independently for D435i calibration in agricultural environments
"""

import sys
import os
from src.utils.automatic_calibration import start_automatic_calibration

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.calibration_helper import quick_agricultural_calibration
import argparse
import time

def main():
    """Main calibration process"""
    parser = argparse.ArgumentParser(
        description="Agricultural Camera Calibration for Intel RealSense D435i",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/calibrate_camera.py                    # Interactive calibration
  python scripts/calibrate_camera.py --auto-save       # Auto-save calibration
  python scripts/calibrate_camera.py --frames 20       # Capture 20 frames
  
Requirements:
  - Intel RealSense D435i camera connected
  - 9x6 checkerboard with 25mm squares
  - Good lighting conditions
  - Stable camera mounting
        """
    )

    parser.add_argument(
    '--automatic',
    action='store_true',
    help='Use automatic intelligent calibration (recommended for 10x10 board)'
)
    
    parser.add_argument(
        '--frames',
        type=int,
        default=15,
        help='Number of calibration frames to capture (default: 15)'
    )
    
    parser.add_argument(
        '--auto-save',
        action='store_true',
        help='Automatically save calibration when complete'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/calibration',
        help='Output directory for calibration files'
    )
    
    args = parser.parse_args()
    
    print("🌾 Agricultural SLAM System - Camera Calibration")
    print("=" * 60)
    print(f"Target frames: {args.frames}")
    print(f"Auto-save: {args.auto_save}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run calibration
    try:
        print("🎯 Starting agricultural camera calibration...")
        print()
        print("📋 Calibration Instructions:")
        print("1. Connect Intel RealSense D435i camera")
        print("2. Use 9x6 checkerboard with 25mm squares")
        print("3. Ensure good lighting (outdoor preferred)")
        print("4. Hold checkerboard steady during capture")
        print("5. Cover different areas of camera field of view")
        print("6. Maintain 0.5-2.0m distance from camera")
        print()
        print("🎮 Controls:")
        print("   'c' - Capture calibration frame")
        print("   'q' - Quit calibration")
        print("   'd' - Done (finish calibration)")
        print()
        
        input("Press Enter to start calibration...")
        
        # Perform calibration
        calibration_result = quick_agricultural_calibration()
        
        if calibration_result and calibration_result.get('success'):
            print("\n✅ Calibration completed successfully!")
            print(f"   Reprojection error: {calibration_result['reprojection_error']:.3f} pixels")
            print(f"   Frames used: {calibration_result['frames_used']}")
            print(f"   Agricultural suitability: {calibration_result['calibration_quality']['outdoor_suitability']:.2f}")
            
            # Save calibration
            if args.auto_save:
                from src.utils.calibration_helper import get_calibration_helper
                helper = get_calibration_helper()
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(args.output_dir, f"d435i_agricultural_calibration_{timestamp}.json")
                
                saved_file = helper.save_calibration(calibration_result, output_file)
                
                if saved_file:
                    print(f"📁 Calibration saved: {saved_file}")
                    
                    # Also save to config directory for system use
                    config_file = "config/current_calibration.json"
                    helper.save_calibration(calibration_result, config_file)
                    print(f"📁 Active calibration updated: {config_file}")
                else:
                    print("❌ Failed to save calibration")
            else:
                save_choice = input("\n💾 Save calibration? (y/n): ")
                if save_choice.lower() in ['y', 'yes']:
                    from src.utils.calibration_helper import get_calibration_helper
                    helper = get_calibration_helper()
                    
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_file = os.path.join(args.output_dir, f"d435i_agricultural_calibration_{timestamp}.json")
                    
                    saved_file = helper.save_calibration(calibration_result, output_file)
                    if saved_file:
                        print(f"✅ Calibration saved: {saved_file}")
                    else:
                        print("❌ Failed to save calibration")
            
            print("\n🎯 Calibration Quality Assessment:")
            quality = calibration_result['calibration_quality']
            print(f"   Reprojection Error: {quality['reprojection_error']:.3f} pixels")
            print(f"   Coverage Score: {quality['coverage_score']:.2f}/1.0")
            print(f"   Stability Score: {quality['stability_score']:.2f}/1.0")
            print(f"   Agricultural Suitability: {quality['outdoor_suitability']:.2f}/1.0")
            
            if quality['outdoor_suitability'] >= 0.8:
                print("✅ Excellent for agricultural use!")
            elif quality['outdoor_suitability'] >= 0.6:
                print("✅ Good for agricultural use")
            else:
                print("⚠️  May need recalibration for optimal agricultural performance")
                
        else:
            print("\n❌ Calibration failed or was cancelled")
            print("   Please ensure:")
            print("   - Camera is properly connected")
            print("   - Checkerboard is clearly visible")
            print("   - Lighting conditions are adequate")
            print("   - At least 10 good calibration frames are captured")
            
    except KeyboardInterrupt:
        print("\n🛑 Calibration interrupted by user")
    except Exception as e:
        print(f"\n❌ Calibration error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())