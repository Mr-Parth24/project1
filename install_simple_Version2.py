#!/usr/bin/env python3
"""
Simplified installation script that avoids problematic dependencies
"""

import os
import sys
import subprocess
import platform

def run_command(cmd, check=True):
    """Run system command"""
    print(f"Running: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return False

def install_dependencies():
    """Install core dependencies without problematic packages"""
    
    print("📦 Installing core dependencies...")
    
    # Core packages that should work everywhere
    core_packages = [
        "pyrealsense2>=2.54.1",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "PyQt5>=5.15.0",
        "pyqtgraph>=0.13.0",
        "matplotlib>=3.7.0",
        "scikit-image>=0.20.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "Pillow>=9.5.0"
    ]
    
    for package in core_packages:
        success = run_command(f"pip install {package}", check=False)
        if not success:
            print(f"⚠️  Warning: Failed to install {package}")
    
    # Try to install optional packages
    optional_packages = [
        "open3d>=0.17.0",
        "pyopengl>=3.1.0",
        "h5py>=3.8.0"
    ]
    
    for package in optional_packages:
        print(f"🔧 Trying to install optional package: {package}")
        success = run_command(f"pip install {package}", check=False)
        if not success:
            print(f"⚠️  Optional package {package} could not be installed (this is OK)")

def main():
    """Main installation process"""
    print("🚀 RealSense Visual Odometry - Simplified Installation")
    print("=" * 55)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Create directories
    print("📁 Creating directories...")
    for directory in ['logs', 'data/exports', 'data/calibration', 'config']:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    # Install dependencies
    install_dependencies()
    
    print("\n🎉 Installation complete!")
    print("\nTo run the application:")
    print("   python main.py")
    print("\nNote: This installation uses a simplified SLAM system")
    print("      that doesn't require g2o (which can be problematic to install)")
    
    return True

if __name__ == "__main__":
    main()