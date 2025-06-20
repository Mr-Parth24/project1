#!/usr/bin/env python3
"""
Dependency installation helper for Agricultural SLAM System
Checks system requirements and installs dependencies with appropriate error handling
"""

import subprocess
import sys
import platform
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    
    print(f"✅ Python version: {sys.version}")
    return True

def install_package(package, optional=False):
    """Install a single package with error handling"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        if optional:
            print(f"⚠️  Optional package {package} failed to install: {e}")
            return False
        else:
            print(f"❌ Failed to install {package}: {e}")
            return False

def install_system_dependencies():
    """Install system-specific dependencies"""
    system = platform.system().lower()
    
    if system == "linux":
        print("🐧 Linux system detected")
        try:
            # Check for apt-get (Ubuntu/Debian)
            subprocess.run(["which", "apt-get"], check=True, capture_output=True)
            print("Installing system dependencies...")
            
            # Intel RealSense dependencies
            subprocess.run([
                "sudo", "apt-get", "update"
            ], check=True)
            
            subprocess.run([
                "sudo", "apt-get", "install", "-y",
                "librealsense2-dev", "librealsense2-utils",
                "python3-opencv", "python3-pyqt6"
            ], check=True)
            
            print("✅ System dependencies installed")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Could not install system dependencies automatically")
            print("   Please install Intel RealSense SDK manually:")
            print("   https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md")
            return False
    
    elif system == "windows":
        print("🪟 Windows system detected")
        print("💡 Please install Intel RealSense SDK manually:")
        print("   https://github.com/IntelRealSense/librealsense/releases")
        return True
    
    elif system == "darwin":
        print("🍎 macOS system detected")
        try:
            # Check for Homebrew
            subprocess.run(["which", "brew"], check=True, capture_output=True)
            print("Installing dependencies via Homebrew...")
            
            subprocess.run([
                "brew", "install", "librealsense"
            ], check=True)
            
            print("✅ System dependencies installed")
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️  Homebrew not found. Please install manually:")
            print("   https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_osx.md")
            return False
    
    return True

def main():
    """Main installation process"""
    print("🌾 Agricultural SLAM System - Dependency Installer")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install system dependencies
    print("\n📦 Installing system dependencies...")
    install_system_dependencies()
    
    # Core Python packages
    print("\n🐍 Installing core Python packages...")
    core_packages = [
        "numpy>=1.21.0",
        "opencv-python>=4.8.0",
        "PyQt6>=6.5.0", 
        "pyrealsense2>=2.54.0",
        "PyYAML>=6.0",
        "psutil>=5.9.0"
    ]
    
    core_success = True
    for package in core_packages:
        if not install_package(package):
            core_success = False
    
    # Optional packages
    print("\n🔧 Installing optional packages...")
    optional_packages = [
        ("PyOpenGL>=3.1.6", "3D visualization"),
        ("PyOpenGL-accelerate>=3.1.6", "3D acceleration"),
        ("GPUtil>=1.4.0", "GPU monitoring"),
        ("scipy>=1.9.0", "Advanced filtering"),
        ("scikit-learn>=1.1.0", "Outlier detection"),
        ("matplotlib>=3.6.0", "Plotting"),
        ("Pillow>=9.0.0", "Image processing")
    ]
    
    for package, description in optional_packages:
        print(f"Installing {package} ({description})...")
        install_package(package, optional=True)
    
    # Installation summary
    print("\n" + "=" * 60)
    if core_success:
        print("✅ Installation completed successfully!")
        print("\n🚀 You can now run the Agricultural SLAM System:")
        print("   python main.py")
        print("\n💡 For first-time setup, consider running calibration:")
        print("   python -c \"from src.utils.calibration_helper import quick_agricultural_calibration; quick_agricultural_calibration()\"")
    else:
        print("❌ Some core packages failed to install")
        print("   Please check error messages above and install manually")
        sys.exit(1)
    
    print("\n📚 Documentation:")
    print("   README.md - Complete system documentation")
    print("   config/ - Configuration files")
    print("   examples/ - Usage examples")

if __name__ == "__main__":
    main()