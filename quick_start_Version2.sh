#!/bin/bash

# Quick Start Script for RealSense D435i Visual Odometry
# Enhanced version with comprehensive setup and troubleshooting

echo "🚀 RealSense D435i Visual Odometry - Quick Start"
echo "================================================"
echo "User: Mr-Parth24"
echo "Date: 2025-06-16 22:26:01 UTC"
echo "================================================"

# Function to check command success
check_success() {
    if [ $? -eq 0 ]; then
        echo "✅ $1"
    else
        echo "❌ $1 failed"
        return 1
    fi
}

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    check_success "Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
check_success "Virtual environment activated"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
check_success "Pip upgraded"

# Install core dependencies
echo "📦 Installing core dependencies..."
pip install pyrealsense2 opencv-python numpy scipy PyQt5 pyqtgraph matplotlib > /dev/null 2>&1
check_success "Core dependencies installed"

# Install additional packages
echo "📦 Installing additional packages..."
pip install pandas scikit-image tqdm Pillow psutil > /dev/null 2>&1
check_success "Additional packages installed"

# Create directories
echo "📁 Creating directories..."
mkdir -p logs data/exports data/calibration config
check_success "Directories created"

# Run diagnostics
echo ""
echo "🧪 Running system diagnostics..."
python debug_tools.py

# Check if diagnostics passed
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 System check completed successfully!"
    echo ""
    echo "🚀 Starting RealSense Visual Odometry..."
    echo "   • Make sure your D435i camera is connected"
    echo "   • Check the Debug Information panel for real-time logs"
    echo "   • Try different movement patterns to test tracking"
    echo ""
    
    # Start the application
    python main.py
else
    echo ""
    echo "⚠️  System diagnostics found issues."
    echo "   Please check the diagnostic output above and fix any problems."
    echo ""
    echo "💡 Troubleshooting tips:"
    echo "   1. Ensure RealSense SDK is installed: sudo apt install librealsense2-dev"
    echo "   2. Check camera connection: lsusb | grep Intel"
    echo "   3. Try different USB port (USB 3.0+ required)"
    echo "   4. Check logs/errors.log for detailed error information"
    echo ""
    echo "🔧 You can still try running the application:"
    echo "   python main.py"
fi

echo ""
echo "📚 For more help:"
echo "   • Check TROUBLESHOOTING.md"
echo "   • Run: python debug_tools.py camera  (to test camera only)"
echo "   • Run: python test_installation.py  (to verify installation)"