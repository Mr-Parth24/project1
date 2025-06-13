#!/bin/bash

# Visual Odometry System Launcher
# Author: Mr-Parth24
# Date: 2025-06-13

echo "Starting Visual Odometry System..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p data logs

# Run the system
echo "Launching application..."
python main_application.py

echo "System shutdown complete."