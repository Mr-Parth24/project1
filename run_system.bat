@echo off
REM Visual Odometry System Launcher for Windows
REM Author: Mr-Parth24
REM Date: 2025-06-13

echo Starting Visual Odometry System...

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create necessary directories
if not exist "data" mkdir data
if not exist "logs" mkdir logs

REM Run the system
echo Launching application...
python main_application.py

echo System shutdown complete.
pause