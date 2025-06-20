# Agricultural SLAM System

A Visual SLAM (Simultaneous Localization and Mapping) system using Intel RealSense D435i camera for agricultural equipment tracking. This system provides real-time camera pose estimation and trajectory tracking without requiring IMU or GPS sensors.

## Features

- **Real-time Visual Odometry**: Track camera movement using only visual features
- **2D Trajectory Visualization**: Live path display with distance measurement
- **Feature Detection**: ORB-based feature detection with overlay visualization
- **Agricultural Optimization**: Specifically tuned for field environments with limited visual features
- **Professional GUI**: Multi-panel interface with real-time monitoring
- **Data Export**: Save trajectories and performance metrics
- **No External Dependencies**: Works without IMU, GPS, or external markers

## System Requirements

### Hardware
- Intel RealSense D435i camera
- Windows 10/11 or Ubuntu 18.04+
- CPU: Intel i5+ or equivalent (multi-core recommended)
- RAM: 8GB minimum, 16GB recommended
- USB 3.0 port for camera connection

### Software
- Python 3.8+
- Intel RealSense SDK 2.0
- OpenCV 4.8+
- PyQt6 6.5+

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-org/agricultural-slam.git
cd agricultural-slam
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Intel RealSense SDK
Follow the [Intel RealSense installation guide](https://github.com/IntelRealSense/librealsense) for your platform.

### 4. Verify Installation
```bash
python -c "import pyrealsense2 as rs; print('RealSense SDK installed successfully')"
```

## Quick Start

### 1. Connect Camera
Connect your Intel RealSense D435i to a USB 3.0 port.

### 2. Run Application
```bash
python main.py
```

### 3. Start SLAM
1. Click **"Start Camera"** to begin video stream
2. Check **"Enable SLAM"** to start tracking
3. Check **"Show Features"** to see detected features
4. Move camera to build trajectory

## Usage Guide

### GUI Overview

The interface consists of three main panels:

#### Left Panel - Live Camera Feed
- Real-time color/depth video stream
- Feature detection overlay (green circles)
- SLAM status indicators
- Control buttons for camera and SLAM

#### Center Panel - Trajectory Visualization
- 2D trajectory plot showing camera path
- Real-time distance and position display
- Auto-scaling and grid options
- Reset and save functionality

#### Right Panel - System Information
- Performance metrics and FPS
- SLAM statistics (features, keyframes, map points)
- System logs and error messages
- Configuration details

### Basic Operations

#### Starting SLAM
1. **Start Camera**: Initialize RealSense camera
2. **Enable SLAM**: Begin visual odometry processing
3. **Show Features**: Display detected ORB features
4. **Move Camera**: Walk/drive to build trajectory

#### Viewing Results
- **Green Line**: Your movement path in 2D
- **Distance Counter**: Total distance traveled
- **Position Display**: Current X, Z coordinates
- **Status Indicators**: Tracking quality and performance

#### Saving Data
- **Save Trajectory**: Export path data to NPZ/CSV format
- **Auto-save**: Optional automatic data logging
- **Performance Logs**: System metrics and statistics

### Advanced Configuration

#### Camera Settings (`config/camera_config.yaml`)
```yaml
camera:
  color_width: 640
  color_height: 480
  fps: 30
  auto_exposure: true
```

#### SLAM Parameters (`config/slam_config.yaml`)
```yaml
slam:
  max_features: 1000
  min_features_for_tracking: 30
  keyframe_distance_threshold: 0.3
  max_translation_per_frame: 10.0
```

## Performance Optimization

### For Best Results
- **Good Lighting**: Ensure adequate illumination
- **Textured Surfaces**: Point camera at objects with visual features
- **Smooth Movement**: Avoid rapid camera motions
- **Stable Mounting**: Minimize vibration when vehicle-mounted

### Troubleshooting

#### Common Issues

**"Tracking Lost" Message**
- Move camera more slowly
- Point at textured surfaces (not blank walls)
- Improve lighting conditions
- Check camera connection

**Poor Distance Accuracy**
- Ensure proper camera calibration
- Avoid scenes with insufficient depth variation
- Check for camera obstruction or dirt

**Low Frame Rate**
- Reduce max_features in SLAM config
- Close other applications
- Use faster computer hardware

#### Debug Mode
Enable detailed logging by setting environment variable:
```bash
export SLAM_DEBUG=1
python main.py
```

## Agricultural Applications

### Equipment Tracking
- **Combine Harvesters**: Track field coverage patterns
- **Tractors**: Monitor tillage and planting routes
- **Sprayers**: Verify application coverage
- **General Farm Equipment**: GPS-independent backup tracking

### Field Mapping
- **Coverage Analysis**: Ensure complete field coverage
- **Overlap Monitoring**: Minimize redundant passes
- **Boundary Detection**: Track field edge navigation
- **Work Pattern Analysis**: Optimize operational efficiency

### Benefits for Agriculture
- **GPS-Independent**: Works when GPS is jammed or unavailable
- **High Accuracy**: Centimeter-level precision for short distances
- **Real-time Feedback**: Immediate tracking results
- **Cost-effective**: Uses standard camera hardware

## File Structure

```
agricultural-slam/
├── src/
│   ├── core/                    # Core SLAM algorithms
│   │   ├── camera_manager.py    # RealSense interface
│   │   ├── feature_detector.py  # ORB feature detection
│   │   ├── visual_odometry.py   # Pose estimation
│   │   └── slam_engine.py       # Main SLAM processing
│   ├── gui/                     # User interface
│   │   ├── main_window.py       # Main application window
│   │   ├── camera_widget.py     # Camera display widget
│   │   └── trajectory_widget.py # Trajectory visualization
│   ├── algorithms/              # SLAM implementations
│   │   └── custom_visual_slam.py # Custom SLAM system
│   └── utils/                   # Utilities
│       ├── config_manager.py    # Configuration handling
│       ├── data_logger.py       # Data save/load
│       └── coordinate_transform.py # Coordinate systems
├── config/                      # Configuration files
│   ├── camera_config.yaml       # Camera parameters
│   └── slam_config.yaml         # SLAM settings
├── data/                        # Data storage
│   ├── trajectories/           # Saved trajectories
│   ├── maps/                   # SLAM maps
│   └── logs/                   # Performance logs
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Development

### Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull Request

### Testing
```bash
# Run individual component tests
python src/core/feature_detector.py
python src/core/visual_odometry.py
python src/algorithms/custom_visual_slam.py

# Test with sample data
python tests/test_slam_pipeline.py
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings for all functions
- Include error handling

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Intel RealSense team for SDK and documentation
- OpenCV community for computer vision algorithms
- PyQt team for GUI framework
- Agricultural technology research community

## Support

For issues and questions:
- Create GitHub issue for bugs
- Check documentation for usage questions
- Contact development team for custom applications

## Roadmap

### Phase 3 (Planned)
- 3D map visualization
- Loop closure detection
- Map persistence and loading
- Multi-session mapping

### Future Enhancements
- Machine learning-based feature detection
- Multi-camera support
- Cloud data synchronization
- Advanced agricultural analytics

---

**Agricultural SLAM System v2.0** - Bringing precision tracking to modern agriculture.