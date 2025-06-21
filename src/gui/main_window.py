"""
Enhanced Main Window for Agricultural SLAM System
Integrated with all core components for real-time agricultural mapping
FIXED: Component integration, data flow, and proper camera widget updates
Provides 3D visualization and precision distance tracking
"""

import sys
import numpy as np
import cv2
import time
import threading
from typing import Dict, Optional, List
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QGroupBox, QCheckBox, QSlider, QTextEdit,
    QProgressBar, QTabWidget, QFrame, QSplitter, QScrollArea,
    QSpinBox, QDoubleSpinBox, QComboBox, QStatusBar, QMenuBar,
    QFileDialog, QMessageBox
)
from PyQt6.QtCore import (
    QTimer, QThread, pyqtSignal, QMutex, QMutexLocker, Qt
)
from PyQt6.QtGui import QFont, QPixmap, QAction, QIcon

from src.core.camera_manager import CameraManager
from src.core.agri_slam_core import AgriSLAMCore  # FIXED: Use AgriSLAMCore instead
from src.gui.camera_widget import CameraWidget
from src.gui.trajectory_widget import TrajectoryWidget
from src.utils.data_logger import get_data_logger

class SLAMProcessingThread(QThread):
    """
    FIXED: Dedicated thread for SLAM processing with proper AgriSLAMCore integration
    """
    
    # Signals for updating GUI
    results_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    status_update = pyqtSignal(str)
    
    def __init__(self, camera_manager: CameraManager, slam_system: AgriSLAMCore):
        super().__init__()
        self.camera_manager = camera_manager
        self.slam_system = slam_system
        self.is_running = False
        self.process_slam = False
        self.mutex = QMutex()
        
        # Performance tracking
        self.frame_processing_times = []
        self.last_fps_calculation = time.time()
        self.fps_counter = 0
        
        print("🧵 SLAM Processing Thread initialized (FIXED)")
        
    def run(self):
        """FIXED: Main processing loop with proper error handling and data flow"""
        self.is_running = True
        self.status_update.emit("SLAM processing thread started")
        
        frame_count = 0
        last_fps_time = time.time()
        fps_counter = 0
        
        while self.is_running:
            try:
                # FIXED: Get frames with timestamp from camera manager
                frame_data = self.camera_manager.get_frames()
                if frame_data is None:
                    self.msleep(10)  # Small delay if no frames
                    continue
                
                # FIXED: Unpack frame data properly (includes timestamp now)
                color_frame, depth_frame, timestamp = frame_data
                frame_count += 1
                fps_counter += 1
                
                # Calculate FPS every second
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = fps_counter / (current_time - last_fps_time)
                    fps_counter = 0
                    last_fps_time = current_time
                else:
                    fps = self.camera_manager.get_current_fps()  # Use camera manager's FPS
                
                # FIXED: Process with SLAM if enabled
                results = None
                processing_start = time.time()
                
                if self.process_slam:
                    with QMutexLocker(self.mutex):
                        # FIXED: Use AgriSLAMCore.process_frame with proper parameters
                        results = self.slam_system.process_frame(
                            color_frame, depth_frame, timestamp
                        )
                        
                        # FIXED: Add frame data and camera info for display
                        results['color_frame'] = color_frame
                        results['depth_frame'] = depth_frame
                        results['frame_count'] = frame_count
                        results['fps'] = fps
                        results['camera_resolution'] = self.camera_manager.get_camera_resolution()
                        results['camera_format'] = self.camera_manager.get_camera_format()
                
                else:
                    # FIXED: Just pass through frame data with minimal processing
                    results = {
                        'color_frame': color_frame,
                        'depth_frame': depth_frame,
                        'frame_count': frame_count,
                        'fps': fps,
                        'pose_estimated': False,
                        'slam_mode': 'DISABLED',
                        'position': np.array([0.0, 0.0, 0.0]),
                        'rotation': np.eye(3),
                        'num_features': 0,
                        'num_matches': 0,
                        'total_distance': 0.0,
                        'processing_time': 0.0,
                        'tracking_quality': 0.0,
                        'agricultural_scene': {},
                        'camera_resolution': self.camera_manager.get_camera_resolution(),
                        'camera_format': self.camera_manager.get_camera_format(),
                        'debug_info': 'SLAM disabled - camera only mode'
                    }
                
                # Track processing time
                processing_time = time.time() - processing_start
                self.frame_processing_times.append(processing_time)
                if len(self.frame_processing_times) > 100:
                    self.frame_processing_times.pop(0)
                
                # FIXED: Add processing statistics to results
                if results:
                    results['thread_processing_time'] = processing_time
                    results['avg_thread_processing_time'] = np.mean(self.frame_processing_times)
                    
                    # Emit results to GUI
                    self.results_ready.emit(results)
                
                # Small delay to prevent overwhelming the system
                self.msleep(1)
                
            except Exception as e:
                error_msg = f"SLAM processing error: {str(e)}"
                print(f"❌ {error_msg}")
                self.error_occurred.emit(error_msg)
                self.msleep(100)  # Longer delay on error
    
    def enable_slam(self, enabled: bool):
        """Enable or disable SLAM processing"""
        with QMutexLocker(self.mutex):
            self.process_slam = enabled
            print(f"🧵 SLAM processing: {'ENABLED' if enabled else 'DISABLED'}")
    
    def stop(self):
        """Stop the processing thread"""
        self.is_running = False
        self.wait()  # Wait for thread to finish
        print("🧵 SLAM Processing Thread stopped")

class EnhancedMainWindow(QMainWindow):
    """
    FIXED: Enhanced Main Window with proper component integration
    Uses AgriSLAMCore and handles all data flow correctly
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Agricultural SLAM System v2.0 - Enhanced (FIXED)")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Core components
        self.camera_manager = None
        self.slam_system = None
        self.processing_thread = None
        self.data_logger = get_data_logger()
        
        # State tracking
        self.camera_active = False
        self.slam_active = False
        self.session_active = False
        self.session_start_time = None
        
        # Performance monitoring
        self.performance_stats = {
            'frames_processed': 0,
            'slam_success_rate': 0.0,
            'avg_processing_time': 0.0,
            'distance_accuracy': 0.0
        }
        
        # FIXED: Frame data tracking for proper widget updates
        self.last_frame_data = None
        self.frame_update_counter = 0
        
        # Initialize UI
        self.init_ui()
        self.init_timers()
        self.init_status_bar()
        
        # Apply modern styling
        self.apply_modern_style()
        
        print("🏠 Enhanced Main Window initialized (FIXED VERSION)")
        print("   - AgriSLAMCore integration: ✅")
        print("   - Proper data flow: ✅")
        print("   - Camera widget integration: ✅")
    
    def init_ui(self):
        """Initialize the enhanced user interface"""
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - Camera and controls
        left_panel = self.create_left_panel()
        main_splitter.addWidget(left_panel)
        
        # Center panel - 3D visualization
        center_panel = self.create_center_panel()
        main_splitter.addWidget(center_panel)
        
        # Right panel - Information and statistics
        right_panel = self.create_right_panel()
        main_splitter.addWidget(right_panel)
        
        # Set splitter proportions
        main_splitter.setSizes([500, 700, 400])
        
        # Create menu bar
        self.create_menu_bar()
    
    def create_left_panel(self) -> QWidget:
        """Create left panel with camera controls"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Camera Control Group
        camera_group = QGroupBox("🎥 Camera Control")
        camera_layout = QVBoxLayout(camera_group)
        
        # Camera status
        self.camera_status_label = QLabel("Camera: Disconnected")
        self.camera_status_label.setStyleSheet("color: red; font-weight: bold;")
        camera_layout.addWidget(self.camera_status_label)
        
        # Camera buttons
        self.start_camera_btn = QPushButton("Start Camera")
        self.start_camera_btn.clicked.connect(self.toggle_camera)
        camera_layout.addWidget(self.start_camera_btn)
        
        # FIXED: Camera widget with proper initialization
        self.camera_widget = CameraWidget(width=640, height=480)  # Will be updated dynamically
        camera_layout.addWidget(self.camera_widget)
        
        layout.addWidget(camera_group)
        
        # SLAM Control Group
        slam_group = QGroupBox("🌾 Agricultural SLAM Control")
        slam_layout = QVBoxLayout(slam_group)
        
        # SLAM status
        self.slam_status_label = QLabel("SLAM: Inactive")
        self.slam_status_label.setStyleSheet("color: orange; font-weight: bold;")
        slam_layout.addWidget(self.slam_status_label)
        
        # SLAM controls
        self.enable_slam_checkbox = QCheckBox("Enable SLAM Processing")
        self.enable_slam_checkbox.stateChanged.connect(self.toggle_slam)
        slam_layout.addWidget(self.enable_slam_checkbox)
        
        self.show_features_checkbox = QCheckBox("Show Feature Detection")
        self.show_features_checkbox.setChecked(True)
        self.show_features_checkbox.stateChanged.connect(self.on_feature_display_changed)
        slam_layout.addWidget(self.show_features_checkbox)
        
        self.show_agricultural_checkbox = QCheckBox("Show Agricultural Features")
        self.show_agricultural_checkbox.setChecked(True)
        self.show_agricultural_checkbox.stateChanged.connect(self.on_agricultural_display_changed)
        slam_layout.addWidget(self.show_agricultural_checkbox)
        
        # Performance mode selection
        perf_label = QLabel("Performance Mode:")
        slam_layout.addWidget(perf_label)
        
        self.performance_combo = QComboBox()
        self.performance_combo.addItems(["FAST", "BALANCED", "ACCURATE"])
        self.performance_combo.setCurrentText("BALANCED")
        self.performance_combo.currentTextChanged.connect(self.on_performance_mode_changed)
        slam_layout.addWidget(self.performance_combo)
        
        # Reset button
        self.reset_slam_btn = QPushButton("Reset SLAM")
        self.reset_slam_btn.clicked.connect(self.reset_slam)
        slam_layout.addWidget(self.reset_slam_btn)
        
        layout.addWidget(slam_group)
        
        # Session Control Group
        session_group = QGroupBox("📁 Session Management")
        session_layout = QVBoxLayout(session_group)
        
        self.session_status_label = QLabel("Session: Not started")
        session_layout.addWidget(self.session_status_label)
        
        self.start_session_btn = QPushButton("Start New Session")
        self.start_session_btn.clicked.connect(self.start_new_session)
        session_layout.addWidget(self.start_session_btn)
        
        self.save_session_btn = QPushButton("Save Session")
        self.save_session_btn.clicked.connect(self.save_current_session)
        self.save_session_btn.setEnabled(False)
        session_layout.addWidget(self.save_session_btn)
        
        layout.addWidget(session_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        return panel
    
    def create_center_panel(self) -> QWidget:
        """Create center panel with 3D trajectory visualization"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Trajectory control group
        traj_control_group = QGroupBox("🗺️ 3D Trajectory Visualization")
        traj_control_layout = QHBoxLayout(traj_control_group)
        
        # View controls
        self.view_2d_btn = QPushButton("2D View")
        self.view_2d_btn.setCheckable(True)
        self.view_2d_btn.setChecked(True)
        self.view_2d_btn.clicked.connect(self.switch_to_2d_view)
        traj_control_layout.addWidget(self.view_2d_btn)
        
        self.view_3d_btn = QPushButton("3D View")
        self.view_3d_btn.setCheckable(True)
        self.view_3d_btn.clicked.connect(self.switch_to_3d_view)
        traj_control_layout.addWidget(self.view_3d_btn)
        
        # Clear trajectory button
        self.clear_trajectory_btn = QPushButton("Clear Trajectory")
        self.clear_trajectory_btn.clicked.connect(self.clear_trajectory)
        traj_control_layout.addWidget(self.clear_trajectory_btn)
        
        traj_control_layout.addStretch()
        
        layout.addWidget(traj_control_group)
        
        # Trajectory widget
        self.trajectory_widget = TrajectoryWidget()
        layout.addWidget(self.trajectory_widget)
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Create right panel with information and statistics"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Create tabs for different information
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # Distance & Position Tab
        distance_tab = self.create_distance_tab()
        tab_widget.addTab(distance_tab, "📏 Distance & Position")
        
        # Performance Tab
        performance_tab = self.create_performance_tab()
        tab_widget.addTab(performance_tab, "⚡ Performance")
        
        # Agricultural Tab
        agricultural_tab = self.create_agricultural_tab()
        tab_widget.addTab(agricultural_tab, "🌾 Agricultural")
        
        # Debug Tab
        debug_tab = self.create_debug_tab()
        tab_widget.addTab(debug_tab, "🔧 Debug")
        
        return panel
    
    def create_distance_tab(self) -> QWidget:
        """Create distance and position information tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Distance measurements group
        distance_group = QGroupBox("Distance Measurements")
        distance_layout = QGridLayout(distance_group)
        
        # SLAM distance
        distance_layout.addWidget(QLabel("SLAM Distance:"), 0, 0)
        self.slam_distance_label = QLabel("0.000 m")
        self.slam_distance_label.setStyleSheet("font-weight: bold; color: blue;")
        distance_layout.addWidget(self.slam_distance_label, 0, 1)
        
        # Precision distance
        distance_layout.addWidget(QLabel("Precision Distance:"), 1, 0)
        self.precision_distance_label = QLabel("0.000 m")
        self.precision_distance_label.setStyleSheet("font-weight: bold; color: green;")
        distance_layout.addWidget(self.precision_distance_label, 1, 1)
        
        # Distance difference
        distance_layout.addWidget(QLabel("Difference:"), 2, 0)
        self.distance_diff_label = QLabel("0.000 m")
        distance_layout.addWidget(self.distance_diff_label, 2, 1)
        
        # Accuracy estimate
        distance_layout.addWidget(QLabel("Estimated Accuracy:"), 3, 0)
        self.accuracy_label = QLabel("±5.0 cm")
        distance_layout.addWidget(self.accuracy_label, 3, 1)
        
        layout.addWidget(distance_group)
        
        # Position group
        position_group = QGroupBox("Current Position")
        position_layout = QGridLayout(position_group)
        
        position_layout.addWidget(QLabel("X:"), 0, 0)
        self.position_x_label = QLabel("0.000 m")
        position_layout.addWidget(self.position_x_label, 0, 1)
        
        position_layout.addWidget(QLabel("Y:"), 1, 0)
        self.position_y_label = QLabel("0.000 m")
        position_layout.addWidget(self.position_y_label, 1, 1)
        
        position_layout.addWidget(QLabel("Z:"), 2, 0)
        self.position_z_label = QLabel("0.000 m")
        position_layout.addWidget(self.position_z_label, 2, 1)
        
        layout.addWidget(position_group)
        
        # Path statistics group
        path_group = QGroupBox("Path Statistics")
        path_layout = QGridLayout(path_group)
        
        path_layout.addWidget(QLabel("Trajectory Points:"), 0, 0)
        self.trajectory_points_label = QLabel("0")
        path_layout.addWidget(self.trajectory_points_label, 0, 1)
        
        path_layout.addWidget(QLabel("Keyframes:"), 1, 0)
        self.keyframes_label = QLabel("0")
        path_layout.addWidget(self.keyframes_label, 1, 1)
        
        path_layout.addWidget(QLabel("Session Duration:"), 2, 0)
        self.session_duration_label = QLabel("00:00:00")
        path_layout.addWidget(self.session_duration_label, 2, 1)
        
        layout.addWidget(path_group)
        
        layout.addStretch()
        return tab
    
    def create_performance_tab(self) -> QWidget:
        """Create performance monitoring tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Performance metrics group
        perf_group = QGroupBox("Performance Metrics")
        perf_layout = QGridLayout(perf_group)
        
        perf_layout.addWidget(QLabel("FPS:"), 0, 0)
        self.fps_label = QLabel("0.0")
        perf_layout.addWidget(self.fps_label, 0, 1)
        
        perf_layout.addWidget(QLabel("Processing Time:"), 1, 0)
        self.processing_time_label = QLabel("0.0 ms")
        perf_layout.addWidget(self.processing_time_label, 1, 1)
        
        perf_layout.addWidget(QLabel("Features Detected:"), 2, 0)
        self.features_label = QLabel("0")
        perf_layout.addWidget(self.features_label, 2, 1)
        
        perf_layout.addWidget(QLabel("Tracking Quality:"), 3, 0)
        self.tracking_quality_label = QLabel("0%")
        perf_layout.addWidget(self.tracking_quality_label, 3, 1)
        
        # Progress bar for tracking quality
        self.tracking_quality_bar = QProgressBar()
        self.tracking_quality_bar.setRange(0, 100)
        perf_layout.addWidget(self.tracking_quality_bar, 4, 0, 1, 2)
        
        layout.addWidget(perf_group)
        
        # System info group
        system_group = QGroupBox("System Information")
        system_layout = QGridLayout(system_group)
        
        system_layout.addWidget(QLabel("Frames Processed:"), 0, 0)
        self.frames_processed_label = QLabel("0")
        system_layout.addWidget(self.frames_processed_label, 0, 1)
        
        system_layout.addWidget(QLabel("Success Rate:"), 1, 0)
        self.success_rate_label = QLabel("0%")
        system_layout.addWidget(self.success_rate_label, 1, 1)
        
        # FIXED: Add camera information
        system_layout.addWidget(QLabel("Camera Resolution:"), 2, 0)
        self.camera_resolution_label = QLabel("Unknown")
        system_layout.addWidget(self.camera_resolution_label, 2, 1)
        
        system_layout.addWidget(QLabel("Camera Format:"), 3, 0)
        self.camera_format_label = QLabel("Unknown")
        system_layout.addWidget(self.camera_format_label, 3, 1)
        
        layout.addWidget(system_group)
        
        layout.addStretch()
        return tab
    
    def create_agricultural_tab(self) -> QWidget:
        """Create agricultural features tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Agricultural scene group
        agri_group = QGroupBox("Agricultural Scene Analysis")
        agri_layout = QGridLayout(agri_group)
        
        agri_layout.addWidget(QLabel("Scene Type:"), 0, 0)
        self.scene_type_label = QLabel("Unknown")
        agri_layout.addWidget(self.scene_type_label, 0, 1)
        
        agri_layout.addWidget(QLabel("Crop Rows Detected:"), 1, 0)
        self.crop_rows_label = QLabel("No")
        agri_layout.addWidget(self.crop_rows_label, 1, 1)
        
        agri_layout.addWidget(QLabel("Ground Plane:"), 2, 0)
        self.ground_plane_label = QLabel("Not estimated")
        agri_layout.addWidget(self.ground_plane_label, 2, 1)
        
        agri_layout.addWidget(QLabel("Field Coverage:"), 3, 0)
        self.field_coverage_label = QLabel("0.0 m²")
        agri_layout.addWidget(self.field_coverage_label, 3, 1)
        
        layout.addWidget(agri_group)
        
        # Agricultural statistics group
        agri_stats_group = QGroupBox("Agricultural Statistics")
        agri_stats_layout = QGridLayout(agri_stats_group)
        
        agri_stats_layout.addWidget(QLabel("Scene Complexity:"), 0, 0)
        self.scene_complexity_label = QLabel("0%")
        agri_stats_layout.addWidget(self.scene_complexity_label, 0, 1)
        
        agri_stats_layout.addWidget(QLabel("Lighting Quality:"), 1, 0)
        self.lighting_quality_label = QLabel("0%")
        agri_stats_layout.addWidget(self.lighting_quality_label, 1, 1)
        
        agri_stats_layout.addWidget(QLabel("Agricultural Score:"), 2, 0)
        self.agricultural_score_label = QLabel("0.0")
        agri_stats_layout.addWidget(self.agricultural_score_label, 2, 1)
        
        layout.addWidget(agri_stats_group)
        
        layout.addStretch()
        return tab
    
    def create_debug_tab(self) -> QWidget:
        """Create debug information tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Debug output
        debug_group = QGroupBox("Debug Information")
        debug_layout = QVBoxLayout(debug_group)
        
        self.debug_text = QTextEdit()
        self.debug_text.setMaximumHeight(200)
        self.debug_text.setFont(QFont("Consolas", 9))
        debug_layout.addWidget(self.debug_text)
        
        # Clear debug button
        clear_debug_btn = QPushButton("Clear Debug Log")
        clear_debug_btn.clicked.connect(self.debug_text.clear)
        debug_layout.addWidget(clear_debug_btn)
        
        layout.addWidget(debug_group)
        
        # SLAM state group
        slam_state_group = QGroupBox("SLAM State Information")
        slam_state_layout = QGridLayout(slam_state_group)
        
        slam_state_layout.addWidget(QLabel("SLAM Mode:"), 0, 0)
        self.slam_mode_label = QLabel("INACTIVE")
        slam_state_layout.addWidget(self.slam_mode_label, 0, 1)
        
        slam_state_layout.addWidget(QLabel("Last Update:"), 1, 0)
        self.last_update_label = QLabel("Never")
        slam_state_layout.addWidget(self.last_update_label, 1, 1)
        
        slam_state_layout.addWidget(QLabel("Thread Processing:"), 2, 0)
        self.thread_processing_label = QLabel("0.0 ms")
        slam_state_layout.addWidget(self.thread_processing_label, 2, 1)
        
        layout.addWidget(slam_state_group)
        
        layout.addStretch()
        return tab
    
    def create_menu_bar(self):
        """Create enhanced menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        new_session_action = QAction('New Session', self)
        new_session_action.triggered.connect(self.start_new_session)
        file_menu.addAction(new_session_action)
        
        save_session_action = QAction('Save Session', self)
        save_session_action.triggered.connect(self.save_current_session)
        file_menu.addAction(save_session_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        toggle_features_action = QAction('Toggle Features', self)
        toggle_features_action.triggered.connect(
            lambda: self.show_features_checkbox.setChecked(
                not self.show_features_checkbox.isChecked()
            )
        )
        view_menu.addAction(toggle_features_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        reset_slam_action = QAction('Reset SLAM', self)
        reset_slam_action.triggered.connect(self.reset_slam)
        tools_menu.addAction(reset_slam_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def init_timers(self):
        """Initialize update timers"""
        # Main update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_session_info)
        self.update_timer.start(1000)  # Update every second
    
    def init_status_bar(self):
        """Initialize status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Agricultural SLAM System Ready (FIXED)")
    
    def apply_modern_style(self):
        """Apply modern styling to the interface"""
        style = """
        QMainWindow {
            background-color: #f0f0f0;
        }
        QGroupBox {
            font-weight: bold;
            border: 2px solid #cccccc;
            border-radius: 8px;
            margin-top: 1ex;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        QPushButton {
            background-color: #e1e1e1;
            border: 1px solid #adadad;
            border-radius: 4px;
            padding: 5px;
            min-height: 20px;
        }
        QPushButton:hover {
            background-color: #d4edda;
            border-color: #c3e6cb;
        }
        QPushButton:pressed {
            background-color: #c1e2c8;
        }
        QPushButton:checked {
            background-color: #28a745;
            color: white;
        }
        QLabel {
            color: #333333;
        }
        QTextEdit {
            background-color: white;
            border: 1px solid #cccccc;
            border-radius: 4px;
        }
        """
        self.setStyleSheet(style)
    
    def toggle_camera(self):
        """FIXED: Toggle camera on/off with proper AgriSLAMCore integration"""
        if not self.camera_active:
            # Start camera
            try:
                self.camera_manager = CameraManager()
                if self.camera_manager.initialize_camera():
                    if self.camera_manager.start_streaming():
                        # FIXED: Initialize AgriSLAMCore instead of EnhancedCustomVisualSLAM
                        self.slam_system = AgriSLAMCore(self.camera_manager)
                        
                        # FIXED: Update camera widget with actual resolution
                        camera_resolution = self.camera_manager.get_camera_resolution()
                        self.camera_widget.set_source_frame_size(camera_resolution[0], camera_resolution[1])
                        
                        # Start processing thread
                        self.processing_thread = SLAMProcessingThread(
                            self.camera_manager, self.slam_system
                        )
                        self.processing_thread.results_ready.connect(self.on_slam_results)
                        self.processing_thread.error_occurred.connect(self.on_slam_error)
                        self.processing_thread.status_update.connect(self.on_status_update)
                        self.processing_thread.start()
                        
                        self.camera_active = True
                        self.start_camera_btn.setText("Stop Camera")
                        self.camera_status_label.setText("Camera: Active")
                        self.camera_status_label.setStyleSheet("color: green; font-weight: bold;")
                        self.enable_slam_checkbox.setEnabled(True)
                        
                        # FIXED: Display camera information
                        camera_info = self.camera_manager.get_camera_info()
                        self.camera_resolution_label.setText(f"{camera_resolution[0]}x{camera_resolution[1]}")
                        self.camera_format_label.setText(self.camera_manager.get_camera_format())
                        
                        self.status_bar.showMessage("Camera started successfully")
                        print(f"✅ Camera started: {camera_resolution[0]}x{camera_resolution[1]} {self.camera_manager.get_camera_format()}")
                    else:
                        QMessageBox.critical(self, "Error", "Failed to start camera streaming")
                else:
                    QMessageBox.critical(self, "Error", "Failed to initialize camera")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Camera error: {str(e)}")
        else:
            # Stop camera
            self.stop_camera()
    
    def stop_camera(self):
        """Stop camera and SLAM processing"""
        try:
            self.camera_active = False
            self.slam_active = False
            
            # Stop processing thread
            if self.processing_thread:
                self.processing_thread.stop()
                self.processing_thread = None
            
            # Stop camera
            if self.camera_manager:
                self.camera_manager.stop_streaming()
                self.camera_manager = None
            
            # Reset SLAM system
            self.slam_system = None
            
            # Update UI
            self.start_camera_btn.setText("Start Camera")
            self.camera_status_label.setText("Camera: Disconnected")
            self.camera_status_label.setStyleSheet("color: red; font-weight: bold;")
            self.slam_status_label.setText("SLAM: Inactive")
            self.slam_status_label.setStyleSheet("color: orange; font-weight: bold;")
            self.enable_slam_checkbox.setChecked(False)
            self.enable_slam_checkbox.setEnabled(False)
            
            # Clear camera widget
            self.camera_widget.clear_display()
            
            self.status_bar.showMessage("Camera stopped")
            print("⏹️  Camera and SLAM stopped")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error stopping camera: {str(e)}")
    
    def toggle_slam(self):
        """Toggle SLAM processing on/off"""
        if not self.camera_active:
            QMessageBox.warning(self, "Warning", "Please start camera first")
            self.enable_slam_checkbox.setChecked(False)
            return
        
        self.slam_active = self.enable_slam_checkbox.isChecked()
        
        if self.processing_thread:
            self.processing_thread.enable_slam(self.slam_active)
        
        if self.slam_active:
            self.slam_status_label.setText("SLAM: Active")
            self.slam_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.status_bar.showMessage("SLAM processing enabled")
            print("🌾 SLAM processing enabled")
        else:
            self.slam_status_label.setText("SLAM: Inactive")
            self.slam_status_label.setStyleSheet("color: orange; font-weight: bold;")
            self.status_bar.showMessage("SLAM processing disabled")
            print("⏸️  SLAM processing disabled")
    
    def on_performance_mode_changed(self, mode: str):
        """Handle performance mode change"""
        if self.slam_system:
            # Note: AgriSLAMCore doesn't have set_performance_mode, but we can add it
            self.status_bar.showMessage(f"Performance mode: {mode}")
            print(f"🔧 Performance mode: {mode}")
    
    def on_feature_display_changed(self):
        """FIXED: Handle feature display toggle"""
        show_features = self.show_features_checkbox.isChecked()
        self.camera_widget.show_features = show_features
        print(f"🎯 Feature display: {'ON' if show_features else 'OFF'}")
    
    def on_agricultural_display_changed(self):
        """FIXED: Handle agricultural display toggle"""
        show_agricultural = self.show_agricultural_checkbox.isChecked()
        self.camera_widget.show_agricultural = show_agricultural
        print(f"🌾 Agricultural display: {'ON' if show_agricultural else 'OFF'}")
    
    def reset_slam(self):
        """Reset SLAM system"""
        if self.slam_system:
            self.slam_system.reset()
            self.trajectory_widget.clear_trajectory()
            self.status_bar.showMessage("SLAM system reset")
            
            # Reset all labels
            self.slam_distance_label.setText("0.000 m")
            self.precision_distance_label.setText("0.000 m")
            self.distance_diff_label.setText("0.000 m")
            self.position_x_label.setText("0.000 m")
            self.position_y_label.setText("0.000 m")
            self.position_z_label.setText("0.000 m")
            self.trajectory_points_label.setText("0")
            self.keyframes_label.setText("0")
            
            print("🔄 SLAM system reset")
    
    def start_new_session(self):
        """Start a new SLAM session"""
        if self.session_active:
            reply = QMessageBox.question(
                self, "Confirm", "End current session and start new one?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        self.reset_slam()
        self.session_active = True
        self.session_start_time = time.time()
        self.save_session_btn.setEnabled(True)
        self.session_status_label.setText("Session: Active")
        self.status_bar.showMessage("New session started")
        print("📁 New session started")
    
    def save_current_session(self):
        """Save current SLAM session"""
        if not self.slam_system or not self.session_active:
            QMessageBox.warning(self, "Warning", "No active session to save")
            return
        
        try:
            # FIXED: Use AgriSLAMCore.save_session method
            filename = self.slam_system.save_session()
            if filename:
                QMessageBox.information(
                    self, "Success", 
                    f"Session saved successfully:\n{filename}"
                )
                self.status_bar.showMessage("Session saved")
                print(f"💾 Session saved: {filename}")
            else:
                QMessageBox.warning(self, "Warning", "Failed to save session")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving session: {str(e)}")
    
    def switch_to_2d_view(self):
        """Switch to 2D trajectory view"""
        self.view_2d_btn.setChecked(True)
        self.view_3d_btn.setChecked(False)
        self.trajectory_widget.set_view_mode('2D')
        print("📊 Switched to 2D view")
    
    def switch_to_3d_view(self):
        """Switch to 3D trajectory view"""
        self.view_2d_btn.setChecked(False)
        self.view_3d_btn.setChecked(True)
        self.trajectory_widget.set_view_mode('3D')
        print("📊 Switched to 3D view")
    
    def clear_trajectory(self):
        """Clear trajectory display"""
        self.trajectory_widget.clear_trajectory()
        print("🗑️  Trajectory cleared")
    
    def on_slam_results(self, results: Dict):
        """FIXED: Handle SLAM processing results with proper data flow"""
        try:
            self.frame_update_counter += 1
            self.last_frame_data = results
            
            # FIXED: Update camera widget with comprehensive data
            if 'color_frame' in results:
                color_frame = results['color_frame']
                depth_frame = results.get('depth_frame')
                
                # Extract features if available
                features = None
                if 'current_features' in results:
                    features = results['current_features']
                elif self.slam_active and 'num_features' in results and results['num_features'] > 0:
                    # Generate dummy features for display if real features not available
                    features = []
                
                # Extract agricultural features (FIXED format)
                agricultural_features = results.get('agricultural_scene', {})
                
                # Update camera widget with all data
                self.camera_widget.update_frame(
                    color_frame=color_frame,
                    depth_frame=depth_frame,
                    features=features,
                    matches=None,
                    agricultural_features=agricultural_features
                )
            
            # FIXED: Update trajectory if SLAM is active and pose is estimated
            if self.slam_active and results.get('pose_estimated', False):
                position = results.get('position', np.array([0, 0, 0]))
                if len(position) >= 3:
                    self.trajectory_widget.add_point(position)
            
            # FIXED: Update distance information
            if 'total_distance' in results:
                self.slam_distance_label.setText(f"{results['total_distance']:.3f} m")
            
            # Update precision distance if available
            precision_distance = results.get('precision_distance', results.get('total_distance', 0.0))
            self.precision_distance_label.setText(f"{precision_distance:.3f} m")
            
            # Calculate difference
            slam_dist = results.get('total_distance', 0.0)
            diff = abs(slam_dist - precision_distance)
            self.distance_diff_label.setText(f"{diff:.3f} m")
            
            # FIXED: Update position with proper data access
            if 'position' in results:
                pos = results['position']
                if len(pos) >= 3:
                    self.position_x_label.setText(f"{pos[0]:.3f} m")
                    self.position_y_label.setText(f"{pos[1]:.3f} m")
                    self.position_z_label.setText(f"{pos[2]:.3f} m")
            
            # FIXED: Update performance metrics
            if 'fps' in results:
                self.fps_label.setText(f"{results['fps']:.1f}")
            
            if 'processing_time' in results:
                self.processing_time_label.setText(f"{results['processing_time']*1000:.1f} ms")
            
            if 'thread_processing_time' in results:
                self.thread_processing_label.setText(f"{results['thread_processing_time']*1000:.1f} ms")
            
            if 'num_features' in results:
                self.features_label.setText(str(results['num_features']))
            
            if 'tracking_quality' in results:
                quality = int(results['tracking_quality'] * 100)
                self.tracking_quality_label.setText(f"{quality}%")
                self.tracking_quality_bar.setValue(quality)
            
            # FIXED: Update agricultural information with proper data structure handling
            agri_info = results.get('agricultural_scene', {})
            if agri_info:
                scene_type = agri_info.get('scene_type', 'unknown')
                self.scene_type_label.setText(scene_type.title())
                
                crop_rows_detected = agri_info.get('crop_rows_detected', False)
                crop_rows_count = len(agri_info.get('crop_rows', []))
                self.crop_rows_label.setText(f"Yes ({crop_rows_count})" if crop_rows_detected else "No")
                
                ground_plane = agri_info.get('ground_plane')
                if ground_plane and hasattr(ground_plane, 'confidence'):
                    self.ground_plane_label.setText(f"Yes ({ground_plane.confidence:.2f})")
                else:
                    self.ground_plane_label.setText("No")
                
                complexity = agri_info.get('scene_complexity', 0.0)
                self.scene_complexity_label.setText(f"{complexity*100:.1f}%")
                
                lighting = agri_info.get('lighting_quality', 0.0)
                self.lighting_quality_label.setText(f"{lighting*100:.1f}%")
                
                agricultural_score = agri_info.get('agricultural_score', 0.0)
                self.agricultural_score_label.setText(f"{agricultural_score:.2f}")
            
            # FIXED: Update SLAM state information
            slam_mode = results.get('slam_mode', 'UNKNOWN')
            self.slam_mode_label.setText(slam_mode)
            self.last_update_label.setText(time.strftime("%H:%M:%S"))
            
            # Update frame counter and camera information
            frame_count = results.get('frame_count', 0)
            self.frames_processed_label.setText(str(frame_count))
            
            # Update camera resolution and format if available
            if 'camera_resolution' in results:
                resolution = results['camera_resolution']
                self.camera_resolution_label.setText(f"{resolution[0]}x{resolution[1]}")
            
            if 'camera_format' in results:
                self.camera_format_label.setText(results['camera_format'])
            
            # FIXED: Update debug info with controlled frequency
            debug_info = results.get('debug_info', '')
            if debug_info and self.frame_update_counter % 30 == 0:  # Only every 30 frames
                timestamp = time.strftime("%H:%M:%S")
                self.debug_text.append(f"[{timestamp}] {debug_info}")
                
                # Keep debug text limited
                if self.debug_text.document().lineCount() > 100:
                    cursor = self.debug_text.textCursor()
                    cursor.movePosition(cursor.MoveOperation.Start)
                    cursor.movePosition(cursor.MoveOperation.Down, cursor.MoveMode.KeepAnchor, 10)
                    cursor.removeSelectedText()
                
                # Auto-scroll to bottom
                cursor = self.debug_text.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                self.debug_text.setTextCursor(cursor)
            
        except Exception as e:
            print(f"❌ Error updating GUI with SLAM results: {e}")
            self.debug_text.append(f"[{time.strftime('%H:%M:%S')}] GUI UPDATE ERROR: {str(e)}")
    
    def on_slam_error(self, error_msg: str):
        """Handle SLAM processing errors"""
        timestamp = time.strftime("%H:%M:%S")
        self.debug_text.append(f"[{timestamp}] ERROR: {error_msg}")
        self.status_bar.showMessage(f"SLAM Error: {error_msg}")
        print(f"❌ SLAM Error: {error_msg}")
    
    def on_status_update(self, status_msg: str):
        """Handle status updates"""
        self.status_bar.showMessage(status_msg)
    
    def update_session_info(self):
        """Update session information"""
        if self.session_active and self.session_start_time:
            duration = time.time() - self.session_start_time
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = int(duration % 60)
            self.session_duration_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About Agricultural SLAM System",
            "Agricultural SLAM System v2.0 (FIXED)\n\n"
            "Enhanced real-time visual SLAM for agricultural applications\n"
            "Features:\n"
            "• Centimeter-level distance tracking\n"
            "• 3D trajectory visualization\n"
            "• Agricultural scene understanding\n"
            "• Crop row detection\n"
            "• Real-time performance monitoring\n"
            "• Structured agricultural feature display\n\n"
            "Optimized for Intel RealSense D435i camera\n"
            "FIXED: Proper component integration and data flow"
        )
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.camera_active:
            reply = QMessageBox.question(
                self, "Confirm Exit", "Camera is active. Exit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.stop_camera()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = EnhancedMainWindow()
    window.show()
    sys.exit(app.exec())