"""
Control panel widget with tracking controls and settings
"""

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import logging

class ControlPanel(QWidget):
    """Control panel for tracking operations"""
    
    # Signals
    start_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    export_clicked = pyqtSignal()
    calibrate_clicked = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Tracking state
        self.is_tracking = False
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize user interface"""
        layout = QVBoxLayout(self)
        
        # Main controls group
        controls_group = QGroupBox("Tracking Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        # Start/Stop buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("🟢 Start Tracking")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                border: 2px solid #229954;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
            QPushButton:pressed {
                background-color: #1e8449;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
                border-color: #5d6d7e;
            }
        """)
        self.start_btn.clicked.connect(self.on_start_clicked)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("🔴 Stop Tracking")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                border: 2px solid #c0392b;
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #ec7063;
            }
            QPushButton:pressed {
                background-color: #a93226;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
                border-color: #5d6d7e;
            }
        """)
        self.stop_btn.clicked.connect(self.on_stop_clicked)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        controls_layout.addLayout(button_layout)
        
        # Reset button
        self.reset_btn = QPushButton("🔄 Reset Tracking")
        self.reset_btn.setMinimumHeight(40)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #f39c12;
                border: 2px solid #e67e22;
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #f5b041;
            }
            QPushButton:pressed {
                background-color: #d68910;
            }
        """)
        self.reset_btn.clicked.connect(self.on_reset_clicked)
        controls_layout.addWidget(self.reset_btn)
        
        layout.addWidget(controls_group)
        
        # Data management group
        data_group = QGroupBox("Data Management")
        data_layout = QVBoxLayout(data_group)
        
        self.export_btn = QPushButton("💾 Export Data")
        self.export_btn.setMinimumHeight(35)
        self.export_btn.clicked.connect(self.on_export_clicked)
        data_layout.addWidget(self.export_btn)
        
        layout.addWidget(data_group)
        
        # Camera settings group
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QVBoxLayout(camera_group)
        
        self.calibrate_btn = QPushButton("📐 Calibrate Camera")
        self.calibrate_btn.setMinimumHeight(35)
        self.calibrate_btn.clicked.connect(self.on_calibrate_clicked)
        camera_layout.addWidget(self.calibrate_btn)
        
        # Feature detector selection
        detector_layout = QHBoxLayout()
        detector_layout.addWidget(QLabel("Feature Detector:"))
        
        self.detector_combo = QComboBox()
        self.detector_combo.addItems(["ORB", "SIFT", "SURF", "AKAZE"])
        self.detector_combo.setCurrentText("ORB")
        detector_layout.addWidget(self.detector_combo)
        
        camera_layout.addLayout(detector_layout)
        
        # Max features setting
        features_layout = QHBoxLayout()
        features_layout.addWidget(QLabel("Max Features:"))
        
        self.features_spin = QSpinBox()
        self.features_spin.setRange(100, 5000)
        self.features_spin.setValue(1000)
        self.features_spin.setSingleStep(100)
        features_layout.addWidget(self.features_spin)
        
        camera_layout.addLayout(features_layout)
        
        layout.addWidget(camera_group)
        
        # SLAM settings group
        slam_group = QGroupBox("SLAM Settings")
        slam_layout = QVBoxLayout(slam_group)
        
        # Loop closure threshold
        loop_layout = QHBoxLayout()
        loop_layout.addWidget(QLabel("Loop Threshold:"))
        
        self.loop_threshold_spin = QDoubleSpinBox()
        self.loop_threshold_spin.setRange(0.1, 1.0)
        self.loop_threshold_spin.setValue(0.15)
        self.loop_threshold_spin.setSingleStep(0.05)
        self.loop_threshold_spin.setDecimals(2)
        loop_layout.addWidget(self.loop_threshold_spin)
        
        slam_layout.addLayout(loop_layout)
        
        # Use IMU checkbox
        self.use_imu_check = QCheckBox("Use IMU Data")
        self.use_imu_check.setChecked(True)
        slam_layout.addWidget(self.use_imu_check)
        
        # Use depth checkbox
        self.use_depth_check = QCheckBox("Use Depth Data")
        self.use_depth_check.setChecked(True)
        slam_layout.addWidget(self.use_depth_check)
        
        layout.addWidget(slam_group)
        
        # Add stretch to push everything to top
        layout.addStretch()
        
        # Status indicator
        self.status_label = QLabel("🔴 Ready to Start")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #34495e;
                border: 2px solid #2c3e50;
                border-radius: 5px;
                padding: 8px;
                color: #ffffff;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.status_label)
    
    def on_start_clicked(self):
        """Handle start button click"""
        self.start_clicked.emit()
    
    def on_stop_clicked(self):
        """Handle stop button click"""
        self.stop_clicked.emit()
    
    def on_reset_clicked(self):
        """Handle reset button click"""
        self.reset_clicked.emit()
    
    def on_export_clicked(self):
        """Handle export button click"""
        self.export_clicked.emit()
    
    def on_calibrate_clicked(self):
        """Handle calibrate button click"""
        self.calibrate_clicked.emit()
    
    def set_tracking_state(self, is_tracking: bool):
        """Update UI based on tracking state"""
        self.is_tracking = is_tracking
        
        if is_tracking:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status_label.setText("🟢 Tracking Active")
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #27ae60;
                    border: 2px solid #229954;
                    border-radius: 5px;
                    padding: 8px;
                    color: #ffffff;
                    font-weight: bold;
                }
            """)
        else:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("🔴 Ready to Start")
            self.status_label.setStyleSheet("""
                QLabel {
                    background-color: #34495e;
                    border: 2px solid #2c3e50;
                    border-radius: 5px;
                    padding: 8px;
                    color: #ffffff;
                    font-weight: bold;
                }
            """)
    
    def get_settings(self) -> dict:
        """Get current settings"""
        return {
            'feature_detector': self.detector_combo.currentText(),
            'max_features': self.features_spin.value(),
            'loop_threshold': self.loop_threshold_spin.value(),
            'use_imu': self.use_imu_check.isChecked(),
            'use_depth': self.use_depth_check.isChecked()
        }