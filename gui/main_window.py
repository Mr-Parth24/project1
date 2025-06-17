"""
Improved Main GUI Window with Better Visibility and Modern Design
Enhanced readability and user experience

Author: Enhanced for Mr-Parth24
Date: 2025-06-17 18:21:25 UTC
"""

import sys
import cv2
import numpy as np
import threading
import time
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl

from core.camera_manager import CameraManager
from core.visual_odometry import VisualOdometry
from core.slam_processor import SLAMProcessor
from core.trajectory_tracker import TrajectoryTracker
from gui.widgets.camera_widget import CameraWidget
from gui.widgets.trajectory_widget import TrajectoryWidget
from gui.widgets.control_panel import ControlPanel
from utils.data_export import DataExporter
from config.camera_config import CameraConfig
import logging

class ModernDebugWidget(QWidget):
    """Modern debug widget with better visibility"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header with icon and title
        header_layout = QHBoxLayout()
        
        # Debug icon
        debug_icon = QLabel("🐛")
        debug_icon.setFont(QFont("Arial", 16))
        header_layout.addWidget(debug_icon)
        
        # Title
        title = QLabel("Debug Information")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setStyleSheet("color: #ffffff; margin-left: 8px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Clear button
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.setFixedSize(80, 28)
        clear_btn.clicked.connect(self.clear_debug)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                border: none;
                border-radius: 4px;
                color: white;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        header_layout.addWidget(clear_btn)
        
        layout.addLayout(header_layout)
        
        # Debug text area with modern styling
        self.debug_text = QTextEdit()
        self.debug_text.setMaximumHeight(180)
        self.debug_text.setFont(QFont("Consolas", 10))
        self.debug_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 2px solid #404040;
                border-radius: 8px;
                padding: 8px;
                selection-background-color: #264f78;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #5a5a5a;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #6a6a6a;
            }
        """)
        layout.addWidget(self.debug_text)
        
        # Status bar
        self.status_bar = QLabel("Ready to debug...")
        self.status_bar.setFont(QFont("Segoe UI", 9))
        self.status_bar.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                color: #cccccc;
                padding: 4px 8px;
                border-radius: 4px;
                border: 1px solid #404040;
            }        """)
        layout.addWidget(self.status_bar)
    
    def add_debug_info(self, message: str, level: str = "INFO"):
        """Add debug information with color coding - thread safe"""
        # Use QTimer.singleShot to ensure GUI updates happen in main thread
        QTimer.singleShot(0, lambda: self._add_debug_info_safe(message, level))
    
    def _add_debug_info_safe(self, message: str, level: str = "INFO"):
        """Thread-safe debug info addition"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Color coding based on level
        colors = {
            "INFO": "#4fc3f7",
            "WARNING": "#ffb74d", 
            "ERROR": "#f44336",
            "SUCCESS": "#66bb6a",
            "DEBUG": "#ab47bc"
        }
        
        color = colors.get(level, "#ffffff")
        
        # Format message with HTML
        formatted_message = f"""
        <div style="margin: 2px 0; font-family: Consolas;">
            <span style="color: #888888;">[{timestamp}]</span>
            <span style="color: {color}; font-weight: bold;">[{level}]</span>
            <span style="color: #d4d4d4;"> {message}</span>
        </div>
        """
        
        self.debug_text.append(formatted_message)
        
        # Auto-scroll to bottom
        scrollbar = self.debug_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
        # Update status
        self.status_bar.setText(f"Last: {level} - {message[:50]}{'...' if len(message) > 50 else ''}")
        
        # Limit text length
        if self.debug_text.document().blockCount() > 150:
            cursor = self.debug_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.movePosition(cursor.Down, cursor.KeepAnchor, 30)
            cursor.removeSelectedText()
    
    def clear_debug(self):
        """Clear debug text"""
        self.debug_text.clear()
        self.status_bar.setText("Debug log cleared")

class ModernStatsWidget(QWidget):
    """Modern statistics widget with better layout"""
    
    def __init__(self):
        super().__init__()
        self.stats_labels = {}
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Header
        header_layout = QHBoxLayout()
        
        stats_icon = QLabel("📊")
        stats_icon.setFont(QFont("Arial", 16))
        header_layout.addWidget(stats_icon)
        
        title = QLabel("Tracking Statistics")
        title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        title.setStyleSheet("color: #ffffff; margin-left: 8px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Create statistics cards
        self.create_stats_cards(layout)
        
        # Progress indicators
        self.create_progress_section(layout)
        
        layout.addStretch()
    
    def create_stats_cards(self, parent_layout):
        """Create modern stats cards"""
        
        # Position card
        pos_card = self.create_stat_card("📍 Position", "position", "(0.000, 0.000, 0.000)")
        parent_layout.addWidget(pos_card)
        
        # Movement cards in grid
        movement_grid = QGridLayout()
        
        # Distance cards
        dist_card = self.create_stat_card("📏 Distance", "distance", "0.000 m")
        movement_grid.addWidget(dist_card, 0, 0)
        
        speed_card = self.create_stat_card("⚡ Speed", "speed", "0.000 m/s")
        movement_grid.addWidget(speed_card, 0, 1)
        
        # Quality cards
        quality_card = self.create_stat_card("🎯 Quality", "quality", "Unknown")
        movement_grid.addWidget(quality_card, 1, 0)
        
        fps_card = self.create_stat_card("⏱️ FPS", "fps", "0.0")
        movement_grid.addWidget(fps_card, 1, 1)
        
        movement_widget = QWidget()
        movement_widget.setLayout(movement_grid)
        parent_layout.addWidget(movement_widget)
        
        # Technical details (collapsible)
        self.create_technical_section(parent_layout)
    
    def create_stat_card(self, title: str, key: str, default_value: str = "--"):
        """Create a modern stat card"""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        
        layout = QVBoxLayout(card)
        layout.setSpacing(4)
        layout.setContentsMargins(8, 8, 8, 8)
        
        # Title
        title_label = QLabel(title)
        title_label.setFont(QFont("Segoe UI", 9, QFont.Bold))
        title_label.setStyleSheet("color: #cccccc;")
        layout.addWidget(title_label)
        
        # Value
        value_label = QLabel(default_value)
        value_label.setFont(QFont("Consolas", 10, QFont.Bold))
        value_label.setStyleSheet("color: #4fc3f7; margin-top: 2px;")
        value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(value_label)
        
        self.stats_labels[key] = value_label
        
        return card
    
    def create_technical_section(self, parent_layout):
        """Create technical details section"""
        # Collapsible technical details
        technical_group = QGroupBox("🔧 Technical Details")
        technical_group.setFont(QFont("Segoe UI", 10, QFont.Bold))
        technical_group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                border: 2px solid #404040;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                color: #4fc3f7;
                background-color: #252525;
            }
        """)
        
        tech_layout = QVBoxLayout(technical_group)
        tech_layout.setSpacing(6)
        
        # Technical stats in compact layout
        tech_stats = [
            ("Features", "features"),
            ("Matches", "matches"), 
            ("Success Rate", "success_rate"),
            ("Trajectory Points", "trajectory_points")
        ]
        
        for i in range(0, len(tech_stats), 2):
            row_layout = QHBoxLayout()
            
            # First stat
            stat1 = tech_stats[i]
            self.add_tech_stat(row_layout, stat1[0], stat1[1])
            
            # Second stat if exists
            if i + 1 < len(tech_stats):
                stat2 = tech_stats[i + 1]
                self.add_tech_stat(row_layout, stat2[0], stat2[1])
            
            tech_layout.addLayout(row_layout)
        
        parent_layout.addWidget(technical_group)
    
    def add_tech_stat(self, layout, label_text, key):
        """Add a technical stat to layout"""
        label = QLabel(f"{label_text}:")
        label.setFont(QFont("Segoe UI", 9))
        label.setStyleSheet("color: #cccccc;")
        label.setMinimumWidth(80)
        layout.addWidget(label)
        
        value_label = QLabel("--")
        value_label.setFont(QFont("Consolas", 9, QFont.Bold))
        value_label.setStyleSheet("color: #4fc3f7;")
        value_label.setAlignment(Qt.AlignRight)
        layout.addWidget(value_label)
        
        self.stats_labels[key] = value_label
    
    def create_progress_section(self, parent_layout):
        """Create progress indicators section"""
        progress_group = QGroupBox("📈 Progress Indicators")
        progress_group.setFont(QFont("Segoe UI", 10, QFont.Bold))
        progress_group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                border: 2px solid #404040;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 8px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px 0 8px;
                color: #66bb6a;
                background-color: #252525;
            }
        """)
        
        progress_layout = QVBoxLayout(progress_group)
        progress_layout.setSpacing(8)
        
        # Features progress bar
        features_label = QLabel("🔍 Features Detected")
        features_label.setFont(QFont("Segoe UI", 9))
        features_label.setStyleSheet("color: #cccccc;")
        progress_layout.addWidget(features_label)
        
        self.features_progress = QProgressBar()
        self.features_progress.setMaximum(1000)
        self.features_progress.setTextVisible(True)
        self.features_progress.setFormat("%v / %m features")
        self.features_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #404040;
                border-radius: 6px;
                text-align: center;
                background-color: #1e1e1e;
                color: #ffffff;
                font-weight: bold;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4fc3f7, stop:1 #29b6f6);
                border-radius: 4px;
            }
        """)
        progress_layout.addWidget(self.features_progress)
        
        # Matches progress bar
        matches_label = QLabel("🎯 Feature Matches")
        matches_label.setFont(QFont("Segoe UI", 9))
        matches_label.setStyleSheet("color: #cccccc;")
        progress_layout.addWidget(matches_label)
        
        self.matches_progress = QProgressBar()
        self.matches_progress.setMaximum(500)
        self.matches_progress.setTextVisible(True)
        self.matches_progress.setFormat("%v / %m matches")
        self.matches_progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #404040;
                border-radius: 6px;
                text-align: center;
                background-color: #1e1e1e;
                color: #ffffff;
                font-weight: bold;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #66bb6a, stop:1 #4caf50);
                border-radius: 4px;
            }
        """)
        progress_layout.addWidget(self.matches_progress)
        
        parent_layout.addWidget(progress_group)
    
    def update_stat(self, key: str, value: str, color: str = "#4fc3f7"):
        """Update a statistic with optional color"""
        if key in self.stats_labels:
            self.stats_labels[key].setText(str(value))
            self.stats_labels[key].setStyleSheet(f"color: {color}; font-weight: bold;")

class MainWindow(QMainWindow):
    """Improved main application window with modern design"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.camera_manager = CameraManager()
        self.config = CameraConfig()
        
        # Initialize processing components
        self.visual_odometry = None
        self.slam_processor = None
        self.trajectory_tracker = TrajectoryTracker()
        self.data_exporter = DataExporter()
        
        # Processing state
        self.is_tracking = False
        self.processing_thread = None
        self.processing_lock = threading.Lock()
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.last_update_time = time.time()
        
        # Timer for GUI updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_gui)
        
        # Current data
        self.current_frame_data = None
        self.current_vo_result = None
        self.current_slam_result = None
        
        # Debug and monitoring
        self.frame_processing_times = []
        self.tracking_failures = 0
          # Initialize UI
        self.init_modern_ui()
        
        # Connect signals
        self.connect_signals()
        
        self.logger.info("Modern main window initialized")
    
    def init_modern_ui(self):
        """Initialize modern user interface with better visibility"""
        self.setWindowTitle("RealSense D435i Visual Odometry - Mr-Parth24")
        
        # Set minimum and default sizes for better responsiveness
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Set window icon
        self.setWindowIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        
        # Create central widget with splitter for better layout management
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(4)
        main_layout.setContentsMargins(4, 4, 4, 4)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #404040;
                width: 3px;
            }
            QSplitter::handle:hover {
                background-color: #4fc3f7;
            }
        """)
        
        # Left panel - Camera and Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Trajectory and Stats
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions (40% left, 60% right)
        splitter.setSizes([600, 900])
        splitter.setStretchFactor(0, 2)  # Left panel
        splitter.setStretchFactor(1, 3)  # Right panel
        
        main_layout.addWidget(splitter)
        
        # Create modern status bar
        self.create_modern_status_bar()
        
        # Create modern menu bar
        self.create_modern_menu_bar()
          # Apply modern theme
        self.apply_modern_theme()
    
    def create_left_panel(self) -> QWidget:
        """Create left panel with camera and controls"""
        # Create scroll area for left panel
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setMaximumWidth(600)
        left_scroll.setMinimumWidth(500)
        
        # Style the scroll area
        left_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #5a5a5a;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #6a6a6a;
            }
            QScrollBar:horizontal {
                background-color: #2d2d2d;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #5a5a5a;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #6a6a6a;
            }
        """)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(8)
        left_layout.setContentsMargins(8, 8, 8, 8)
        
        # Camera section
        camera_group = QGroupBox("📹 Camera Feed")
        camera_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        camera_group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                border: 2px solid #404040;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px 0 8px;
                color: #4fc3f7;
                background-color: #252525;
            }
        """)
        
        camera_layout = QVBoxLayout(camera_group)
        self.camera_widget = CameraWidget()
        # Set minimum size for camera widget
        self.camera_widget.setMinimumSize(480, 360)
        camera_layout.addWidget(self.camera_widget)
        
        left_layout.addWidget(camera_group)
        
        # Controls section
        controls_group = QGroupBox("🎮 Control Panel")
        controls_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        controls_group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                border: 2px solid #404040;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px 0 8px;
                color: #66bb6a;
                background-color: #252525;
            }
        """)
        
        controls_layout = QVBoxLayout(controls_group)
        self.control_panel = ControlPanel()
        controls_layout.addWidget(self.control_panel)
        
        left_layout.addWidget(controls_group)        
        # Add stretch to prevent widgets from expanding unnecessarily
        left_layout.addStretch()
        
        # Set the widget in the scroll area
        left_scroll.setWidget(left_widget)
        
        return left_scroll
    
    def create_right_panel(self) -> QWidget:
        """Create right panel with trajectory and stats"""
        # Create scroll area for right panel
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        right_scroll.setMinimumWidth(800)
        
        # Style the scroll area
        right_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #5a5a5a;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #6a6a6a;
            }
            QScrollBar:horizontal {
                background-color: #2d2d2d;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #5a5a5a;
                border-radius: 6px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #6a6a6a;
            }
        """)
        
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(8)
        right_layout.setContentsMargins(8, 8, 8, 8)
        
        # Trajectory section
        trajectory_group = QGroupBox("🗺️ 3D Trajectory View")
        trajectory_group.setFont(QFont("Segoe UI", 11, QFont.Bold))
        trajectory_group.setStyleSheet("""
            QGroupBox {
                color: #ffffff;
                border: 2px solid #404040;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 12px;
                background-color: #252525;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px 0 8px;
                color: #ff9800;
                background-color: #252525;
            }
        """)
        
        trajectory_layout = QVBoxLayout(trajectory_group)
        self.trajectory_widget = TrajectoryWidget()
        # Set minimum size for trajectory widget
        self.trajectory_widget.setMinimumSize(600, 400)
        trajectory_layout.addWidget(self.trajectory_widget)
        
        right_layout.addWidget(trajectory_group)
        
        # Statistics and Debug panel in tabs for better organization
        info_tabs = QTabWidget()
        info_tabs.setStyleSheet("""
            QTabWidget {
                background-color: #252525;
                border-radius: 8px;
            }
            QTabWidget::pane {
                border: 2px solid #404040;
                border-radius: 8px;
                background-color: #252525;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #4fc3f7;
                color: #000000;
            }
            QTabBar::tab:hover {
                background-color: #3d3d3d;
            }
        """)
        
        # Statistics tab
        self.stats_widget = ModernStatsWidget()
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setWidget(self.stats_widget)
        stats_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        info_tabs.addTab(stats_scroll, "📊 Statistics")
        
        # Debug tab
        self.debug_widget = ModernDebugWidget()
        debug_scroll = QScrollArea()
        debug_scroll.setWidgetResizable(True)
        debug_scroll.setWidget(self.debug_widget)
        debug_scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        info_tabs.addTab(debug_scroll, "🐛 Debug Log")        
        right_layout.addWidget(info_tabs)
        
        # Set the widget in the scroll area
        right_scroll.setWidget(right_widget)
        
        return right_scroll
    
    def create_modern_status_bar(self):
        """Create modern status bar"""
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #1e1e1e;
                border-top: 2px solid #404040;
                color: #cccccc;
                font-weight: bold;
            }
        """)
        
        # Camera status with icon
        self.camera_status_label = QLabel("📷 Camera: Disconnected")
        self.camera_status_label.setStyleSheet("color: #f44336; margin: 4px;")
        self.status_bar.addWidget(self.camera_status_label)
        
        # Processing status
        self.processing_status_label = QLabel("⚙️ Processing: Stopped")
        self.processing_status_label.setStyleSheet("color: #ff9800; margin: 4px;")
        self.status_bar.addWidget(self.processing_status_label)
        
        # Separator
        separator = QLabel("|")
        separator.setStyleSheet("color: #666666; margin: 4px;")
        self.status_bar.addPermanentWidget(separator)
        
        # FPS counter
        self.fps_label = QLabel("⏱️ FPS: 0.0")
        self.fps_label.setStyleSheet("color: #4fc3f7; margin: 4px;")
        self.status_bar.addPermanentWidget(self.fps_label)
        
        # Frame counter
        self.frame_counter_label = QLabel("🎬 Frames: 0")
        self.frame_counter_label.setStyleSheet("color: #66bb6a; margin: 4px;")
        self.status_bar.addPermanentWidget(self.frame_counter_label)
        
        # Memory usage
        self.memory_label = QLabel("💾 Memory: 0 MB")
        self.memory_label.setStyleSheet("color: #ab47bc; margin: 4px;")
        self.status_bar.addPermanentWidget(self.memory_label)
    
    def create_modern_menu_bar(self):
        """Create modern menu bar"""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #1e1e1e;
                color: #ffffff;
                border-bottom: 2px solid #404040;
                font-weight: bold;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 12px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background-color: #4fc3f7;
                color: #000000;
            }
            QMenu {
                background-color: #2d2d2d;
                border: 2px solid #404040;
                border-radius: 8px;
                color: #ffffff;
                font-weight: bold;
            }
            QMenu::item {
                padding: 8px 16px;
                border-radius: 4px;
                margin: 2px;
            }
            QMenu::item:selected {
                background-color: #4fc3f7;
                color: #000000;
            }
        """)
        
        # File menu
        file_menu = menubar.addMenu('📁 File')
        
        export_action = QAction('💾 Export Data', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)
        
        export_debug_action = QAction('📋 Export Debug Log', self)
        export_debug_action.triggered.connect(self.export_debug_log)
        file_menu.addAction(export_debug_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('🚪 Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('🔧 Tools')
        
        reset_action = QAction('🔄 Reset Tracking', self)
        reset_action.triggered.connect(self.reset_tracking)
        tools_menu.addAction(reset_action)
        
        camera_info_action = QAction('📷 Camera Info', self)
        camera_info_action.triggered.connect(self.show_camera_info)
        tools_menu.addAction(camera_info_action)
        
        # Help menu
        help_menu = menubar.addMenu('❓ Help')
        
        about_action = QAction('ℹ️ About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def apply_modern_theme(self):
        """Apply modern dark theme with high contrast"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            
            QWidget {
                background-color: #1a1a1a;
                color: #ffffff;
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
            
            QLabel {
                color: #ffffff;
            }
            
            QPushButton {
                background-color: #2d2d2d;
                border: 2px solid #404040;
                border-radius: 8px;
                padding: 10px 20px;
                color: #ffffff;
                font-weight: bold;
                font-size: 11px;
                min-height: 20px;
            }
            
            QPushButton:hover {
                background-color: #404040;
                border-color: #4fc3f7;
                transform: translateY(-1px);
            }
            
            QPushButton:pressed {
                background-color: #1e1e1e;
                transform: translateY(1px);
            }
            
            QPushButton:disabled {
                background-color: #1e1e1e;
                color: #666666;
                border-color: #333333;
            }
            
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                border-radius: 6px;
                margin: 0;
            }
            
            QScrollBar::handle:vertical {
                background-color: #5a5a5a;
                border-radius: 6px;
                min-height: 20px;
                margin: 0;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #6a6a6a;
            }
            
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
                background: none;
            }
        """)
    
    def connect_signals(self):
        """Connect widget signals"""
        self.control_panel.start_clicked.connect(self.start_tracking)
        self.control_panel.stop_clicked.connect(self.stop_tracking)
        self.control_panel.reset_clicked.connect(self.reset_tracking)
        self.control_panel.export_clicked.connect(self.export_data)
        self.control_panel.calibrate_clicked.connect(self.calibrate_camera)
    
    def start_tracking(self):
        """Start visual odometry tracking"""
        if self.is_tracking:
            return
        
        self.debug_widget.add_debug_info("Initializing tracking system...", "INFO")
        
        try:
            # Start camera
            if not self.camera_manager.start_streaming():
                error_msg = "Failed to start camera"
                self.debug_widget.add_debug_info(error_msg, "ERROR")
                QMessageBox.critical(self, "Error", error_msg)
                return
            
            self.debug_widget.add_debug_info("Camera started successfully", "SUCCESS")
            
            # Get camera intrinsics
            color_matrix, depth_matrix = self.camera_manager.get_camera_intrinsics()
            if color_matrix is None:
                error_msg = "Failed to get camera parameters"
                self.debug_widget.add_debug_info(error_msg, "ERROR")
                QMessageBox.critical(self, "Error", error_msg)
                return
            
            # Initialize visual odometry
            settings = self.control_panel.get_settings()
            self.visual_odometry = VisualOdometry(
                camera_matrix=color_matrix,
                depth_scale=self.camera_manager.get_depth_scale(),
                detector_type=settings['feature_detector'],
                max_features=settings['max_features']
            )
            
            self.debug_widget.add_debug_info(f"Visual odometry initialized: {settings['feature_detector']}, {settings['max_features']} features", "SUCCESS")
            
            # Initialize SLAM processor
            self.slam_processor = SLAMProcessor(self.visual_odometry)
            
            # Reset trajectory tracker
            self.trajectory_tracker.reset()
            
            # Start processing
            self.is_tracking = True
            self.processing_thread = threading.Thread(target=self.enhanced_processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Start GUI updates
            self.update_timer.start(50)  # 20 FPS
            
            # Update UI
            self.control_panel.set_tracking_state(True)
            self.camera_status_label.setText("📷 Camera: Connected")
            self.camera_status_label.setStyleSheet("color: #66bb6a; margin: 4px;")
            self.processing_status_label.setText("⚙️ Processing: Running")
            self.processing_status_label.setStyleSheet("color: #66bb6a; margin: 4px;")
            
            self.debug_widget.add_debug_info("Tracking started successfully!", "SUCCESS")
            
        except Exception as e:
            error_msg = f"Failed to start tracking: {e}"
            self.debug_widget.add_debug_info(error_msg, "ERROR")
            QMessageBox.critical(self, "Error", error_msg)
    
    def enhanced_processing_loop(self):
        """Enhanced processing loop"""
        frame_count = 0
        start_time = time.time()
        
        while self.is_tracking:
            try:
                # Get frame
                frame_data = self.camera_manager.get_frame()
                if frame_data is None:
                    time.sleep(0.001)
                    continue
                
                # Process with visual odometry
                vo_result = self.visual_odometry.process_frame(frame_data)
                
                # Process with SLAM
                slam_result = self.slam_processor.process_frame(frame_data, vo_result)
                
                # Update trajectory
                trajectory_result = self.trajectory_tracker.add_point(
                    slam_result['pose'], frame_data.timestamp
                )
                
                # Thread-safe update
                with self.processing_lock:
                    self.current_frame_data = frame_data
                    self.current_vo_result = vo_result
                    self.current_slam_result = slam_result
                
                frame_count += 1
                
                # Log important events
                if vo_result.get('pose_updated', False):
                    distance = slam_result.get('distance_traveled', 0)
                    self.debug_widget.add_debug_info(f"Pose updated - Distance: {distance:.3f}m", "DEBUG")
                elif frame_count % 30 == 0:  # Log every 30 frames if no pose update
                    quality = vo_result.get('tracking_quality', 'Unknown')
                    self.debug_widget.add_debug_info(f"Tracking quality: {quality}", "WARNING")
                
                time.sleep(0.001)
                
            except Exception as e:
                self.debug_widget.add_debug_info(f"Processing error: {e}", "ERROR")
                time.sleep(0.1)
    
    def update_gui(self):
        """Update GUI with current data"""
        try:
            with self.processing_lock:
                if self.current_frame_data is None:
                    return
                
                frame_data = self.current_frame_data
                vo_result = self.current_vo_result
                slam_result = self.current_slam_result
            
            # Update camera display
            self.camera_widget.update_frame(frame_data.color_image, frame_data.depth_image)
            
            # Update trajectory
            trajectory_points = self.trajectory_tracker.get_recent_trajectory(1000)
            if len(trajectory_points) > 1:
                current_pos = trajectory_points[-1]
                self.trajectory_widget.update_trajectory(trajectory_points, current_pos)
            
            # Update statistics
            if slam_result:
                self.update_modern_statistics(slam_result, vo_result)
            
            # Update memory usage
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.memory_label.setText(f"💾 Memory: {memory_mb:.0f} MB")
            except:
                pass
                
        except Exception as e:
            self.debug_widget.add_debug_info(f"GUI update error: {e}", "ERROR")
    
    def update_modern_statistics(self, slam_result: dict, vo_result: dict):
        """Update modern statistics display"""
        try:
            # Position
            pos = slam_result['position']
            self.stats_widget.update_stat('position', f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
            
            # Distance
            distance = slam_result['distance_traveled']
            self.stats_widget.update_stat('distance', f"{distance:.3f} m", "#4fc3f7")
            
            # Speed
            stats = self.trajectory_tracker.get_statistics()
            speed = stats['current_speed']
            speed_color = "#66bb6a" if speed > 0.01 else "#ff9800"
            self.stats_widget.update_stat('speed', f"{speed:.3f} m/s", speed_color)
            
            # Quality
            quality = vo_result.get('tracking_quality', 'Unknown')
            quality_colors = {
                'Excellent': '#66bb6a',
                'Good': '#4fc3f7', 
                'Fair': '#ff9800',
                'Poor': '#f44336'
            }
            quality_color = quality_colors.get(quality.split(' - ')[0], '#cccccc')
            self.stats_widget.update_stat('quality', quality, quality_color)
            
            # FPS
            if hasattr(self.camera_manager, 'get_fps'):
                fps = self.camera_manager.get_fps()
                self.stats_widget.update_stat('fps', f"{fps:.1f}")
            
            # Technical details
            if hasattr(self.visual_odometry, 'feature_tracker'):
                features = self.visual_odometry.feature_tracker.total_features_detected
                matches = self.visual_odometry.feature_tracker.total_matches_found
                
                self.stats_widget.update_stat('features', str(features))
                self.stats_widget.update_stat('matches', str(matches))
                
                # Update progress bars
                self.stats_widget.features_progress.setValue(min(features, 1000))
                self.stats_widget.matches_progress.setValue(min(matches, 500))
            
            # Success rate
            if hasattr(self.visual_odometry, 'get_statistics'):
                vo_stats = self.visual_odometry.get_statistics()
                success_rate = vo_stats['success_rate']
                self.stats_widget.update_stat('success_rate', f"{success_rate:.1%}")
            
            # Trajectory points
            trajectory_points = len(self.trajectory_tracker.trajectory_points)
            self.stats_widget.update_stat('trajectory_points', str(trajectory_points))
            
        except Exception as e:
            self.debug_widget.add_debug_info(f"Statistics update error: {e}", "ERROR")
    
    def stop_tracking(self):
        """Stop tracking"""
        if not self.is_tracking:
            return
        
        self.debug_widget.add_debug_info("Stopping tracking...", "INFO")
        
        self.is_tracking = False
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        self.camera_manager.stop_streaming()
        self.update_timer.stop()
        
        # Update UI
        self.control_panel.set_tracking_state(False)
        self.camera_status_label.setText("📷 Camera: Disconnected")
        self.camera_status_label.setStyleSheet("color: #f44336; margin: 4px;")
        self.processing_status_label.setText("⚙️ Processing: Stopped")
        self.processing_status_label.setStyleSheet("color: #ff9800; margin: 4px;")
        
        self.debug_widget.add_debug_info("Tracking stopped", "SUCCESS")
    
    def reset_tracking(self):
        """Reset tracking system"""
        was_tracking = self.is_tracking
        
        self.debug_widget.add_debug_info("Resetting tracking system...", "INFO")
        
        if was_tracking:
            self.stop_tracking()
        
        # Reset components
        if self.visual_odometry:
            self.visual_odometry.reset()
        if self.slam_processor:
            self.slam_processor.reset()
        
        self.trajectory_tracker.reset()
        self.trajectory_widget.clear_trajectory()
        self.camera_widget.clear_display()
        
        # Reset stats
        for key in self.stats_widget.stats_labels.keys():
            self.stats_widget.update_stat(key, "--")
        
        self.stats_widget.features_progress.setValue(0)
        self.stats_widget.matches_progress.setValue(0)
        
        if was_tracking:
            self.start_tracking()
        
        self.debug_widget.add_debug_info("Reset complete", "SUCCESS")
    
    def export_data(self):
        """Export data"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Data",
                f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "JSON files (*.json);;CSV files (*.csv)"
            )
            
            if filename:
                if filename.endswith('.json'):
                    self.trajectory_tracker.export_trajectory(filename)
                else:
                    self.data_exporter.export_to_csv(
                        self.trajectory_tracker.trajectory_points, filename
                    )
                
                self.debug_widget.add_debug_info(f"Data exported: {filename}", "SUCCESS")
                QMessageBox.information(self, "Export Complete", f"Data exported to {filename}")
                
        except Exception as e:
            self.debug_widget.add_debug_info(f"Export error: {e}", "ERROR")
    
    def export_debug_log(self):
        """Export debug log"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Debug Log",
                f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "Text files (*.txt)"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.debug_widget.debug_text.toPlainText())
                QMessageBox.information(self, "Export Complete", f"Debug log exported to {filename}")
                
        except Exception as e:
            self.debug_widget.add_debug_info(f"Debug export error: {e}", "ERROR")
    
    def calibrate_camera(self):
        """Camera calibration placeholder"""
        QMessageBox.information(self, "Camera Calibration", "Camera calibration feature coming soon!")
    
    def show_camera_info(self):
        """Show camera information"""
        if self.camera_manager.is_streaming:
            color_matrix, _ = self.camera_manager.get_camera_intrinsics()
            info = f"""
Camera Information:

Intrinsics:
fx: {color_matrix[0,0]:.2f}
fy: {color_matrix[1,1]:.2f}
cx: {color_matrix[0,2]:.2f}
cy: {color_matrix[1,2]:.2f}

Resolution: {self.camera_manager.width}x{self.camera_manager.height}
FPS: {self.camera_manager.get_fps():.1f}
            """
            QMessageBox.information(self, "Camera Info", info)
        else:
            QMessageBox.information(self, "Camera Info", "Camera not connected")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About", """
<h3>RealSense D435i Visual Odometry</h3>
<p><b>Enhanced GUI Version for Mr-Parth24</b></p>
<p><b>Date:</b> 2025-06-17 18:21:25 UTC</p>
<p>Modern, high-contrast interface for better visibility</p>
        """)
    
    def closeEvent(self, event):
        """Handle close event"""
        if self.is_tracking:
            self.stop_tracking()
        
        self.config.save_config()
        event.accept()