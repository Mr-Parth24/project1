"""
Enhanced Main Application with Live Camera Feed and 3D Markers
Author: Mr-Parth24
Date: 2025-06-13
Time: 20:52:08 UTC
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from typing import Optional
import cv2
import numpy as np
import os
import json

from enhanced_camera_integration import EnhancedRealCameraManager
from enhanced_feature_tracker import AdvancedFeatureTracker
from enhanced_pose_estimator import PrecisePoseEstimator
from live_feed_visualizer import LiveFeedVisualizer

class LiveVisualOdometrySystem:
    """Complete Visual Odometry System with Live Feed and 3D Markers"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎥 Live Visual Odometry System - Intel RealSense D435i [ENHANCED]")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Enhanced system components
        self.camera_manager = EnhancedRealCameraManager()
        self.feature_tracker = AdvancedFeatureTracker()
        self.pose_estimator = PrecisePoseEstimator()
        self.live_visualizer = LiveFeedVisualizer()
        
        # System state
        self.is_running = False
        self.processing_thread = None
        self.live_feed_active = False
        
        # Enhanced statistics
        self.stats = {
            'fps': 0.0,
            'features': 0,
            'matches': 0,
            'total_distance': 0.0,
            'displacement_from_start': 0.0,
            'current_speed': 0.0,
            'direction_angle': 0.0,
            'frame_count': 0,
            'quality_score': 0.0,
            'tracking_confidence': 0.0,
            'x_displacement': 0.0,
            'y_displacement': 0.0,
            'z_displacement': 0.0,
            'session_time': 0.0
        }
        
        # Session management
        self.session_start_time = time.time()
        self.session_data = []
        self.auto_save_enabled = False
        
        # Create enhanced GUI
        self._create_enhanced_gui()
        
        # Start updates
        self._schedule_updates()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        print("🚀 Live Visual Odometry System initialized!")
        print("📺 Enhanced live camera feed with 3D markers")
        print("📏 Precise distance tracking without IMU/GPS")
        print("🎯 Real-time feature visualization")
    
    def _create_enhanced_gui(self):
        """Create enhanced GUI with modern styling"""
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 24, 'bold'), foreground='#ffffff', background='#2b2b2b')
        style.configure('Heading.TLabel', font=('Arial', 12, 'bold'), foreground='#ffffff', background='#3b3b3b')
        style.configure('Status.TLabel', font=('Arial', 10), foreground='#00ff00', background='#2b2b2b')
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Enhanced title section
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(title_frame, 
                               text="🎥 Live Visual Odometry System with 3D Markers",
                               style='Title.TLabel')
        title_label.pack(side=tk.LEFT)
        
        # Status indicators frame
        status_frame = ttk.Frame(title_frame)
        status_frame.pack(side=tk.RIGHT)
        
        self.system_status = tk.Canvas(status_frame, width=35, height=35, bg='#2b2b2b', highlightthickness=0)
        self.system_status.pack(side=tk.LEFT, padx=(15, 8))
        self.system_status.create_oval(2, 2, 33, 33, fill="red", outline="darkred", width=3)
        
        self.status_text = ttk.Label(status_frame, text="OFFLINE", 
                                   font=("Arial", 14, "bold"), foreground="red", background='#2b2b2b')
        self.status_text.pack(side=tk.LEFT)
        
        # Enhanced control panel
        control_frame = ttk.LabelFrame(main_frame, text="🎮 System Controls", padding="15")
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Main controls row
        controls_row1 = ttk.Frame(control_frame)
        controls_row1.pack(fill=tk.X, pady=(0, 10))
        
        self.start_btn = ttk.Button(controls_row1, text="🎥 Start Live Tracking", 
                                   command=self._start_system, width=20)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_btn = ttk.Button(controls_row1, text="⏹️ Stop System", 
                                  command=self._stop_system, state="disabled", width=15)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.reset_btn = ttk.Button(controls_row1, text="🔄 Reset Origin", 
                                   command=self._reset_origin, width=15)
        self.reset_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.live_feed_btn = ttk.Button(controls_row1, text="📺 Open Live Feed", 
                                       command=self._toggle_live_feed, width=15)
        self.live_feed_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Display options row
        controls_row2 = ttk.Frame(control_frame)
        controls_row2.pack(fill=tk.X, pady=(5, 0))
        
        self.show_3d_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_row2, text="📐 3D Coordinate System", 
                       variable=self.show_3d_var).pack(side=tk.LEFT, padx=(0, 15))
        
        self.show_features_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_row2, text="🎯 Feature Points", 
                       variable=self.show_features_var).pack(side=tk.LEFT, padx=(0, 15))
        
        self.show_motion_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_row2, text="➡️ Motion Vectors", 
                       variable=self.show_motion_var).pack(side=tk.LEFT, padx=(0, 15))
        
        self.auto_save_var = tk.BooleanVar()
        ttk.Checkbutton(controls_row2, text="💾 Auto-save Data", 
                       variable=self.auto_save_var, command=self._toggle_auto_save).pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Button(controls_row2, text="📊 Export Session", 
                  command=self._export_session_data, width=15).pack(side=tk.LEFT)
        
        # Enhanced metrics panel
        metrics_frame = ttk.LabelFrame(main_frame, text="📊 Real-time Performance & Position Metrics", padding="15")
        metrics_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Performance metrics section
        perf_frame = ttk.Frame(metrics_frame)
        perf_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(perf_frame, text="Performance:", style='Heading.TLabel').pack(anchor=tk.W)
        
        perf_grid = ttk.Frame(perf_frame)
        perf_grid.pack(fill=tk.X, pady=(5, 0))
        
        # Performance metrics
        ttk.Label(perf_grid, text="FPS:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.fps_label = ttk.Label(perf_grid, text="0.0", foreground="blue", font=("Arial", 12, "bold"))
        self.fps_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 25))
        
        ttk.Label(perf_grid, text="Features:", font=("Arial", 10, "bold")).grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.features_label = ttk.Label(perf_grid, text="0", foreground="green", font=("Arial", 11))
        self.features_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 25))
        
        ttk.Label(perf_grid, text="Matches:", font=("Arial", 10, "bold")).grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.matches_label = ttk.Label(perf_grid, text="0", foreground="orange", font=("Arial", 11))
        self.matches_label.grid(row=0, column=5, sticky=tk.W, padx=(0, 25))
        
        ttk.Label(perf_grid, text="Quality:", font=("Arial", 10, "bold")).grid(row=0, column=6, sticky=tk.W, padx=(0, 5))
        self.quality_label = ttk.Label(perf_grid, text="0.00", foreground="purple", font=("Arial", 11))
        self.quality_label.grid(row=0, column=7, sticky=tk.W)
        
        # Position metrics section
        pos_frame = ttk.Frame(metrics_frame)
        pos_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(pos_frame, text="Position & Movement:", style='Heading.TLabel').pack(anchor=tk.W)
        
        pos_grid = ttk.Frame(pos_frame)
        pos_grid.pack(fill=tk.X, pady=(5, 0))
        
        # Distance metrics
        ttk.Label(pos_grid, text="Total Distance:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.distance_label = ttk.Label(pos_grid, text="0.00 m", foreground="red", font=("Arial", 14, "bold"))
        self.distance_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 25))
        
        ttk.Label(pos_grid, text="From Start:", font=("Arial", 10, "bold")).grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.displacement_label = ttk.Label(pos_grid, text="0.00 m", foreground="darkred", font=("Arial", 13, "bold"))
        self.displacement_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 25))
        
        ttk.Label(pos_grid, text="Speed:", font=("Arial", 10, "bold")).grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.speed_label = ttk.Label(pos_grid, text="0.00 m/s", foreground="darkblue", font=("Arial", 11))
        self.speed_label.grid(row=0, column=5, sticky=tk.W, padx=(0, 25))
        
        ttk.Label(pos_grid, text="Direction:", font=("Arial", 10, "bold")).grid(row=0, column=6, sticky=tk.W, padx=(0, 5))
        self.direction_label = ttk.Label(pos_grid, text="0° N", foreground="darkgreen", font=("Arial", 11))
        self.direction_label.grid(row=0, column=7, sticky=tk.W)
        
        # Coordinate details
        coord_grid = ttk.Frame(pos_frame)
        coord_grid.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(coord_grid, text="Coordinates (m):", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, padx=(0, 15))
        
        ttk.Label(coord_grid, text="X:", font=("Arial", 10)).grid(row=0, column=1, sticky=tk.W, padx=(0, 5))
        self.x_label = ttk.Label(coord_grid, text="0.000", foreground="red", font=("Arial", 10, "bold"))
        self.x_label.grid(row=0, column=2, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(coord_grid, text="Y:", font=("Arial", 10)).grid(row=0, column=3, sticky=tk.W, padx=(0, 5))
        self.y_label = ttk.Label(coord_grid, text="0.000", foreground="green", font=("Arial", 10, "bold"))
        self.y_label.grid(row=0, column=4, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(coord_grid, text="Z:", font=("Arial", 10)).grid(row=0, column=5, sticky=tk.W, padx=(0, 5))
        self.z_label = ttk.Label(coord_grid, text="0.000", foreground="blue", font=("Arial", 10, "bold"))
        self.z_label.grid(row=0, column=6, sticky=tk.W, padx=(0, 20))
        
        ttk.Label(coord_grid, text="Session:", font=("Arial", 10)).grid(row=0, column=7, sticky=tk.W, padx=(0, 5))
        self.session_time_label = ttk.Label(coord_grid, text="00:00", foreground="purple", font=("Arial", 10, "bold"))
        self.session_time_label.grid(row=0, column=8, sticky=tk.W)
        
        # Enhanced camera info panel
        camera_frame = ttk.LabelFrame(main_frame, text="📹 Camera Information & Live Feed Status", padding="15")
        camera_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.camera_info_text = tk.Text(camera_frame, height=5, wrap=tk.WORD, 
                                       font=("Consolas", 10), bg='#1e1e1e', fg='#ffffff',
                                       insertbackground='white')
        camera_scrollbar = ttk.Scrollbar(camera_frame, orient="vertical", command=self.camera_info_text.yview)
        self.camera_info_text.configure(yscrollcommand=camera_scrollbar.set)
        
        self.camera_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        camera_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.camera_info_text.insert(tk.END, "🔴 Camera System Status: OFFLINE\n")
        self.camera_info_text.insert(tk.END, "📋 Ready for Intel RealSense D435i connection\n")
        self.camera_info_text.insert(tk.END, "🎯 Enhanced tracking with 3D visualization ready\n")
        self.camera_info_text.insert(tk.END, "📺 Live feed with real-time overlays available\n")
        self.camera_info_text.insert(tk.END, "🚀 Click 'Start Live Tracking' to begin!")
        self.camera_info_text.config(state=tk.DISABLED)
        
        # Enhanced trajectory display
        trajectory_frame = ttk.LabelFrame(main_frame, text="🗺️ Live 3D Trajectory Visualization (Top-Down View)", padding="15")
        trajectory_frame.pack(fill=tk.BOTH, expand=True)
        
        # Trajectory canvas with enhanced styling
        canvas_frame = ttk.Frame(trajectory_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.trajectory_canvas = tk.Canvas(canvas_frame, bg="#000000", height=280, 
                                         highlightthickness=2, highlightbackground="#555555",
                                         cursor="crosshair")
        self.trajectory_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Instructions panel
        instructions_frame = ttk.LabelFrame(main_frame, text="📋 Operating Instructions & Controls", padding="10")
        instructions_frame.pack(fill=tk.X, pady=(15, 0))
        
        instructions_text = ("🔹 Connect Intel RealSense D435i camera  "
                           "🔹 Click 'Start Live Tracking' to begin  "
                           "🔹 Click 'Open Live Feed' to see camera view with 3D markers  "
                           "🔹 Move camera smoothly to track trajectory  "
                           "🔹 Green dot = current position, Red dot = start position  "
                           "🔹 Use keyboard shortcuts in live feed: F=features, 3=3D markers, M=motion vectors")
        
        ttk.Label(instructions_frame, text=instructions_text, font=("Arial", 10), 
                 wraplength=1200, justify=tk.LEFT).pack()
    
    def _start_system(self):
        """Enhanced system startup"""
        try:
            self.start_btn.config(state="disabled", text="🔄 Initializing System...")
            self.status_text.config(text="INITIALIZING", foreground="orange")
            
            # Reset session
            self.session_start_time = time.time()
            self.session_data.clear()
            
            # Initialize in separate thread
            init_thread = threading.Thread(target=self._initialize_enhanced_system, daemon=True)
            init_thread.start()
            
        except Exception as e:
            messagebox.showerror("Startup Error", f"Failed to start system: {str(e)}")
            self._reset_buttons()
    
    def _initialize_enhanced_system(self):
        """Enhanced system initialization"""
        try:
            max_retries = 3
            for attempt in range(max_retries):
                self.root.after(0, lambda: self._update_init_status(f"Camera init attempt {attempt + 1}/{max_retries}..."))
                
                success = self.camera_manager.initialize()
                if success:
                    break
                elif attempt < max_retries - 1:
                    time.sleep(2)
            
            if success:
                # Start camera capture
                self.camera_manager.start_capture()
                
                # Update GUI on main thread
                self.root.after(0, self._system_initialized_success)
                
                # Start enhanced processing
                self.is_running = True
                self.processing_thread = threading.Thread(target=self._enhanced_processing_loop, daemon=True)
                self.processing_thread.start()
                
            else:
                self.root.after(0, self._system_initialization_failed)
                
        except Exception as e:
            self.root.after(0, lambda: self._system_initialization_failed(str(e)))
    
    def _update_init_status(self, status):
        """Update initialization status"""
        self.start_btn.config(text=f"🔄 {status}")
    
    def _system_initialized_success(self):
        """Handle successful system initialization"""
        self.start_btn.config(text="🟢 System Active", state="disabled")
        self.stop_btn.config(state="normal")
        self.live_feed_btn.config(state="normal")
        self.status_text.config(text="TRACKING", foreground="green")
        
        # Animate status indicator
        self.system_status.delete("all")
        self.system_status.create_oval(2, 2, 33, 33, fill="lime", outline="darkgreen", width=3)
        
        # Enhanced camera info
        camera_info = self.camera_manager.get_enhanced_camera_info()
        info_text = f"""🎥 Camera: Intel RealSense D435i (CONNECTED & ACTIVE)
📏 Resolution: {camera_info.get('width', 0)}x{camera_info.get('height', 0)} @ {camera_info.get('fps', 0)}fps
🔧 Intrinsics: fx={camera_info.get('fx', 0):.1f}, fy={camera_info.get('fy', 0):.1f}
📐 Principal Point: cx={camera_info.get('cx', 0):.1f}, cy={camera_info.get('cy', 0):.1f}
✅ Status: ACTIVE - Enhanced 6DOF tracking enabled
🎯 Features: Advanced multi-method detection active
📺 Live Feed: Ready (click 'Open Live Feed' to view)
🕒 Session started: {time.strftime('%H:%M:%S')}
🚀 System ready for precise motion tracking!"""
        
        self.camera_info_text.config(state=tk.NORMAL)
        self.camera_info_text.delete(1.0, tk.END)
        self.camera_info_text.insert(tk.END, info_text)
        self.camera_info_text.config(state=tk.DISABLED)
        
        messagebox.showinfo("🎉 System Ready", 
                           "Enhanced Visual Odometry System initialized successfully!\n\n"
                           "✅ Intel RealSense D435i camera active\n"
                           "✅ Enhanced 6DOF tracking enabled\n"
                           "✅ Precise distance measurement ready\n"
                           "✅ Live feed with 3D markers available\n\n"
                           "🎯 Move the camera to begin tracking your trajectory!")
    
    def _enhanced_processing_loop(self):
        """Enhanced processing loop with live feed support"""
        last_time = time.time()
        frame_times = []
        
        while self.is_running:
            try:
                # Get frame from enhanced camera
                frame_data = self.camera_manager.get_frame()
                
                if frame_data is not None:
                    color_frame, depth_frame = frame_data
                    
                    # Enhanced tracking with 3D pose estimation
                    tracking_result = self.feature_tracker.process_frame_enhanced(
                        color_frame, depth_frame, self.camera_manager.camera_matrix
                    )
                    
                    # Calculate enhanced FPS
                    current_time = time.time()
                    frame_time = current_time - last_time
                    last_time = current_time
                    
                    frame_times.append(frame_time)
                    if len(frame_times) > 30:
                        frame_times.pop(0)
                    
                    avg_frame_time = sum(frame_times) / len(frame_times)
                    fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
                    
                    # Update enhanced statistics
                    self.stats.update({
                        'fps': fps,
                        'features': tracking_result['num_features'],
                        'matches': tracking_result['num_matches'],
                        'total_distance': tracking_result['total_distance'],
                        'displacement_from_start': tracking_result['displacement_from_start'],
                        'current_speed': tracking_result['current_speed'],
                        'direction_angle': tracking_result['direction_angle'],
                        'frame_count': self.camera_manager.frame_count,
                        'quality_score': tracking_result['quality_score'],
                        'tracking_confidence': tracking_result['tracking_confidence'],
                        'x_displacement': tracking_result['x_displacement'],
                        'y_displacement': tracking_result['y_displacement'],
                        'z_displacement': tracking_result['z_displacement'],
                        'session_time': current_time - self.session_start_time
                    })
                    
                    # Update live feed if active
                    if self.live_feed_active:
                        self.live_visualizer.update_display(
                            color_frame, depth_frame, tracking_result, 
                            self.camera_manager.camera_matrix
                        )
                    
                    # Save session data if enabled
                    if self.auto_save_enabled:
                        self._log_session_data(tracking_result, current_time)
                    
                    # Schedule GUI update
                    self.root.after(0, lambda: self._update_enhanced_trajectory_display(tracking_result['trajectory']))
                
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                print(f"Enhanced processing error: {e}")
                time.sleep(0.1)
    
    def _toggle_live_feed(self):
        """Toggle live camera feed window"""
        try:
            if not self.live_feed_active:
                success = self.live_visualizer.start_visualization()
                if success:
                    self.live_feed_active = True
                    self.live_feed_btn.config(text="📺 Close Live Feed")
                    messagebox.showinfo("Live Feed", 
                                       "Live camera feed opened!\n\n"
                                       "🎯 Features: F key to toggle\n"
                                       "📐 3D markers: 3 key to toggle\n"
                                       "➡️ Motion vectors: M key to toggle\n"
                                       "🗺️ Trajectory: T key to toggle\n"
                                       "🔄 Reset view: R key")
                else:
                    messagebox.showerror("Error", "Failed to start live feed visualization")
            else:
                self.live_visualizer.stop_visualization()
                self.live_feed_active = False
                self.live_feed_btn.config(text="📺 Open Live Feed")
                
        except Exception as e:
            messagebox.showerror("Live Feed Error", f"Error toggling live feed: {str(e)}")
    
    def _toggle_auto_save(self):
        """Toggle automatic data saving"""
        self.auto_save_enabled = self.auto_save_var.get()
        if self.auto_save_enabled:
            print("✅ Auto-save enabled - session data will be logged")
        else:
            print("❌ Auto-save disabled")
    
    def _log_session_data(self, tracking_result, timestamp):
        """Log session data for analysis"""
        data_point = {
            'timestamp': timestamp,
            'frame_count': self.stats['frame_count'],
            'position': tracking_result['current_position'],
            'total_distance': tracking_result['total_distance'],
            'displacement_from_start': tracking_result['displacement_from_start'],
            'speed': tracking_result['current_speed'],
            'direction': tracking_result['direction_angle'],
            'features': tracking_result['num_features'],
            'matches': tracking_result['num_matches'],
            'quality': tracking_result['quality_score'],
            'confidence': tracking_result['tracking_confidence']
        }
        self.session_data.append(data_point)
    
    def _update_enhanced_trajectory_display(self, trajectory):
        """Enhanced trajectory visualization"""
        try:
            self.trajectory_canvas.delete("all")
            
            if len(trajectory) < 2:
                return
            
            canvas_width = self.trajectory_canvas.winfo_width()
            canvas_height = self.trajectory_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            # Draw enhanced grid background
            self._draw_enhanced_grid(canvas_width, canvas_height)
            
            # Convert trajectory with enhanced scaling
            traj_array = np.array(trajectory)
            x_coords, z_coords = traj_array[:, 0], traj_array[:, 2]
            
            if len(x_coords) > 1:
                # Enhanced scaling and positioning
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                z_min, z_max = np.min(z_coords), np.max(z_coords)
                
                x_range = max(x_max - x_min, 0.1)
                z_range = max(z_max - z_min, 0.1)
                
                margin = 40
                scale_x = (canvas_width - 2 * margin) / x_range
                scale_z = (canvas_height - 2 * margin) / z_range
                scale = min(scale_x, scale_z)
                
                # Convert to canvas coordinates
                canvas_points = []
                for x, z in zip(x_coords, z_coords):
                    canvas_x = margin + (x - x_min) * scale
                    canvas_y = margin + (z - z_min) * scale
                    canvas_points.append((canvas_x, canvas_y))
                
                # Draw enhanced trajectory
                self._draw_enhanced_trajectory(canvas_points)
                
                # Draw enhanced position markers
                self._draw_enhanced_position_markers(canvas_points)
                
                # Draw scale and info
                self._draw_trajectory_info(canvas_width, canvas_height, x_range, z_range)
        
        except Exception as e:
            print(f"Enhanced trajectory display error: {e}")
    
    def _draw_enhanced_grid(self, width, height):
        """Draw enhanced grid background"""
        # Major grid lines
        major_spacing = 40
        for i in range(0, width, major_spacing):
            self.trajectory_canvas.create_line(i, 0, i, height, fill="#333333", width=1)
        for i in range(0, height, major_spacing):
            self.trajectory_canvas.create_line(0, i, width, i, fill="#333333", width=1)
        
        # Minor grid lines
        minor_spacing = 20
        for i in range(0, width, minor_spacing):
            self.trajectory_canvas.create_line(i, 0, i, height, fill="#1a1a1a", width=1)
        for i in range(0, height, minor_spacing):
            self.trajectory_canvas.create_line(0, i, width, i, fill="#1a1a1a", width=1)
        
        # Center lines
        center_x, center_y = width // 2, height // 2
        self.trajectory_canvas.create_line(center_x, 0, center_x, height, fill="#555555", width=2)
        self.trajectory_canvas.create_line(0, center_y, width, center_y, fill="#555555", width=2)
    
    def _draw_enhanced_trajectory(self, points):
        """Draw trajectory with gradient and enhanced styling"""
        if len(points) < 2:
            return
        
        total_points = len(points)
        for i in range(len(points) - 1):
            progress = i / max(1, total_points - 1)
            
            # Enhanced color gradient
            if progress < 0.33:
                # Blue to cyan
                r = 0
                g = int(255 * (progress * 3))
                b = 255
            elif progress < 0.67:
                # Cyan to yellow
                r = int(255 * ((progress - 0.33) * 3))
                g = 255
                b = int(255 * (1 - (progress - 0.33) * 3))
            else:
                # Yellow to red
                r = 255
                g = int(255 * (1 - (progress - 0.67) * 3))
                b = 0
            
            color = f"#{r:02x}{g:02x}{b:02x}"
            
            # Variable line width
            width = max(2, int(4 * (progress + 0.5)))
            
            self.trajectory_canvas.create_line(
                points[i][0], points[i][1],
                points[i+1][0], points[i+1][1],
                fill=color, width=width, capstyle=tk.ROUND
            )
    
    def _draw_enhanced_position_markers(self, points):
        """Draw enhanced position markers"""
        if not points:
            return
        
        # Start position with pulsing effect
        start_x, start_y = points[0]
        self.trajectory_canvas.create_oval(
            start_x-8, start_y-8, start_x+8, start_y+8,
            fill="red", outline="white", width=3
        )
        self.trajectory_canvas.create_oval(
            start_x-4, start_y-4, start_x+4, start_y+4,
            fill="darkred", outline="red", width=2
        )
        self.trajectory_canvas.create_text(
            start_x, start_y-25, text="🏁 START", fill="white", 
            font=("Arial", 12, "bold")
        )
        
        # Current position with enhanced styling
        curr_x, curr_y = points[-1]
        self.trajectory_canvas.create_oval(
            curr_x-10, curr_y-10, curr_x+10, curr_y+10,
            fill="lime", outline="white", width=4
        )
        self.trajectory_canvas.create_oval(
            curr_x-6, curr_y-6, curr_x+6, curr_y+6,
            fill="yellow", outline="lime", width=2
        )
        self.trajectory_canvas.create_oval(
            curr_x-2, curr_y-2, curr_x+2, curr_y+2,
            fill="white", outline="yellow", width=1
        )
        self.trajectory_canvas.create_text(
            curr_x, curr_y-30, text="📍 NOW", fill="lime", 
            font=("Arial", 12, "bold")
        )
        
        # Distance and direction info
        distance_text = f"{self.stats['total_distance']:.2f}m"
        direction_text = f"{self.stats['direction_angle']:.0f}°"
        
        self.trajectory_canvas.create_text(
            curr_x, curr_y+25, text=distance_text, fill="yellow", 
            font=("Arial", 11, "bold")
        )
        self.trajectory_canvas.create_text(
            curr_x, curr_y+40, text=direction_text, fill="cyan", 
            font=("Arial", 10, "bold")
        )
    
    def _draw_trajectory_info(self, width, height, x_range, z_range):
        """Draw trajectory information and scale"""
        # Scale indicator
        scale_length = 60
        scale_meters = max(x_range, z_range) * (scale_length / (width - 80))
        
        scale_x = width - 90
        scale_y = height - 40
        
        # Scale bar background
        self.trajectory_canvas.create_rectangle(
            scale_x - 5, scale_y - 15, scale_x + scale_length + 15, scale_y + 15,
            fill="#2a2a2a", outline="#666666", width=1
        )
        
        # Scale bar
        self.trajectory_canvas.create_line(
            scale_x, scale_y, scale_x + scale_length, scale_y,
            fill="white", width=4
        )
        self.trajectory_canvas.create_line(
            scale_x, scale_y - 5, scale_x, scale_y + 5,
            fill="white", width=2
        )
        self.trajectory_canvas.create_line(
            scale_x + scale_length, scale_y - 5, scale_x + scale_length, scale_y + 5,
            fill="white", width=2
        )
        
        self.trajectory_canvas.create_text(
            scale_x + scale_length/2, scale_y - 20,
            text=f"{scale_meters:.2f}m", fill="white", 
            font=("Arial", 10, "bold")
        )
        
        # Compass indicator
        compass_x, compass_y = 40, 40
        compass_radius = 25
        
        # Compass background
        self.trajectory_canvas.create_oval(
            compass_x - compass_radius, compass_y - compass_radius,
            compass_x + compass_radius, compass_y + compass_radius,
            fill="#2a2a2a", outline="#666666", width=2
        )
        
        # Compass directions
        directions = [("N", 0), ("E", 90), ("S", 180), ("W", 270)]
        for direction, angle in directions:
            angle_rad = math.radians(angle - 90)
            text_x = compass_x + int((compass_radius - 8) * math.cos(angle_rad))
            text_y = compass_y + int((compass_radius - 8) * math.sin(angle_rad))
            
            color = "red" if direction == "N" else "white"
            self.trajectory_canvas.create_text(
                text_x, text_y, text=direction, fill=color, 
                font=("Arial", 9, "bold")
            )
        
        # Current direction indicator
        direction_angle = self.stats['direction_angle']
        dir_rad = math.radians(direction_angle - 90)
        dir_x = compass_x + int((compass_radius - 12) * math.cos(dir_rad))
        dir_y = compass_y + int((compass_radius - 12) * math.sin(dir_rad))
        
        self.trajectory_canvas.create_line(
            compass_x, compass_y, dir_x, dir_y,
            fill="lime", width=3, arrow=tk.LAST, arrowshape=(8, 10, 3)
        )
    
    def _system_initialization_failed(self, error_msg="Unknown error"):
        """Handle system initialization failure"""
        self._reset_buttons()
        
        error_text = f"""🚨 Enhanced System Initialization Failed!

🔍 Advanced Troubleshooting:
1️⃣ Intel RealSense D435i camera connection
   • Check USB 3.0 connection (blue port recommended)
   • Try different USB ports and cables
   • Verify camera LED indicators

2️⃣ Driver and Software Issues
   • Install Intel RealSense SDK 2.0
   • Update camera firmware
   • Check Windows device manager

3️⃣ System Resources
   • Close other camera applications
   • Check USB bandwidth availability
   • Restart application as administrator

4️⃣ Hardware Verification
   • Test with Intel RealSense Viewer
   • Verify camera functionality
   • Check for hardware conflicts

📝 Technical Error Details: {error_msg}

💡 Additional Resources:
• Intel RealSense Support: https://support.intel.com/
• Driver Download: https://github.com/IntelRealSense/librealsense
• Documentation: https://dev.intelrealsense.com/

🔧 If issues persist, try running the system in compatibility mode."""
        
        messagebox.showerror("🚨 System Initialization Failed", error_text)
    
    def _stop_system(self):
        """Enhanced system stop with comprehensive cleanup"""
        try:
            self.is_running = False
            self.status_text.config(text="STOPPING", foreground="orange")
            
            # Stop live feed if active
            if self.live_feed_active:
                self.live_visualizer.stop_visualization()
                self.live_feed_active = False
            
            # Stop camera system
            self.camera_manager.stop_capture()
            
            # Wait for processing thread
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=3.0)
            
            # Calculate session summary
            session_duration = time.time() - self.session_start_time
            
            # Update GUI
            self._reset_buttons()
            
            # Update status
            self.system_status.delete("all")
            self.system_status.create_oval(2, 2, 33, 33, fill="red", outline="darkred", width=3)
            self.status_text.config(text="OFFLINE", foreground="red")
            
            # Enhanced session summary
            summary_text = f"""🔴 Enhanced Visual Odometry System - Session Complete

📊 Session Performance Summary:
   ⏱️  Duration: {session_duration:.1f} seconds ({session_duration/60:.1f} minutes)
   📏 Total Distance Traveled: {self.stats['total_distance']:.3f} meters
   📐 Displacement from Start: {self.stats['displacement_from_start']:.3f} meters
   🎞️  Total Frames Processed: {self.stats['frame_count']:,}
   📈 Average FPS: {self.stats['fps']:.1f}
   🎯 Average Features per Frame: {self.stats['features']}
   🔗 Average Matches per Frame: {self.stats['matches']}
   ⭐ Final Tracking Quality: {self.stats['quality_score']:.2f}
   🎪 Peak Speed Achieved: {self.stats['current_speed']:.3f} m/s

📍 Final Position Coordinates:
   🔴 X: {self.stats['x_displacement']:.3f} meters
   🟢 Y: {self.stats['y_displacement']:.3f} meters  
   🔵 Z: {self.stats['z_displacement']:.3f} meters

🧭 Final Direction: {self.stats['direction_angle']:.1f}° from start

💾 Session Data: {'Saved' if self.auto_save_enabled else 'Not saved'} ({len(self.session_data)} data points)

Click 'Start Live Tracking' to begin a new session."""
            
            self.camera_info_text.config(state=tk.NORMAL)
            self.camera_info_text.delete(1.0, tk.END)
            self.camera_info_text.insert(tk.END, summary_text)
            self.camera_info_text.config(state=tk.DISABLED)
            
            # Show completion message
            messagebox.showinfo("🎉 Session Complete", 
                               f"Enhanced Visual Odometry Session Completed!\n\n"
                               f"📊 Session Statistics:\n"
                               f"⏱️ Duration: {session_duration:.1f}s\n"
                               f"📏 Total Distance: {self.stats['total_distance']:.3f}m\n"
                               f"📐 Final Displacement: {self.stats['displacement_from_start']:.3f}m\n"
                               f"🎞️ Frames Processed: {self.stats['frame_count']:,}\n"
                               f"📈 Average FPS: {self.stats['fps']:.1f}\n\n"
                               f"Thank you for using the Enhanced Visual Odometry System!")
            
        except Exception as e:
            messagebox.showerror("Stop Error", f"Error stopping system: {str(e)}")
    
    def _reset_origin(self):
        """Reset coordinate origin and trajectory"""
        try:
            if self.is_running:
                # Reset tracking components
                self.feature_tracker.reset_trajectory()
                self.pose_estimator.reset()
                
                # Reset statistics
                self.stats.update({
                    'total_distance': 0.0,
                    'displacement_from_start': 0.0,
                    'current_speed': 0.0,
                    'direction_angle': 0.0,
                    'x_displacement': 0.0,
                    'y_displacement': 0.0,
                    'z_displacement': 0.0
                })
                
                # Clear trajectory display
                self.trajectory_canvas.delete("all")
                
                # Reset session data
                self.session_data.clear()
                self.session_start_time = time.time()
                
                messagebox.showinfo("🔄 Origin Reset", 
                                   "✅ Coordinate origin reset successfully!\n"
                                   "✅ Trajectory tracking restarted\n"
                                   "✅ All distance measurements reset\n"
                                   "✅ Session data cleared\n\n"
                                   "🎯 Current position is now the new origin (0,0,0)")
            else:
                messagebox.showwarning("⚠️ System Not Running", 
                                     "Please start the tracking system before resetting origin.")
                
        except Exception as e:
            messagebox.showerror("Reset Error", f"Error resetting origin: {str(e)}")
    
    def _export_session_data(self):
        """Export comprehensive session data"""
        try:
            if not self.session_data and not self.is_running:
                messagebox.showwarning("No Data", 
                                     "No session data available to export.\n"
                                     "Enable 'Auto-save Data' and run a tracking session first.")
                return
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_visual_odometry_session_{timestamp}.json"
            
            # Comprehensive export data
            export_data = {
                'session_info': {
                    'start_time': self.session_start_time,
                    'export_time': time.time(),
                    'duration_seconds': time.time() - self.session_start_time,
                    'system_version': 'Enhanced Visual Odometry v2.0',
                    'camera_model': 'Intel RealSense D435i',
                    'user': 'Mr-Parth24'
                },
                'session_summary': {
                    'total_distance': self.stats['total_distance'],
                    'displacement_from_start': self.stats['displacement_from_start'],
                    'final_position': [
                        self.stats['x_displacement'],
                        self.stats['y_displacement'], 
                        self.stats['z_displacement']
                    ],
                    'final_direction_degrees': self.stats['direction_angle'],
                    'total_frames': self.stats['frame_count'],
                    'average_fps': self.stats['fps'],
                    'final_quality_score': self.stats['quality_score'],
                    'final_tracking_confidence': self.stats['tracking_confidence']
                },
                'camera_info': self.camera_manager.get_enhanced_camera_info() if self.is_running else {},
                'trajectory_data': self.session_data,
                'settings': {
                    'auto_save_enabled': self.auto_save_enabled,
                    'live_feed_was_active': self.live_feed_active,
                    'show_3d_markers': self.show_3d_var.get(),
                    'show_features': self.show_features_var.get(),
                    'show_motion_vectors': self.show_motion_var.get()
                }
            }
            
            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            filepath = os.path.join("data", filename)
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            messagebox.showinfo("📊 Export Successful", 
                               f"Session data exported successfully!\n\n"
                               f"📁 File: {filename}\n"
                               f"📂 Location: {os.path.abspath(filepath)}\n"
                               f"📊 Data Points: {len(self.session_data)}\n"
                               f"📏 Total Distance: {self.stats['total_distance']:.3f}m\n"
                               f"⏱️ Session Duration: {(time.time() - self.session_start_time):.1f}s\n\n"
                               f"💡 You can analyze this data with external tools!")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export session data:\n{str(e)}")
    
    def _reset_buttons(self):
        """Reset button states to initial"""
        self.start_btn.config(text="🎥 Start Live Tracking", state="normal")
        self.stop_btn.config(state="disabled")
        self.live_feed_btn.config(text="📺 Open Live Feed", state="disabled")
    
    def _schedule_updates(self):
        """Schedule periodic GUI updates"""
        self._update_enhanced_statistics_display()
        self.root.after(100, self._schedule_updates)  # Update every 100ms
    
    def _update_enhanced_statistics_display(self):
        """Update enhanced statistics display"""
        try:
            # Performance metrics
            self.fps_label.config(text=f"{self.stats['fps']:.1f}")
            self.features_label.config(text=str(self.stats['features']))
            self.matches_label.config(text=str(self.stats['matches']))
            self.quality_label.config(text=f"{self.stats['quality_score']:.2f}")
            
            # Distance and position metrics
            self.distance_label.config(text=f"{self.stats['total_distance']:.3f} m")
            self.displacement_label.config(text=f"{self.stats['displacement_from_start']:.3f} m")
            self.speed_label.config(text=f"{self.stats['current_speed']:.3f} m/s")
            
            # Direction with compass text
            direction_text = self._get_direction_text(self.stats['direction_angle'])
            self.direction_label.config(text=f"{self.stats['direction_angle']:.0f}° {direction_text}")
            
            # Coordinate display
            self.x_label.config(text=f"{self.stats['x_displacement']:.3f}")
            self.y_label.config(text=f"{self.stats['y_displacement']:.3f}")
            self.z_label.config(text=f"{self.stats['z_displacement']:.3f}")
            
            # Session time
            if self.is_running:
                session_duration = time.time() - self.session_start_time
                minutes = int(session_duration // 60)
                seconds = int(session_duration % 60)
                self.session_time_label.config(text=f"{minutes:02d}:{seconds:02d}")
            
        except Exception as e:
            print(f"Statistics update error: {e}")
    
    def _get_direction_text(self, angle: float) -> str:
        """Convert angle to compass direction"""
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        index = int((angle + 22.5) / 45) % 8
        return directions[index]
    
    def _on_closing(self):
        """Enhanced application closing with data preservation"""
        try:
            if self.is_running:
                result = messagebox.askyesnocancel(
                    "🔄 Exit Enhanced Visual Odometry System", 
                    "🎥 The tracking system is currently active.\n\n"
                    "💾 Would you like to export your session data before exiting?\n\n"
                    "✅ YES - Export session data and exit\n"
                    "❌ NO - Exit without saving session data\n"
                    "🚫 CANCEL - Continue running the system"
                )
                
                if result is None:  # Cancel
                    return
                elif result:  # Yes - export data
                    if self.session_data or self.is_running:
                        self._export_session_data()
                    
                # Stop system and cleanup
                self._stop_system()
                time.sleep(1.5)  # Allow time for cleanup
            
            # Final cleanup
            if self.live_feed_active:
                self.live_visualizer.stop_visualization()
            
            print("🎯 Enhanced Visual Odometry System shutdown complete")
            print("Thank you for using our advanced tracking system!")
            
            self.root.quit()
            
        except Exception as e:
            print(f"Shutdown error: {e}")
            self.root.quit()
    
    def run(self):
        """Run the enhanced Visual Odometry System"""
        print("🚀 Starting Enhanced Visual Odometry System GUI...")
        self.root.mainloop()


def main():
    """Enhanced main application entry point"""
    try:
        print("=" * 60)
        print("🎥 Enhanced Visual Odometry System v2.0")
        print("📅 Current Date: 2025-06-13 20:56:39 UTC")
        print("👤 User: Mr-Parth24")
        print("=" * 60)
        print("🚀 Features:")
        print("   ✅ Real-time 6DOF camera tracking")
        print("   ✅ Live camera feed with 3D coordinate overlays")
        print("   ✅ Enhanced feature detection and matching")
        print("   ✅ Precise distance measurement (no IMU/GPS)")
        print("   ✅ Real-time direction and speed calculation")
        print("   ✅ Advanced trajectory visualization")
        print("   ✅ Session data export and analysis")
        print("   ✅ Intel RealSense D435i integration")
        print("=" * 60)
        
        # Create and run enhanced application
        app = LiveVisualOdometrySystem()
        app.run()
        
    except Exception as e:
        print(f"❌ Enhanced application failed to start: {e}")
        print("🔧 Please check your system requirements:")
        print("   • Intel RealSense D435i camera")
        print("   • Intel RealSense SDK 2.0")
        print("   • Python 3.8+ with required packages")
        print("   • USB 3.0 connection")
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()