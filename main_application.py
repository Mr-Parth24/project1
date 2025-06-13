"""
Enhanced Main Application with Real Camera Integration
Author: Mr-Parth24
Date: 2025-06-13
"""

import sys
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from typing import Optional
import cv2
import numpy as np

from camera_integration import RealCameraManager, SimpleFeatureTracker

class EnhancedVisualOdometryGUI:
    """Enhanced GUI with real camera integration"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Visual Odometry System - Intel RealSense D435i [LIVE]")
        self.root.geometry("1000x700")
        
        # System components
        self.camera_manager = RealCameraManager()
        self.feature_tracker = SimpleFeatureTracker()
        
        # System state
        self.is_running = False
        self.processing_thread = None
        
        # Statistics
        self.stats = {
            'fps': 0.0,
            'features': 0,
            'matches': 0,
            'distance': 0.0,
            'total_distance': 0.0,
            'frame_count': 0
        }
        
        # Create GUI
        self._create_enhanced_widgets()
        
        # Start status updates
        self._schedule_status_update()
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        print("Enhanced GUI initialized!")
    
    def _create_enhanced_widgets(self):
        """Create enhanced GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title and status
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(title_frame, text="Visual Odometry System - LIVE", 
                 font=("Arial", 16, "bold")).pack(side=tk.LEFT)
        
        self.status_indicator = tk.Canvas(title_frame, width=20, height=20)
        self.status_indicator.pack(side=tk.RIGHT)
        self.status_indicator.create_oval(2, 2, 18, 18, fill="red", outline="black")
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="System Control", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.start_button = ttk.Button(control_frame, text="Start Camera", 
                                      command=self._start_system)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop", 
                                     command=self._stop_system, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.reset_button = ttk.Button(control_frame, text="Reset Trajectory", 
                                      command=self._reset_trajectory)
        self.reset_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Statistics panel
        stats_frame = ttk.LabelFrame(main_frame, text="Real-time Statistics", padding="10")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create statistics grid
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X)
        
        # FPS
        ttk.Label(stats_grid, text="FPS:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.fps_label = ttk.Label(stats_grid, text="0.0", foreground="blue", font=("Arial", 10, "bold"))
        self.fps_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        # Features
        ttk.Label(stats_grid, text="Features:").grid(row=0, column=2, sticky=tk.W, padx=(0, 5))
        self.features_label = ttk.Label(stats_grid, text="0", foreground="green")
        self.features_label.grid(row=0, column=3, sticky=tk.W, padx=(0, 20))
        
        # Matches
        ttk.Label(stats_grid, text="Matches:").grid(row=0, column=4, sticky=tk.W, padx=(0, 5))
        self.matches_label = ttk.Label(stats_grid, text="0", foreground="orange")
        self.matches_label.grid(row=0, column=5, sticky=tk.W, padx=(0, 20))
        
        # Distance
        ttk.Label(stats_grid, text="Total Distance:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.distance_label = ttk.Label(stats_grid, text="0.00 m", foreground="red", font=("Arial", 10, "bold"))
        self.distance_label.grid(row=1, column=1, sticky=tk.W, padx=(0, 20))
        
        # Frame count
        ttk.Label(stats_grid, text="Frames:").grid(row=1, column=2, sticky=tk.W, padx=(0, 5))
        self.frames_label = ttk.Label(stats_grid, text="0", foreground="purple")
        self.frames_label.grid(row=1, column=3, sticky=tk.W, padx=(0, 20))
        
        # Camera info
        camera_frame = ttk.LabelFrame(main_frame, text="Camera Information", padding="10")
        camera_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.camera_info_text = tk.Text(camera_frame, height=4, wrap=tk.WORD)
        self.camera_info_text.pack(fill=tk.X)
        self.camera_info_text.insert(tk.END, "Camera not initialized. Click 'Start Camera' to begin.")
        self.camera_info_text.config(state=tk.DISABLED)
        
        # Trajectory display
        trajectory_frame = ttk.LabelFrame(main_frame, text="Live Trajectory (Top View)", padding="10")
        trajectory_frame.pack(fill=tk.BOTH, expand=True)
        
        self.trajectory_canvas = tk.Canvas(trajectory_frame, bg="black", height=200)
        self.trajectory_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions_frame = ttk.LabelFrame(main_frame, text="Instructions", padding="5")
        instructions_frame.pack(fill=tk.X, pady=(10, 0))
        
        instructions = "1. Connect Intel RealSense D435i camera  2. Click 'Start Camera'  3. Move camera to see trajectory  4. Green dot = current position"
        ttk.Label(instructions_frame, text=instructions, font=("Arial", 9)).pack()
    
    def _start_system(self):
        """Start the camera system"""
        try:
            self.start_button.config(state="disabled")
            self.start_button.config(text="Initializing...")
            
            # Initialize camera in separate thread
            init_thread = threading.Thread(target=self._initialize_camera, daemon=True)
            init_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start system: {str(e)}")
            self._reset_buttons()
    
    def _initialize_camera(self):
        """Initialize camera (runs in separate thread)"""
        try:
            # Initialize camera
            success = self.camera_manager.initialize()
            
            if success:
                # Start capture
                self.camera_manager.start_capture()
                
                # Update GUI
                self.root.after(0, self._camera_initialized_success)
                
                # Start processing
                self.is_running = True
                self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
                self.processing_thread.start()
                
            else:
                self.root.after(0, self._camera_initialization_failed)
                
        except Exception as e:
            self.root.after(0, lambda: self._camera_initialization_failed(str(e)))
    
    def _camera_initialized_success(self):
        """Handle successful camera initialization"""
        self.start_button.config(text="Camera Running", state="disabled")
        self.stop_button.config(state="normal")
        
        # Update status indicator
        self.status_indicator.delete("all")
        self.status_indicator.create_oval(2, 2, 18, 18, fill="green", outline="black")
        
        # Update camera info
        camera_info = self.camera_manager.get_camera_info()
        info_text = f"""Camera: Intel RealSense D435i
Resolution: {camera_info.get('width', 0)}x{camera_info.get('height', 0)} @ {camera_info.get('fps', 0)}fps
Intrinsics: fx={camera_info.get('fx', 0):.1f}, fy={camera_info.get('fy', 0):.1f}
Status: ACTIVE - Real-time tracking enabled"""
        
        self.camera_info_text.config(state=tk.NORMAL)
        self.camera_info_text.delete(1.0, tk.END)
        self.camera_info_text.insert(tk.END, info_text)
        self.camera_info_text.config(state=tk.DISABLED)
        
        messagebox.showinfo("Success", "Camera initialized successfully!\nReal-time tracking is now active.")
    
    def _camera_initialization_failed(self, error_msg="Unknown error"):
        """Handle camera initialization failure"""
        self._reset_buttons()
        
        error_text = f"""Camera initialization failed!

Possible issues:
1. Intel RealSense D435i not connected
2. Camera drivers not installed
3. Camera in use by another application

Error: {error_msg}

Please check your camera connection and try again."""
        
        messagebox.showerror("Camera Error", error_text)
    
    def _processing_loop(self):
        """Main processing loop for camera frames"""
        last_time = time.time()
        
        while self.is_running:
            try:
                # Get frame from camera
                frame_data = self.camera_manager.get_frame()
                
                if frame_data is not None:
                    color_frame, depth_frame = frame_data
                    
                    # Process frame
                    results = self.feature_tracker.process_frame(color_frame, depth_frame)
                    
                    # Calculate FPS
                    current_time = time.time()
                    if current_time - last_time > 0:
                        fps = 1.0 / (current_time - last_time)
                        last_time = current_time
                    else:
                        fps = 0.0
                    
                    # Update statistics
                    self.stats.update({
                        'fps': fps,
                        'features': results['num_features'],
                        'matches': results['num_matches'],
                        'distance': results['distance_moved'],
                        'total_distance': results['total_distance'],
                        'frame_count': self.camera_manager.frame_count
                    })
                    
                    # Schedule GUI update
                    self.root.after(0, lambda: self._update_trajectory_display(results['trajectory']))
                
                else:
                    time.sleep(0.01)  # Small delay if no frame available
                    
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.1)
    
    def _update_trajectory_display(self, trajectory):
        """Update trajectory visualization"""
        try:
            # Clear canvas
            self.trajectory_canvas.delete("all")
            
            if len(trajectory) < 2:
                return
            
            # Get canvas dimensions
            canvas_width = self.trajectory_canvas.winfo_width()
            canvas_height = self.trajectory_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            # Convert trajectory to canvas coordinates
            traj_array = np.array(trajectory)
            
            # Use X and Z for top-down view
            x_coords = traj_array[:, 0]
            z_coords = traj_array[:, 2]
            
            if len(x_coords) > 1:
                # Scale to fit canvas
                x_min, x_max = np.min(x_coords), np.max(x_coords)
                z_min, z_max = np.min(z_coords), np.max(z_coords)
                
                x_range = max(x_max - x_min, 0.1)
                z_range = max(z_max - z_min, 0.1)
                
                margin = 50
                scale_x = (canvas_width - 2 * margin) / x_range
                scale_z = (canvas_height - 2 * margin) / z_range
                scale = min(scale_x, scale_z)
                
                # Convert to canvas coordinates
                canvas_points = []
                for i, (x, z) in enumerate(zip(x_coords, z_coords)):
                    canvas_x = margin + (x - x_min) * scale
                    canvas_y = margin + (z - z_min) * scale
                    canvas_points.append((canvas_x, canvas_y))
                
                # Draw trajectory line
                if len(canvas_points) > 1:
                    for i in range(len(canvas_points) - 1):
                        self.trajectory_canvas.create_line(
                            canvas_points[i][0], canvas_points[i][1],
                            canvas_points[i+1][0], canvas_points[i+1][1],
                            fill="cyan", width=2
                        )
                
                # Draw current position
                if canvas_points:
                    curr_x, curr_y = canvas_points[-1]
                    self.trajectory_canvas.create_oval(
                        curr_x-5, curr_y-5, curr_x+5, curr_y+5,
                        fill="lime", outline="white", width=2
                    )
                
                # Draw start position
                start_x, start_y = canvas_points[0]
                self.trajectory_canvas.create_oval(
                    start_x-4, start_y-4, start_x+4, start_y+4,
                    fill="red", outline="white", width=2
                )
                
                # Add labels
                self.trajectory_canvas.create_text(
                    start_x, start_y-15, text="START", fill="white", font=("Arial", 8, "bold")
                )
                self.trajectory_canvas.create_text(
                    curr_x, curr_y-15, text="NOW", fill="lime", font=("Arial", 8, "bold")
                )
        
        except Exception as e:
            print(f"Trajectory display error: {e}")
    
    def _stop_system(self):
        """Stop the camera system"""
        try:
            self.is_running = False
            
            # Stop camera
            self.camera_manager.stop_capture()
            
            # Wait for processing thread
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=2.0)
            
            # Update GUI
            self._reset_buttons()
            
            # Update status indicator
            self.status_indicator.delete("all")
            self.status_indicator.create_oval(2, 2, 18, 18, fill="red", outline="black")
            
            # Update camera info
            self.camera_info_text.config(state=tk.NORMAL)
            self.camera_info_text.delete(1.0, tk.END)
            self.camera_info_text.insert(tk.END, "Camera stopped. Click 'Start Camera' to resume.")
            self.camera_info_text.config(state=tk.DISABLED)
            
            messagebox.showinfo("Stopped", "Camera system stopped successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error stopping system: {str(e)}")
    
    def _reset_trajectory(self):
        """Reset trajectory tracking"""
        try:
            self.feature_tracker.trajectory = [[0, 0, 0]]
            self.feature_tracker.total_distance = 0.0
            
            # Clear trajectory display
            self.trajectory_canvas.delete("all")
            
            # Reset statistics
            self.stats['total_distance'] = 0.0
            self.stats['distance'] = 0.0
            
            messagebox.showinfo("Reset", "Trajectory reset successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error resetting trajectory: {str(e)}")
    
    def _reset_buttons(self):
        """Reset button states"""
        self.start_button.config(text="Start Camera", state="normal")
        self.stop_button.config(state="disabled")
    
    def _schedule_status_update(self):
        """Schedule periodic status updates"""
        self._update_statistics_display()
        self.root.after(100, self._schedule_status_update)  # Update every 100ms
    
    def _update_statistics_display(self):
        """Update statistics display"""
        try:
            self.fps_label.config(text=f"{self.stats['fps']:.1f}")
            self.features_label.config(text=str(self.stats['features']))
            self.matches_label.config(text=str(self.stats['matches']))
            self.distance_label.config(text=f"{self.stats['total_distance']:.2f} m")
            self.frames_label.config(text=str(self.stats['frame_count']))
            
        except Exception as e:
            print(f"Statistics update error: {e}")
    
    def _on_closing(self):
        """Handle application closing"""
        if self.is_running:
            if messagebox.askokcancel("Quit", "Camera is running. Stop and quit?"):
                self._stop_system()
                time.sleep(1)  # Give time for cleanup
                self.root.quit()
        else:
            self.root.quit()
    
    def run(self):
        """Run the enhanced GUI application"""
        self.root.mainloop()

def main():
    """Main application entry point"""
    try:
        print("Starting Enhanced Visual Odometry System...")
        
        # Create and run enhanced GUI
        app = EnhancedVisualOdometryGUI()
        app.run()
        
    except Exception as e:
        print(f"Application failed to start: {e}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()