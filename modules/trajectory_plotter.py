"""
Real-time Trajectory Plotter
Visualizes the camera path in 3D
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import threading
import time

class TrajectoryPlotter:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.trajectory_points = []
        self.current_pos = np.array([0.0, 0.0, 0.0])
        self.is_plotting = False
        self.lock = threading.Lock()
        
        # Plot settings
        self.max_points = 1000  # Limit points for performance
        
    def start_plotting(self):
        """Start the real-time plotting"""
        if self.is_plotting:
            return
            
        self.is_plotting = True
        
        # Set up the plot
        plt.ion()  # Interactive mode
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (meters)')
        self.ax.set_title('Real-time Camera Trajectory')
        
        # Set equal aspect ratio
        self.ax.set_box_aspect([1,1,1])
        
        print("📈 3D trajectory plotter started")
        
        # Start update loop
        self.update_loop()
        
    def update_loop(self):
        """Main update loop for plotting"""
        while self.is_plotting:
            try:
                with self.lock:
                    if len(self.trajectory_points) > 0:
                        self.update_plot()
                        
                plt.pause(0.1)  # Small delay
                
            except Exception as e:
                print(f"❌ Plotting error: {e}")
                break
                
        plt.ioff()
        
    def update_plot(self):
        """Update the 3D plot"""
        if not self.trajectory_points:
            return
            
        # Clear previous plot
        self.ax.clear()
        
        # Convert trajectory to numpy array
        trajectory_array = np.array(self.trajectory_points)
        
        # Limit points for performance
        if len(trajectory_array) > self.max_points:
            trajectory_array = trajectory_array[-self.max_points:]
            
        # Plot trajectory
        self.ax.plot(trajectory_array[:, 0], 
                    trajectory_array[:, 1], 
                    trajectory_array[:, 2], 
                    'b-', linewidth=2, alpha=0.7, label='Path')
        
        # Plot start point
        if len(trajectory_array) > 0:
            self.ax.scatter(trajectory_array[0, 0], 
                          trajectory_array[0, 1], 
                          trajectory_array[0, 2], 
                          color='green', s=100, label='Start')
            
        # Plot current position
        self.ax.scatter(self.current_pos[0], 
                       self.current_pos[1], 
                       self.current_pos[2], 
                       color='red', s=100, label='Current')
        
        # Set labels and title
        self.ax.set_xlabel('X (meters)')
        self.ax.set_ylabel('Y (meters)')
        self.ax.set_zlabel('Z (meters)')
        self.ax.set_title(f'Camera Trajectory ({len(trajectory_array)} points)')
        
        # Add legend
        self.ax.legend()
        
        # Set reasonable axis limits
        if len(trajectory_array) > 1:
            margin = 1.0
            x_min, x_max = trajectory_array[:, 0].min() - margin, trajectory_array[:, 0].max() + margin
            y_min, y_max = trajectory_array[:, 1].min() - margin, trajectory_array[:, 1].max() + margin
            z_min, z_max = trajectory_array[:, 2].min() - margin, trajectory_array[:, 2].max() + margin
            
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            self.ax.set_zlim(z_min, z_max)
        else:
            self.ax.set_xlim(-2, 2)
            self.ax.set_ylim(-2, 2)
            self.ax.set_zlim(-2, 2)
            
        # Set equal aspect ratio
        self.ax.set_box_aspect([1,1,1])
        
        # Draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def update_trajectory(self, trajectory, current_position):
        """Update trajectory data"""
        with self.lock:
            self.trajectory_points = trajectory.copy()
            self.current_pos = current_position.copy()
            
    def clear_trajectory(self):
        """Clear the trajectory"""
        with self.lock:
            self.trajectory_points = []
            self.current_pos = np.array([0.0, 0.0, 0.0])
            
    def stop_plotting(self):
        """Stop the plotting"""
        self.is_plotting = False
        if self.fig:
            plt.close(self.fig)