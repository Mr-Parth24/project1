"""
Trajectory Tracking and Analysis
Handles trajectory storage, analysis, and loop detection
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from scipy.spatial.distance import euclidean
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TrajectoryPoint:
    """Single trajectory point with metadata"""
    
    def __init__(self, position: np.ndarray, orientation: np.ndarray, 
                 timestamp: float, velocity: Optional[np.ndarray] = None):
        self.position = position.copy()  # [x, y, z]
        self.orientation = orientation.copy()  # 3x3 rotation matrix
        self.timestamp = timestamp
        self.velocity = velocity.copy() if velocity is not None else np.zeros(3)
        self.distance_from_start = 0.0
        self.cumulative_distance = 0.0

class TrajectoryTracker:
    """Advanced trajectory tracking and analysis"""
    
    def __init__(self, max_points: int = 10000):
        self.logger = logging.getLogger(__name__)
        self.max_points = max_points
        
        # Trajectory data
        self.trajectory_points: List[TrajectoryPoint] = []
        self.total_distance = 0.0
        self.start_position = None
        self.current_position = np.zeros(3)
        
        # Loop detection
        self.loop_closures: List[Dict[str, Any]] = []
        self.loop_detection_threshold = 0.5  # meters
        self.min_loop_points = 100  # Minimum points before checking loops
        
        # Smoothing and filtering
        self.position_buffer = deque(maxlen=10)
        self.enable_smoothing = True
        
        # Statistics
        self.max_speed = 0.0
        self.avg_speed = 0.0
        self.direction_changes = 0
        
    def add_point(self, pose: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """Add a new trajectory point"""
        
        position = pose[:3, 3]
        orientation = pose[:3, :3]
        
        # Calculate velocity if we have previous points
        velocity = np.zeros(3)
        if len(self.trajectory_points) > 0:
            prev_point = self.trajectory_points[-1]
            dt = timestamp - prev_point.timestamp
            if dt > 0:
                velocity = (position - prev_point.position) / dt
        
        # Create trajectory point
        point = TrajectoryPoint(position, orientation, timestamp, velocity)
        
        # Calculate distances
        if self.start_position is None:
            self.start_position = position.copy()
            point.distance_from_start = 0.0
        else:
            point.distance_from_start = np.linalg.norm(position - self.start_position)
        
        if len(self.trajectory_points) > 0:
            distance_delta = np.linalg.norm(position - self.trajectory_points[-1].position)
            self.total_distance += distance_delta
            point.cumulative_distance = self.total_distance
        
        # Add to trajectory
        self.trajectory_points.append(point)
        self.current_position = position.copy()
        
        # Maintain maximum points
        if len(self.trajectory_points) > self.max_points:
            self.trajectory_points.pop(0)
        
        # Update position buffer for smoothing
        self.position_buffer.append(position)
        
        # Update statistics
        self._update_statistics()
        
        # Check for loop closure
        loop_detected = self._check_loop_closure(point)
        
        return {
            'position': position,
            'distance_from_start': point.distance_from_start,
            'cumulative_distance': point.cumulative_distance,
            'velocity': velocity,
            'speed': np.linalg.norm(velocity),
            'loop_detected': loop_detected,
            'total_points': len(self.trajectory_points)
        }
    
    def _update_statistics(self):
        """Update trajectory statistics"""
        if len(self.trajectory_points) < 2:
            return
        
        # Calculate speeds
        speeds = [np.linalg.norm(point.velocity) for point in self.trajectory_points]
        self.max_speed = max(speeds) if speeds else 0.0
        self.avg_speed = np.mean(speeds) if speeds else 0.0
        
        # Count direction changes (simplified)
        if len(self.trajectory_points) >= 3:
            recent_points = self.trajectory_points[-3:]
            v1 = recent_points[1].position - recent_points[0].position
            v2 = recent_points[2].position - recent_points[1].position
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1))
                if angle > np.pi / 4:  # 45 degrees
                    self.direction_changes += 1
    
    def _check_loop_closure(self, current_point: TrajectoryPoint) -> bool:
        """Check if current point closes a loop"""
        
        if len(self.trajectory_points) < self.min_loop_points:
            return False
        
        # Check against points that are not too recent
        check_points = self.trajectory_points[:-20]  # Skip last 20 points
        
        for i, point in enumerate(check_points):
            distance = np.linalg.norm(current_point.position - point.position)
            
            if distance < self.loop_detection_threshold:
                # Verify it's actually a loop (not just oscillation)
                if self._verify_loop_closure(current_point, point, i):
                    loop_info = {
                        'start_index': i,
                        'end_index': len(self.trajectory_points) - 1,
                        'start_position': point.position.copy(),
                        'end_position': current_point.position.copy(),
                        'loop_distance': distance,
                        'loop_length': current_point.cumulative_distance - point.cumulative_distance,
                        'timestamp': current_point.timestamp
                    }
                    
                    self.loop_closures.append(loop_info)
                    self.logger.info(f"Loop closure detected: {distance:.2f}m gap, {loop_info['loop_length']:.2f}m loop")
                    return True
        
        return False
    
    def _verify_loop_closure(self, current_point: TrajectoryPoint, 
                           loop_point: TrajectoryPoint, loop_index: int) -> bool:
        """Verify that detected proximity is actually a meaningful loop"""
        
        # Check if enough distance was traveled
        distance_traveled = current_point.cumulative_distance - loop_point.cumulative_distance
        if distance_traveled < 2.0:  # At least 2 meters
            return False
        
        # Check if enough time passed
        time_elapsed = current_point.timestamp - loop_point.timestamp
        if time_elapsed < 10.0:  # At least 10 seconds
            return False
        
        # Check trajectory shape (should form a rough loop)
        loop_points = self.trajectory_points[loop_index:-1]
        if len(loop_points) < 10:
            return False
        
        return True
    
    def get_smoothed_trajectory(self, window_size: int = 5) -> np.ndarray:
        """Get smoothed trajectory points"""
        if len(self.trajectory_points) < window_size:
            return np.array([point.position for point in self.trajectory_points])
        
        positions = np.array([point.position for point in self.trajectory_points])
        
        # Apply Savitzky-Golay filter for smoothing
        try:
            smoothed_positions = np.zeros_like(positions)
            for i in range(3):  # x, y, z
                smoothed_positions[:, i] = savgol_filter(positions[:, i], window_size, 3)
            return smoothed_positions
        except:
            return positions
    
    def get_trajectory_segment(self, start_time: float, end_time: float) -> List[TrajectoryPoint]:
        """Get trajectory segment between two timestamps"""
        segment = []
        for point in self.trajectory_points:
            if start_time <= point.timestamp <= end_time:
                segment.append(point)
        return segment
    
    def get_recent_trajectory(self, num_points: int = 100) -> np.ndarray:
        """Get recent trajectory points"""
        recent_points = self.trajectory_points[-num_points:] if len(self.trajectory_points) > num_points else self.trajectory_points
        return np.array([point.position for point in recent_points])
    
    def calculate_path_efficiency(self) -> float:
        """Calculate path efficiency (straight line distance / actual distance)"""
        if len(self.trajectory_points) < 2:
            return 1.0
        
        start_pos = self.trajectory_points[0].position
        end_pos = self.trajectory_points[-1].position
        straight_distance = np.linalg.norm(end_pos - start_pos)
        
        if self.total_distance == 0:
            return 1.0
        
        return straight_distance / self.total_distance
    
    def get_movement_direction(self) -> str:
        """Get current movement direction"""
        if len(self.trajectory_points) < 2:
            return "Unknown"
        
        recent_velocity = self.trajectory_points[-1].velocity
        
        if np.linalg.norm(recent_velocity) < 0.01:
            return "Stationary"
        
        # Determine primary direction
        abs_velocity = np.abs(recent_velocity)
        max_axis = np.argmax(abs_velocity)
        
        directions = ["X", "Y", "Z"]
        sign = "+" if recent_velocity[max_axis] > 0 else "-"
        
        return f"{sign}{directions[max_axis]}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive trajectory statistics"""
        return {
            'total_points': len(self.trajectory_points),
            'total_distance': self.total_distance,
            'distance_from_start': self.trajectory_points[-1].distance_from_start if self.trajectory_points else 0.0,
            'max_speed': self.max_speed,
            'avg_speed': self.avg_speed,
            'current_speed': np.linalg.norm(self.trajectory_points[-1].velocity) if self.trajectory_points else 0.0,
            'path_efficiency': self.calculate_path_efficiency(),
            'movement_direction': self.get_movement_direction(),
            'direction_changes': self.direction_changes,
            'loop_closures': len(self.loop_closures),
            'duration': self.trajectory_points[-1].timestamp - self.trajectory_points[0].timestamp if len(self.trajectory_points) > 1 else 0.0
        }
    
    def export_trajectory(self, filename: str):
        """Export trajectory to file"""
        import json
        
        export_data = {
            'metadata': {
                'total_points': len(self.trajectory_points),
                'total_distance': self.total_distance,
                'duration': self.trajectory_points[-1].timestamp - self.trajectory_points[0].timestamp if len(self.trajectory_points) > 1 else 0.0,
                'export_timestamp': self.trajectory_points[-1].timestamp if self.trajectory_points else 0.0
            },
            'trajectory': [],
            'loop_closures': self.loop_closures,
            'statistics': self.get_statistics()
        }
        
        for point in self.trajectory_points:
            export_data['trajectory'].append({
                'timestamp': point.timestamp,
                'position': point.position.tolist(),
                'orientation': point.orientation.tolist(),
                'velocity': point.velocity.tolist(),
                'distance_from_start': point.distance_from_start,
                'cumulative_distance': point.cumulative_distance
            })
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Trajectory exported to {filename}")
    
    def reset(self):
        """Reset trajectory tracker"""
        self.trajectory_points.clear()
        self.loop_closures.clear()
        self.position_buffer.clear()
        
        self.total_distance = 0.0
        self.start_position = None
        self.current_position = np.zeros(3)
        self.max_speed = 0.0
        self.avg_speed = 0.0
        self.direction_changes = 0
        
        self.logger.info("Trajectory tracker reset")