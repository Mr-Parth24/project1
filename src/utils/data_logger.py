"""
Data Logger for SLAM System
Handles saving and loading trajectory, map, and performance data
"""

import os
import json
import numpy as np
import datetime
from typing import Dict, List, Any, Optional
import pickle

class DataLogger:
    """
    Handles data logging for the SLAM system
    Saves trajectories, maps, keyframes, and performance metrics
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data logger
        
        Args:
            data_dir: Root directory for data storage
        """
        self.data_dir = data_dir
        self.trajectories_dir = os.path.join(data_dir, "trajectories")
        self.maps_dir = os.path.join(data_dir, "maps")
        self.logs_dir = os.path.join(data_dir, "logs")
        self.calibration_dir = os.path.join(data_dir, "calibration")
        
        # Create directories
        self._create_directories()
        
        print(f"Data Logger initialized with directory: {data_dir}")
    
    def _create_directories(self):
        """Create necessary data directories"""
        dirs = [self.data_dir, self.trajectories_dir, self.maps_dir, 
                self.logs_dir, self.calibration_dir]
        
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
    
    def save_trajectory(self, trajectory_points: np.ndarray, 
                       metadata: Dict[str, Any] = None) -> str:
        """
        Save trajectory data
        
        Args:
            trajectory_points: Nx3 array of trajectory points
            metadata: Additional metadata (distance, timestamps, etc.)
            
        Returns:
            Path to saved file
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.npz"
            filepath = os.path.join(self.trajectories_dir, filename)
            
            # Prepare data
            save_data = {
                'trajectory': trajectory_points,
                'timestamp': timestamp,
                'num_points': len(trajectory_points)
            }
            
            # Add metadata if provided
            if metadata:
                save_data.update(metadata)
            
            # Save as NPZ file
            np.savez_compressed(filepath, **save_data)
            
            print(f"Trajectory saved: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving trajectory: {e}")
            return ""
    
    def load_trajectory(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load trajectory data
        
        Args:
            filepath: Path to trajectory file
            
        Returns:
            Dictionary with trajectory data
        """
        try:
            data = np.load(filepath, allow_pickle=True)
            
            result = {
                'trajectory': data['trajectory'],
                'timestamp': str(data.get('timestamp', 'unknown')),
                'num_points': int(data.get('num_points', 0))
            }
            
            # Load additional metadata
            for key in data.files:
                if key not in ['trajectory', 'timestamp', 'num_points']:
                    result[key] = data[key]
            
            print(f"Trajectory loaded: {filepath}")
            return result
            
        except Exception as e:
            print(f"Error loading trajectory: {e}")
            return None
    
    def save_slam_map(self, map_points: List[np.ndarray], 
                     keyframes: List[Dict], metadata: Dict = None) -> str:
        """
        Save SLAM map data
        
        Args:
            map_points: List of 3D map points
            keyframes: List of keyframe data
            metadata: Additional metadata
            
        Returns:
            Path to saved file
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"slam_map_{timestamp}.pkl"
            filepath = os.path.join(self.maps_dir, filename)
            
            # Prepare map data
            map_data = {
                'map_points': map_points,
                'keyframes': keyframes,
                'timestamp': timestamp,
                'num_map_points': len(map_points),
                'num_keyframes': len(keyframes)
            }
            
            if metadata:
                map_data.update(metadata)
            
            # Save as pickle file
            with open(filepath, 'wb') as f:
                pickle.dump(map_data, f)
            
            print(f"SLAM map saved: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving SLAM map: {e}")
            return ""
    
    def load_slam_map(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load SLAM map data
        
        Args:
            filepath: Path to map file
            
        Returns:
            Dictionary with map data
        """
        try:
            with open(filepath, 'rb') as f:
                map_data = pickle.load(f)
            
            print(f"SLAM map loaded: {filepath}")
            return map_data
            
        except Exception as e:
            print(f"Error loading SLAM map: {e}")
            return None
    
    def save_performance_log(self, performance_data: Dict[str, Any]) -> str:
        """
        Save performance metrics
        
        Args:
            performance_data: Dictionary with performance metrics
            
        Returns:
            Path to saved file
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_{timestamp}.json"
            filepath = os.path.join(self.logs_dir, filename)
            
            # Add timestamp
            performance_data['timestamp'] = timestamp
            performance_data['log_time'] = datetime.datetime.now().isoformat()
            
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(performance_data, f, indent=2, default=str)
            
            print(f"Performance log saved: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving performance log: {e}")
            return ""
    
    def save_camera_calibration(self, camera_matrix: np.ndarray, 
                              dist_coeffs: np.ndarray, metadata: Dict = None) -> str:
        """
        Save camera calibration data
        
        Args:
            camera_matrix: 3x3 camera matrix
            dist_coeffs: Distortion coefficients
            metadata: Additional calibration metadata
            
        Returns:
            Path to saved file
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_calibration_{timestamp}.npz"
            filepath = os.path.join(self.calibration_dir, filename)
            
            # Prepare calibration data
            calib_data = {
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs,
                'timestamp': timestamp
            }
            
            if metadata:
                calib_data.update(metadata)
            
            # Save as NPZ file
            np.savez_compressed(filepath, **calib_data)
            
            print(f"Camera calibration saved: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving camera calibration: {e}")
            return ""
    
    def get_trajectory_files(self) -> List[str]:
        """Get list of available trajectory files"""
        try:
            files = []
            for filename in os.listdir(self.trajectories_dir):
                if filename.endswith('.npz') and filename.startswith('trajectory_'):
                    files.append(os.path.join(self.trajectories_dir, filename))
            return sorted(files, reverse=True)  # Most recent first
        except:
            return []
    
    def get_map_files(self) -> List[str]:
        """Get list of available map files"""
        try:
            files = []
            for filename in os.listdir(self.maps_dir):
                if filename.endswith('.pkl') and filename.startswith('slam_map_'):
                    files.append(os.path.join(self.maps_dir, filename))
            return sorted(files, reverse=True)  # Most recent first
        except:
            return []
    
    def export_trajectory_csv(self, trajectory_points: np.ndarray, 
                            filepath: str = None) -> str:
        """
        Export trajectory to CSV format
        
        Args:
            trajectory_points: Nx3 array of trajectory points
            filepath: Output file path (optional)
            
        Returns:
            Path to exported file
        """
        try:
            if filepath is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trajectory_{timestamp}.csv"
                filepath = os.path.join(self.trajectories_dir, filename)
            
            # Create CSV content
            import csv
            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Point_ID', 'X', 'Y', 'Z', 'Distance_From_Start'])
                
                # Calculate cumulative distance
                total_distance = 0.0
                for i, point in enumerate(trajectory_points):
                    if i > 0:
                        dist = np.linalg.norm(point - trajectory_points[i-1])
                        total_distance += dist
                    
                    writer.writerow([i, point[0], point[1], point[2], total_distance])
            
            print(f"Trajectory exported to CSV: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error exporting trajectory to CSV: {e}")
            return ""
    
    def cleanup_old_files(self, max_files: int = 50):
        """
        Clean up old data files to save space
        
        Args:
            max_files: Maximum number of files to keep per category
        """
        try:
            # Clean trajectory files
            traj_files = self.get_trajectory_files()
            if len(traj_files) > max_files:
                for filepath in traj_files[max_files:]:
                    os.remove(filepath)
                    print(f"Removed old trajectory file: {filepath}")
            
            # Clean map files
            map_files = self.get_map_files()
            if len(map_files) > max_files:
                for filepath in map_files[max_files:]:
                    os.remove(filepath)
                    print(f"Removed old map file: {filepath}")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Global data logger instance
data_logger = DataLogger()

def get_data_logger() -> DataLogger:
    """Get the global data logger instance"""
    return data_logger

# Test function
def test_data_logger():
    """Test data logger functionality"""
    logger = DataLogger()
    
    # Test trajectory save/load
    test_trajectory = np.array([
        [0, 0, 0],
        [1, 0, 1],
        [2, 0, 2],
        [2, 1, 2]
    ])
    
    # Save trajectory
    filepath = logger.save_trajectory(test_trajectory, {'distance': 4.0, 'duration': 10.5})
    
    # Load trajectory
    loaded_data = logger.load_trajectory(filepath)
    if loaded_data:
        print(f"Loaded trajectory with {loaded_data['num_points']} points")
    
    # Export to CSV
    csv_path = logger.export_trajectory_csv(test_trajectory)
    print(f"CSV exported to: {csv_path}")

if __name__ == "__main__":
    test_data_logger()