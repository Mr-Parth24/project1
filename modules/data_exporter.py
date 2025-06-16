"""
Enhanced Data Export Module
Saves trajectory data to various formats with GUI support
"""

import json
import csv
import numpy as np
import os
from datetime import datetime

class DataExporter:
    def __init__(self):
        self.output_dir = "data/output_paths"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def save_trajectory(self, trajectory, total_distance):
        """Save trajectory to JSON and CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data
        trajectory_data = {
            "timestamp": timestamp,
            "total_distance": float(total_distance),
            "total_points": len(trajectory),
            "start_position": {
                "x": float(trajectory[0][0]) if len(trajectory) > 0 else 0.0,
                "y": float(trajectory[0][1]) if len(trajectory) > 0 else 0.0,
                "z": float(trajectory[0][2]) if len(trajectory) > 0 else 0.0
            },
            "end_position": {
                "x": float(trajectory[-1][0]) if len(trajectory) > 0 else 0.0,
                "y": float(trajectory[-1][1]) if len(trajectory) > 0 else 0.0,
                "z": float(trajectory[-1][2]) if len(trajectory) > 0 else 0.0
            },
            "trajectory": []
        }
        
        # Convert trajectory to list format
        for i, point in enumerate(trajectory):
            trajectory_data["trajectory"].append({
                "index": i,
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "timestamp": i * 0.033  # Approximate 30fps timing
            })
            
        # Calculate additional metrics
        if len(trajectory) > 1:
            net_displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
            trajectory_data["net_displacement"] = float(net_displacement)
            trajectory_data["loop_closure_error"] = float(net_displacement)
            
        # Save JSON
        json_filename = os.path.join(self.output_dir, f"trajectory_{timestamp}.json")
        with open(json_filename, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
            
        # Save CSV
        csv_filename = os.path.join(self.output_dir, f"trajectory_{timestamp}.csv")
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "X", "Y", "Z", "Timestamp"])
            
            for i, point in enumerate(trajectory):
                writer.writerow([i, point[0], point[1], point[2], i * 0.033])
                
        # Save summary
        summary_filename = os.path.join(self.output_dir, f"summary_{timestamp}.txt")
        with open(summary_filename, 'w') as f:
            f.write(f"SLAM Tracker - Trajectory Summary\n")
            f.write(f"=" * 40 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(f"TRAJECTORY METRICS:\n")
            f.write(f"Total Distance: {total_distance:.3f} meters\n")
            f.write(f"Total Points: {len(trajectory)}\n")
            f.write(f"Duration: {len(trajectory) * 0.033:.1f} seconds\n")
            
            if len(trajectory) > 0:
                f.write(f"\nPOSITION DATA:\n")
                f.write(f"Start Position: ({trajectory[0][0]:.3f}, {trajectory[0][1]:.3f}, {trajectory[0][2]:.3f})\n")
                f.write(f"End Position: ({trajectory[-1][0]:.3f}, {trajectory[-1][1]:.3f}, {trajectory[-1][2]:.3f})\n")
                
                if len(trajectory) > 1:
                    displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
                    f.write(f"Net Displacement: {displacement:.3f} meters\n")
                    f.write(f"Loop Closure Error: {displacement:.3f} meters\n")
                    
                    # Calculate bounding box
                    trajectory_array = np.array(trajectory)
                    mins = np.min(trajectory_array, axis=0)
                    maxs = np.max(trajectory_array, axis=0)
                    f.write(f"\nTRAJECTORY BOUNDS:\n")
                    f.write(f"X Range: {mins[0]:.3f} to {maxs[0]:.3f} ({maxs[0]-mins[0]:.3f} m)\n")
                    f.write(f"Y Range: {mins[1]:.3f} to {maxs[1]:.3f} ({maxs[1]-mins[1]:.3f} m)\n")
                    f.write(f"Z Range: {mins[2]:.3f} to {maxs[2]:.3f} ({maxs[2]-mins[2]:.3f} m)\n")
                    
        return json_filename
        
    def export_csv(self, trajectory, filename):
        """Export trajectory to specific CSV file"""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Index", "X", "Y", "Z", "Timestamp"])
            
            for i, point in enumerate(trajectory):
                writer.writerow([i, point[0], point[1], point[2], i * 0.033])
                
    def export_json(self, trajectory, total_distance, filename):
        """Export trajectory to specific JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        trajectory_data = {
            "exported_at": datetime.now().isoformat(),
            "timestamp": timestamp,
            "total_distance": float(total_distance),
            "total_points": len(trajectory),
            "trajectory": []
        }
        
        for i, point in enumerate(trajectory):
            trajectory_data["trajectory"].append({
                "index": i,
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
                "timestamp": i * 0.033
            })
            
        with open(filename, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
            
    def load_trajectory(self, filename):
        """Load trajectory from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                
            trajectory = []
            for point in data["trajectory"]:
                trajectory.append(np.array([point["x"], point["y"], point["z"]]))
                
            return {
                "trajectory": trajectory,
                "total_distance": data.get("total_distance", 0.0),
                "timestamp": data.get("timestamp", "unknown")
            }
            
        except Exception as e:
            print(f"❌ Error loading trajectory: {e}")
            return None