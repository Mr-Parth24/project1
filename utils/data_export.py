"""
Data export utilities for trajectory and session data
"""

import csv
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import os

class DataExporter:
    """Data export functionality for trajectory and analysis data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def export_to_csv(self, trajectory_points: List, filename: str, 
                     include_orientation: bool = True, include_velocity: bool = True):
        """Export trajectory data to CSV format"""
        try:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'x', 'y', 'z', 'distance_from_start', 'cumulative_distance']
                
                if include_orientation:
                    fieldnames.extend(['r00', 'r01', 'r02', 'r10', 'r11', 'r12', 'r20', 'r21', 'r22'])
                
                if include_velocity:
                    fieldnames.extend(['vx', 'vy', 'vz', 'speed'])
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for point in trajectory_points:
                    row = {
                        'timestamp': point.timestamp,
                        'x': point.position[0],
                        'y': point.position[1],
                        'z': point.position[2],
                        'distance_from_start': point.distance_from_start,
                        'cumulative_distance': point.cumulative_distance
                    }
                    
                    if include_orientation:
                        orientation = point.orientation.flatten()
                        for i, val in enumerate(orientation):
                            row[f'r{i//3}{i%3}'] = val
                    
                    if include_velocity:
                        row['vx'] = point.velocity[0]
                        row['vy'] = point.velocity[1]
                        row['vz'] = point.velocity[2]
                        row['speed'] = np.linalg.norm(point.velocity)
                    
                    writer.writerow(row)
            
            self.logger.info(f"Trajectory exported to CSV: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"CSV export failed: {e}")
            return False
    
    def export_to_json(self, trajectory_points: List, filename: str, 
                      metadata: Optional[Dict[str, Any]] = None):
        """Export trajectory data to JSON format"""
        try:
            export_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'total_points': len(trajectory_points),
                    'format_version': '1.0'
                },
                'trajectory': []
            }
            
            # Add custom metadata
            if metadata:
                export_data['metadata'].update(metadata)
            
            # Export trajectory points
            for point in trajectory_points:
                point_data = {
                    'timestamp': point.timestamp,
                    'position': point.position.tolist(),
                    'orientation': point.orientation.tolist(),
                    'velocity': point.velocity.tolist(),
                    'distance_from_start': point.distance_from_start,
                    'cumulative_distance': point.cumulative_distance
                }
                export_data['trajectory'].append(point_data)
            
            with open(filename, 'w') as jsonfile:
                json.dump(export_data, jsonfile, indent=2)
            
            self.logger.info(f"Trajectory exported to JSON: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"JSON export failed: {e}")
            return False
    
    def export_analysis_report(self, trajectory_points: List, stats: Dict[str, Any], 
                             filename: str):
        """Export comprehensive analysis report"""
        try:
            report = {
                'session_info': {
                    'export_time': datetime.now().isoformat(),
                    'total_points': len(trajectory_points),
                    'duration': stats.get('duration', 0),
                    'software_version': '1.0.0'
                },
                'trajectory_statistics': stats,
                'analysis': self._compute_advanced_analysis(trajectory_points),
                'raw_data': {
                    'positions': [point.position.tolist() for point in trajectory_points],
                    'timestamps': [point.timestamp for point in trajectory_points],
                    'distances': [point.cumulative_distance for point in trajectory_points]
                }
            }
            
            with open(filename, 'w') as jsonfile:
                json.dump(report, jsonfile, indent=2)
            
            self.logger.info(f"Analysis report exported: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Analysis report export failed: {e}")
            return False
    
    def _compute_advanced_analysis(self, trajectory_points: List) -> Dict[str, Any]:
        """Compute advanced trajectory analysis"""
        if len(trajectory_points) < 2:
            return {}
        
        try:
            positions = np.array([point.position for point in trajectory_points])
            timestamps = np.array([point.timestamp for point in trajectory_points])
            
            # Compute derivatives
            velocities = np.diff(positions, axis=0) / np.diff(timestamps).reshape(-1, 1)
            speeds = np.linalg.norm(velocities, axis=1)
            
            # Compute accelerations
            accelerations = np.diff(velocities, axis=0) / np.diff(timestamps[1:]).reshape(-1, 1)
            
            # Compute curvature
            curvatures = self._compute_curvature(positions)
            
            analysis = {
                'velocity_stats': {
                    'max_speed': float(np.max(speeds)),
                    'min_speed': float(np.min(speeds)),
                    'avg_speed': float(np.mean(speeds)),
                    'std_speed': float(np.std(speeds))
                },
                'acceleration_stats': {
                    'max_acceleration': float(np.max(np.linalg.norm(accelerations, axis=1))),
                    'avg_acceleration': float(np.mean(np.linalg.norm(accelerations, axis=1)))
                },
                'path_analysis': {
                    'total_distance': float(trajectory_points[-1].cumulative_distance),
                    'straight_line_distance': float(np.linalg.norm(positions[-1] - positions[0])),
                    'path_efficiency': float(np.linalg.norm(positions[-1] - positions[0]) / trajectory_points[-1].cumulative_distance),
                    'max_curvature': float(np.max(curvatures)) if len(curvatures) > 0 else 0.0,
                    'avg_curvature': float(np.mean(curvatures)) if len(curvatures) > 0 else 0.0
                },
                'bounding_box': {
                    'min': positions.min(axis=0).tolist(),
                    'max': positions.max(axis=0).tolist(),
                    'center': positions.mean(axis=0).tolist(),
                    'size': (positions.max(axis=0) - positions.min(axis=0)).tolist()
                }
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Advanced analysis computation failed: {e}")
            return {}
    
    def _compute_curvature(self, positions: np.ndarray) -> np.ndarray:
        """Compute curvature along trajectory"""
        if len(positions) < 3:
            return np.array([])
        
        try:
            # Compute first and second derivatives
            dx = np.gradient(positions[:, 0])
            dy = np.gradient(positions[:, 1])
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            
            # Compute curvature
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
            
            # Handle division by zero
            curvature = np.nan_to_num(curvature)
            
            return curvature
            
        except Exception as e:
            self.logger.error(f"Curvature computation failed: {e}")
            return np.array([])
    
    def export_kml(self, trajectory_points: List, filename: str, 
                  name: str = "RealSense Trajectory"):
        """Export trajectory to KML format for Google Earth"""
        try:
            kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{name}</name>
    <description>Trajectory from RealSense D435i Visual Odometry</description>
    <Style id="yellowLineGreenPoly">
      <LineStyle>
        <color>7f00ffff</color>
        <width>4</width>
      </LineStyle>
      <PolyStyle>
        <color>7f00ff00</color>
      </PolyStyle>
    </Style>
    <Placemark>
      <name>Trajectory Path</name>
      <description>Complete trajectory path</description>
      <styleUrl>#yellowLineGreenPoly</styleUrl>
      <LineString>
        <extrude>1</extrude>
        <tessellate>1</tessellate>
        <altitudeMode>absolute</altitudeMode>
        <coordinates>
"""
            
            # Add coordinate points (assuming positions are in local coordinates)
            # In a real application, you'd need to convert to GPS coordinates
            for point in trajectory_points:
                # This is a placeholder - you'd need proper coordinate transformation
                lon = point.position[0] / 111320.0  # Rough conversion for demo
                lat = point.position[1] / 110540.0  # Rough conversion for demo
                alt = point.position[2]
                kml_content += f"          {lon},{lat},{alt}\n"
            
            kml_content += """        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>"""
            
            with open(filename, 'w') as kmlfile:
                kmlfile.write(kml_content)
            
            self.logger.info(f"KML exported: {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"KML export failed: {e}")
            return False
    
    def create_export_summary(self, export_dir: str, session_name: str) -> str:
        """Create export summary file"""
        try:
            summary_file = os.path.join(export_dir, f"{session_name}_summary.txt")
            
            with open(summary_file, 'w') as f:
                f.write(f"RealSense D435i Visual Odometry Export Summary\n")
                f.write(f"=" * 50 + "\n\n")
                f.write(f"Session: {session_name}\n")
                f.write(f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Export Directory: {export_dir}\n\n")
                
                f.write(f"Files in this export:\n")
                f.write(f"- trajectory.json: Complete trajectory data\n")
                f.write(f"- trajectory.csv: Trajectory in spreadsheet format\n")
                f.write(f"- analysis_report.json: Detailed analysis\n")
                f.write(f"- summary.txt: This summary file\n\n")
                
                f.write(f"Data Format Information:\n")
                f.write(f"- Positions in meters (X, Y, Z)\n")
                f.write(f"- Timestamps in Unix time\n")
                f.write(f"- Orientations as 3x3 rotation matrices\n")
                f.write(f"- Velocities in m/s\n\n")
                
                f.write(f"Coordinate System:\n")
                f.write(f"- X: Right (camera frame)\n")
                f.write(f"- Y: Down (camera frame)\n")
                f.write(f"- Z: Forward (camera frame)\n")
            
            return summary_file
            
        except Exception as e:
            self.logger.error(f"Summary creation failed: {e}")
            return ""