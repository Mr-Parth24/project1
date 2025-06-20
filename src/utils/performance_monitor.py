"""
Performance Monitor for Agricultural SLAM System
Real-time performance tracking, optimization, and health monitoring
Provides adaptive performance tuning for field deployment
"""

import time
import threading
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import json
import os

class PerformanceMonitor:
    """
    Comprehensive performance monitoring for agricultural SLAM
    Tracks FPS, memory, CPU, processing times, and agricultural metrics
    """
    
    def __init__(self, history_size: int = 300):
        """Initialize performance monitor"""
        self.history_size = history_size
        
        # Performance data storage
        self.frame_times = deque(maxlen=history_size)
        self.processing_times = deque(maxlen=history_size)
        self.slam_times = deque(maxlen=history_size)
        self.feature_counts = deque(maxlen=history_size)
        self.match_counts = deque(maxlen=history_size)
        self.distance_accuracies = deque(maxlen=history_size)
        
        # System resource tracking
        self.cpu_usage = deque(maxlen=history_size)
        self.memory_usage = deque(maxlen=history_size)
        self.gpu_usage = deque(maxlen=history_size)
        
        # Agricultural specific metrics
        self.agricultural_scores = deque(maxlen=history_size)
        self.crop_row_detections = deque(maxlen=history_size)
        self.tracking_quality = deque(maxlen=history_size)
        
        # Performance thresholds
        self.target_fps = 30.0
        self.min_fps = 15.0
        self.max_cpu_usage = 80.0
        self.max_memory_usage = 4096  # MB
        self.min_feature_count = 100
        
        # Monitoring state
        self.monitoring_active = False
        self.start_time = time.time()
        self.last_update = time.time()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Performance alerts
        self.performance_alerts = []
        self.alert_history = deque(maxlen=50)
        
        print("Performance Monitor initialized")
    
    def start_monitoring(self):
        """Start performance monitoring"""
        with self.lock:
            self.monitoring_active = True
            self.start_time = time.time()
            print("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        with self.lock:
            self.monitoring_active = False
            print("Performance monitoring stopped")
    
    def update_frame_metrics(self, frame_data: Dict):
        """Update frame processing metrics"""
        try:
            current_time = time.time()
            
            with self.lock:
                if not self.monitoring_active:
                    return
                
                # Frame timing
                if 'frame_time' in frame_data:
                    self.frame_times.append(frame_data['frame_time'])
                
                if 'processing_time' in frame_data:
                    self.processing_times.append(frame_data['processing_time'])
                
                if 'slam_time' in frame_data:
                    self.slam_times.append(frame_data['slam_time'])
                
                # Feature metrics
                if 'num_features' in frame_data:
                    self.feature_counts.append(frame_data['num_features'])
                
                if 'num_matches' in frame_data:
                    self.match_counts.append(frame_data['num_matches'])
                
                # Distance accuracy
                if 'distance_accuracy' in frame_data:
                    self.distance_accuracies.append(frame_data['distance_accuracy'])
                
                # Agricultural metrics
                if 'agricultural_score' in frame_data:
                    self.agricultural_scores.append(frame_data['agricultural_score'])
                
                if 'crop_rows_detected' in frame_data:
                    self.crop_row_detections.append(frame_data['crop_rows_detected'])
                
                if 'tracking_quality' in frame_data:
                    self.tracking_quality.append(frame_data['tracking_quality'])
                
                self.last_update = current_time
                
                # Check for performance issues
                self._check_performance_alerts()
        
        except Exception as e:
            print(f"Frame metrics update error: {e}")
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            with self.lock:
                if not self.monitoring_active:
                    return
                
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_usage.append(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)
                self.memory_usage.append(memory_mb)
                
                # GPU usage (if available)
                try:
                    import GPUtil
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_load = gpus[0].load * 100
                        self.gpu_usage.append(gpu_load)
                    else:
                        self.gpu_usage.append(0.0)
                except (ImportError, ModuleNotFoundError):
                    self.gpu_usage.append(0.0)
                except Exception:
                    # Handle any other GPU-related errors
                    self.gpu_usage.append(0.0)
        
        except Exception as e:
            print(f"System metrics update error: {e}")
    
    def get_current_metrics(self) -> Dict:
        """Get current performance metrics"""
        try:
            with self.lock:
                current_time = time.time()
                uptime = current_time - self.start_time
                
                # Calculate FPS
                fps = 0.0
                if len(self.frame_times) >= 2:
                    recent_times = list(self.frame_times)[-10:]
                    if len(recent_times) > 1:
                        avg_frame_time = np.mean(np.diff(recent_times))
                        fps = 1.0 / max(avg_frame_time, 0.001)
                
                # Average processing times
                avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
                avg_slam_time = np.mean(self.slam_times) if self.slam_times else 0.0
                
                # Feature statistics
                avg_features = np.mean(self.feature_counts) if self.feature_counts else 0.0
                avg_matches = np.mean(self.match_counts) if self.match_counts else 0.0
                
                # Agricultural metrics
                avg_agricultural_score = np.mean(self.agricultural_scores) if self.agricultural_scores else 0.0
                crop_row_detection_rate = np.mean(self.crop_row_detections) if self.crop_row_detections else 0.0
                avg_tracking_quality = np.mean(self.tracking_quality) if self.tracking_quality else 0.0
                
                # System resources
                current_cpu = self.cpu_usage[-1] if self.cpu_usage else 0.0
                current_memory = self.memory_usage[-1] if self.memory_usage else 0.0
                current_gpu = self.gpu_usage[-1] if self.gpu_usage else 0.0
                
                # Distance accuracy
                avg_distance_accuracy = np.mean(self.distance_accuracies) if self.distance_accuracies else 0.05
                
                return {
                    'timestamp': current_time,
                    'uptime_seconds': uptime,
                    'performance': {
                        'fps': fps,
                        'target_fps': self.target_fps,
                        'avg_processing_time_ms': avg_processing_time * 1000,
                        'avg_slam_time_ms': avg_slam_time * 1000,
                        'frames_processed': len(self.frame_times)
                    },
                    'features': {
                        'avg_features_detected': avg_features,
                        'avg_matches': avg_matches,
                        'min_feature_threshold': self.min_feature_count
                    },
                    'agricultural': {
                        'avg_agricultural_score': avg_agricultural_score,
                        'crop_row_detection_rate': crop_row_detection_rate,
                        'avg_tracking_quality': avg_tracking_quality,
                        'avg_distance_accuracy_cm': avg_distance_accuracy * 100
                    },
                    'system': {
                        'cpu_usage_percent': current_cpu,
                        'memory_usage_mb': current_memory,
                        'gpu_usage_percent': current_gpu,
                        'memory_limit_mb': self.max_memory_usage
                    },
                    'alerts': {
                        'active_alerts': len(self.performance_alerts),
                        'recent_alerts': list(self.alert_history)[-5:]
                    }
                }
        
        except Exception as e:
            print(f"Get current metrics error: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def get_performance_trends(self, window_size: int = 60) -> Dict:
        """Get performance trends over specified window"""
        try:
            with self.lock:
                if len(self.frame_times) < window_size:
                    window_size = len(self.frame_times)
                
                if window_size < 2:
                    return {'error': 'Insufficient data for trends'}
                
                # Recent data
                recent_processing = list(self.processing_times)[-window_size:]
                recent_features = list(self.feature_counts)[-window_size:]
                recent_agricultural = list(self.agricultural_scores)[-window_size:]
                recent_cpu = list(self.cpu_usage)[-window_size:]
                
                # Calculate trends (positive = improving, negative = degrading)
                processing_trend = self._calculate_trend(recent_processing)
                feature_trend = self._calculate_trend(recent_features)
                agricultural_trend = self._calculate_trend(recent_agricultural)
                cpu_trend = self._calculate_trend(recent_cpu)
                
                return {
                    'window_size': window_size,
                    'trends': {
                        'processing_time': -processing_trend,  # Negative because lower is better
                        'feature_count': feature_trend,
                        'agricultural_score': agricultural_trend,
                        'cpu_usage': -cpu_trend  # Negative because lower is better
                    },
                    'stability': {
                        'processing_stability': 1.0 - (np.std(recent_processing) / max(np.mean(recent_processing), 0.001)),
                        'feature_stability': 1.0 - (np.std(recent_features) / max(np.mean(recent_features), 1)),
                        'agricultural_stability': 1.0 - (np.std(recent_agricultural) / max(np.mean(recent_agricultural), 0.001))
                    }
                }
        
        except Exception as e:
            print(f"Performance trends error: {e}")
            return {'error': str(e)}
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend direction (-1 to 1)"""
        try:
            if len(data) < 2:
                return 0.0
            
            # Simple linear regression slope
            x = np.arange(len(data))
            y = np.array(data)
            
            # Calculate slope
            n = len(data)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x * x) - np.sum(x) ** 2)
            
            # Normalize slope to -1 to 1 range
            max_change = max(abs(max(data) - min(data)), 0.001)
            normalized_slope = np.clip(slope / max_change, -1.0, 1.0)
            
            return normalized_slope
        
        except Exception as e:
            print(f"Trend calculation error: {e}")
            return 0.0
    
    def _check_performance_alerts(self):
        """Check for performance issues and generate alerts"""
        try:
            current_time = time.time()
            new_alerts = []
            
            # Check FPS
            if len(self.frame_times) >= 10:
                recent_times = list(self.frame_times)[-10:]
                avg_frame_time = np.mean(np.diff(recent_times))
                current_fps = 1.0 / max(avg_frame_time, 0.001)
                
                if current_fps < self.min_fps:
                    new_alerts.append({
                        'type': 'low_fps',
                        'severity': 'warning' if current_fps > self.min_fps * 0.7 else 'critical',
                        'message': f'Low FPS: {current_fps:.1f} (target: {self.target_fps})',
                        'timestamp': current_time
                    })
            
            # Check CPU usage
            if self.cpu_usage and self.cpu_usage[-1] > self.max_cpu_usage:
                new_alerts.append({
                    'type': 'high_cpu',
                    'severity': 'warning' if self.cpu_usage[-1] < 90 else 'critical',
                    'message': f'High CPU usage: {self.cpu_usage[-1]:.1f}%',
                    'timestamp': current_time
                })
            
            # Check memory usage
            if self.memory_usage and self.memory_usage[-1] > self.max_memory_usage:
                new_alerts.append({
                    'type': 'high_memory',
                    'severity': 'warning',
                    'message': f'High memory usage: {self.memory_usage[-1]:.0f}MB',
                    'timestamp': current_time
                })
            
            # Check feature count
            if self.feature_counts and self.feature_counts[-1] < self.min_feature_count:
                new_alerts.append({
                    'type': 'low_features',
                    'severity': 'warning',
                    'message': f'Low feature count: {self.feature_counts[-1]}',
                    'timestamp': current_time
                })
            
            # Add new alerts
            for alert in new_alerts:
                if not self._is_duplicate_alert(alert):
                    self.performance_alerts.append(alert)
                    self.alert_history.append(alert)
            
            # Clean old alerts (remove alerts older than 30 seconds)
            self.performance_alerts = [
                alert for alert in self.performance_alerts 
                if current_time - alert['timestamp'] < 30.0
            ]
        
        except Exception as e:
            print(f"Performance alert check error: {e}")
    
    def _is_duplicate_alert(self, new_alert: Dict) -> bool:
        """Check if alert is duplicate of recent alert"""
        try:
            current_time = time.time()
            
            for alert in self.performance_alerts:
                if (alert['type'] == new_alert['type'] and 
                    current_time - alert['timestamp'] < 10.0):  # 10 second window
                    return True
            
            return False
        
        except Exception as e:
            print(f"Duplicate alert check error: {e}")
            return False
    
    def get_optimization_recommendations(self) -> List[Dict]:
        """Get performance optimization recommendations"""
        try:
            recommendations = []
            metrics = self.get_current_metrics()
            
            # FPS optimization
            current_fps = metrics['performance']['fps']
            if current_fps < self.target_fps * 0.8:
                if metrics['system']['cpu_usage_percent'] > 70:
                    recommendations.append({
                        'type': 'reduce_processing',
                        'priority': 'high',
                        'recommendation': 'Reduce feature detection count or enable frame skipping',
                        'expected_improvement': 'Increase FPS by 20-30%'
                    })
                
                if metrics['features']['avg_features_detected'] > 800:
                    recommendations.append({
                        'type': 'optimize_features',
                        'priority': 'medium',
                        'recommendation': 'Reduce maximum feature count to 500-600',
                        'expected_improvement': 'Increase FPS by 10-15%'
                    })
            
            # Memory optimization
            if metrics['system']['memory_usage_mb'] > self.max_memory_usage * 0.8:
                recommendations.append({
                    'type': 'reduce_memory',
                    'priority': 'medium',
                    'recommendation': 'Reduce trajectory history or enable data compression',
                    'expected_improvement': 'Reduce memory usage by 20-30%'
                })
            
            # Agricultural performance
            if metrics['agricultural']['avg_agricultural_score'] < 0.5:
                recommendations.append({
                    'type': 'improve_agricultural',
                    'priority': 'medium',
                    'recommendation': 'Improve lighting conditions or adjust camera position',
                    'expected_improvement': 'Better agricultural scene understanding'
                })
            
            # Distance accuracy
            if metrics['agricultural']['avg_distance_accuracy_cm'] > 5.0:
                recommendations.append({
                    'type': 'improve_accuracy',
                    'priority': 'high',
                    'recommendation': 'Calibrate camera or improve feature matching',
                    'expected_improvement': 'Reduce distance error to <3cm'
                })
            
            return recommendations
        
        except Exception as e:
            print(f"Optimization recommendations error: {e}")
            return []
    
    def export_performance_report(self, filepath: str = None) -> str:
        """Export comprehensive performance report"""
        try:
            if filepath is None:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filepath = f"performance_report_{timestamp}.json"
            
            report = {
                'report_metadata': {
                    'generated_at': time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    'monitoring_duration_hours': (time.time() - self.start_time) / 3600,
                    'total_frames_processed': len(self.frame_times)
                },
                'current_metrics': self.get_current_metrics(),
                'performance_trends': self.get_performance_trends(),
                'optimization_recommendations': self.get_optimization_recommendations(),
                'alert_summary': {
                    'total_alerts': len(self.alert_history),
                    'alert_types': self._summarize_alert_types(),
                    'recent_alerts': list(self.alert_history)[-10:]
                },
                'raw_data': {
                    'frame_times': list(self.frame_times)[-100:],
                    'processing_times': list(self.processing_times)[-100:],
                    'feature_counts': list(self.feature_counts)[-100:],
                    'agricultural_scores': list(self.agricultural_scores)[-100:],
                    'cpu_usage': list(self.cpu_usage)[-100:],
                    'memory_usage': list(self.memory_usage)[-100:]
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"Performance report exported: {filepath}")
            return filepath
        
        except Exception as e:
            print(f"Performance report export error: {e}")
            return ""
    
    def _summarize_alert_types(self) -> Dict:
        """Summarize alert types and frequencies"""
        try:
            alert_counts = {}
            for alert in self.alert_history:
                alert_type = alert.get('type', 'unknown')
                alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
            
            return alert_counts
        
        except Exception as e:
            print(f"Alert summary error: {e}")
            return {}
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        with self.lock:
            self.frame_times.clear()
            self.processing_times.clear()
            self.slam_times.clear()
            self.feature_counts.clear()
            self.match_counts.clear()
            self.distance_accuracies.clear()
            self.cpu_usage.clear()
            self.memory_usage.clear()
            self.gpu_usage.clear()
            self.agricultural_scores.clear()
            self.crop_row_detections.clear()
            self.tracking_quality.clear()
            self.performance_alerts.clear()
            self.alert_history.clear()
            self.start_time = time.time()
            
            print("Performance metrics reset")

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    return performance_monitor

def start_performance_monitoring():
    """Start global performance monitoring"""
    performance_monitor.start_monitoring()

def stop_performance_monitoring():
    """Stop global performance monitoring"""
    performance_monitor.stop_monitoring()

def update_frame_performance(frame_data: Dict):
    """Update frame performance metrics globally"""
    performance_monitor.update_frame_metrics(frame_data)

def get_current_performance() -> Dict:
    """Get current performance metrics globally"""
    return performance_monitor.get_current_metrics()