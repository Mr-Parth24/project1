"""
Performance Monitoring System
Author: Mr-Parth24
Date: 2025-06-13
"""

import time
import psutil
import threading
from collections import deque
from typing import Dict, Any, List
import logging

class PerformanceMonitor:
    """System performance monitoring"""
    
    def __init__(self, window_size: int = 100):
        self.logger = logging.getLogger(__name__)
        self.window_size = window_size
        
        # Performance metrics
        self.frame_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.feature_counts = deque(maxlen=window_size)
        self.match_counts = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.cpu_usage = deque(maxlen=window_size)
        
        # Timing
        self.frame_start_time = None
        self.total_frames = 0
        self.start_time = time.time()
        
        # System monitoring
        self.system_monitor_thread = None
        self.monitoring_active = False
        
        # Start system monitoring
        self._start_system_monitoring()
        
        self.logger.info("Performance monitor initialized")
    
    def _start_system_monitoring(self):
        """Start system resource monitoring"""
        self.monitoring_active = True
        self.system_monitor_thread = threading.Thread(target=self._system_monitor_worker, daemon=True)
        self.system_monitor_thread.start()
    
    def _system_monitor_worker(self):
        """Monitor system resources"""
        while self.monitoring_active:
            try:
                # Get system metrics
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=None)
                
                self.memory_usage.append(memory_percent)
                self.cpu_usage.append(cpu_percent)
                
                time.sleep(1.0)  # Update every second
                
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                time.sleep(5.0)
    
    def start_frame(self):
        """Start timing a frame"""
        self.frame_start_time = time.time()
    
    def end_frame(self, feature_count: int = 0, match_count: int = 0):
        """End timing a frame"""
        if self.frame_start_time is not None:
            frame_time = time.time() - self.frame_start_time
            self.frame_times.append(frame_time)
            self.feature_counts.append(feature_count)
            self.match_counts.append(match_count)
            
            self.total_frames += 1
            self.frame_start_time = None
    
    def get_fps(self) -> float:
        """Get current FPS"""
        if len(self.frame_times) == 0:
            return 0.0
        
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        total_time = time.time() - self.start_time
        
        stats = {
            'fps': self.get_fps(),
            'total_frames': self.total_frames,
            'total_time': total_time,
            'avg_features': sum(self.feature_counts) / len(self.feature_counts) if self.feature_counts else 0,
            'avg_matches': sum(self.match_counts) / len(self.match_counts) if self.match_counts else 0,
            'memory_usage': list(self.memory_usage)[-10:] if self.memory_usage else [],
            'cpu_usage': list(self.cpu_usage)[-10:] if self.cpu_usage else [],
            'current_memory': self.memory_usage[-1] if self.memory_usage else 0,
            'current_cpu': self.cpu_usage[-1] if self.cpu_usage else 0
        }
        
        return stats
    
    def cleanup(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.system_monitor_thread and self.system_monitor_thread.is_alive():
            self.system_monitor_thread.join(timeout=2.0)