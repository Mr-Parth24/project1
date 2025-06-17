"""
Enhanced logging system for comprehensive debugging and monitoring
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
import json
import traceback
from typing import Dict, Any

class EnhancedFormatter(logging.Formatter):
    """Enhanced formatter with color support and detailed information"""
    
    # Color codes for terminal output
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Add color for console output
        if hasattr(record, 'color') and record.color:
            color = self.COLORS.get(record.levelname, '')
            reset = self.COLORS['RESET']
            record.levelname = f"{color}{record.levelname}{reset}"
        
        # Add extra context information
        if hasattr(record, 'component'):
            record.msg = f"[{record.component}] {record.msg}"
        
        return super().format(record)

class PerformanceLogger:
    """Logger for performance metrics and statistics"""
    
    def __init__(self, log_file: str = "logs/performance.json"):
        self.log_file = log_file
        self.metrics = {
            'session_start': datetime.now().isoformat(),
            'frames_processed': 0,
            'tracking_successes': 0,
            'tracking_failures': 0,
            'average_processing_time': 0.0,
            'peak_memory_usage': 0.0,
            'features_detected': [],
            'pose_updates': [],
            'errors': []
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log_frame_processing(self, processing_time: float, success: bool, 
                           features_detected: int, pose_updated: bool):
        """Log frame processing metrics"""
        self.metrics['frames_processed'] += 1
        
        if success:
            self.metrics['tracking_successes'] += 1
        else:
            self.metrics['tracking_failures'] += 1
        
        # Update average processing time
        current_avg = self.metrics['average_processing_time']
        frame_count = self.metrics['frames_processed']
        self.metrics['average_processing_time'] = (
            (current_avg * (frame_count - 1) + processing_time) / frame_count
        )
        
        # Log features detected
        self.metrics['features_detected'].append({
            'timestamp': datetime.now().isoformat(),
            'count': features_detected
        })
        
        # Log pose updates
        if pose_updated:
            self.metrics['pose_updates'].append({
                'timestamp': datetime.now().isoformat(),
                'frame_number': frame_count
            })
        
        # Keep only recent data (last 1000 entries)
        if len(self.metrics['features_detected']) > 1000:
            self.metrics['features_detected'] = self.metrics['features_detected'][-500:]
        if len(self.metrics['pose_updates']) > 1000:
            self.metrics['pose_updates'] = self.metrics['pose_updates'][-500:]
    
    def log_error(self, error_type: str, error_message: str, traceback_str: str = None):
        """Log error information"""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message,
            'frame_number': self.metrics['frames_processed']
        }
        
        if traceback_str:
            error_entry['traceback'] = traceback_str
        
        self.metrics['errors'].append(error_entry)
        
        # Keep only recent errors
        if len(self.metrics['errors']) > 100:
            self.metrics['errors'] = self.metrics['errors'][-50:]
    
    def update_memory_usage(self, memory_mb: float):
        """Update peak memory usage"""
        if memory_mb > self.metrics['peak_memory_usage']:
            self.metrics['peak_memory_usage'] = memory_mb
    
    def save_metrics(self):
        """Save metrics to file"""
        try:
            self.metrics['session_end'] = datetime.now().isoformat()
            
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            print(f"Failed to save performance metrics: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_frames = self.metrics['frames_processed']
        success_rate = (self.metrics['tracking_successes'] / max(1, total_frames)) * 100
        
        return {
            'total_frames': total_frames,
            'success_rate': f"{success_rate:.1f}%",
            'average_processing_time': f"{self.metrics['average_processing_time']*1000:.1f}ms",
            'peak_memory_usage': f"{self.metrics['peak_memory_usage']:.1f}MB",
            'total_errors': len(self.metrics['errors']),
            'pose_updates': len(self.metrics['pose_updates'])
        }

def setup_enhanced_logging(log_level: str = "INFO", 
                         console_output: bool = True,
                         file_output: bool = True,
                         performance_logging: bool = True):
    """Setup enhanced logging system"""
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = EnhancedFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(lambda record: setattr(record, 'color', True) or True)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_output:
        file_handler = logging.handlers.RotatingFileHandler(
            'logs/visual_odometry.log',
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_formatter = EnhancedFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        'logs/errors.log',
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = EnhancedFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n%(exc_info)s'
    )
    error_handler.setFormatter(error_formatter)
    root_logger.addHandler(error_handler)
    
    # Performance logger
    if performance_logging:
        perf_logger = PerformanceLogger()
        return perf_logger
    
    return None

# Exception handler for uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    """Handle uncaught exceptions"""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logger = logging.getLogger("UNCAUGHT_EXCEPTION")
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

# Install exception handler
sys.excepthook = handle_exception