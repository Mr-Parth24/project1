"""
Comprehensive Error Handling and Recovery System
Author: Mr-Parth24
Date: 2025-06-13
"""

import logging
import traceback
import time
from typing import Dict, List, Any, Callable, Optional
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorRecord:
    timestamp: float
    error_type: str
    severity: ErrorSeverity
    message: str
    traceback: str
    component: str
    recovery_attempted: bool = False
    recovery_successful: bool = False

class ErrorHandler:
    """Comprehensive error handling and recovery system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_history = []
        self.error_counts = defaultdict(int)
        self.component_errors = defaultdict(list)
        
        # Recovery strategies
        self.recovery_strategies = {}
        self.max_recovery_attempts = 3
        
        # Error thresholds
        self.error_thresholds = {
            ErrorSeverity.LOW: 10,      # 10 low errors before escalation
            ErrorSeverity.MEDIUM: 5,    # 5 medium errors before escalation
            ErrorSeverity.HIGH: 2,      # 2 high errors before escalation
            ErrorSeverity.CRITICAL: 1   # 1 critical error triggers immediate action
        }
        
        # Callbacks for error notifications
        self.error_callbacks = {}
        
        self._register_default_strategies()
        
        self.logger.info("Error handler initialized")
    
    def _register_default_strategies(self):
        """Register default recovery strategies"""
        self.register_recovery_strategy("camera_error", self._recover_camera_error)
        self.register_recovery_strategy("feature_detection_error", self._recover_feature_error)
        self.register_recovery_strategy("pose_estimation_error", self._recover_pose_error)
        self.register_recovery_strategy("memory_error", self._recover_memory_error)
        self.register_recovery_strategy("visualization_error", self._recover_visualization_error)
    
    def handle_error(self, error: Exception, component: str = "unknown", 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Optional[Dict[str, Any]] = None) -> bool:
        """Handle an error with automatic recovery attempts"""
        try:
            # Create error record
            error_record = ErrorRecord(
                timestamp=time.time(),
                error_type=type(error).__name__,
                severity=severity,
                message=str(error),
                traceback=traceback.format_exc(),
                component=component
            )
            
            # Log error
            self._log_error(error_record, context)
            
            # Track error
            self._track_error(error_record)
            
            # Attempt recovery
            recovery_success = self._attempt_recovery(error_record, context)
            
            # Check if error threshold exceeded
            self._check_error_thresholds(error_record)
            
            # Execute callbacks
            self._execute_error_callbacks(error_record, context)
            
            return recovery_success
            
        except Exception as e:
            self.logger.critical(f"Error handler failed: {e}")
            return False
    
    def _log_error(self, error_record: ErrorRecord, context: Optional[Dict[str, Any]]):
        """Log error with appropriate level"""
        log_message = f"[{error_record.component}] {error_record.error_type}: {error_record.message}"
        
        if context:
            log_message += f" | Context: {context}"
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Log full traceback for debugging
        if error_record.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            self.logger.debug(f"Full traceback:\n{error_record.traceback}")
    
    def _track_error(self, error_record: ErrorRecord):
        """Track error statistics"""
        # Add to history
        self.error_history.append(error_record)
        
        # Update counts
        self.error_counts[error_record.error_type] += 1
        self.component_errors[error_record.component].append(error_record)
        
        # Keep history manageable
        max_history = 1000
        if len(self.error_history) > max_history:
            self.error_history = self.error_history[-max_history:]
    
    def _attempt_recovery(self, error_record: ErrorRecord, 
                         context: Optional[Dict[str, Any]]) -> bool:
        """Attempt to recover from error"""
        try:
            # Determine recovery strategy
            strategy_key = self._get_recovery_strategy_key(error_record)
            
            if strategy_key in self.recovery_strategies:
                recovery_function = self.recovery_strategies[strategy_key]
                
                # Attempt recovery
                error_record.recovery_attempted = True
                recovery_success = recovery_function(error_record, context)
                error_record.recovery_successful = recovery_success
                
                if recovery_success:
                    self.logger.info(f"Recovery successful for {error_record.error_type}")
                else:
                    self.logger.warning(f"Recovery failed for {error_record.error_type}")
                
                return recovery_success
            else:
                self.logger.warning(f"No recovery strategy for {error_record.error_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            return False
    
    def _get_recovery_strategy_key(self, error_record: ErrorRecord) -> str:
        """Determine recovery strategy key based on error"""
        component = error_record.component.lower()
        error_type = error_record.error_type.lower()
        
        # Map errors to recovery strategies
        if "camera" in component or "realsense" in error_record.message.lower():
            return "camera_error"
        elif "feature" in component or "orb" in error_record.message.lower():
            return "feature_detection_error"
        elif "pose" in component or "pnp" in error_record.message.lower():
            return "pose_estimation_error"
        elif "memory" in error_type or "out of memory" in error_record.message.lower():
            return "memory_error"
        elif "visualization" in component or "opencv" in error_record.message.lower():
            return "visualization_error"
        else:
            return "generic_error"
    
# ... (previous code continues)

    def _recover_camera_error(self, error_record: ErrorRecord, 
                             context: Optional[Dict[str, Any]]) -> bool:
        """Recover from camera-related errors"""
        try:
            self.logger.info("Attempting camera error recovery...")
            
            # Wait before retry
            time.sleep(1.0)
            
            # If camera manager is available in context, try to reinitialize
            if context and 'camera_manager' in context:
                camera_manager = context['camera_manager']
                
                # Stop current pipeline
                if hasattr(camera_manager, 'cleanup'):
                    camera_manager.cleanup()
                
                time.sleep(2.0)
                
                # Reinitialize
                if hasattr(camera_manager, 'initialize'):
                    return camera_manager.initialize()
            
            return False
            
        except Exception as e:
            self.logger.error(f"Camera recovery failed: {e}")
            return False
    
    def _recover_feature_error(self, error_record: ErrorRecord, 
                              context: Optional[Dict[str, Any]]) -> bool:
        """Recover from feature detection errors"""
        try:
            self.logger.info("Attempting feature detection recovery...")
            
            # Try different feature detector
            if context and 'feature_processor' in context:
                feature_processor = context['feature_processor']
                
                # Switch to more robust detector temporarily
                if hasattr(feature_processor, 'switch_detector'):
                    return feature_processor.switch_detector('sift')
            
            return True  # Non-critical error, continue processing
            
        except Exception as e:
            self.logger.error(f"Feature recovery failed: {e}")
            return False
    
    def _recover_pose_error(self, error_record: ErrorRecord, 
                           context: Optional[Dict[str, Any]]) -> bool:
        """Recover from pose estimation errors"""
        try:
            self.logger.info("Attempting pose estimation recovery...")
            
            # Use motion model prediction
            if context and 'pose_estimator' in context:
                pose_estimator = context['pose_estimator']
                
                if hasattr(pose_estimator, 'use_motion_prediction'):
                    return pose_estimator.use_motion_prediction()
            
            return True  # Continue with reduced accuracy
            
        except Exception as e:
            self.logger.error(f"Pose recovery failed: {e}")
            return False
    
    def _recover_memory_error(self, error_record: ErrorRecord, 
                             context: Optional[Dict[str, Any]]) -> bool:
        """Recover from memory-related errors"""
        try:
            self.logger.info("Attempting memory error recovery...")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Reduce buffer sizes if possible
            if context:
                for key, component in context.items():
                    if hasattr(component, 'reduce_memory_usage'):
                        component.reduce_memory_usage()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Memory recovery failed: {e}")
            return False
    
    def _recover_visualization_error(self, error_record: ErrorRecord, 
                                    context: Optional[Dict[str, Any]]) -> bool:
        """Recover from visualization errors"""
        try:
            self.logger.info("Attempting visualization recovery...")
            
            # Disable problematic visualization components
            if context and 'visualizer' in context:
                visualizer = context['visualizer']
                
                if hasattr(visualizer, 'disable_3d_plot'):
                    visualizer.disable_3d_plot()
                    return True
            
            return True  # Non-critical, system can continue
            
        except Exception as e:
            self.logger.error(f"Visualization recovery failed: {e}")
            return False
    
    def _check_error_thresholds(self, error_record: ErrorRecord):
        """Check if error thresholds are exceeded"""
        try:
            # Count recent errors of this severity
            recent_time = time.time() - 60  # Last minute
            recent_errors = [
                err for err in self.error_history 
                if err.timestamp > recent_time and err.severity == error_record.severity
            ]
            
            threshold = self.error_thresholds[error_record.severity]
            
            if len(recent_errors) >= threshold:
                self.logger.critical(
                    f"Error threshold exceeded: {len(recent_errors)} "
                    f"{error_record.severity.value} errors in last minute"
                )
                
                # Trigger escalation
                self._escalate_error_handling(error_record, recent_errors)
                
        except Exception as e:
            self.logger.error(f"Error threshold check failed: {e}")
    
    def _escalate_error_handling(self, error_record: ErrorRecord, recent_errors: List[ErrorRecord]):
        """Escalate error handling for serious issues"""
        try:
            if error_record.severity == ErrorSeverity.CRITICAL:
                self.logger.critical("CRITICAL ERROR THRESHOLD EXCEEDED - INITIATING SHUTDOWN")
                # Could trigger system shutdown or safe mode
                
            elif error_record.severity == ErrorSeverity.HIGH:
                self.logger.error("HIGH ERROR THRESHOLD EXCEEDED - ENTERING SAFE MODE")
                # Could reduce system functionality
                
            # Execute escalation callbacks
            for callback_name, callback_func in self.error_callbacks.items():
                try:
                    callback_func("escalation", {
                        'severity': error_record.severity,
                        'recent_errors': recent_errors,
                        'component': error_record.component
                    })
                except Exception as e:
                    self.logger.error(f"Escalation callback {callback_name} failed: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error escalation failed: {e}")
    
    def _execute_error_callbacks(self, error_record: ErrorRecord, 
                                context: Optional[Dict[str, Any]]):
        """Execute registered error callbacks"""
        for callback_name, callback_func in self.error_callbacks.items():
            try:
                callback_func("error", {
                    'error_record': error_record,
                    'context': context
                })
            except Exception as e:
                self.logger.error(f"Error callback {callback_name} failed: {e}")
    
    def register_recovery_strategy(self, error_key: str, recovery_function: Callable):
        """Register a recovery strategy for specific error types"""
        self.recovery_strategies[error_key] = recovery_function
        self.logger.info(f"Registered recovery strategy for {error_key}")
    
    def register_error_callback(self, name: str, callback: Callable):
        """Register an error notification callback"""
        self.error_callbacks[name] = callback
        self.logger.info(f"Registered error callback: {name}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        try:
            recent_time = time.time() - 3600  # Last hour
            recent_errors = [err for err in self.error_history if err.timestamp > recent_time]
            
            severity_counts = defaultdict(int)
            for error in recent_errors:
                severity_counts[error.severity.value] += 1
            
            return {
                'total_errors': len(self.error_history),
                'recent_errors': len(recent_errors),
                'error_by_type': dict(self.error_counts),
                'error_by_severity': dict(severity_counts),
                'error_by_component': {
                    comp: len(errors) for comp, errors in self.component_errors.items()
                },
                'recovery_success_rate': self._calculate_recovery_success_rate()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get error statistics: {e}")
            return {}
    
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate recovery success rate"""
        try:
            recovery_attempts = [
                err for err in self.error_history if err.recovery_attempted
            ]
            
            if not recovery_attempts:
                return 0.0
            
            successful_recoveries = [
                err for err in recovery_attempts if err.recovery_successful
            ]
            
            return len(successful_recoveries) / len(recovery_attempts)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate recovery success rate: {e}")
            return 0.0                 