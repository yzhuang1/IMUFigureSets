"""
Error Monitor System
Monitors log output for ERROR messages and terminates program execution
"""

import sys
import logging
import threading
import io
import atexit
from typing import Set, List, Callable

class ErrorTerminator:
    """
    Monitors logging output and stdout/stderr for ERROR messages.
    Terminates the program immediately when ERROR is detected.
    """
    
    def __init__(self, 
                 terminate_on_error: bool = True,
                 error_keywords: List[str] = None,
                 ignore_patterns: List[str] = None):
        """
        Initialize the error terminator.
        
        Args:
            terminate_on_error: Whether to terminate on error detection
            error_keywords: List of keywords that trigger termination (default: ["ERROR"])
            ignore_patterns: List of patterns to ignore (case-insensitive)
        """
        self.terminate_on_error = terminate_on_error
        self.error_keywords = error_keywords or ["ERROR"]
        # Enhanced ignore patterns to avoid false positives
        default_ignore_patterns = [
            "error handling", "error detection", "no error", "error rate", 
            "error analysis", "[notice]", "❌ error:", "error:", 
            "pipeline error", "compression failed", "❌ compression failed",
            "failed to load", "could not load", "not found", "❌ pipeline error",
            "critical error detected", "terminating program", "🚨",
            "error monitoring", "monitoring activated", "terminate on",
            
            # Training errors that should trigger retry cycles, not termination
            "training execution failed", "hidden_size must be divisible", 
            "training function generation attempt", "debug attempt", 
            "bo training objective failed", "model architecture validation failed",
            "hyperparameter constraint", "divisible by nheads", "divisible by num_heads",
            "failed to parse", "generation attempt", "regenerating code"
        ]
        self.ignore_patterns = [p.lower() for p in (ignore_patterns or [])] + default_ignore_patterns
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.is_monitoring = False
        self.detected_errors: List[str] = []
        
        # Custom logging handler
        self.log_handler = None
        
    def start_monitoring(self):
        """Start monitoring stdout, stderr, and logging for ERROR messages"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        
        # Replace stdout and stderr with monitoring versions
        sys.stdout = self._create_monitoring_stream(self.original_stdout, "STDOUT")
        sys.stderr = self._create_monitoring_stream(self.original_stderr, "STDERR")
        
        # Add logging handler with high priority
        self.log_handler = ErrorDetectionHandler(self)
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        # Set level to ensure we catch ERROR messages
        self.log_handler.setLevel(logging.ERROR)
        
        # Register cleanup on exit
        atexit.register(self.stop_monitoring)
        
        print("🔍 Error monitoring activated - program will terminate on critical errors")
        
    def stop_monitoring(self):
        """Stop monitoring and restore original streams"""
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        
        # Restore original streams
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Remove logging handler
        if self.log_handler:
            logging.getLogger().removeHandler(self.log_handler)
            self.log_handler = None
            
    def _create_monitoring_stream(self, original_stream, stream_name):
        """Create a monitoring wrapper for stdout/stderr"""
        class MonitoringStream:
            def __init__(self, original, terminator, name):
                self.original = original
                self.terminator = terminator
                self.name = name
                
            def write(self, text):
                # Write to original stream first
                self.original.write(text)
                self.original.flush()
                
                # Check for errors
                self.terminator._check_for_error(text, self.name)
                
            def flush(self):
                self.original.flush()
                
            def __getattr__(self, name):
                return getattr(self.original, name)
                
        return MonitoringStream(original_stream, self, stream_name)
        
    def _check_for_error(self, text: str, source: str):
        """Check text for ERROR patterns and terminate if found"""
        if not self.terminate_on_error:
            return
            
        # Check for ERROR pattern (case-sensitive)
        for keyword in self.error_keywords:
            if keyword in text:
                # Check if this should be ignored (case-insensitive for ignore patterns)
                should_ignore = False
                text_lower = text.lower()
                for ignore_pattern in self.ignore_patterns:
                    if ignore_pattern in text_lower:
                        should_ignore = True
                        break
                
                if not should_ignore:
                    error_msg = f"ERROR pattern '{keyword}' detected in {source}: {text.strip()}"
                    self.detected_errors.append(error_msg)
                    self._terminate_program(error_msg)
                    return
                
    def _terminate_program(self, error_msg: str):
        """Terminate the program due to error detection"""
        # Stop monitoring immediately to prevent recursion
        self.stop_monitoring()
        
        # Print termination message to original streams to avoid triggering monitoring
        self.original_stdout.write(f"\n{'='*80}\n")
        self.original_stdout.write(f"🚨 CRITICAL ERROR DETECTED - TERMINATING PROGRAM\n")
        self.original_stdout.write(f"{'='*80}\n")
        self.original_stdout.write(f"Error message: {error_msg}\n")
        self.original_stdout.write(f"Total errors detected: {len(self.detected_errors)}\n")
        self.original_stdout.write(f"Termination reason: Error monitoring detected ERROR keyword\n")
        self.original_stdout.write(f"{'='*80}\n")
        self.original_stdout.flush()
        
        # Force exit
        sys.exit(1)

class ErrorDetectionHandler(logging.Handler):
    """Custom logging handler that monitors for ERROR level messages"""
    
    def __init__(self, terminator: ErrorTerminator):
        super().__init__()
        self.terminator = terminator
        
    def emit(self, record):
        """Check log records for ERROR level messages"""
        try:
            # Check if this is an ERROR level message
            if record.levelno >= logging.ERROR:
                msg = self.format(record)
                self.terminator._check_for_error(msg, "LOGGING")
        except Exception:
            # Don't let errors in error detection break the program
            pass

# Global error terminator instance
_global_terminator = None

def enable_error_termination(error_keywords: List[str] = None, 
                           ignore_patterns: List[str] = None):
    """
    Enable global error termination monitoring.
    
    Args:
        error_keywords: List of keywords that trigger termination (default: ["ERROR"])
        ignore_patterns: List of patterns to ignore
    """
    global _global_terminator
    
    if _global_terminator is None:
        _global_terminator = ErrorTerminator(
            terminate_on_error=True,
            error_keywords=error_keywords,
            ignore_patterns=ignore_patterns
        )
        _global_terminator.start_monitoring()
    
def disable_error_termination():
    """Disable global error termination monitoring"""
    global _global_terminator
    
    if _global_terminator:
        _global_terminator.stop_monitoring()
        _global_terminator = None

def is_error_monitoring_active() -> bool:
    """Check if error monitoring is currently active"""
    global _global_terminator
    return _global_terminator is not None and _global_terminator.is_monitoring

def temporarily_disable_error_monitoring():
    """Temporarily disable error monitoring (for debug cycles)"""
    global _global_terminator
    if _global_terminator:
        _global_terminator.terminate_on_error = False
        print("🔧 Error monitoring temporarily disabled for debug cycles")

def re_enable_error_monitoring():
    """Re-enable error monitoring after debug cycles"""
    global _global_terminator
    if _global_terminator:
        _global_terminator.terminate_on_error = True
        print("🔍 Error monitoring re-enabled after debug cycles")

# Convenience function for enabling with common patterns
def enable_strict_error_monitoring():
    """Enable error monitoring with strict settings for critical errors only"""
    enable_error_termination(
        error_keywords=["ERROR"],
        ignore_patterns=[]
    )

if __name__ == "__main__":
    # Test the error monitoring system
    print("Testing Error Monitoring System")
    
    # Enable monitoring
    enable_strict_error_monitoring()
    
    print("This is a normal message")
    print("This is a warning message")
    
    # This should trigger termination
    print("This contains an ERROR message - program should terminate")