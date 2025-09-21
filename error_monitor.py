"""
Error Monitor System
Monitors log output for ERROR messages and terminates program execution
"""

import sys
import logging
import threading
import io
import atexit
import time
import os
from pathlib import Path
from typing import Set, List, Callable, Optional


class LogFileMonitor:
    """Monitor log files for ERROR messages"""

    def __init__(self, error_terminator):
        self.error_terminator = error_terminator
        self.monitoring = False
        self.monitor_thread = None
        self.log_file_path = None
        self.last_processed_line = 0  # Track which lines we've already processed

    def start_monitoring(self, log_file_path: str):
        """Start monitoring the specified log file"""
        if self.monitoring:
            return

        self.log_file_path = log_file_path
        self.monitoring = True

        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitor_log_file,
            daemon=True,
            name="LogFileMonitor"
        )
        self.monitor_thread.start()
        print(f"[DEBUG] Started log file monitoring: {log_file_path}")

    def stop_monitoring(self):
        """Stop monitoring the log file"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        print("[DEBUG] Stopped log file monitoring")

    def _monitor_log_file(self):
        """Monitor log file for ERROR entries - reads entire file and tracks processed lines"""
        if not self.log_file_path:
            return

        try:
            while self.monitoring:
                # Check if file exists
                if not os.path.exists(self.log_file_path):
                    time.sleep(0.5)
                    continue

                # Read entire file and process new lines
                try:
                    with open(self.log_file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    # Process only new lines since last check
                    for i, line in enumerate(lines):
                        if i >= self.last_processed_line:
                            # Check for ERROR messages
                            if ' - ERROR - ' in line:
                                self.error_terminator._check_for_error(line.strip(), "LOG_FILE")

                    # Update our position
                    self.last_processed_line = len(lines)

                except (IOError, OSError) as e:
                    print(f"[DEBUG] Temporary file read error (will retry): {e}")

                # Wait before next check
                time.sleep(0.1)

        except Exception as e:
            print(f"[DEBUG] Log file monitoring error: {e}")


class LoggingErrorHandler(logging.Handler):
    """Custom logging handler to monitor ERROR level messages"""

    def __init__(self, error_terminator):
        super().__init__()
        self.error_terminator = error_terminator

    def emit(self, record):
        """Called when a log record is emitted"""
        if record.levelname == "ERROR":
            log_message = self.format(record)
            self.error_terminator._check_for_error(log_message, "LOGGING")


class ErrorTerminator:
    """
    Monitors logging output and stdout/stderr for ERROR messages.
    Terminates the program immediately when ERROR is detected.
    """
    
    def __init__(self,
                 terminate_on_error: bool = True,
                 error_keywords: List[str] = None):
        """
        Initialize the error terminator.

        Args:
            terminate_on_error: Whether to terminate on error detection
            error_keywords: List of keywords that trigger termination (default: ["ERROR"])
        """
        self.terminate_on_error = terminate_on_error
        self.error_keywords = error_keywords or ["ERROR"]
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.is_monitoring = False
        self.detected_errors: List[str] = []
        
        # Custom logging handler
        self.log_handler = None

        # Log file monitor
        self.log_file_monitor = LogFileMonitor(self)
        
    def start_monitoring(self, log_file_path: Optional[str] = None):
        """Start monitoring stdout, stderr, logging, and log files for ERROR messages"""
        if self.is_monitoring:
            return

        self.is_monitoring = True

        # Replace stdout and stderr with monitoring versions
        sys.stdout = self._create_monitoring_stream(self.original_stdout, "STDOUT")
        sys.stderr = self._create_monitoring_stream(self.original_stderr, "STDERR")

        # Add logging handler to monitor log messages
        self.log_handler = LoggingErrorHandler(self)
        logging.getLogger().addHandler(self.log_handler)

        # Start log file monitoring if path provided
        if log_file_path and os.path.exists(log_file_path):
            self.log_file_monitor.start_monitoring(log_file_path)

        # Register cleanup on exit
        atexit.register(self.stop_monitoring)

        print("[DEBUG] Error monitoring activated - program will terminate on critical errors")
        
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

        # Stop log file monitoring
        self.log_file_monitor.stop_monitoring()

            
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
        # Skip monitoring our own debug messages to prevent recursion
        if "[DEBUG]" in text or "BO Error detected" in text or "ERROR pattern" in text or "Detected:" in text:
            return

        # Check for ERROR pattern (case-sensitive)
        for keyword in self.error_keywords:
            if keyword in text:
                error_msg = f"ERROR pattern '{keyword}' detected in {source}: {text.strip()}"
                self.detected_errors.append(error_msg)

                # Check if we're in BO process - if so, send to debug GPT instead of terminating
                global _in_bo_process
                if _in_bo_process:
                    self._handle_bo_error(text, source)
                    return

                # Only terminate if termination is enabled
                if self.terminate_on_error:
                    self._terminate_program(error_msg)
                return
                
    def _handle_bo_error(self, text: str, source: str):
        """Handle error during BO process - log but don't terminate (debug handled by training executor)"""
        print(f"[DEBUG] BO Error logged: {text.strip()[:100]}...")

        # Log the error but don't terminate (debug GPT handled directly in training executor)
        error_msg = f"BO Error in {source}: {text.strip()}"
        self.detected_errors.append(error_msg)

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

# Global flag to track if we're in BO process
_in_bo_process = False

def enable_error_termination(error_keywords: List[str] = None, log_file_path: Optional[str] = None):
    """
    Enable global error termination monitoring.

    Args:
        error_keywords: List of keywords that trigger termination (default: ["ERROR"])
        log_file_path: Path to log file to monitor (optional)
    """
    global _global_terminator

    if _global_terminator is None:
        _global_terminator = ErrorTerminator(
            terminate_on_error=True,
            error_keywords=error_keywords,
        )
        _global_terminator.start_monitoring(log_file_path)
    
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

def set_bo_process_mode(enabled: bool, log_file_path: Optional[str] = None):
    """Set whether we're currently in BO process (for error handling)"""
    global _in_bo_process, _global_terminator
    _in_bo_process = enabled
    if enabled:
        print("[DEBUG] BO process mode enabled - errors will be sent to debug GPT")
        # Start error monitoring if not already active
        if not _global_terminator or not _global_terminator.is_monitoring:
            _global_terminator = ErrorTerminator()
            _global_terminator.start_monitoring(log_file_path)
            print("[DEBUG] Error monitoring started for BO process")
    else:
        print("[DEBUG] BO process mode disabled - errors will terminate program")
        # Stop error monitoring when BO process ends
        if _global_terminator and _global_terminator.is_monitoring:
            _global_terminator.stop_monitoring()
            print("[DEBUG] Error monitoring stopped - BO process ended")

def is_in_bo_process() -> bool:
    """Check if we're currently in BO process"""
    global _in_bo_process
    return _in_bo_process

if __name__ == "__main__":
    # Test the error monitoring system
    print("Testing Error Monitoring System")

    # Enable monitoring
    enable_error_termination(
        error_keywords=["ERROR"],
    )
    
    print("This is a normal message")
    print("This is a warning message")
    
    # This should trigger termination
    print("This contains an ERROR message - program should terminate")