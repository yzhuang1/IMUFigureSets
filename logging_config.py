"""
Centralized logging configuration for the ML pipeline
Provides consistent logging setup across all modules
"""

import logging
import os
from datetime import datetime
from typing import Optional


class PipelineLogger:
    """Centralized logger configuration for the ML pipeline"""

    _instance = None
    _logger = None
    _log_file_path = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PipelineLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if PipelineLogger._logger is None:
            self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration with file output only"""
        import multiprocessing as mp

        # Skip logging setup in worker processes to avoid creating multiple log files
        if mp.current_process().name != 'MainProcess':
            self._logger = logging.getLogger()
            self._logger.addHandler(logging.NullHandler())
            self._log_file_path = None
            return

        log_dir = "logs"

        try:
            # Create logs directory
            os.makedirs(log_dir, exist_ok=True)

            # Generate log filename (only in main process)
            log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.log")
            self._log_file_path = os.path.join(log_dir, log_filename)

            # Clear any existing handlers
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            # Setup file handler only (no console duplication)
            file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            file_handler = logging.FileHandler(self._log_file_path, mode="w", encoding="utf-8")
            file_handler.setFormatter(file_formatter)

            # Configure root logger
            root_logger.setLevel(logging.INFO)
            root_logger.addHandler(file_handler)

            self._logger = root_logger

        except Exception as e:
            # Fallback: create a basic logger that doesn't output to console
            self._logger = logging.getLogger()
            self._logger.addHandler(logging.NullHandler())

    def get_logger(self, name: str = __name__) -> logging.Logger:
        """Get a logger for a specific module"""
        return logging.getLogger(name)

    def get_log_file_path(self) -> Optional[str]:
        """Get the current log file path"""
        return self._log_file_path

    def info_to_file(self, message: str, logger_name: str = "pipeline"):
        """Log info message to file only (not console)"""
        logger = self.get_logger(logger_name)
        logger.info(message)

    def warning_to_file(self, message: str, logger_name: str = "pipeline"):
        """Log warning message to file only (not console)"""
        logger = self.get_logger(logger_name)
        logger.warning(message)

    def error_to_file(self, message: str, logger_name: str = "pipeline"):
        """Log error message to file only (not console)"""
        logger = self.get_logger(logger_name)
        logger.error(message)


# Global instance
pipeline_logger = PipelineLogger()


def get_pipeline_logger(name: str = __name__) -> logging.Logger:
    """Get a configured pipeline logger for a module"""
    return pipeline_logger.get_logger(name)


def log_info(message: str, logger_name: str = "pipeline"):
    """Log info message to file"""
    pipeline_logger.info_to_file(message, logger_name)


def log_warning(message: str, logger_name: str = "pipeline"):
    """Log warning message to file"""
    pipeline_logger.warning_to_file(message, logger_name)


def log_error(message: str, logger_name: str = "pipeline"):
    """Log error message to file"""
    pipeline_logger.error_to_file(message, logger_name)


def get_log_file_path() -> Optional[str]:
    """Get the current log file path"""
    return pipeline_logger.get_log_file_path()