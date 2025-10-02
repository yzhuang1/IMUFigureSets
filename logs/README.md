# Execution Logs

Timestamped execution logs from the ML pipeline with comprehensive INFO-level logging.

## Overview

All pipeline execution logs are stored here with automatic timestamping and centralized configuration via `logging_config.py`.

## File Structure

```
logs/
├── YYYY-MM-DD_HH-MM-SS.log
├── 2025-01-02_14-30-22.log
├── 2025-01-02_15-45-10.log
└── ...
```

Each pipeline run creates a new log file named with the start timestamp.

## Log Format

```
YYYY-MM-DD HH:MM:SS - LEVEL - MODULE - MESSAGE
```

**Example:**
```
2025-01-02 14:30:22 - INFO - main - Starting AI-enhanced training (single attempt, fail-fast)
2025-01-02 14:30:23 - INFO - adapters.universal_converter - Data conversion: numpy_array -> torch_tensor
2025-01-02 14:30:25 - INFO - _models.ai_code_generator - AI generated training function: BiLSTM_Attention
2025-01-02 14:30:45 - INFO - bo.run_bo - BO Trial 1: Initial random exploration
2025-01-02 14:31:10 - INFO - _models.training_function_executor - Model: 524,288 parameters, 187.3KB storage
```

## Log Levels

- **INFO**: Normal pipeline operations, progress updates
- **WARNING**: Non-critical issues, fallback behaviors
- **ERROR**: Failures and exceptions with stack traces
- **DEBUG**: Detailed diagnostic information (disabled by default)

## Configuration

Centralized in `logging_config.py`:

```python
from logging_config import get_pipeline_logger, get_log_file_path

# Get logger for any module
logger = get_pipeline_logger(__name__)

# Get current log file path
log_path = get_log_file_path()
print(f"Logging to: {log_path}")
```

**Features:**
- Automatic timestamped filenames
- INFO level by default
- Console output disabled (logs to file only)
- UTF-8 encoding for special characters
- Creates `logs/` directory if missing

## Usage

### In Code
```python
from logging_config import get_pipeline_logger

logger = get_pipeline_logger(__name__)

logger.info("Pipeline started")
logger.warning("Using CPU fallback")
logger.error(f"Training failed: {error}")
```

### Viewing Logs
```bash
# View latest log
tail -f logs/$(ls -t logs/ | head -1)

# Search for errors
grep "ERROR" logs/*.log

# Filter by module
grep "_models.ai_code_generator" logs/2025-01-02_14-30-22.log
```

## What's Logged

### Data Processing
- Data conversion details
- Profile analysis results
- Standardization statistics

### AI Code Generation
- GPT API calls and responses
- Generated model names
- Confidence scores
- JSON parsing and fixes

### Bayesian Optimization
- Trial suggestions and results
- Surrogate model updates
- Best parameters found
- Convergence status

### Training Execution
- Model architecture details
- Parameter counts and storage size
- Training progress (loss, accuracy per epoch)
- GPU/CPU device selection

### Error Handling
- Exception stack traces
- GPT debugging attempts
- Applied corrections
- System issue detection

## Error Monitoring

During BO, logs are monitored in real-time:

```python
from error_monitor import set_bo_process_mode

# Enable real-time monitoring
set_bo_process_mode(True, log_file_path)

# Errors are automatically:
# 1. Detected in log file
# 2. Sent to GPT for analysis
# 3. Corrections stored for application

set_bo_process_mode(False)  # Disable after BO
```

## Cleanup

```bash
# Remove logs older than 7 days
find logs -name "*.log" -mtime +7 -delete

# Archive old logs
tar -czf logs_archive_$(date +%Y%m%d).tar.gz logs/*.log
```

## Troubleshooting

**No logs created:**
- Check `logs/` directory exists and is writable
- Verify `logging_config.py` is imported before other modules

**Logs not updating:**
- Flush handlers: `logger.handlers[0].flush()`
- Check file permissions

**Too verbose:**
- Increase log level in `logging_config.py`: `level=logging.WARNING`

**Missing timestamps:**
- Ensure format string includes `%(asctime)s`
