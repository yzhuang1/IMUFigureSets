# Logs

This folder contains training and execution logs from the ML pipeline.

## Contents

- **Timestamped Log Files** - Comprehensive logging of pipeline execution
- **Training Logs** - Model training progress and metrics
- **Error Logs** - Debugging information and error traces
- **Execution Logs** - Step-by-step pipeline progress

## Structure

Log files are named with pattern: `YYYY-MM-DD_HH-MM-SS.log`

## Configuration

- Logging is configured at INFO level
- Comprehensive logging covers all pipeline stages
- Automatic log rotation and timestamping
- Configured through `logging_config.py`

## Purpose

- Debugging and troubleshooting pipeline issues
- Monitoring training progress and performance
- Audit trail of pipeline execution
- Performance analysis and optimization