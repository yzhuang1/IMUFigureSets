# Adapters

This folder contains data conversion utilities that transform various data formats into PyTorch tensors.

## Key Components

- **Universal Data Converter** (`universal_converter.py`) - Main conversion function that handles any data format
- **DataProfile** class - Analyzes and describes data characteristics
- Supports tabular, image, sequence, and custom data types

## Main Function

- `convert_to_torch_dataset()` - The primary conversion function that takes raw data and converts it to PyTorch-compatible format

This module enables the pipeline to work with diverse data sources by providing a unified interface for data conversion.