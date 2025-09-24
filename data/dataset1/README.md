# Data

This folder contains datasets used for training and evaluation.

## Structure

- **`mitdb/`** - MIT-BIH Arrhythmia Database for ECG analysis
- Raw data files in various formats (CSV, images, sequences, etc.)
- Data is processed by the universal converter in the `adapters/` module

## Usage

The pipeline automatically detects and converts data from this folder using the Universal Data Converter, which:
- Analyzes data characteristics
- Creates appropriate DataProfile objects
- Converts to PyTorch tensor format
- Handles tabular, image, sequence, and custom data types

Place your datasets here for automatic processing by the ML pipeline.