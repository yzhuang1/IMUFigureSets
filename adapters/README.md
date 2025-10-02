# Data Adapters

Universal data conversion system that automatically detects and transforms various data formats into PyTorch-compatible tensors with intelligent data profiling.

## Overview

The adapters module provides a unified interface for converting diverse data sources (NumPy arrays, Pandas DataFrames, sequences, images) into PyTorch datasets with automatic type detection, standardization, and profiling.

## Core Components

### 1. UniversalConverter
Main converter class with automatic format detection and registration system.

```python
from adapters.universal_converter import convert_to_torch_dataset

# Automatic conversion with profiling
dataset, collate_fn, profile = convert_to_torch_dataset(data, labels)
```

**Supported Formats:**
- NumPy arrays (1D-4D)
- Pandas DataFrames
- PyTorch tensors
- Python lists (sequences)
- Custom formats via registration

### 2. DataProfile
Analyzes and describes data characteristics for AI model selection.

```python
from adapters.universal_converter import analyze_data_profile

profile = analyze_data_profile(X, y)
print(f"Type: {profile.data_type}")           # numpy_array, pandas_dataframe, etc.
print(f"Shape: {profile.shape}")              # (samples, features)
print(f"Is sequence: {profile.is_sequence}")  # True/False
print(f"Is image: {profile.is_image}")        # True/False
print(f"Is tabular: {profile.is_tabular}")    # True/False
print(f"Classes: {profile.label_count}")      # Number of unique labels
```

**Profile Attributes:**
- `data_type`: Format identifier (numpy_array, pandas_dataframe, sequence_list, etc.)
- `shape`: Data dimensions
- `dtype`: Data type (float32, int64, etc.)
- `sample_count`: Number of samples
- `feature_count`: Number of features
- `is_sequence`: Boolean for time series data
- `is_image`: Boolean for image data
- `is_tabular`: Boolean for structured data
- `label_count`: Number of unique classes
- `sequence_lengths`: List of sequence lengths (if applicable)
- `channels/height/width`: Image dimensions (if applicable)

### 3. UniversalDataset
PyTorch Dataset wrapper supporting standardization and sequence handling.

```python
from adapters.universal_converter import UniversalDataset

# Create dataset with standardization
dataset = UniversalDataset(
    data, labels, profile,
    standardize=True,  # Enable z-score normalization
    standardization_stats={'mean': train_mean, 'std': train_std}  # Use train stats
)
```

**Features:**
- Automatic label encoding (string → integer)
- Optional standardization (z-score normalization)
- Sequence padding for variable-length data
- Preserves standardization stats to prevent data leakage

### 4. Collate Functions
Custom collate functions for DataLoader to handle sequences.

```python
from adapters.universal_converter import universal_collate_fn
from torch.utils.data import DataLoader

# Create DataLoader with sequence support
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_fn,  # Handles variable-length sequences
    shuffle=True
)
```

## Usage Examples

### Basic Conversion
```python
import numpy as np
from adapters.universal_converter import convert_to_torch_dataset

# Tabular data
X = np.random.randn(1000, 20).astype('float32')
y = np.random.choice([0, 1, 2], size=1000)

dataset, collate_fn, profile = convert_to_torch_dataset(X, y)
print(f"Converted {profile.sample_count} samples with {profile.feature_count} features")
```

### Sequence Data
```python
# Variable-length sequences
sequences = [
    np.random.randn(100, 5).astype('float32'),  # 100 timesteps, 5 features
    np.random.randn(150, 5).astype('float32'),  # 150 timesteps, 5 features
    np.random.randn(80, 5).astype('float32')    # 80 timesteps, 5 features
]
labels = [0, 1, 0]

dataset, collate_fn, profile = convert_to_torch_dataset(sequences, labels)
print(f"Max sequence length: {max(profile.sequence_lengths)}")

# Use collate_fn in DataLoader for proper padding
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
```

### Image Data
```python
# Image batch (N, H, W, C) or (N, C, H, W)
images = np.random.randint(0, 255, (100, 32, 32, 3), dtype=np.uint8)
labels = np.random.choice([0, 1], size=100)

dataset, collate_fn, profile = convert_to_torch_dataset(images, labels)
print(f"Image size: {profile.height}x{profile.width}, Channels: {profile.channels}")
```

### Pandas DataFrames
```python
import pandas as pd

df = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'feature3': np.random.choice(['A', 'B', 'C'], 1000)
})
labels = np.random.choice([0, 1], size=1000)

dataset, collate_fn, profile = convert_to_torch_dataset(df, labels)
print(f"DataFrame columns: {profile.metadata['columns']}")
```

### Standardization (Prevents Data Leakage)
```python
# Train set: compute and apply standardization
train_dataset = UniversalDataset(
    X_train, y_train, profile,
    standardize=True,
    standardization_stats=None  # Computes from training data
)

# Val/Test sets: use training stats
val_dataset = UniversalDataset(
    X_val, y_val, profile,
    standardize=True,
    standardization_stats=train_dataset.standardization_stats  # Reuse train stats
)
```

### Custom Data Converter
```python
from adapters.universal_converter import universal_converter

def convert_custom_format(data, labels=None, **kwargs):
    # Your conversion logic
    profile = analyze_data_profile(data, labels)
    dataset = UniversalDataset(data, labels, profile, **kwargs)
    return dataset, None, profile

# Register custom converter
universal_converter.register_converter("my_custom_type", convert_custom_format)

# Use it
dataset, collate_fn, profile = convert_to_torch_dataset(
    custom_data, labels, data_type="my_custom_type"
)
```

## Data Type Detection

The converter automatically detects data types based on:

1. **Tabular Data** (2D arrays):
   - NumPy arrays with shape (N, F)
   - Pandas DataFrames
   - Sets `is_tabular=True`

2. **Sequence Data** (3D arrays or lists):
   - Shape (N, T, F) - N samples, T timesteps, F features
   - Variable-length sequences in lists
   - Sets `is_sequence=True`
   - Provides `sequence_lengths` list

3. **Image Data** (3D/4D arrays):
   - Shape (H, W, C) or (C, H, W) for single images
   - Shape (N, H, W, C) or (N, C, H, W) for batches
   - Detects RGB (3 channels) or grayscale (1 channel)
   - Sets `is_image=True`

## Integration with Pipeline

The data profile is used throughout the pipeline:

1. **AI Code Generator**: Uses profile to select appropriate model architectures
   ```python
   # GPT uses profile to determine model type
   if profile.is_sequence:
       model = LSTM/Transformer
   elif profile.is_image:
       model = CNN/ResNet
   else:
       model = MLP
   ```

2. **Training Executor**: Validates input shapes match generated code
3. **Visualization**: Adapts charts based on data characteristics

## Key Files

- `universal_converter.py` - Main conversion logic and classes
- `README.md` - This documentation

## Design Principles

1. **Automatic Detection**: No manual format specification required
2. **Unified Interface**: Single function for all conversions
3. **Extensibility**: Easy to add custom converters
4. **Data Integrity**: Standardization stats prevent leakage
5. **Sequence Support**: Proper handling of variable-length data