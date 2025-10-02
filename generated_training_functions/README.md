# Generated Training Functions Cache

AI-generated training function storage with complete executable PyTorch code, hyperparameter search spaces, and metadata.

## Overview

This directory stores AI-generated training functions as JSON files, enabling code reuse, caching, and auditing of GPT-5 generated architectures.

## File Structure

```
training_function_{data_type}_{model_name}_{timestamp}.json
```

**Examples:**
- `training_function_numpy_array_BiLSTM_Attention_1704067200.json`
- `training_function_sequence_list_TransformerEncoder_1704070800.json`

## JSON Format

```json
{
  "model_name": "BiLSTM_Attention",
  "training_code": "def train_model(X_train, y_train, X_val, y_val, device, **hyperparams):\n    ...",
  "bo_config": {
    "lr": {"default": 0.001, "type": "Real", "low": 1e-6, "high": 1e-1, "prior": "log-uniform"},
    "hidden_size": {"default": 128, "type": "Integer", "low": 32, "high": 512},
    "quantization_bits": {"default": 32, "type": "Categorical", "categories": [8, 16, 32]}
  },
  "confidence": 0.92,
  "data_profile": {...},
  "timestamp": 1704067200,
  "metadata": {
    "generated_by": "AI Code Generator",
    "api_model": "gpt-5",
    "version": "1.0"
  }
}
```

## Usage

### Load and Execute
```python
from _models.training_function_executor import training_executor

# Load cached function
training_data = training_executor.load_training_function(
    'generated_training_functions/training_function_numpy_array_BiLSTM_1234.json'
)

# Execute
model, metrics = training_executor.execute_training_function(
    training_data, X_train, y_train, X_val, y_val, device='cuda'
)
```

### List Available Functions
```python
functions = training_executor.list_available_training_functions()

for func in functions:
    print(f"{func['model_name']}: Confidence {func['confidence']:.2f}")
    print(f"  File: {func['filepath']}")
    print(f"  BO params: {func['bo_parameters']}")
```

## Contents

Each JSON file contains:

1. **training_code**: Complete executable Python function
   - Model architecture definition
   - Optimizer and loss function
   - Training loop with logging
   - Quantization logic
   - Returns model + metrics

2. **bo_config**: Hyperparameter search space
   - Parameter types (Real, Integer, Categorical)
   - Value ranges and priors
   - Default values

3. **confidence**: GPT's confidence score (0-1)

4. **data_profile**: Original data characteristics
   - Shape, type, features
   - Sequence/image/tabular flags
   - Sample and class counts

5. **metadata**: Generation details
   - Timestamp
   - GPT model used
   - Generator version

## Benefits

- **Caching**: Avoid regenerating identical architectures
- **Reusability**: Load successful models for similar datasets
- **Auditing**: Track all AI-generated code
- **Debugging**: Inspect generated functions
- **Versioning**: Compare different architecture iterations

## Automatic Generation

Functions are automatically saved when generated:

```python
from _models.ai_code_generator import ai_code_generator

code_rec = ai_code_generator.generate_training_function(...)
filepath = ai_code_generator.save_training_function(code_rec, data_profile)
# Prints: 💾 SAVED TRAINING FUNCTION: filename.json
```

## Cleanup

Old or unsuccessful functions can be manually removed:
```bash
# Remove functions older than 30 days
find generated_training_functions -name "*.json" -mtime +30 -delete
```