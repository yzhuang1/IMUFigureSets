# Pipeline Orchestration & Evaluation

Main orchestrator for the AI-enhanced ML pipeline that coordinates code generation, Bayesian optimization, training execution, and evaluation in a single-pass, fail-fast architecture.

## Overview

The `CodeGenerationPipelineOrchestrator` manages the complete ML workflow: generating training code with GPT-5, running BO to optimize hyperparameters, executing final training, and evaluating results - all with centralized data splitting to prevent leakage.

## Core Component

### CodeGenerationPipelineOrchestrator

```python
from evaluation.code_generation_pipeline_orchestrator import CodeGenerationPipelineOrchestrator

# Initialize orchestrator
orchestrator = CodeGenerationPipelineOrchestrator(
    data_profile={'data_type': 'numpy_array', 'is_sequence': True, ...}
)

# Run complete pipeline
model, results = orchestrator.run_complete_pipeline(
    X, y,
    device='cuda',
    input_shape=(1000, 2),
    num_classes=5
)

print(f"BO Best Score: {results['bo_best_score']:.4f}")
print(f"Final Accuracy: {results['final_metrics']['acc']:.4f}")
```

## Pipeline Execution Flow

### Single-Pass Architecture
```
1. Centralized Data Splitting
   ├─ Train/Val/Test splits created once
   └─ Standardization stats computed from train only

2. AI Code Generation
   ├─ Optional literature review (GPT-5 + web search)
   └─ Generate complete training function

3. JSON Storage
   └─ Cache generated function for reuse

4. Bayesian Optimization
   ├─ Initialize from GPT search space
   ├─ BO trials with error monitoring
   └─ Real-time GPT debugging

5. Final Training Execution
   ├─ Train with best hyperparameters
   └─ Evaluate on test set

6. Performance Analysis
   └─ Return model + comprehensive metrics
```

### Key Features
- **Single Attempt**: No retry logic, fail-fast design
- **Centralized Splits**: Consistent data across all stages
- **Error Monitoring**: Real-time log parsing during BO
- **BO Termination**: Automatic stop on unfixable errors
- **Auto Visualization**: Charts generated in `charts/`

## Usage Examples

### Basic Pipeline Execution
```python
from evaluation.code_generation_pipeline_orchestrator import CodeGenerationPipelineOrchestrator
import numpy as np

# Prepare data
X = np.random.randn(1000, 20).astype('float32')
y = np.random.choice([0, 1, 2], size=1000)

# Create data profile
from adapters.universal_converter import analyze_data_profile
profile = analyze_data_profile(X, y)

# Run pipeline
orchestrator = CodeGenerationPipelineOrchestrator(profile.to_dict())
model, results = orchestrator.run_complete_pipeline(
    X, y,
    device='cuda',
    input_shape=(20,),
    num_classes=3
)
```

### Access Pipeline History
```python
# After execution
history = orchestrator.pipeline_history

for record in history:
    print(f"Attempt: {record['attempt']}")
    print(f"Model: {record['model_name']}")
    print(f"BO Score: {record['bo_best_score']:.4f}")
    print(f"Final Score: {record['performance_score']:.4f}")
```

### Integration with Main Pipeline
```python
# In main.py
from main import train_with_iterative_selection

results = train_with_iterative_selection(
    data, labels,
    device='cuda',
    epochs=20
)

# Results include:
# - model: Final trained model
# - pipeline_results: Full orchestrator output
# - final_metrics: Test set performance
# - attempt_summary: Pipeline summary
```

## Centralized Data Splitting

Prevents data leakage by creating splits once at pipeline start:

```python
# In orchestrator.run_complete_pipeline()
from data_splitting import create_consistent_splits, get_current_splits

# Create splits once
splits = create_consistent_splits(X, y, test_size=0.2, val_size=0.2)

# All components use same splits
current = get_current_splits()
X_train, y_train = current.X_train, current.y_train
X_val, y_val = current.X_val, current.y_val
X_test, y_test = current.X_test, current.y_test
```

## Error Monitoring

Real-time log monitoring during BO:

```python
from error_monitor import set_bo_process_mode

# Enable monitoring before BO
set_bo_process_mode(True, log_file_path)

# BO runs with error detection
try:
    bo_results = run_bayesian_optimization(...)
except Exception as e:
    # Error monitor has already:
    # 1. Parsed log file for errors
    # 2. Called GPT for analysis
    # 3. Stored corrections or identified system issues
    handle_error(e)

# Disable after BO
set_bo_process_mode(False)
```

## Pipeline Results Structure

```python
{
    'model_name': 'BiLSTM_Attention',
    'bo_best_params': {'lr': 0.001, 'batch_size': 64, ...},
    'bo_best_score': 0.8542,
    'final_metrics': {
        'acc': 0.8731,
        'train_losses': [0.5, 0.3, 0.2, ...],
        'val_losses': [0.6, 0.4, 0.3, ...],
        'val_acc': [0.75, 0.82, 0.87, ...]
    },
    'model_storage_size_kb': 187.3,
    'model_parameter_count': 524288,
    'pipeline_history': [{...}],
    'total_attempts': 1,
    'successful_attempts': 1
}
```

## Visualization

Auto-generated charts in `charts/`:

```python
from visualization import generate_bo_charts, create_charts_folder

# Orchestrator automatically calls:
charts_dir = create_charts_folder(model_name="BiLSTM")
generate_bo_charts(bo_history, charts_dir)

# Creates:
# - convergence_plot.png
# - parameter_importance.png
# - training_curves.png
```

## Configuration

```bash
# In .env
MAX_BO_TRIALS=40        # BO trial limit
DEBUG_CHANCES=4         # GPT debug attempts
BO_SAMPLE_NUM=5000      # BO subset size
SKIP_LITERATURE_REVIEW=false
```

## Key Files

- `code_generation_pipeline_orchestrator.py` - Main orchestrator
- `evaluate.py` - Evaluation utilities
- `README.md` - This documentation

## Design Principles

1. **Single Pass**: One attempt, fail-fast architecture
2. **Centralized Splits**: Consistent data across all stages
3. **Error Awareness**: Real-time monitoring and GPT recovery
4. **Comprehensive Metrics**: Full training history and final results
5. **Auto Visualization**: Charts generated automatically
