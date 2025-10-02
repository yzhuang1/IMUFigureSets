# Bayesian Optimization

Proper Bayesian Optimization implementation using scikit-optimize with Random Forest surrogate model and Expected Improvement acquisition function.

## Overview

This module provides efficient hyperparameter optimization using Bayesian Optimization with a Random Forest surrogate model. It integrates seamlessly with GPT-generated search spaces and includes real-time error recovery.

## Core Components

### BayesianOptimizer Class

```python
from bo.run_bo import BayesianOptimizer

# Initialize with GPT-generated search space
optimizer = BayesianOptimizer(gpt_search_space=code_rec.bo_config)

# Suggest hyperparameters
hparams = optimizer.suggest()

# Train and observe result
objective_value, metrics = train_function(**hparams)
optimizer.observe(hparams, objective_value)

# Get best parameters
best_params, best_value = optimizer.get_best_params()
```

**Key Features:**
- Random Forest surrogate model (more robust than Gaussian Process)
- Expected Improvement acquisition function
- GPT-generated search space conversion
- Automatic initial random exploration (default: 8 points)
- Parameter type handling (Real, Integer, Categorical)

### Search Space Format

GPT generates search spaces in this format:
```python
{
    "lr": {
        "type": "Real",
        "low": 1e-6,
        "high": 1e-1,
        "prior": "log-uniform"  # or "uniform"
    },
    "batch_size": {
        "type": "Categorical",
        "categories": [8, 16, 32, 64, 128]
    },
    "hidden_size": {
        "type": "Integer",
        "low": 32,
        "high": 512
    }
}
```

Automatically converted to scikit-optimize dimensions:
```python
from skopt.space import Real, Integer, Categorical

[
    Real(1e-6, 1e-1, prior='log-uniform', name='lr'),
    Categorical([8, 16, 32, 64, 128], name='batch_size'),
    Integer(32, 512, name='hidden_size')
]
```

## Global BO Interface

### Suggest-Observe Pattern

```python
from bo.run_bo import suggest, observe, reset_optimizer_from_code_recommendation

# Initialize BO from GPT code
reset_optimizer_from_code_recommendation(code_rec)

# BO loop
for trial in range(max_trials):
    # Get next hyperparameters
    hparams = suggest()

    # Evaluate (train model)
    objective_value, metrics = train_model(**hparams)

    # Update surrogate model
    observe(hparams, objective_value)

# Best result stored in optimizer
from bo.run_bo import get_optimizer_info
info = get_optimizer_info()
print(f"Best value: {info['best_value']}")
```

### Configuration

```bash
# In .env
MAX_BO_TRIALS=40        # Maximum BO iterations
BO_SAMPLE_NUM=5000      # Subset size for efficiency
```

## Usage Examples

### Basic BO
```python
from bo.run_bo import BayesianOptimizer

search_space = {
    "lr": {"type": "Real", "low": 1e-5, "high": 1e-1, "prior": "log-uniform"},
    "batch_size": {"type": "Categorical", "categories": [16, 32, 64]},
    "dropout": {"type": "Real", "low": 0.0, "high": 0.5}
}

optimizer = BayesianOptimizer(gpt_search_space=search_space, n_initial_points=5)

for i in range(20):
    hparams = optimizer.suggest()
    value = objective_function(**hparams)
    optimizer.observe(hparams, value)

best_params, best_value = optimizer.get_best_params()
```

### With Training Executor
```python
from bo.run_bo import BayesianOptimizer
from _models.training_function_executor import BO_TrainingObjective

# Create objective function
objective = BO_TrainingObjective(training_data, device='cuda')

# Initialize BO
optimizer = BayesianOptimizer(gpt_search_space=training_data['bo_config'])

# Optimize
for trial in range(40):
    hparams = optimizer.suggest()
    value, metrics = objective(hparams)
    optimizer.observe(hparams, value)
    print(f"Trial {trial}: {value:.4f}")
```

### Convergence Monitoring
```python
info = optimizer.get_convergence_info()

print(f"Status: {info['status']}")  # 'exploring' or 'converging'
print(f"Best value: {info['best_value']}")
print(f"Recent improvement: {info['recent_avg_improvement']}")
print(f"History: {info['improvement_history']}")
```

## Model Size Penalty

The BO objective includes a penalty for oversized models:

```python
def _calculate_size_penalty(storage_size_kb):
    """Penalty for models exceeding 256KB"""
    if storage_size_kb <= 256:
        return 0.0
    else:
        excess_ratio = (storage_size_kb - 256) / 256
        penalty = min(excess_ratio * 0.5, 0.8)  # Cap at 0.8
        return penalty

# Final objective
objective_value = base_performance - size_penalty
```

This encourages BO to find smaller models without sacrificing too much performance.

## Error Recovery

BO integrates with GPT debugging:

```python
# During BO trial
try:
    value, metrics = objective(hparams)
    observe(hparams, value)
except Exception as e:
    # GPT analyzes error and suggests fix
    corrections = debug_with_gpt(error=str(e))

    if "bo_config" in corrections:
        # Apply hyperparameter fix
        apply_corrections(corrections)
    elif "system_issue" in corrections:
        # Stop BO gracefully
        raise SystemExit("Unfixable error detected")
```

## Integration with Pipeline

```python
# In code_generation_pipeline_orchestrator.py
from bo.run_bo import reset_optimizer_from_code_recommendation, suggest, observe

# Step 1: Initialize from GPT code
reset_optimizer_from_code_recommendation(code_rec)

# Step 2: BO loop with error monitoring
for trial in range(config.max_bo_trials):
    hparams = suggest()

    try:
        value, metrics = objective(hparams)
        observe(hparams, value)
    except Exception as e:
        # Error monitoring and recovery
        handle_error(e)
```

## Advanced Features

### Custom Search Spaces
```python
from skopt.space import Real, Integer, Categorical

custom_space = [
    Real(0.0001, 0.1, prior='log-uniform', name='lr'),
    Integer(1, 10, name='num_layers'),
    Categorical(['relu', 'tanh', 'sigmoid'], name='activation')
]

optimizer = BayesianOptimizer(search_space=custom_space)
```

### Parameter Validation
```python
validated_params = optimizer.validate_params(raw_params)
# Converts:
# - Floats to integers for integer parameters
# - Ensures correct types for all parameters
```

## Key Files

- `run_bo.py` - Main BO implementation
- `README.md` - This documentation

## Design Principles

1. **Proper BO**: Uses scikit-optimize (not random search)
2. **GPT Integration**: Accepts AI-generated search spaces
3. **Efficiency**: Random Forest faster than Gaussian Process
4. **Robustness**: Handles parameter constraints automatically
5. **Size-Aware**: Built-in penalty for model compression
