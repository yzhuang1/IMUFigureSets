# Bayesian Optimization (BO)

This folder contains the Bayesian optimization implementation for hyperparameter tuning using scikit-optimize.

## Key Components

- **`run_bo.py`** - Proper BO implementation using scikit-optimize with Random Forest surrogate model
- Template-aware search spaces for different model architectures
- Expected Improvement acquisition function with comprehensive hyperparameter spaces
- `BO_TrainingObjective` - Bayesian optimization objective function for hyperparameter tuning

## Features

- Uses Random Forest as surrogate model for efficient optimization
- Configurable through `MAX_BO_TRIALS` setting (default: 40 trials)
- Automatic hyperparameter space definition based on model architecture
- Integration with AI-generated training functions

This module optimizes model hyperparameters to achieve the best possible performance automatically.