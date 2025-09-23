# Charts

This folder contains visualization outputs from Bayesian optimization and model training results.

## Contents

- **BO Results Visualizations** - Automatically generated charts showing optimization progress
- **Model Performance Plots** - Training metrics and validation curves
- **Hyperparameter Analysis** - Visual analysis of parameter importance and convergence

## Structure

Charts are organized chronologically by timestamp:
- `{Timestamp}_BO_{ModelName}/` - Individual BO run results (newest at bottom when sorted)
- Contains convergence plots, parameter importance charts, and optimization summaries

## Generation

Charts are automatically created by the `visualization.py` module during pipeline execution and saved here for review and analysis.