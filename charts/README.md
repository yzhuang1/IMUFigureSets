# Visualization Charts

Auto-generated charts and visualizations from Bayesian optimization and training results.

## Overview

This directory contains timestamped visualization outputs showing BO convergence, parameter importance, and training curves.

## Directory Structure

```
charts/
├── {Timestamp}_BO_{ModelName}/
│   ├── convergence_plot.png
│   ├── parameter_importance.png
│   ├── training_curves.png
│   └── bo_summary.txt
└── ...
```

**Example:**
```
20250102_143022_BO_BiLSTM_Attention/
├── convergence_plot.png
├── parameter_importance.png
├── training_curves.png
└── bo_summary.txt
```

## Chart Types

### 1. Convergence Plot
Shows BO objective value over trials:
- X-axis: Trial number
- Y-axis: Objective value (validation metric)
- Highlights best trial found
- Shows convergence trend

### 2. Parameter Importance
Visualizes hyperparameter impact:
- Bar chart of parameter influence
- Based on Random Forest feature importance
- Identifies critical hyperparameters

### 3. Training Curves
Model training progress:
- Training loss over epochs
- Validation loss over epochs
- Validation accuracy over epochs
- Helps identify overfitting/underfitting

### 4. BO Summary (Text)
Textual summary including:
- Best hyperparameters found
- Best objective value
- Total trials run
- Convergence status

## Automatic Generation

Charts are auto-created by the pipeline orchestrator:

```python
from visualization import generate_bo_charts, create_charts_folder

# Create charts directory
charts_dir = create_charts_folder(model_name="BiLSTM_Attention")

# Generate all visualizations
generate_bo_charts(
    bo_history=bo_results['history'],
    output_dir=charts_dir,
    model_name="BiLSTM_Attention"
)
```

## Manual Chart Generation

```python
import matplotlib.pyplot as plt
from visualization import plot_convergence, plot_parameter_importance

# Load BO results
bo_results = {...}  # From BO run

# Create custom charts
plot_convergence(bo_results['history'], save_path='my_convergence.png')
plot_parameter_importance(
    bo_results['optimizer'],
    param_names=['lr', 'batch_size', 'hidden_size'],
    save_path='my_importance.png'
)
```

## Accessing Charts

Charts are organized chronologically (newest at bottom when sorted):

```bash
# List all chart directories
ls -lt charts/

# View specific run
open charts/20250102_143022_BO_BiLSTM_Attention/convergence_plot.png
```

## Integration

Charts are automatically linked to:
- BO results in pipeline output
- Log files (chart path logged)
- Training function JSON (via timestamp)

## Customization

Modify `visualization.py` to customize:
- Chart styles and colors
- Figure sizes and DPI
- Additional plot types
- Export formats (PNG, PDF, SVG)

## Cleanup

```bash
# Remove old charts (>30 days)
find charts -type d -mtime +30 -exec rm -rf {} +
```
