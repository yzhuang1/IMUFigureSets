# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-enhanced machine learning pipeline that automatically converts various data formats to PyTorch tensors, uses OpenAI's GPT models to recommend optimal neural network architectures, and supports dynamic model registration with Bayesian optimization.

## Development Setup

### Dependencies Installation
```bash
pip install -r requirements.txt
```

### OpenAI API Configuration
Required for AI model selection functionality:
```bash
# Interactive setup (recommended)
python setup_api_key.py

# Or set environment variable
export OPENAI_API_KEY='your-api-key'

# Or create .env file
echo "OPENAI_API_KEY=your-api-key" > .env
```

### Running the Pipeline
```bash
python main.py
```

## Core Architecture

### Data Processing Flow
1. **Universal Data Converter** (`adapters/universal_converter.py`) - Converts any data format to PyTorch tensors
2. **AI Model Selector** (`models/ai_model_selector.py`) - Uses GPT to recommend optimal models based on data characteristics
3. **Dynamic Model Registry** (`models/dynamic_model_registry.py`) - Manages and builds neural network models dynamically
4. **Training & Optimization** - Integrates Gaussian Process-based Bayesian optimization for hyperparameter tuning

### Key Components

**Data Conversion:**
- `convert_to_torch_dataset()` - Main conversion function
- `DataProfile` class - Analyzes and describes data characteristics
- Supports tabular, image, sequence, and custom data types

**AI Model Selection:**
- `select_model_for_data()` - GPT-powered model recommendation
- `ModelRecommendation` class - Contains model choice with reasoning
- Configurable through prompts in `prompts/` directory

**Model Building:**
- `build_model_from_recommendation()` - Creates models from AI recommendations
- `register_model()` - Add custom model architectures
- Pre-built models: `TabularMLP`, `ImageCNN`, etc.

## Entry Points

### Main Pipeline
- `main.py` - Advanced AI-enhanced pipeline with iterative model selection
- `process_data_with_ai_enhanced_evaluation()` - Full automated processing with iterative AI evaluation
- `train_with_iterative_selection()` - Complete training workflow with AI feedback loop

### Bayesian Optimization
- `bo/run_ai_enhanced_bo.py` - AI-enhanced BO runner
- `bo/ai_enhanced_objective.py` - Objective function with AI recommendations


## Configuration Management

Configuration is handled through `config.py`:
- Loads from environment variables and `.env` file
- `config.is_openai_configured()` - Check API key status
- OpenAI model defaults to GPT-4

**API Call Limits (prevents infinite loops and controls costs):**
- `MAX_MODEL_ATTEMPTS=3` - Maximum model architectures to try
- `MAX_BO_TRIALS=10` - Maximum Gaussian Process Bayesian Optimization trials
- `MAX_EVAL_RETRIES=2` - Maximum evaluation retries

See `API_LIMITS.md` for detailed cost estimates and configuration guide.

## Extending the System

### Adding New Data Converters
Register in `UniversalConverter` class:
```python
def _convert_new_format(self, data, labels=None):
    # Implementation
    return dataset, data_profile
```

### Adding New Models
```python
from models.dynamic_model_registry import register_model

register_model("MyModel", MyModelClass, {
    "type": "custom",
    "description": "Custom model description"
})
```

### Customizing AI Prompts
Edit templates in `prompts/` directory:
- `system_prompt.txt` - System behavior
- `model_selection_prompt.txt` - Model recommendation instructions

## Project Structure

- `adapters/` - Data conversion utilities
- `models/` - Model definitions and AI selection
- `bo/` - Bayesian optimization components  
- `prompts/` - AI prompt templates
- `evaluation/` - Model evaluation utilities
- `config.py` - Configuration management
- `requirements.txt` - Python dependencies

## Important Notes

- The project uses PyTorch as the primary ML framework
- OpenAI API integration requires valid API key for AI features
- All original APIs remain backward compatible
- Logging is configured at INFO level by default
- No formal testing framework is configured - testing is done through example scripts