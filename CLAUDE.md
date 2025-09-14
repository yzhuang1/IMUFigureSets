# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-enhanced machine learning pipeline that automatically converts various data formats to PyTorch tensors, uses OpenAI's GPT models to generate complete training functions as executable code, and performs Bayesian optimization with AI-generated model architectures. The pipeline features iterative AI evaluation and feedback loops for automated machine learning.

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
2. **AI Code Generator** (`models/ai_code_generator.py`) - Uses GPT to generate complete training functions as executable code
3. **Training Function Executor** (`models/training_function_executor.py`) - Executes AI-generated training code with error handling
4. **Code Generation Pipeline Orchestrator** (`evaluation/code_generation_pipeline_orchestrator.py`) - Manages ML pipeline with AI-generated training functions
5. **Bayesian Optimization** (`bo/run_bo.py`) - Proper BO implementation using scikit-optimize with Random Forest surrogate model

### Key Components

**Data Conversion:**
- `convert_to_torch_dataset()` - Main conversion function
- `DataProfile` class - Analyzes and describes data characteristics
- Supports tabular, image, sequence, and custom data types

**AI Code Generation:**
- `generate_training_code_for_data()` - GPT-powered complete training function generation
- `CodeRecommendation` class - Contains generated code, hyperparameters, and reasoning
- `AICodeGenerator` class - Manages code generation with template-based prompting
- Configurable through prompts in `prompts/` directory

**Training Function Execution:**
- `training_executor` - Executes AI-generated training functions with comprehensive error handling
- `BO_TrainingObjective` - Bayesian optimization objective function for hyperparameter tuning
- Dynamic model instantiation from generated code
- Automatic GPU/CPU device selection and memory management

## Entry Points

### Main Pipeline
- `main.py` - AI-enhanced pipeline with code generation → BO → evaluation → feedback loop
- `train_with_iterative_selection()` - Complete training workflow with iterative AI code generation
- `CodeGenerationPipelineOrchestrator` - Orchestrates ML pipeline with AI-generated training functions

### Bayesian Optimization
- `bo/run_bo.py` - Proper BO implementation using scikit-optimize with Random Forest surrogate model
- Template-aware search spaces for different model architectures
- Expected Improvement acquisition function with comprehensive hyperparameter spaces


## Configuration Management

Configuration is handled through `config.py`:
- Loads from environment variables and `.env` file
- `config.is_openai_configured()` - Check API key status
- OpenAI model defaults to GPT-4

**API Call Limits (prevents infinite loops and controls costs):**
- `MAX_MODEL_ATTEMPTS=4` - Maximum model architectures to try (default: 4)
- `MAX_BO_TRIALS=10` - Maximum Bayesian Optimization trials (default: 10)
- `MAX_EVAL_RETRIES=2` - Maximum evaluation retries (default: 2)

See `API_LIMITS.md` for detailed cost estimates and configuration guide.

## Extending the System

### Adding New Data Converters
Register in `UniversalConverter` class:
```python
def _convert_new_format(self, data, labels=None):
    # Implementation
    return dataset, data_profile
```

### Adding New Training Templates
Create new templates in the `AICodeGenerator`:
```python
# Add new template to TRAINING_TEMPLATES dictionary
"NewArchitecture": {
    "template": "your_template_code_here",
    "description": "Description of the new architecture",
    "bo_parameters": ["param1", "param2"]
}
```

### Customizing AI Prompts
Edit templates in `prompts/` directory:
- `system_prompt.txt` - System behavior
- `model_selection_prompt.txt` - Model recommendation instructions

## Project Structure

- `adapters/` - Data conversion utilities
- `models/` - AI code generation and training function execution
  - `ai_code_generator.py` - GPT-powered training function generation
  - `training_function_executor.py` - Training function execution with error handling
- `bo/` - Bayesian optimization with scikit-optimize
  - `run_bo.py` - Proper BO implementation with Random Forest surrogate
- `prompts/` - AI prompt templates for code generation
- `evaluation/` - Model evaluation and pipeline orchestration
  - `code_generation_pipeline_orchestrator.py` - Main pipeline orchestrator
  - `evaluate.py` - Model evaluation utilities
- `generated_training_functions/` - AI-generated training function cache (JSON files)
- `charts/` - Bayesian optimization visualization outputs
- `logs/` - Training and execution logs
- `config.py` - Configuration management with API limits
- `requirements.txt` - Python dependencies
- `visualization.py` - BO results visualization and charting

## Important Notes

- The project uses PyTorch as the primary ML framework
- OpenAI API integration requires valid API key for AI code generation features
- AI-generated training functions are cached in `generated_training_functions/` as JSON files
- Bayesian optimization uses scikit-optimize with Random Forest surrogate model and Expected Improvement acquisition
- Comprehensive logging is configured at INFO level with timestamped log files in `logs/`
- Visualization outputs (charts and BO summaries) are automatically generated in `charts/` directory
- API call limits prevent infinite loops and control costs - configurable via `.env` file
- No formal testing framework is configured - testing is done through example scripts and pipeline execution