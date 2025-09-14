# AI-Enhanced ML Pipeline

An AI-powered machine learning pipeline that generates complete training functions as executable code, performs Bayesian optimization, and provides comprehensive training and evaluation with iterative feedback loops.

## New Architecture Features

### 🚀 Core Features
- **Universal Data Converter**: Automatically converts various data formats to PyTorch tensors
- **AI Code Generator**: Uses GPT to generate complete training functions as executable code
- **Training Function Executor**: Executes AI-generated training code with comprehensive error handling
- **Bayesian Optimization**: Proper BO implementation using scikit-optimize with Random Forest surrogate model
- **Comprehensive Training & Evaluation**: Full train/validation/test pipeline with performance analysis
- **Iterative AI Feedback Loop**: AI-enhanced pipeline with code generation → BO → evaluation → feedback

### 📊 Supported Data Types
- **Tabular Data**: NumPy arrays, Pandas DataFrames
- **Image Data**: 2D/3D/4D arrays, supports different channel formats
- **Sequence Data**: Regular and irregular time series
- **Custom Data**: Supports arbitrary data formats through registry

### 🤖 AI Training & Evaluation
- **Complete Training Functions**: AI generates full training loops with models, optimizers, and validation
- **Automated Hyperparameter Optimization**: Bayesian optimization with template-aware search spaces
- **Comprehensive Evaluation**: Accuracy, F1-score, with configurable performance thresholds
- **Performance Analysis**: Automatic model comparison and best model selection
- **Training Function Caching**: Generated functions saved as JSON for reuse and analysis

## Quick Start

### 1. Install Dependencies
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set OpenAI API Key
```bash
# Option 1: Use the setup script (recommended)
python setup_api_key.py

# Option 2: Set environment variable
export OPENAI_API_KEY='your-api-key'

# Option 3: Create .env file
echo "OPENAI_API_KEY=your-api-key" > .env
```

### 3. Basic Usage
```python
from main import train_with_iterative_selection
import numpy as np

# Prepare data
X = np.random.randn(1000, 20).astype("float32")
y = np.random.choice([0, 1, 2], size=1000)  # Integer labels for classification

# AI automatically generates training functions, optimizes, and evaluates
best_model, results = train_with_iterative_selection(X, y, epochs=10, max_model_attempts=3)
print(f"Best model: {results['model_name']}")
print(f"Final metrics: {results['final_metrics']}")
print(f"BO best score: {results['bo_best_score']}")
```

### 4. Run Main Pipeline
```python
python main.py
```

## Detailed Usage Guide

### Data Conversion
```python
from adapters.universal_converter import convert_to_torch_dataset

# Automatically detect data type and convert
dataset, collate_fn, profile = convert_to_torch_dataset(data, labels)
print(f"Data profile: {profile}")
```

### AI Code Generation
```python
from models.ai_code_generator import generate_training_code_for_data

# Generate complete training function
code_rec = generate_training_code_for_data(data_profile, input_shape, num_classes)
print(f"Generated model: {code_rec.model_name}")
print(f"BO parameters: {code_rec.bo_parameters}")
print(f"Reasoning: {code_rec.reasoning}")
```

### Training Function Execution
```python
from models.training_function_executor import training_executor

# Execute AI-generated training function
model, metrics = training_executor.execute_training_function(
    training_data, X_train, y_train, X_val, y_val, device='cpu', **hyperparams
)
print(f"Training metrics: {metrics}")
```

### Bayesian Optimization
```python
from bo.run_bo import run_bo

# Run proper BO with scikit-optimize
result = run_bo(X, y, training_data, n_trials=10, template_name="CustomLSTMClassifier")
print(f"Best parameters: {result['best_params']}")
print(f"Best score: {result['best_value']}")
```

### Pipeline Orchestration
```python
from evaluation.code_generation_pipeline_orchestrator import CodeGenerationPipelineOrchestrator

# Run complete pipeline with iterative AI feedback
orchestrator = CodeGenerationPipelineOrchestrator(data_profile, max_model_attempts=3)
best_model, results = orchestrator.run_complete_pipeline(
    X, y, device='cpu', input_shape=input_shape, num_classes=num_classes
)
print(f"Pipeline history: {len(orchestrator.pipeline_history)} attempts")
print(f"Best model: {results['model_name']}")
```

### Customizing AI Prompts
```python
from prompts import prompt_loader

# Load and customize prompts
system_prompt = prompt_loader.load_system_prompt()
model_prompt = prompt_loader.format_model_selection_prompt(data_profile)

# Edit prompt templates in prompts/ directory
# - prompts/system_prompt.txt
# - prompts/model_selection_prompt.txt
```

### Configuration Management
```python
from config import config

# Check if OpenAI is configured
if config.is_openai_configured():
    print("OpenAI is ready to use")
    print(f"Model: {config.openai_model}")
else:
    print("Please configure your OpenAI API key")

# Configuration is automatically loaded from:
# 1. Environment variables
# 2. .env file
# 3. Default values
```

## Project Structure

```
ml_pipeline/
├── adapters/
│   └── universal_converter.py              # Universal data converter
├── models/
│   ├── ai_code_generator.py                # GPT-powered training function generation
│   └── training_function_executor.py       # Training function execution with error handling
├── bo/
│   └── run_bo.py                           # Proper BO implementation with scikit-optimize
├── evaluation/
│   ├── code_generation_pipeline_orchestrator.py # Main pipeline orchestrator
│   └── evaluate.py                         # Model evaluation utilities
├── prompts/
│   └── prompt_loader.py                    # Prompt loading utilities
├── generated_training_functions/           # AI-generated training function cache (JSON)
├── charts/                                 # BO visualization outputs
├── logs/                                   # Training and execution logs
├── config.py                              # Configuration management with API limits
├── setup_api_key.py                       # API key setup script
├── main.py                                # Main AI-enhanced pipeline
├── visualization.py                       # BO results visualization
└── requirements.txt                       # Dependencies list
```

## Dependencies

- torch (PyTorch for neural networks)
- numpy (numerical computing)
- pandas (data manipulation)
- scikit-learn (ML utilities and metrics)
- tqdm (progress bars)
- openai>=1.0.0 (GPT API for code generation)
- python-dotenv>=1.0.0 (environment variable management)
- scikit-optimize (Bayesian optimization)
- requests (HTTP requests)
- opencv-python (optional, for image processing)
- pillow (optional, for image processing)

## Training & Evaluation Pipeline

The system provides comprehensive training and evaluation capabilities:

### Training Flow
1. **Data Preprocessing**: Automatic train/validation/test splits with proper stratification
2. **AI Code Generation**: GPT generates complete training functions with models, loss functions, optimizers
3. **Bayesian Optimization**: Hyperparameter optimization using Random Forest surrogate model
4. **Final Training**: Training with optimized hyperparameters on full training set
5. **Comprehensive Evaluation**: Test set evaluation with accuracy and macro F1-score

### Evaluation Features
- **Multi-stage Evaluation**: Validation during training, BO performance tracking, final test evaluation
- **Performance Thresholds**: Configurable minimum acceptable performance (F1: 0.3, Accuracy: 0.4)
- **Automatic Model Selection**: Compares multiple AI-generated architectures and selects best performer
- **Detailed Logging**: Comprehensive logging with timestamped files in `logs/` directory
- **Visualization**: Automatic BO charts and performance summaries in `charts/` directory

## API Configuration

Configure API limits in `.env` file to control costs and prevent infinite loops:
```
MAX_MODEL_ATTEMPTS=4    # Maximum model architectures to try
MAX_BO_TRIALS=10        # Maximum BO trials per model
MAX_EVAL_RETRIES=2      # Maximum evaluation retries
```

## Contributing Guide

1. **Add new data converters**: Register in `adapters/universal_converter.py`
2. **Add new training templates**: Add to `TRAINING_TEMPLATES` in `models/ai_code_generator.py`
3. **Extend evaluation metrics**: Modify `evaluation/evaluate.py`
4. **Customize BO search spaces**: Update `get_search_space_for_template()` in `bo/run_bo.py`

## License

MIT License

