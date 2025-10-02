# AI-Enhanced ML Pipeline

A fully automated machine learning pipeline that uses GPT-5 to generate complete training functions as executable code, performs Bayesian optimization with Random Forest surrogate models, and provides comprehensive training and evaluation in a single-pass, fail-fast architecture.

## Core Features

### AI Code Generation
- **GPT-5 Powered Code Generation**: Generates complete PyTorch training functions with models, optimizers, loss functions, and training loops
- **Literature Review Integration**: Optional AI-powered research analysis to inform model architecture selection
- **Intelligent Prompt Engineering**: Template-based prompts with dataset context and constraint validation
- **JSON-Based Code Storage**: Generated functions cached with hyperparameters and metadata
- **Self-Debugging**: GPT automatically fixes code errors and hyperparameter incompatibilities

### Universal Data Conversion
- **Automatic Format Detection**: Supports NumPy arrays, Pandas DataFrames, PyTorch tensors, and custom formats
- **Smart Data Profiling**: Analyzes data characteristics (tabular, sequence, image) for optimal model selection
- **Sequence Handling**: Variable-length sequence support with padding and collate functions
- **Data Standardization**: Centralized statistics computation to prevent data leakage

### Bayesian Optimization
- **Scikit-Optimize Integration**: Random Forest surrogate model with Expected Improvement acquisition
- **GPT-Generated Search Spaces**: AI defines hyperparameter ranges and types (Real, Integer, Categorical)
- **Smart Constraint Handling**: Automatic validation of hyperparameter compatibility
- **Model Size Optimization**: Built-in penalty for models exceeding 256KB storage limit
- **Real-time Error Recovery**: Monitors training errors and applies GPT corrections during BO

### Training & Quantization
- **Post-Training Quantization**: Support for 8/16/32-bit quantization with configurable weight/activation compression
- **GPU Auto-Detection**: Automatic CUDA availability check with fallback to CPU
- **Model Size Validation**: Ensures models meet 256KB storage constraint
- **Centralized Data Splitting**: Consistent train/val/test splits across all experiments
- **Comprehensive Metrics**: Tracks training loss, validation loss, accuracy, F1-score across epochs

### Pipeline Orchestration
- **Single-Pass Execution**: Fail-fast architecture with no retry logic
- **Error Monitoring**: Real-time log monitoring for training failures
- **BO Process Termination**: Automatic pipeline stop on unfixable errors
- **Visualization**: Auto-generated charts for convergence, parameter importance, training curves
- **Extensive Logging**: Timestamped logs with INFO-level detail for all pipeline stages

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API
```bash
# Option 1: Interactive setup (recommended)
python setup_api_key.py

# Option 2: Create .env file
echo "OPENAI_API_KEY=your-api-key" > .env
echo "MAX_BO_TRIALS=40" >> .env
echo "DEBUG_CHANCES=4" >> .env
echo "BO_SAMPLE_NUM=5000" >> .env
```

### 3. Run the Pipeline
```bash
python main.py
```

### 4. Python API Usage
```python
from main import train_with_iterative_selection
import numpy as np

# Prepare data
X = np.random.randn(1000, 20).astype("float32")
y = np.random.choice([0, 1, 2], size=1000)

# Run complete AI-enhanced pipeline
results = train_with_iterative_selection(X, y, epochs=10, device="cuda")

print(f"Model: {results['pipeline_results']['model_name']}")
print(f"Accuracy: {results['final_metrics']['acc']:.4f}")
print(f"BO Best Score: {results['pipeline_results']['bo_best_score']:.4f}")
```

## Architecture Overview

### Pipeline Flow
```
Data → Universal Converter → AI Code Generator → BO Optimization → Training Execution → Evaluation
                                      ↓
                          Literature Review (optional)
                                      ↓
                          GPT-5 Training Function Generation
                                      ↓
                          JSON Storage & Caching
                                      ↓
                          BO with Random Forest Surrogate
                                      ↓
                          Error Monitoring & GPT Debugging
                                      ↓
                          Final Model + Metrics
```

### Key Components

#### 1. Universal Data Converter (`adapters/universal_converter.py`)
```python
from adapters.universal_converter import convert_to_torch_dataset

# Automatically detect and convert any data format
dataset, collate_fn, profile = convert_to_torch_dataset(data, labels)
print(f"Type: {profile.data_type}")
print(f"Shape: {profile.shape}")
print(f"Is sequence: {profile.is_sequence}")
```

**Features:**
- Auto-detects tabular, sequence, and image data
- Handles variable-length sequences with padding
- Computes data statistics for standardization
- Creates PyTorch Dataset and DataLoader

#### 2. AI Code Generator (`_models/ai_code_generator.py`)
```python
from _models.ai_code_generator import generate_training_code_for_data

# Generate complete training function with GPT-5
code_rec = generate_training_code_for_data(
    data_profile.to_dict(),
    input_shape=(1000, 2),
    num_classes=5,
    include_literature_review=True
)

print(f"Model: {code_rec.model_name}")
print(f"Confidence: {code_rec.confidence}")
print(f"Hyperparameters: {list(code_rec.bo_config.keys())}")
```

**Features:**
- GPT-5 generates complete PyTorch training loops
- Optional literature review for research-informed architectures
- Self-debugging with automatic error correction
- Supports quantization (8/16/32-bit)
- JSON caching in `generated_training_functions/`

#### 3. Bayesian Optimization (`bo/run_bo.py`)
```python
from bo.run_bo import BayesianOptimizer, reset_optimizer_from_code_recommendation

# Initialize BO with GPT-generated search space
reset_optimizer_from_code_recommendation(code_rec)

# Suggest hyperparameters using RF surrogate + Expected Improvement
from bo.run_bo import suggest, observe

hparams = suggest()
# ... train model with hparams ...
observe(hparams, validation_score)
```

**Features:**
- Random Forest surrogate model (scikit-optimize)
- Expected Improvement acquisition function
- GPT-defined search spaces (Real, Integer, Categorical)
- Model size penalty for compression
- Real-time error recovery

#### 4. Training Executor (`_models/training_function_executor.py`)
```python
from _models.training_function_executor import training_executor, BO_TrainingObjective

# Execute AI-generated training code
model, metrics = training_executor.execute_training_function(
    training_data, X_train, y_train, X_val, y_val,
    device='cuda', lr=0.001, batch_size=64
)

print(f"Storage: {metrics['model_storage_size_kb']:.1f}KB")
print(f"Parameters: {metrics['model_parameter_count']:,}")
```

**Features:**
- Dynamic code execution from JSON
- GPU/CPU auto-detection
- Quantization support
- Storage size validation
- Comprehensive error handling

#### 5. Pipeline Orchestrator (`evaluation/code_generation_pipeline_orchestrator.py`)
```python
from evaluation.code_generation_pipeline_orchestrator import CodeGenerationPipelineOrchestrator

# Run complete single-pass pipeline
orchestrator = CodeGenerationPipelineOrchestrator(data_profile.to_dict())
model, results = orchestrator.run_complete_pipeline(
    X, y, device='cuda', input_shape=(1000, 2), num_classes=5
)

print(f"BO Score: {results['bo_best_score']:.4f}")
print(f"Final Accuracy: {results['final_metrics']['acc']:.4f}")
```

**Features:**
- Single-pass, fail-fast execution
- Centralized data splitting
- Error monitoring with log parsing
- BO process termination on failures
- Auto-generated visualizations

## Project Structure

```
ml_pipeline/
├── adapters/                               # Data conversion utilities
│   ├── universal_converter.py              # Universal data converter with auto-detection
│   └── README.md                           # Adapter documentation
├── _models/                                # AI code generation & execution
│   ├── ai_code_generator.py                # GPT-5 training function generation
│   ├── training_function_executor.py       # Execute AI-generated training code
│   ├── literature_review.py                # AI-powered research analysis
│   └── README.md                           # Models documentation
├── bo/                                     # Bayesian optimization
│   ├── run_bo.py                           # Random Forest BO with scikit-optimize
│   └── README.md                           # BO documentation
├── evaluation/                             # Pipeline orchestration
│   ├── code_generation_pipeline_orchestrator.py # Main orchestrator
│   ├── evaluate.py                         # Evaluation utilities
│   └── README.md                           # Evaluation documentation
├── prompts/                                # AI prompt templates
│   ├── prompt_loader.py                    # Prompt loading utilities
│   ├── system_prompt.txt                   # System prompt template
│   ├── model_selection_prompt.txt          # Model selection template
│   └── README.md                           # Prompts documentation
├── generated_training_functions/           # AI-generated code cache
│   ├── *.json                              # Cached training functions
│   └── README.md                           # Cache documentation
├── charts/                                 # Visualization outputs
│   ├── *_BO_*/                             # BO convergence charts
│   └── README.md                           # Charts documentation
├── logs/                                   # Execution logs
│   ├── *.log                               # Timestamped log files
│   └── README.md                           # Logging documentation
├── trained_models/                         # Model checkpoints
│   ├── *.pth                               # Saved model weights
│   └── README.md                           # Models documentation
├── literature_reviews/                     # AI research summaries
│   ├── *.json                              # Literature review files
│   └── README.md                           # Reviews documentation
├── config.py                               # Configuration with API limits
├── data_splitting.py                       # Centralized train/val/test splits
├── error_monitor.py                        # Real-time error monitoring
├── logging_config.py                       # Centralized logging setup
├── visualization.py                        # BO charts generation
├── main.py                                 # Main pipeline entry point
├── setup_api_key.py                        # Interactive API setup
├── requirements.txt                        # Python dependencies
└── CLAUDE.md                               # Development instructions
```

## Configuration

### Environment Variables (.env)
```bash
# OpenAI API
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-5

# Dataset Context (for GPT prompts)
DATASET_NAME="MIT-BIH Arrhythmia Database"
DATASET_SOURCE="PhysioNet"

# API Limits (cost control)
MAX_BO_TRIALS=40          # Max Bayesian optimization trials
DEBUG_CHANCES=4           # Max GPT debugging attempts
BO_SAMPLE_NUM=5000        # BO subset size for efficiency

# Optional Features
SKIP_LITERATURE_REVIEW=false   # Enable/disable AI literature review
DATA_FOLDER=dataset1           # Data directory name
```

### Cost Control
The pipeline includes built-in safeguards to prevent runaway costs:
- `MAX_BO_TRIALS`: Limits total BO iterations (default: 40)
- `DEBUG_CHANCES`: Limits GPT error correction attempts (default: 4)
- `BO_SAMPLE_NUM`: Uses subset for BO efficiency (default: 5000 samples)

## Key Dependencies

```
torch>=2.0.0              # PyTorch for neural networks
numpy>=1.24.0             # Numerical computing
pandas>=2.0.0             # Data manipulation
scikit-learn>=1.3.0       # ML utilities and metrics
scikit-optimize>=0.9.0    # Bayesian optimization
openai>=1.0.0             # GPT API
python-dotenv>=1.0.0      # Environment variables
matplotlib>=3.7.0         # Visualization
tqdm>=4.65.0              # Progress bars
```

Install all dependencies: `pip install -r requirements.txt`

## Advanced Features

### Literature Review Integration
```python
# Enable AI-powered literature review before code generation
from _models.ai_code_generator import generate_training_code_for_data

code_rec = generate_training_code_for_data(
    data_profile, input_shape, num_classes,
    include_literature_review=True  # Conducts research analysis
)
```

### Custom Quantization
```python
# Configure quantization in hyperparameters
hyperparams = {
    'quantization_bits': 8,           # 8/16/32-bit precision
    'quantize_weights': True,         # Compress model weights
    'quantize_activations': False     # Keep activations full precision
}
```

### Error Monitoring & Recovery
The pipeline automatically:
1. Monitors training logs in real-time
2. Detects errors during BO trials
3. Calls GPT for error analysis and fixes
4. Applies corrections and retries or terminates gracefully

### Centralized Data Splitting
```python
from data_splitting import create_consistent_splits, get_current_splits

# Create splits once at pipeline start
splits = create_consistent_splits(X, y, test_size=0.2, val_size=0.2)

# All components use same splits (prevents data leakage)
current_splits = get_current_splits()
X_train, y_train = current_splits.X_train, current_splits.y_train
```

## Extending the Pipeline

### 1. Add Custom Data Converter
```python
# In adapters/universal_converter.py
def _convert_custom_format(self, data, labels=None, **kwargs):
    profile = analyze_data_profile(data, labels)
    dataset = UniversalDataset(data, labels, profile, **kwargs)
    return dataset, None, profile

# Register the converter
universal_converter.register_converter("custom_type", _convert_custom_format)
```

### 2. Customize AI Prompts
Edit template files in `prompts/` directory:
- `system_prompt.txt` - System behavior
- `model_selection_prompt.txt` - Model recommendations

### 3. Add Custom Metrics
```python
# In evaluation/evaluate.py
def custom_metric(y_true, y_pred):
    # Your metric calculation
    return score
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce `BO_SAMPLE_NUM` in `.env`
- Lower batch size in hyperparameters
- Use CPU instead: `device='cpu'`

**2. GPT API Errors**
- Verify `OPENAI_API_KEY` is set correctly
- Check API rate limits
- Review `logs/` for detailed error messages

**3. Model Size Exceeds 256KB**
- Enable quantization in generated code
- GPT will automatically suggest smaller architectures
- Check `model_storage_size_kb` in metrics

**4. BO Not Converging**
- Increase `MAX_BO_TRIALS`
- Check hyperparameter ranges in `bo_config`
- Review BO charts in `charts/` directory

## Citation

If you use this pipeline in your research, please cite:
```
@software{ai_ml_pipeline,
  title={AI-Enhanced ML Pipeline with GPT-5 Code Generation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/ml_pipeline}
}
```

## License

MIT License - see LICENSE file for details

