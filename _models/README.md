# AI Code Generation & Training Execution

GPT-5 powered system for generating complete PyTorch training functions, executing them with comprehensive error handling, and conducting AI-driven literature reviews.

## Overview

This module is the heart of the AI-enhanced pipeline, using GPT-5 to generate executable training code, handle errors intelligently, and optionally conduct research-informed model selection through literature reviews.

## Core Components

### 1. AI Code Generator (`ai_code_generator.py`)

Generates complete PyTorch training functions using GPT-5 with intelligent prompting and self-debugging capabilities.

```python
from _models.ai_code_generator import generate_training_code_for_data

# Generate training function with literature review
code_rec = generate_training_code_for_data(
    data_profile={'data_type': 'numpy_array', 'is_sequence': True, ...},
    input_shape=(1000, 2),
    num_classes=5,
    include_literature_review=True  # Optional research analysis
)

print(f"Model: {code_rec.model_name}")
print(f"Confidence: {code_rec.confidence}")
print(f"BO params: {list(code_rec.bo_config.keys())}")
```

**Key Features:**
- GPT-5 generates complete training loops (model, optimizer, loss, training logic)
- Dataset-aware prompting with context from `.env` (DATASET_NAME, DATASET_SOURCE)
- Automatic constraint handling (e.g., patch_size divisibility for Transformers)
- Post-training quantization support (8/16/32-bit)
- Self-debugging with GPT error correction
- JSON caching in `generated_training_functions/`

**Generated Code Includes:**
```python
def train_model(X_train, y_train, X_val, y_val, device, **hyperparams):
    # 1. Model architecture built from scratch
    # 2. Optimizer and loss function
    # 3. Training loop with epoch-by-epoch logging
    # 4. Quantization logic
    # 5. Return model + metrics (train_losses, val_losses, val_acc)
```

#### CodeRecommendation Class
```python
@dataclass
class CodeRecommendation:
    model_name: str          # Architecture name (e.g., "BiLSTM_Attention")
    training_code: str       # Complete executable Python code
    bo_config: Dict          # Hyperparameter search space
    confidence: float        # GPT confidence (0-1)
```

**BO Config Format:**
```python
{
    "lr": {
        "default": 0.001,
        "type": "Real",
        "low": 1e-6,
        "high": 1e-1,
        "prior": "log-uniform"
    },
    "batch_size": {
        "default": 64,
        "type": "Categorical",
        "categories": [8, 16, 32, 64, 128]
    },
    "quantization_bits": {
        "default": 32,
        "type": "Categorical",
        "categories": [8, 16, 32]
    }
}
```

#### Self-Debugging Feature
The code generator automatically debugs errors:

```python
# Automatic error correction during training
try:
    # Training fails due to hyperparameter issue
    model, metrics = train_model(...)
except Exception as e:
    # GPT analyzes error and suggests fixes
    corrections = ai_code_generator._debug_json_with_gpt(
        training_code, str(e), bo_config
    )
    # Returns one of:
    # - {"bo_config": {...}}        # Hyperparameter fix
    # - {"training_code": "..."}    # Code fix
    # - {"system_issue": "STOP_PIPELINE"}  # Unfixable error
```

**Debug Response Types:**
1. **Hyperparameter Fix**: Adjusts values in bo_config (e.g., reduce model size)
2. **Code Fix**: Provides corrected training function
3. **System Issue**: Identifies unfixable problems (CUDA OOM, missing dependencies)

### 2. Training Function Executor (`training_function_executor.py`)

Executes AI-generated training code with comprehensive error handling and validation.

```python
from _models.training_function_executor import training_executor

# Load and execute training function
training_data = training_executor.load_training_function('path/to/function.json')
model, metrics = training_executor.execute_training_function(
    training_data,
    X_train, y_train,
    X_val, y_val,
    device='cuda',
    lr=0.001,
    batch_size=64,
    quantization_bits=8
)

print(f"Storage: {metrics['model_storage_size_kb']:.1f}KB")
print(f"Validation: {metrics['model_size_validation']}")  # PASS/FAIL
```

**Key Features:**
- Dynamic code execution from JSON
- GPU/CPU auto-detection with fallback
- NumPy type conversion for PyTorch compatibility
- Hyperparameter validation and fixing
- Model storage size calculation and validation (256KB limit)
- Automatic GPT debugging on errors

#### BO_TrainingObjective Class
Objective function for Bayesian Optimization:

```python
from _models.training_function_executor import BO_TrainingObjective

# Create BO objective
objective = BO_TrainingObjective(
    training_data,
    X_subset=None,  # Uses centralized splits
    y_subset=None,
    device='cuda'
)

# Evaluate hyperparameters
value, metrics = objective({'lr': 0.001, 'batch_size': 64})
```

**Features:**
- Uses centralized data splits (prevents leakage)
- BO subset support for efficiency (configurable via `BO_SAMPLE_NUM`)
- Model size penalty for compression
- Returns objective = performance - size_penalty

#### Storage Size Calculation
```python
def calculate_model_storage_size_kb(model, quantization_bits=32,
                                    quantize_weights=False, quantize_activations=False):
    """
    Calculate actual storage size considering quantization:
    - 8-bit: 1 byte per parameter
    - 16-bit: 2 bytes per parameter
    - 32-bit: 4 bytes per parameter
    """
    total_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = quantization_bits / 8 if quantize_weights else 4
    storage_kb = (total_params * bytes_per_param * 1.1) / 1024  # 10% overhead
    return storage_kb
```

### 3. Literature Review Generator (`literature_review.py`)

Optional AI-powered research analysis to inform model architecture selection.

```python
from _models.literature_review import generate_literature_review_for_data

# Conduct literature review
review = generate_literature_review_for_data(
    data_profile={'is_sequence': True, 'sample_count': 5000, ...},
    input_shape=(1000, 2),
    num_classes=5
)

print(f"Query: {review.query}")
print(f"Recommended: {review.recommended_approaches}")
print(f"Confidence: {review.confidence}")
```

**LiteratureReview Class:**
```python
@dataclass
class LiteratureReview:
    query: str                          # Research query
    review_text: str                    # Comprehensive summary
    key_findings: List[str]             # Key insights
    recommended_approaches: List[str]   # Recommended models
    recent_papers: List[Dict]           # Paper citations
    confidence: float                   # Review confidence
    timestamp: int                      # Generation time
```

**Features:**
- GPT-5 with web search for current research
- Dataset-specific queries (uses DATASET_NAME from .env)
- Focuses on papers with similar data characteristics
- Saves reviews as JSON in `literature_reviews/`
- Can be disabled via `SKIP_LITERATURE_REVIEW=true` in .env

**Integration with Code Generation:**
```python
# Literature review informs prompt
code_rec = generate_training_code_for_data(
    data_profile, input_shape, num_classes,
    include_literature_review=True
)
# GPT prompt includes: "Recommend: {first_recommended_approach}"
```

## Usage Examples

### Basic Code Generation
```python
from _models.ai_code_generator import AICodeGenerator

generator = AICodeGenerator(api_key="your-key", model="gpt-5")

# Generate training function
code_rec = generator.generate_training_function(
    data_profile={'data_type': 'numpy_array', 'shape': (1000, 20), ...},
    input_shape=(20,),
    num_classes=3
)

# Save to JSON
filepath = generator.save_training_function(code_rec, data_profile)
print(f"Saved to: {filepath}")
```

### Execute from JSON
```python
from _models.training_function_executor import training_executor
import torch

# Load from cache
training_data = training_executor.load_training_function(
    'generated_training_functions/training_function_numpy_array_MLP_123456.json'
)

# Prepare data
X_train = torch.randn(800, 20)
y_train = torch.randint(0, 3, (800,))
X_val = torch.randn(200, 20)
y_val = torch.randint(0, 3, (200,))

# Execute training
model, metrics = training_executor.execute_training_function(
    training_data, X_train, y_train, X_val, y_val,
    device='cuda',
    lr=0.001,
    batch_size=64,
    epochs=20,
    quantization_bits=16,
    quantize_weights=True
)

print(f"Final accuracy: {metrics['val_acc'][-1]:.4f}")
print(f"Model size: {metrics['model_storage_size_kb']:.1f}KB")
```

### Literature Review with Code Generation
```python
from _models.ai_code_generator import generate_training_code_for_data

# Full pipeline with research
code_rec = generate_training_code_for_data(
    data_profile={
        'data_type': 'numpy_array',
        'is_sequence': True,
        'sample_count': 5000,
        'feature_count': 2
    },
    input_shape=(1000, 2),
    num_classes=5,
    include_literature_review=True  # Conducts research first
)

# Check what GPT recommended based on research
print(f"Generated: {code_rec.model_name}")
print(f"Confidence: {code_rec.confidence}")
```

### List Available Training Functions
```python
from _models.training_function_executor import training_executor

# Get all cached functions
functions = training_executor.list_available_training_functions()

for func in functions:
    print(f"Model: {func['model_name']}")
    print(f"Confidence: {func['confidence']}")
    print(f"BO params: {func['bo_parameters']}")
    print(f"File: {func['filepath']}\n")
```

### Custom Error Handling
```python
from _models.training_function_executor import training_executor

try:
    model, metrics = training_executor.execute_training_function(...)
except Exception as e:
    # GPT has already analyzed the error
    # Check for suggested corrections in error_monitor
    from error_monitor import _global_terminator
    if _global_terminator and _global_terminator.last_json_corrections:
        corrections = json.loads(_global_terminator.last_json_corrections)
        print(f"GPT suggested: {corrections}")
```

## Configuration

### Environment Variables
```bash
# In .env file
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-5
DATASET_NAME="MIT-BIH Arrhythmia Database"
DATASET_SOURCE="PhysioNet"
SKIP_LITERATURE_REVIEW=false
DEBUG_CHANCES=4
```

### Global Instances
```python
# Pre-configured instances
from _models.ai_code_generator import ai_code_generator
from _models.training_function_executor import training_executor
from _models.literature_review import literature_review_generator

# Use directly
code_rec = ai_code_generator.generate_training_function(...)
model, metrics = training_executor.execute_training_function(...)
review = literature_review_generator.generate_literature_review(...)
```

## Key Files

- `ai_code_generator.py` - GPT-5 code generation with self-debugging
- `training_function_executor.py` - Dynamic code execution with error handling
- `literature_review.py` - AI-powered research analysis
- `README.md` - This documentation

## Design Principles

1. **Complete Automation**: GPT generates entire training pipelines
2. **Self-Correcting**: Automatic error detection and fixing
3. **Research-Informed**: Optional literature review for better architectures
4. **Quantization-Aware**: Built-in compression for 256KB constraint
5. **Fail-Fast**: Identifies unfixable errors quickly to save costs
