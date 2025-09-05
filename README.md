# AI-Enhanced ML Pipeline

A flexible AI-enhanced machine learning pipeline that supports multiple data types and automatic model selection.

## New Architecture Features

### 🚀 Core Features
- **Universal Data Converter**: Automatically converts various data formats to PyTorch tensors
- **AI Model Selector**: Uses ChatGPT API to automatically recommend the most suitable neural network based on data characteristics
- **Dynamic Model Registry System**: Supports dynamic addition of new neural network architectures
- **Intelligent Bayesian Optimization**: Integrates AI-recommended BO process
- **Automatic Data Feature Analysis**: Intelligently analyzes data characteristics and generates detailed profiles

### 📊 Supported Data Types
- **Tabular Data**: NumPy arrays, Pandas DataFrames
- **Image Data**: 2D/3D/4D arrays, supports different channel formats
- **Sequence Data**: Regular and irregular time series
- **Custom Data**: Supports arbitrary data formats through registry

### 🤖 AI Model Recommendations
- Intelligent model selection based on data characteristics
- Supports multiple predefined model types
- Provides detailed recommendation reasons and confidence scores
- Extensible model registry system

## Quick Start

### 1. Install Dependencies
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set OpenAI API Key (Optional)
```bash
export OPENAI_API_KEY='your-api-key'
```

### 3. Basic Usage
```python
from main_new import train_and_evaluate
import numpy as np

# Prepare data
X = np.random.randn(1000, 20).astype("float32")
y = np.random.choice(["A", "B", "C"], size=1000)

# AI automatically processes data and selects model
result = train_and_evaluate(X, y, epochs=5)
print(f"Recommended model: {result['recommendation'].model_name}")
print(f"Training results: {result['metrics']}")
```

### 4. Run Demo
```python
python example_usage.py
```

## Detailed Usage Guide

### Data Conversion
```python
from adapters.universal_converter import convert_to_torch_dataset

# Automatically detect data type and convert
dataset, collate_fn, profile = convert_to_torch_dataset(data, labels)
print(f"Data profile: {profile}")
```

### AI Model Selection
```python
from models.ai_model_selector import select_model_for_data

# Select model based on data characteristics
recommendation = select_model_for_data(profile.to_dict())
print(f"Recommended model: {recommendation.model_name}")
print(f"Recommendation reason: {recommendation.reasoning}")
```

### Bayesian Optimization
```python
from bo.run_ai_enhanced_bo import run_ai_enhanced_bo

# Run AI-enhanced BO
result = run_ai_enhanced_bo(data, labels, n_trials=20)
print(f"Best parameters: {result['best_params']}")
```

### Custom Model Registration
```python
from models.dynamic_model_registry import register_model

# Register custom model
register_model("MyModel", MyModelClass, {
    "type": "custom",
    "description": "My custom model"
})
```

## Project Structure

```
ml_pipeline/
├── adapters/
│   ├── universal_converter.py    # Universal data converter
│   └── unified_adapter.py        # Original adapter (backward compatible)
├── models/
│   ├── ai_model_selector.py      # AI model selector
│   ├── dynamic_model_registry.py # Dynamic model registry system
│   ├── model_picker.py           # Original model selector (backward compatible)
│   └── ...                       # Various model implementations
├── bo/
│   ├── ai_enhanced_objective.py  # AI-enhanced objective function
│   ├── run_ai_enhanced_bo.py     # AI-enhanced BO runner
│   └── ...                       # Original BO code
├── main_new.py                   # New main process
├── example_usage.py              # Usage examples
└── requirements.txt              # Dependencies list
```

## Dependencies

- torch
- numpy
- scikit-learn
- tqdm
- requests (for OpenAI API)
- pandas (optional)
- opencv-python (optional, for image processing)
- pillow (optional, for image processing)

## Backward Compatibility

The new architecture is fully backward compatible, and the original code still works:
- `main.py` - Original main process
- `adapters/unified_adapter.py` - Original data converter
- `models/model_picker.py` - Original model selector
- `bo/objective.py` - Original objective function

## Contributing Guide

1. Add new data converters: Register in `adapters/universal_converter.py`
2. Add new models: Register in `models/dynamic_model_registry.py`
3. Extend AI recommendations: Add new model types in `models/ai_model_selector.py`

## License

MIT License

