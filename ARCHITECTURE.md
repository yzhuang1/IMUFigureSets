# AI-Enhanced Machine Learning Pipeline Architecture

## Overall Architecture

```
Input Data (Any Format)
    ↓
Universal Data Converter
    ↓
Data Feature Analysis (Data Profile)
    ↓
AI Model Selector
    ↓
Dynamic Model Registry System
    ↓
Model Builder
    ↓
Training & Evaluation
    ↓
Bayesian Optimization
    ↓
Output Results
```

## Core Components

### 1. Universal Data Converter (`adapters/universal_converter.py`)

**Function**: Automatically converts various data formats to PyTorch tensor format

**Features**:
- Automatic data type detection (tabular, image, sequence, etc.)
- Supports multiple input formats (NumPy, Pandas, lists, etc.)
- Intelligent data preprocessing and standardization
- Generates detailed data feature profiles

**Main Classes**:
- `DataProfile`: Data feature description class
- `UniversalDataset`: Universal dataset class
- `UniversalConverter`: Main data converter class

### 2. AI Model Selector (`models/ai_model_selector.py`)

**Function**: Uses ChatGPT API to automatically recommend the most suitable neural network based on data characteristics

**Features**:
- Intelligent recommendations based on data characteristics
- Supports multiple predefined model types
- Provides detailed recommendation reasons and confidence scores
- Configurable API calls

**Main Classes**:
- `ModelRecommendation`: Model recommendation result class
- `AIModelSelector`: Main AI model selector class

### 3. Dynamic Model Registry System (`models/dynamic_model_registry.py`)

**Function**: Supports dynamic registration and creation of neural network models

**Features**:
- Dynamic model registration and discovery
- Model factory pattern
- Parameter validation and automatic building
- Metadata management

**Main Classes**:
- `ModelRegistry`: Model registry
- `ModelBuilder`: Model builder

### 4. AI-Enhanced Objective Function (`bo/ai_enhanced_objective.py`)

**Function**: Integrates AI-recommended Bayesian optimization objective function

**Features**:
- Automatic data preprocessing
- AI model recommendation integration
- Intelligent hyperparameter setting
- Detailed evaluation metrics

**Main Classes**:
- `AIEnhancedObjective`: AI-enhanced objective function class

### 5. Main Process (`main_new.py`)

**Function**: Integrates all components into a complete machine learning workflow

**Features**:
- One-click data processing and model selection
- Automatic training and evaluation
- Supports multiple data types
- Detailed logging and result output

## Data Flow

### 1. Data Input Stage
```
Raw Data → Data Type Detection → Data Feature Analysis → Data Conversion → PyTorch Dataset
```

### 2. Model Selection Stage
```
Data Features → AI Analysis → Model Recommendation → Model Building → Model Instance
```

### 3. Training Optimization Stage
```
Model + Data → Hyperparameter Optimization → Model Training → Performance Evaluation → Result Output
```

## Extensibility Design

### 1. Data Converter Extension
- Register new converters in `UniversalConverter`
- Implement custom `_convert_*` methods
- Support new data formats and preprocessing logic

### 2. Model Registry Extension
- Use `register_model()` to register new models
- Implement model factory functions
- Add model metadata

### 3. AI Recommendation Extension
- Add new model types in `AIModelSelector`
- Update recommendation prompt templates
- Extend model parameter mapping

## Backward Compatibility

The new architecture is fully backward compatible with original code:
- Preserves all original API interfaces
- Original files remain usable
- Supports gradual migration

## Usage Examples

### Basic Usage
```python
from main_new import train_and_evaluate

# Automatically process data and select model
result = train_and_evaluate(data, labels)
```

### Advanced Usage
```python
from adapters.universal_converter import convert_to_torch_dataset
from models.ai_model_selector import select_model_for_data
from models.dynamic_model_registry import build_model_from_recommendation

# 1. Convert data
dataset, collate_fn, profile = convert_to_torch_dataset(data, labels)

# 2. AI model selection
recommendation = select_model_for_data(profile.to_dict())

# 3. Build model
model = build_model_from_recommendation(recommendation, input_shape, num_classes)
```

### Bayesian Optimization
```python
from bo.run_ai_enhanced_bo import run_ai_enhanced_bo

# Run AI-enhanced BO
result = run_ai_enhanced_bo(data, labels, n_trials=20)
```

## Configuration Options

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key (for AI model selection)

### Parameter Configuration
- Data preprocessing parameters (standardization, normalization, etc.)
- Model hyperparameters (learning rate, hidden layer size, etc.)
- BO optimization parameters (number of trials, search space, etc.)

## Performance Considerations

### 1. Data Conversion Optimization
- Lazy loading and memory optimization
- Batch processing support
- Parallel data preprocessing

### 2. Model Selection Optimization
- API call caching
- Local model recommendation fallback
- Asynchronous processing support

### 3. Training Optimization
- GPU acceleration support
- Mixed precision training
- Distributed training preparation

## Error Handling

### 1. Data Conversion Errors
- Automatic fallback to default converter
- Detailed error logging
- Data validation and repair

### 2. AI Recommendation Errors
- Local default recommendation fallback
- API call retry mechanism
- Recommendation result validation

### 3. Model Building Errors
- Parameter validation and repair
- Model compatibility checking
- Automatic parameter adjustment

## Monitoring and Logging

### 1. Logging System
- Hierarchical logging
- Structured log output
- Performance metrics tracking

### 2. Monitoring Metrics
- Data conversion performance
- AI recommendation accuracy
- Model training effectiveness
- BO optimization progress

## Future Extensions

### 1. Planned Features
- More data format support
- Richer model types
- Automated hyperparameter tuning
- Model interpretability analysis

### 2. Technical Improvements
- Smarter AI recommendations
- More efficient BO algorithms
- Better error recovery
- Stronger extensibility
