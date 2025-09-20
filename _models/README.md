# Models (_models)

This folder contains the core AI code generation and training function execution components.

## Key Components

- **`ai_code_generator.py`** - GPT-powered training function generation using OpenAI API
- **`training_function_executor.py`** - Training function execution with comprehensive error handling
- **AICodeGenerator** class - Manages code generation with template-based prompting
- **CodeRecommendation** class - Contains generated code, hyperparameters, and reasoning

## Functionality

- **AI Code Generation**: Uses GPT models to generate complete training functions as executable code
- **Training Execution**: Executes AI-generated training code with proper error handling
- **Dynamic Model Instantiation**: Creates models from generated code at runtime
- **GPU/CPU Management**: Automatic device selection and memory management
- **Template Management**: Configurable training templates for different architectures

## Integration

- Works with prompts from `prompts/` directory
- Integrates with Bayesian optimization for hyperparameter tuning
- Supports caching of generated functions in `generated_training_functions/`

This is the core module that enables AI-powered automatic machine learning pipeline generation.