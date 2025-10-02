# AI Prompt Templates

Centralized prompt management system for GPT-5 code generation with template-based formatting and easy customization.

## Overview

This module provides a clean separation between prompts and code, enabling easy customization of AI behavior without modifying application logic. All prompts are stored as text files and loaded dynamically.

## Structure

```
prompts/
├── prompt_loader.py              # Utility for loading and formatting prompts
├── system_prompt.txt             # System-level behavior instructions
├── model_selection_prompt.txt    # Model recommendation template
└── README.md                     # This documentation
```

## Core Component

### PromptLoader Class

```python
from prompts import prompt_loader

# Load system prompt
system_prompt = prompt_loader.load_system_prompt()

# Load and format model selection prompt
model_prompt = prompt_loader.format_model_selection_prompt(
    data_profile={'data_type': 'numpy_array', 'is_sequence': True, ...}
)
```

## Usage Examples

### Basic Prompt Loading
```python
from prompts.prompt_loader import PromptLoader

loader = PromptLoader()

# Load any prompt template
system_prompt = loader.load_prompt('system_prompt')
model_prompt = loader.load_prompt('model_selection_prompt')
```

### Formatted Model Selection
```python
from prompts import prompt_loader

data_profile = {
    'data_type': 'numpy_array',
    'shape': (1000, 20),
    'sample_count': 1000,
    'feature_count': 20,
    'has_labels': True,
    'label_count': 3,
    'is_sequence': False,
    'is_image': False,
    'is_tabular': True
}

# Format with data profile
formatted_prompt = prompt_loader.format_model_selection_prompt(data_profile)

# Optionally exclude specific models
formatted_prompt = prompt_loader.format_model_selection_prompt(
    data_profile,
    exclude_models=['SimpleRNN', 'BasicCNN']
)
```

### Custom Prompt Addition

1. Create new template file:
```bash
# In prompts/ directory
echo "Your custom prompt template with {placeholder}" > custom_prompt.txt
```

2. Add loader method:
```python
# In prompt_loader.py
def load_custom_prompt(self, param1, param2):
    template = self.load_prompt("custom_prompt")
    return template.format(param1=param1, param2=param2)
```

3. Use it:
```python
custom_prompt = prompt_loader.load_custom_prompt("value1", "value2")
```

## Prompt Templates

### system_prompt.txt
Defines GPT's role and behavior:
- Expert ML engineer persona
- PyTorch code generation rules
- JSON response format requirements
- Error handling guidelines

### model_selection_prompt.txt
Template for architecture selection:
- Data profile placeholders: `{data_type}`, `{shape}`, etc.
- Model exclusion support
- Architecture recommendations
- Hyperparameter search space definition

## Template Placeholders

Common placeholders used in templates:

```python
{data_profile_json}    # Full data profile as JSON
{data_type}            # Type: numpy_array, pandas_dataframe, etc.
{data_shape}           # Shape tuple
{sample_count}         # Number of samples
{feature_count}        # Number of features
{has_labels}           # Boolean for supervised learning
{label_count}          # Number of classes
{is_sequence}          # Boolean for time series
{is_image}             # Boolean for image data
{is_tabular}           # Boolean for structured data
{exclusion_text}       # Excluded models list (optional)
```

## Integration with AI Code Generator

```python
# In ai_code_generator.py
from prompts import prompt_loader

def _create_prompt(self, data_profile, input_shape, num_classes):
    # Load base prompt
    system_prompt = prompt_loader.load_system_prompt()
    model_prompt = prompt_loader.format_model_selection_prompt(data_profile)

    # Combine with additional context
    full_prompt = f"{system_prompt}\n\n{model_prompt}\n\n{additional_instructions}"
    return full_prompt
```

## Customization Guide

### Modify Existing Prompts

Edit template files directly:

```bash
# Example: Customize system behavior
nano prompts/system_prompt.txt

# Add new requirements:
# - Focus on energy efficiency
# - Prefer sparse architectures
# - Emphasize explainability
```

### Add Dataset-Specific Context

Prompts automatically include dataset context from `.env`:

```python
# In prompt template
dataset_context = ""
if config.dataset_name:
    dataset_context = f"\nDataset: {config.dataset_name}\nSource: {config.dataset_source}"
```

### Version Control

Prompts are plain text files, making them:
- Easy to track changes with git
- Reviewable in pull requests
- Testable independently
- Shareable across teams

## Best Practices

1. **Use Placeholders**: Keep prompts reusable with `{placeholders}`
2. **Document Changes**: Comment why prompts were modified
3. **Test Iterations**: Verify prompt changes with sample data
4. **Version Sensitive Info**: Don't hardcode API keys or secrets
5. **Keep Focused**: Each prompt should have a single purpose

## Global Instance

```python
# Pre-configured loader available globally
from prompts import prompt_loader

# Use directly
system = prompt_loader.load_system_prompt()
model = prompt_loader.format_model_selection_prompt(profile)
```

## Key Files

- `prompt_loader.py` - Loading and formatting utilities
- `system_prompt.txt` - System behavior template
- `model_selection_prompt.txt` - Model selection template
- `README.md` - This documentation

## Design Principles

1. **Separation of Concerns**: Prompts separate from application logic
2. **Easy Customization**: Edit text files without code changes
3. **Template Reuse**: Placeholders for dynamic content
4. **Version Control**: Plain text for git tracking
5. **Maintainability**: Non-developers can modify prompts
