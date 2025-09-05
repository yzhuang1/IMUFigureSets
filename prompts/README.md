# Prompts Directory

This directory contains prompt templates and utilities for the AI-enhanced machine learning pipeline.

## Structure

- `prompt_loader.py` - Utility class for loading and formatting prompts
- `model_selection_prompt.txt` - Template for AI model selection prompts
- `system_prompt.txt` - System prompt for OpenAI API calls
- `__init__.py` - Package initialization

## Usage

### Loading Prompts Programmatically

```python
from prompts import prompt_loader

# Load system prompt
system_prompt = prompt_loader.load_system_prompt()

# Load and format model selection prompt
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

model_selection_prompt = prompt_loader.format_model_selection_prompt(data_profile)
```

### Customizing Prompts

To modify prompts, simply edit the corresponding `.txt` files:

- `system_prompt.txt` - Change the system role and behavior
- `model_selection_prompt.txt` - Modify the model selection instructions and available model types

The prompt templates use Python string formatting with placeholders like `{data_type}`, `{shape}`, etc.

### Adding New Prompts

1. Create a new `.txt` file with your prompt template
2. Add a method to `PromptLoader` class to load and format the new prompt
3. Use the new method in your code

Example:

```python
# In prompt_loader.py
def load_custom_prompt(self, param1, param2):
    template = self.load_prompt("custom_prompt")
    return template.format(param1=param1, param2=param2)
```

## Benefits

- **Maintainability**: Prompts are separated from code logic
- **Version Control**: Easy to track changes to prompts
- **Collaboration**: Non-developers can modify prompts without touching code
- **Testing**: Prompts can be tested independently
- **Reusability**: Prompts can be shared across different modules
