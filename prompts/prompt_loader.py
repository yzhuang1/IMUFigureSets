"""
Prompt Loader Utility
Loads and formats prompts from template files
"""

import os
from typing import Dict, Any
import json


class PromptLoader:
    """Utility class for loading and formatting prompts"""
    
    def __init__(self, prompts_dir: str = None):
        """
        Initialize prompt loader
        
        Args:
            prompts_dir: Directory containing prompt templates
        """
        if prompts_dir is None:
            # Get the directory of this file and use prompts subdirectory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_dir = current_dir
        
        self.prompts_dir = prompts_dir
    
    def load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt template from file
        
        Args:
            prompt_name: Name of the prompt file (without .txt extension)
        
        Returns:
            str: Prompt template content
        """
        prompt_file = os.path.join(self.prompts_dir, f"{prompt_name}.txt")
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        except Exception as e:
            raise Exception(f"Error loading prompt {prompt_name}: {e}")
    
    def format_model_selection_prompt(self, data_profile: Dict[str, Any]) -> str:
        """
        Load and format the model selection prompt
        
        Args:
            data_profile: Data profile dictionary
        
        Returns:
            str: Formatted prompt
        """
        template = self.load_prompt("model_selection_prompt")
        
        # Format the template with data profile values
        formatted_prompt = template.format(
            data_profile_json=json.dumps(data_profile, indent=2, ensure_ascii=False),
            data_type=data_profile.get('data_type', 'unknown'),
            data_shape=data_profile.get('shape', 'unknown'),
            sample_count=data_profile.get('sample_count', 0),
            feature_count=data_profile.get('feature_count', 0),
            has_labels=data_profile.get('has_labels', False),
            label_count=data_profile.get('label_count', 0),
            is_sequence=data_profile.get('is_sequence', False),
            is_image=data_profile.get('is_image', False),
            is_tabular=data_profile.get('is_tabular', False)
        )
        
        return formatted_prompt
    
    def load_system_prompt(self) -> str:
        """
        Load the system prompt
        
        Returns:
            str: System prompt content
        """
        return self.load_prompt("system_prompt")


# Global prompt loader instance
prompt_loader = PromptLoader()
